from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from diagnose_pcalm_dual_precision import quantize_dual
from predcircuit.pcalm import (
    ResidualMLP,
    Schedule,
    al_energy_shifted,
    constraint_residuals,
    free_init,
    gradient_cosine,
    gradient_relative_error,
    method_grad,
    zero_duals_like,
)


def run(
    model, x, y, *, budget: int, state_lr: float, dual_leak: float, interval: int, accumulator: str
):
    rho, alpha = 1.0, 0.925
    effective_lr = state_lr * x.shape[0]
    free = free_init(model, x)
    duals = zero_duals_like(constraint_residuals(model, x, free))
    acc_sat = acc_total = state_sat = state_total = 0
    finite = True

    for outer_ix in range(budget):
        variables = [z.detach().requires_grad_(True) for z in free]
        energy = al_energy_shifted(model, x, y, variables, [d.detach() for d in duals], rho=rho)
        raw_grads = torch.autograd.grad(energy, variables)
        writeback = (outer_ix + 1) % interval == 0 or outer_ix == budget - 1
        new_free = []
        for z, grad in zip(variables, raw_grads, strict=True):
            q_grad, _, _, _ = quantize_dual(grad, "fixed14_i1")
            target = z.detach() - effective_lr * q_grad
            q_acc, sat, total, _ = quantize_dual(target, accumulator)
            acc_sat += sat
            acc_total += total
            if writeback:
                q_state, sat_state, total_state, _ = quantize_dual(q_acc, "fixed15_i3")
                state_sat += sat_state
                state_total += total_state
                new_free.append(q_state)
            else:
                new_free.append(q_acc)
        free = new_free

        residuals = constraint_residuals(model, x, free)
        q_residuals = [quantize_dual(r, "fixed14_i1")[0] for r in residuals]
        duals_after = [
            quantize_dual((1.0 - dual_leak) * lam + alpha * r, "fixed12_i1")[0]
            for lam, r in zip(duals, q_residuals, strict=True)
        ]
        finite = finite and all(bool(torch.isfinite(t).all()) for t in [*free, *duals_after])
        if outer_ix == budget - 1:
            credit_duals = duals
            break
        duals = duals_after

    loss = al_energy_shifted(
        model, x, y, [z.detach() for z in free], [d.detach() for d in credit_duals], rho=rho
    )
    grads = [g.detach() for g in torch.autograd.grad(loss, tuple(model.weights))]
    return grads, {
        "finite": finite and all(bool(torch.isfinite(g).all()) for g in grads),
        "residual_total": sum(float(r.norm()) for r in constraint_residuals(model, x, free)),
        "accumulator_saturation_rate": acc_sat / acc_total if acc_total else 0.0,
        "state_writeback_saturation_rate": state_sat / state_total if state_total else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=860)
    p.add_argument("--budget", type=int, default=128)
    p.add_argument("--state-lr", type=float, default=0.25)
    p.add_argument("--dual-leak", type=float, default=0.02)
    p.add_argument("--interval", type=int, default=32)
    p.add_argument(
        "--accumulators", default="fp32,fixed24_i3,fixed20_i3,fixed18_i3,fixed16_i3,fixed15_i3"
    )
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    depth, width = 32, 64
    model_seed = a.seed + 1000 * width + depth
    data_seed = a.seed + 10_000 + 1000 * width + depth
    gen = torch.Generator().manual_seed(data_seed)
    x, y = torch.randn(4, 8, generator=gen), torch.randn(4, 4, generator=gen)
    bp_model = ResidualMLP(
        depth=depth, width=width, input_dim=8, output_dim=4, activation="relu", seed=model_seed
    )
    bp = method_grad(bp_model, x, y, Schedule("bp", budget=0), state_lr=a.state_lr, rho=1.0)
    bp_first, bp_norm = bp[0], float(bp[0].norm())
    rows = []
    for accumulator in [v for v in a.accumulators.split(",") if v]:
        model = ResidualMLP(
            depth=depth, width=width, input_dim=8, output_dim=4, activation="relu", seed=model_seed
        )
        grads, stats = run(
            model,
            x,
            y,
            budget=a.budget,
            state_lr=a.state_lr,
            dual_leak=a.dual_leak,
            interval=a.interval,
            accumulator=accumulator,
        )
        first = grads[0]
        cosine = gradient_cosine([first], [bp_first])
        ratio = float(first.norm()) / bp_norm if bp_norm else math.nan
        rel = gradient_relative_error([first], [bp_first])
        rows.append(
            {
                "seed": a.seed,
                "interval": a.interval,
                "accumulator": accumulator,
                **stats,
                "first_layer_cosine_to_bp": cosine,
                "first_layer_grad_norm_ratio_to_bp": ratio,
                "first_layer_relative_error_to_bp": rel,
                "useful_first_layer_credit": bool(stats["finite"])
                and cosine >= 0.9
                and 0.5 <= ratio <= 2.0
                and rel <= 0.6,
            }
        )
    frame = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(a.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
