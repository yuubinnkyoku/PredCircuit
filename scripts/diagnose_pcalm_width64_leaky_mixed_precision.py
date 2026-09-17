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


def quantize(value: torch.Tensor, precision: str) -> tuple[torch.Tensor, int, int]:
    quantized, saturated, total, _ = quantize_dual(value, precision)
    return quantized, saturated, total


def run(
    model: ResidualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    update_precision: str,
    state_precision: str,
    dual_precision: str,
    budget: int,
    state_lr: float,
    dual_leak: float,
) -> tuple[list[torch.Tensor], dict[str, float | bool]]:
    rho = 1.0
    alpha = 0.925
    effective_lr = state_lr * x.shape[0]
    free = free_init(model, x)
    duals = zero_duals_like(constraint_residuals(model, x, free))
    finite = True
    state_sat = state_total = 0
    update_sat = update_total = 0
    dual_sat = dual_total = 0
    max_abs_dual = 0.0

    for outer_ix in range(budget):
        variables = [z.detach().requires_grad_(True) for z in free]
        energy = al_energy_shifted(model, x, y, variables, [d.detach() for d in duals], rho=rho)
        raw_grads = torch.autograd.grad(energy, variables)
        new_free: list[torch.Tensor] = []
        for z, grad in zip(variables, raw_grads, strict=True):
            q_grad, sat, total = quantize(grad, update_precision)
            update_sat += sat
            update_total += total
            q_state, sat, total = quantize(z.detach() - effective_lr * q_grad, state_precision)
            state_sat += sat
            state_total += total
            new_free.append(q_state)
        free = new_free

        raw_residuals = constraint_residuals(model, x, free)
        update_residuals: list[torch.Tensor] = []
        for residual in raw_residuals:
            q_residual, sat, total = quantize(residual, update_precision)
            update_sat += sat
            update_total += total
            update_residuals.append(q_residual)

        duals_after: list[torch.Tensor] = []
        for lam, residual in zip(duals, update_residuals, strict=True):
            q_dual, sat, total = quantize(
                (1.0 - dual_leak) * lam + alpha * residual,
                dual_precision,
            )
            dual_sat += sat
            dual_total += total
            max_abs_dual = max(max_abs_dual, float(q_dual.abs().max()))
            duals_after.append(q_dual)

        finite = finite and all(
            bool(torch.isfinite(t).all())
            for t in [*free, *raw_residuals, *update_residuals, *duals_after]
        )
        if outer_ix == budget - 1:
            credit_duals = duals
            break
        duals = duals_after

    loss = al_energy_shifted(
        model,
        x,
        y,
        [z.detach() for z in free],
        [d.detach() for d in credit_duals],
        rho=rho,
    )
    grads = [g.detach() for g in torch.autograd.grad(loss, tuple(model.weights))]
    residual = sum(float(r.norm()) for r in constraint_residuals(model, x, free))
    finite = finite and all(bool(torch.isfinite(g).all()) for g in grads)
    return grads, {
        "finite": finite,
        "residual_total": residual,
        "max_abs_dual": max_abs_dual,
        "state_saturation_rate": state_sat / state_total if state_total else 0.0,
        "update_saturation_rate": update_sat / update_total if update_total else 0.0,
        "dual_saturation_rate": dual_sat / dual_total if dual_total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hold out leaky PC-ALM at width 64 in 14/14/12 fixed point."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--state-lr", type=float, default=0.234285)
    parser.add_argument("--dual-leak", type=float, default=0.02)
    args = parser.parse_args()

    depth = 32
    width = 64
    model_seed = args.seed + 1000 * width + depth
    data_seed = args.seed + 10_000 + 1000 * width + depth
    generator = torch.Generator().manual_seed(data_seed)
    x = torch.randn(4, 8, generator=generator)
    y = torch.randn(4, 4, generator=generator)

    bp_model = ResidualMLP(
        depth=depth, width=width, input_dim=8, output_dim=4, activation="relu", seed=model_seed
    )
    bp = method_grad(bp_model, x, y, Schedule("bp", budget=0), state_lr=args.state_lr, rho=1.0)
    bp_first = bp[0]
    bp_norm = float(bp_first.norm())

    rows: list[dict[str, object]] = []
    fp32_grads: list[torch.Tensor] | None = None
    configs = {
        "fp32_leaky": ("fp32", "fp32", "fp32"),
        "fixed14_14_12_leaky": ("fixed14_i2", "fixed14_i3", "fixed12_i1"),
    }
    for name, (update_precision, state_precision, dual_precision) in configs.items():
        model = ResidualMLP(
            depth=depth, width=width, input_dim=8, output_dim=4, activation="relu", seed=model_seed
        )
        grads, stats = run(
            model,
            x,
            y,
            update_precision=update_precision,
            state_precision=state_precision,
            dual_precision=dual_precision,
            budget=args.budget,
            state_lr=args.state_lr,
            dual_leak=args.dual_leak,
        )
        if fp32_grads is None:
            fp32_grads = grads
        first = grads[0]
        ratio = float(first.norm()) / bp_norm if bp_norm else math.nan
        cosine = gradient_cosine([first], [bp_first])
        relative_error = gradient_relative_error([first], [bp_first])
        finite = bool(stats["finite"])
        rows.append(
            {
                "seed": args.seed,
                "config": name,
                "budget": args.budget,
                "state_lr": args.state_lr,
                "dual_leak": args.dual_leak,
                "update_precision": update_precision,
                "state_precision": state_precision,
                "dual_precision": dual_precision,
                **stats,
                "useful_first_layer_credit": finite
                and cosine >= 0.9
                and 0.5 <= ratio <= 2.0
                and relative_error <= 0.6,
                "first_layer_cosine_to_bp": cosine,
                "first_layer_grad_norm_ratio_to_bp": ratio,
                "first_layer_relative_error_to_bp": relative_error,
                "all_gradient_relative_error_to_fp32": gradient_relative_error(grads, fp32_grads),
            }
        )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
