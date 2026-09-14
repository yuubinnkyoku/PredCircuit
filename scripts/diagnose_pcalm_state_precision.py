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
    _solve_inner,
    al_energy_shifted,
    constraint_residuals,
    free_init,
    gradient_cosine,
    gradient_relative_error,
    method_grad,
    zero_duals_like,
)


def run_precision(
    model: ResidualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    state_lr: float,
    rho: float,
    alpha: float,
    dual_leak: float,
    budget: int,
    state_precision: str,
) -> tuple[list[torch.Tensor], float, float, int, int, bool]:
    free = free_init(model, x)
    duals = zero_duals_like(constraint_residuals(model, x, free))
    sat = total = 0
    max_state = 0.0
    finite = True
    for outer_ix in range(budget):
        free = _solve_inner(
            model,
            x,
            y,
            free,
            duals,
            state_lr=state_lr,
            rho=rho,
            steps=1,
        )
        quantized_free = []
        for z in free:
            q, s, n, pre = quantize_dual(z, state_precision)
            quantized_free.append(q)
            sat += s
            total += n
            max_state = max(max_state, pre)
        free = quantized_free
        residuals = constraint_residuals(model, x, free)
        duals_after = []
        for lam, residual in zip(duals, residuals, strict=True):
            updated = (1.0 - dual_leak) * lam + alpha * residual
            q, _, _, _ = quantize_dual(updated, "fixed12_i1")
            duals_after.append(q)
        finite = finite and all(
            bool(torch.isfinite(t).all()) for t in [*free, *residuals, *duals_after]
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
    residual_total = sum(float(r.norm()) for r in constraint_residuals(model, x, free))
    finite = finite and all(bool(torch.isfinite(g).all()) for g in grads)
    return grads, residual_total, max_state, sat, total, finite


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--precisions",
        default="fp32,fp16,bf16,fixed16_i2,fixed12_i2,fixed12_i3",
    )
    a = p.parse_args()
    depth = 32
    width = 8
    state_lr = 0.25
    rho = 1.0
    alpha = 0.925
    leak = 0.01
    budget = 112
    model = ResidualMLP(
        depth=depth,
        width=width,
        input_dim=8,
        output_dim=4,
        activation="relu",
        seed=a.seed + depth,
    )
    gen = torch.Generator().manual_seed(a.seed + 10000 + depth)
    x = torch.randn(4, 8, generator=gen)
    y = torch.randn(4, 4, generator=gen)
    bp = method_grad(
        model,
        x,
        y,
        Schedule("bp", budget=0),
        state_lr=state_lr,
        rho=rho,
    )
    bp_first = bp[0]
    bp_norm = float(bp_first.norm())
    rows = []
    ref = None
    for precision in a.precisions.split(","):
        grad, residual, max_state, sat, total, finite = run_precision(
            model,
            x,
            y,
            state_lr=state_lr,
            rho=rho,
            alpha=alpha,
            dual_leak=leak,
            budget=budget,
            state_precision=precision,
        )
        first = grad[0]
        ratio = float(first.norm()) / bp_norm if bp_norm else math.nan
        cosine = gradient_cosine([first], [bp_first])
        rel = gradient_relative_error([first], [bp_first])
        if ref is None:
            ref = grad
        err = gradient_relative_error(grad, ref)
        useful = finite and cosine >= 0.9 and 0.5 <= ratio <= 2.0 and rel <= 0.6
        rows.append(
            {
                "seed": a.seed,
                "state_precision": precision,
                "dual_precision": "fixed12_i1",
                "finite": finite,
                "useful_first_layer_credit": useful,
                "first_layer_cosine_to_bp": cosine,
                "first_layer_grad_norm_ratio_to_bp": ratio,
                "first_layer_relative_error_to_bp": rel,
                "all_gradient_relative_error_to_fp32_state": err,
                "residual_total": residual,
                "max_abs_state_pre_quant": max_state,
                "saturated_state_values": sat,
                "quantized_state_values": total,
                "state_saturation_rate": sat / total if total else 0.0,
            }
        )
    frame = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(a.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
