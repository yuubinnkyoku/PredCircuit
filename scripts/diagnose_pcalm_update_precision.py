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

STATE_PRECISION = "fixed12_i3"
DUAL_PRECISION = "fixed12_i1"


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
    update_precision: str,
) -> tuple[list[torch.Tensor], dict[str, float | int | bool]]:
    free = free_init(model, x)
    duals = zero_duals_like(constraint_residuals(model, x, free))
    effective_lr = state_lr * x.shape[0]

    grad_sat = grad_total = residual_sat = residual_total = 0
    state_sat = state_total = dual_sat = dual_total = 0
    max_grad = max_residual = max_state = max_dual = 0.0
    finite = True

    for outer_ix in range(budget):
        variables = [z.detach().requires_grad_(True) for z in free]
        fixed_duals = [d.detach() for d in duals]
        energy = al_energy_shifted(model, x, y, variables, fixed_duals, rho=rho)
        raw_grads = torch.autograd.grad(energy, variables)

        updated_free: list[torch.Tensor] = []
        for z, raw_grad in zip(variables, raw_grads, strict=True):
            q_grad, sat, total, pre = quantize_dual(raw_grad, update_precision)
            grad_sat += sat
            grad_total += total
            max_grad = max(max_grad, pre)
            raw_state = z.detach() - effective_lr * q_grad
            q_state, sat, total, pre = quantize_dual(raw_state, STATE_PRECISION)
            state_sat += sat
            state_total += total
            max_state = max(max_state, pre)
            updated_free.append(q_state)
        free = updated_free

        raw_residuals = constraint_residuals(model, x, free)
        update_residuals: list[torch.Tensor] = []
        for raw_residual in raw_residuals:
            q_residual, sat, total, pre = quantize_dual(raw_residual, update_precision)
            residual_sat += sat
            residual_total += total
            max_residual = max(max_residual, pre)
            update_residuals.append(q_residual)

        duals_after: list[torch.Tensor] = []
        for lam, residual in zip(duals, update_residuals, strict=True):
            raw_dual = (1.0 - dual_leak) * lam + alpha * residual
            q_dual, sat, total, pre = quantize_dual(raw_dual, DUAL_PRECISION)
            dual_sat += sat
            dual_total += total
            max_dual = max(max_dual, pre)
            duals_after.append(q_dual)

        tensors = [*free, *raw_residuals, *update_residuals, *duals_after]
        finite = finite and all(bool(torch.isfinite(t).all()) for t in tensors)
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
    final_residual = sum(float(r.norm()) for r in constraint_residuals(model, x, free))
    finite = finite and all(bool(torch.isfinite(g).all()) for g in grads)
    return grads, {
        "finite": finite,
        "residual_total": final_residual,
        "max_abs_update_grad_pre_quant": max_grad,
        "max_abs_update_residual_pre_quant": max_residual,
        "max_abs_state_pre_quant": max_state,
        "max_abs_dual_pre_quant": max_dual,
        "update_grad_saturation_rate": grad_sat / grad_total if grad_total else 0.0,
        "update_residual_saturation_rate": (
            residual_sat / residual_total if residual_total else 0.0
        ),
        "state_saturation_rate": state_sat / state_total if state_total else 0.0,
        "dual_saturation_rate": dual_sat / dual_total if dual_total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--precisions",
        default="fp32,fp16,bf16,fixed16_i2,fixed12_i2,fixed12_i3",
    )
    args = parser.parse_args()

    depth = 32
    width = 8
    state_lr = 0.25
    rho = 1.0
    alpha = 0.925
    dual_leak = 0.01
    budget = 112
    model = ResidualMLP(
        depth=depth,
        width=width,
        input_dim=8,
        output_dim=4,
        activation="relu",
        seed=args.seed + depth,
    )
    generator = torch.Generator().manual_seed(args.seed + 10000 + depth)
    x = torch.randn(4, 8, generator=generator)
    y = torch.randn(4, 4, generator=generator)
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

    rows: list[dict[str, object]] = []
    reference: list[torch.Tensor] | None = None
    for precision in args.precisions.split(","):
        grads, metrics = run_precision(
            model,
            x,
            y,
            state_lr=state_lr,
            rho=rho,
            alpha=alpha,
            dual_leak=dual_leak,
            budget=budget,
            update_precision=precision,
        )
        first = grads[0]
        ratio = float(first.norm()) / bp_norm if bp_norm else math.nan
        cosine = gradient_cosine([first], [bp_first])
        relative_error = gradient_relative_error([first], [bp_first])
        if reference is None:
            reference = grads
        error_to_fp32 = gradient_relative_error(grads, reference)
        finite = bool(metrics["finite"])
        useful = finite and cosine >= 0.9 and 0.5 <= ratio <= 2.0 and relative_error <= 0.6
        rows.append(
            {
                "seed": args.seed,
                "update_precision": precision,
                "state_precision": STATE_PRECISION,
                "dual_precision": DUAL_PRECISION,
                "finite": finite,
                "useful_first_layer_credit": useful,
                "first_layer_cosine_to_bp": cosine,
                "first_layer_grad_norm_ratio_to_bp": ratio,
                "first_layer_relative_error_to_bp": relative_error,
                "all_gradient_relative_error_to_fp32_update": error_to_fp32,
                **metrics,
            }
        )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
