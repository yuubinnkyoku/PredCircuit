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

CONFIGS = {
    "fp32": ("fp32", "fp32", "fp32"),
    "update12_only": ("fixed12_i2", "fp32", "fp32"),
    "state12_only": ("fp32", "fixed12_i3", "fp32"),
    "dual12_only": ("fp32", "fp32", "fixed12_i1"),
    "update_state12": ("fixed12_i2", "fixed12_i3", "fp32"),
    "update_dual12": ("fixed12_i2", "fp32", "fixed12_i1"),
    "state_dual12": ("fp32", "fixed12_i3", "fixed12_i1"),
    "all12": ("fixed12_i2", "fixed12_i3", "fixed12_i1"),
}


def q(value: torch.Tensor, precision: str) -> torch.Tensor:
    quantized, _, _, _ = quantize_dual(value, precision)
    return quantized


def run_config(
    model: ResidualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    update_precision: str,
    state_precision: str,
    dual_precision: str,
    budget: int,
) -> tuple[list[torch.Tensor], float, bool]:
    state_lr = 0.25
    rho = 1.0
    alpha = 0.925
    dual_leak = 0.01
    effective_lr = state_lr * x.shape[0]
    free = free_init(model, x)
    duals = zero_duals_like(constraint_residuals(model, x, free))
    finite = True

    for outer_ix in range(budget):
        variables = [z.detach().requires_grad_(True) for z in free]
        energy = al_energy_shifted(model, x, y, variables, [d.detach() for d in duals], rho=rho)
        raw_grads = torch.autograd.grad(energy, variables)
        free = [
            q(z.detach() - effective_lr * q(g, update_precision), state_precision)
            for z, g in zip(variables, raw_grads, strict=True)
        ]
        raw_residuals = constraint_residuals(model, x, free)
        update_residuals = [q(r, update_precision) for r in raw_residuals]
        duals_after = [
            q((1.0 - dual_leak) * lam + alpha * residual, dual_precision)
            for lam, residual in zip(duals, update_residuals, strict=True)
        ]
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
    return grads, residual, finite


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablate update/state/dual quantization for width-64 PC-ALM."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=128)
    args = parser.parse_args()

    depth = 32
    width = 64
    model_seed = args.seed + 1000 * width + depth
    data_seed = args.seed + 10_000 + 1000 * width + depth
    generator = torch.Generator().manual_seed(data_seed)
    x = torch.randn(4, 8, generator=generator)
    y = torch.randn(4, 4, generator=generator)

    bp_model = ResidualMLP(
        depth=depth,
        width=width,
        input_dim=8,
        output_dim=4,
        activation="relu",
        seed=model_seed,
    )
    bp = method_grad(
        bp_model,
        x,
        y,
        Schedule("bp", budget=0),
        state_lr=0.25,
        rho=1.0,
    )
    bp_first = bp[0]
    bp_norm = float(bp_first.norm())

    rows: list[dict[str, object]] = []
    fp32_grads: list[torch.Tensor] | None = None
    for name, (update_precision, state_precision, dual_precision) in CONFIGS.items():
        model = ResidualMLP(
            depth=depth,
            width=width,
            input_dim=8,
            output_dim=4,
            activation="relu",
            seed=model_seed,
        )
        grads, residual, finite = run_config(
            model,
            x,
            y,
            update_precision=update_precision,
            state_precision=state_precision,
            dual_precision=dual_precision,
            budget=args.budget,
        )
        if fp32_grads is None:
            fp32_grads = grads
        first = grads[0]
        ratio = float(first.norm()) / bp_norm if bp_norm else math.nan
        cosine = gradient_cosine([first], [bp_first])
        relative_error = gradient_relative_error([first], [bp_first])
        rows.append(
            {
                "seed": args.seed,
                "config": name,
                "update_precision": update_precision,
                "state_precision": state_precision,
                "dual_precision": dual_precision,
                "finite": finite,
                "useful_first_layer_credit": finite
                and cosine >= 0.9
                and 0.5 <= ratio <= 2.0
                and relative_error <= 0.6,
                "first_layer_cosine_to_bp": cosine,
                "first_layer_grad_norm_ratio_to_bp": ratio,
                "first_layer_relative_error_to_bp": relative_error,
                "all_gradient_relative_error_to_fp32": gradient_relative_error(grads, fp32_grads),
                "residual_total": residual,
            }
        )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
