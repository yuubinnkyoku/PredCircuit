from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from diagnose_pcalm_width64_leaky_mixed_precision import quantize as base_quantize
from diagnose_pcalm_width64_stochastic_update_rounding import stochastic_quantize_fixed
from predcircuit.pcalm import (
    ResidualMLP,
    al_energy_shifted,
    bp_loss,
    constraint_residuals,
    free_init,
    gradient_cosine,
    zero_duals_like,
)

STATE_PRECISION = "fixed15_i3"
UPDATE_PRECISION = "fixed14_i1"
DUAL_PRECISION = "fixed12_i1"
ROLES = ("nearest", "gradient", "residual", "both")


def list_norm(values: list[torch.Tensor]) -> float:
    if not values:
        return 0.0
    squared = torch.stack([torch.sum(value.float() ** 2) for value in values]).sum()
    return float(torch.sqrt(squared))


def zero_fraction(values: list[torch.Tensor]) -> float:
    total = sum(value.numel() for value in values)
    zeros = sum(int(torch.count_nonzero(value == 0)) for value in values)
    return zeros / total if total else 0.0


def run_trace(
    model: ResidualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    role: str,
    generator: torch.Generator,
    budget: int,
    state_lr: float,
    dual_leak: float,
) -> pd.DataFrame:
    rho = 1.0
    alpha = 0.925
    effective_lr = state_lr * x.shape[0]
    free = free_init(model, x)
    duals = zero_duals_like(constraint_residuals(model, x, free))
    rows: list[dict[str, float | int | str | bool]] = []

    bp_grads = [
        grad.detach().clone()
        for grad in torch.autograd.grad(bp_loss(model, x, y), tuple(model.weights))
    ]

    def q_update(value: torch.Tensor, stochastic: bool) -> torch.Tensor:
        if stochastic:
            quantized, _, _ = stochastic_quantize_fixed(
                value,
                UPDATE_PRECISION,
                generator=generator,
                rng_sharing="tensor",
            )
            return quantized
        quantized, _, _ = base_quantize(value, UPDATE_PRECISION)
        return quantized

    for step in range(budget):
        variables = [z.detach().requires_grad_(True) for z in free]
        energy = al_energy_shifted(
            model,
            x,
            y,
            variables,
            [dual.detach() for dual in duals],
            rho=rho,
        )
        raw_grads = list(torch.autograd.grad(energy, variables, retain_graph=True))
        credit_grads = [
            grad.detach().clone()
            for grad in torch.autograd.grad(energy, tuple(model.weights), allow_unused=False)
        ]
        q_grads = [q_update(g, role in ("gradient", "both")) for g in raw_grads]

        new_free: list[torch.Tensor] = []
        realized_steps: list[torch.Tensor] = []
        for z, q_grad in zip(variables, q_grads, strict=True):
            requested = -effective_lr * q_grad
            q_state, _, _ = base_quantize(z.detach() + requested, STATE_PRECISION)
            realized_steps.append(q_state - z.detach())
            new_free.append(q_state)
        free = new_free

        raw_residuals = constraint_residuals(model, x, free)
        q_residuals = [q_update(r, role in ("residual", "both")) for r in raw_residuals]
        duals_after: list[torch.Tensor] = []
        dual_steps: list[torch.Tensor] = []
        for lam, residual in zip(duals, q_residuals, strict=True):
            q_dual, _, _ = base_quantize(
                (1.0 - dual_leak) * lam + alpha * residual,
                DUAL_PRECISION,
            )
            dual_steps.append(q_dual - lam)
            duals_after.append(q_dual)

        rows.append(
            {
                "step": step + 1,
                "role": role,
                "energy": float(energy.detach()),
                "weight_credit_cosine_to_bp": gradient_cosine(credit_grads, bp_grads),
                "first_weight_credit_cosine_to_bp": gradient_cosine(
                    [credit_grads[0]], [bp_grads[0]]
                ),
                "last_weight_credit_cosine_to_bp": gradient_cosine(
                    [credit_grads[-1]], [bp_grads[-1]]
                ),
                "raw_grad_norm": list_norm(raw_grads),
                "q_grad_norm": list_norm(q_grads),
                "q_grad_zero_fraction": zero_fraction(q_grads),
                "state_step_norm": list_norm(realized_steps),
                "state_step_zero_fraction": zero_fraction(realized_steps),
                "raw_residual_norm": list_norm(list(raw_residuals)),
                "q_residual_norm": list_norm(q_residuals),
                "q_residual_zero_fraction": zero_fraction(q_residuals),
                "dual_norm_before": list_norm(duals),
                "dual_step_norm": list_norm(dual_steps),
                "first_residual_norm": float(raw_residuals[0].norm()),
                "last_residual_norm": float(raw_residuals[-1].norm()),
                "first_dual_norm": float(duals_after[0].norm()),
                "last_dual_norm": float(duals_after[-1].norm()),
                "finite": all(
                    bool(torch.isfinite(t).all())
                    for t in [*free, *q_grads, *q_residuals, *duals_after]
                ),
            }
        )
        duals = duals_after

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace PC-ALM state/dual quantization dynamics by stochastic-rounding role."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--rounding-seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--state-lr", type=float, default=0.234285)
    parser.add_argument("--dual-leak", type=float, default=0.02)
    args = parser.parse_args()

    depth, width = 32, 64
    model_seed = args.seed + 1000 * width + depth
    data_seed = args.seed + 10_000 + 1000 * width + depth
    data_generator = torch.Generator().manual_seed(data_seed)
    x = torch.randn(4, 8, generator=data_generator)
    y = torch.randn(4, 4, generator=data_generator)
    model = ResidualMLP(
        depth=depth,
        width=width,
        input_dim=8,
        output_dim=4,
        activation="relu",
        seed=model_seed,
    )
    generator = torch.Generator().manual_seed(args.rounding_seed)
    frame = run_trace(
        model,
        x,
        y,
        role=args.role,
        generator=generator,
        budget=args.budget,
        state_lr=args.state_lr,
        dual_leak=args.dual_leak,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.tail(8).to_string(index=False))


if __name__ == "__main__":
    main()
