from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from diagnose_pcalm_width64_leaky_mixed_precision import quantize
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

CHECKPOINTS = {32, 64, 96, 128, 160, 192, 224, 256}
STATE_CONFIGS = {
    "state_fp32": "fp32",
    "state14_i3": "fixed14_i3",
    "state15_i3": "fixed15_i3",
    "state16_i3": "fixed16_i3",
}


def flat(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def tensor_list_norm(tensors: list[torch.Tensor]) -> float:
    if not tensors:
        return 0.0
    return float(flat(tensors).norm())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm()
    if float(denom) == 0.0:
        return math.nan
    return float(torch.dot(a, b) / denom)


def zero_fraction(tensors: list[torch.Tensor]) -> float:
    total = sum(tensor.numel() for tensor in tensors)
    if total == 0:
        return 0.0
    zeros = sum(int((tensor == 0).sum()) for tensor in tensors)
    return zeros / total


def run_config(
    *,
    model: ResidualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    bp_first: torch.Tensor,
    bp_norm: float,
    state_precision: str,
    budget: int,
    state_lr: float,
    dual_leak: float,
) -> list[dict[str, object]]:
    rho = 1.0
    alpha = 0.925
    update_precision = "fixed14_i1"
    dual_precision = "fixed12_i1"
    effective_lr = state_lr * x.shape[0]

    free = free_init(model, x)
    duals = zero_duals_like(constraint_residuals(model, x, free))
    previous_quantized_delta: torch.Tensor | None = None
    rows: list[dict[str, object]] = []

    for outer_ix in range(budget):
        variables = [z.detach().requires_grad_(True) for z in free]
        energy = al_energy_shifted(
            model,
            x,
            y,
            variables,
            [dual.detach() for dual in duals],
            rho=rho,
        )
        raw_grads = torch.autograd.grad(energy, variables)

        raw_deltas: list[torch.Tensor] = []
        quantized_deltas: list[torch.Tensor] = []
        rounding_errors: list[torch.Tensor] = []
        quantized_grads: list[torch.Tensor] = []
        new_free: list[torch.Tensor] = []
        state_saturated = state_total = 0
        update_saturated = update_total = 0

        for z, grad in zip(variables, raw_grads, strict=True):
            q_grad, saturated, total = quantize(grad, update_precision)
            update_saturated += saturated
            update_total += total
            raw_delta = -effective_lr * q_grad
            candidate = z.detach() + raw_delta
            q_state, saturated, total = quantize(candidate, state_precision)
            state_saturated += saturated
            state_total += total
            quantized_delta = q_state - z.detach()

            quantized_grads.append(q_grad)
            raw_deltas.append(raw_delta)
            quantized_deltas.append(quantized_delta)
            rounding_errors.append(q_state - candidate)
            new_free.append(q_state)

        free = new_free
        raw_residuals = constraint_residuals(model, x, free)
        update_residuals: list[torch.Tensor] = []
        for residual in raw_residuals:
            q_residual, saturated, total = quantize(residual, update_precision)
            update_saturated += saturated
            update_total += total
            update_residuals.append(q_residual)

        duals_after: list[torch.Tensor] = []
        dual_saturated = dual_total = 0
        for lam, residual in zip(duals, update_residuals, strict=True):
            q_dual, saturated, total = quantize(
                (1.0 - dual_leak) * lam + alpha * residual,
                dual_precision,
            )
            dual_saturated += saturated
            dual_total += total
            duals_after.append(q_dual)

        quantized_delta_flat = flat(quantized_deltas)
        previous_delta_cosine = (
            math.nan
            if previous_quantized_delta is None
            else cosine(previous_quantized_delta, quantized_delta_flat)
        )
        previous_quantized_delta = quantized_delta_flat.detach()

        step = outer_ix + 1
        checkpoint_cosine = math.nan
        checkpoint_norm_ratio = math.nan
        checkpoint_relative_error = math.nan
        checkpoint_useful: bool | None = None
        if step in CHECKPOINTS:
            credit_loss = al_energy_shifted(
                model,
                x,
                y,
                [z.detach() for z in free],
                [dual.detach() for dual in duals],
                rho=rho,
            )
            grads = [
                grad.detach()
                for grad in torch.autograd.grad(credit_loss, tuple(model.weights))
            ]
            first = grads[0]
            checkpoint_norm_ratio = float(first.norm()) / bp_norm if bp_norm else math.nan
            checkpoint_cosine = gradient_cosine([first], [bp_first])
            checkpoint_relative_error = gradient_relative_error([first], [bp_first])
            checkpoint_useful = (
                checkpoint_cosine >= 0.9
                and 0.5 <= checkpoint_norm_ratio <= 2.0
                and checkpoint_relative_error <= 0.6
            )

        finite = all(
            bool(torch.isfinite(tensor).all())
            for tensor in [*free, *raw_residuals, *update_residuals, *duals_after]
        )
        rows.append(
            {
                "step": step,
                "state_precision": state_precision,
                "finite": finite,
                "residual_raw_norm": tensor_list_norm(raw_residuals),
                "residual_quantized_norm": tensor_list_norm(update_residuals),
                "dual_norm_before": tensor_list_norm(duals),
                "dual_norm_after": tensor_list_norm(duals_after),
                "dual_step_norm": tensor_list_norm(
                    [
                        after - before
                        for before, after in zip(duals, duals_after, strict=True)
                    ]
                ),
                "raw_state_step_norm": tensor_list_norm(raw_deltas),
                "quantized_state_step_norm": tensor_list_norm(quantized_deltas),
                "state_rounding_error_norm": tensor_list_norm(rounding_errors),
                "update_zero_fraction": zero_fraction(quantized_grads),
                "state_zero_step_fraction": zero_fraction(quantized_deltas),
                "state_step_cosine_previous": previous_delta_cosine,
                "max_abs_state": max(float(z.abs().max()) for z in free),
                "max_abs_dual": max(float(dual.abs().max()) for dual in duals_after),
                "state_saturation_rate": state_saturated / state_total if state_total else 0.0,
                "update_saturation_rate": (
                    update_saturated / update_total if update_total else 0.0
                ),
                "dual_saturation_rate": dual_saturated / dual_total if dual_total else 0.0,
                "checkpoint_cosine_to_bp": checkpoint_cosine,
                "checkpoint_norm_ratio_to_bp": checkpoint_norm_ratio,
                "checkpoint_relative_error_to_bp": checkpoint_relative_error,
                "checkpoint_useful": checkpoint_useful,
            }
        )
        duals = duals_after

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace the non-monotone 14/15/16-bit state lattice dynamics at width 64."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--state-lr", type=float, default=0.234285)
    parser.add_argument("--dual-leak", type=float, default=0.02)
    args = parser.parse_args()

    depth, width = 32, 64
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
        state_lr=args.state_lr,
        rho=1.0,
    )
    bp_first = bp[0]
    bp_norm = float(bp_first.norm())

    rows: list[dict[str, object]] = []
    for config, state_precision in STATE_CONFIGS.items():
        model = ResidualMLP(
            depth=depth,
            width=width,
            input_dim=8,
            output_dim=4,
            activation="relu",
            seed=model_seed,
        )
        config_rows = run_config(
            model=model,
            x=x,
            y=y,
            bp_first=bp_first,
            bp_norm=bp_norm,
            state_precision=state_precision,
            budget=args.budget,
            state_lr=args.state_lr,
            dual_leak=args.dual_leak,
        )
        for row in config_rows:
            row["seed"] = args.seed
            row["config"] = config
        rows.extend(config_rows)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame[frame["step"].isin(CHECKPOINTS)].to_string(index=False))


if __name__ == "__main__":
    main()
