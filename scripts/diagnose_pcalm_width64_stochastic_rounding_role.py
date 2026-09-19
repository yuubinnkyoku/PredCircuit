from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

import diagnose_pcalm_width64_update_lattice_alignment as alignment
from diagnose_pcalm_width64_leaky_mixed_precision import quantize as base_quantize
from diagnose_pcalm_width64_stochastic_update_rounding import stochastic_quantize_fixed
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

STATE_PRECISION = "fixed15_i3"
UPDATE_PRECISION = "fixed14_i1"
DUAL_PRECISION = "fixed12_i1"
ROLES = ("gradient", "residual", "both")


def run_role_alignment(
    model: ResidualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    role: str,
    generator: torch.Generator,
    rng_sharing: str,
    budget: int,
    state_lr: float,
    dual_leak: float,
) -> tuple[list[torch.Tensor], dict[str, float | bool]]:
    rho = 1.0
    alpha = 0.925
    effective_lr = state_lr * x.shape[0]
    free = free_init(model, x)
    duals = zero_duals_like(constraint_residuals(model, x, free))

    state_saturated = state_total = 0
    update_saturated = update_total = 0
    dual_saturated = dual_total = 0
    late_requested_step = 0.0
    late_realized_step = 0.0
    late_update_zero = 0.0
    late_state_zero = 0.0
    late_count = 0
    finite = True

    def q_update(value: torch.Tensor, stochastic: bool) -> tuple[torch.Tensor, int, int]:
        if stochastic:
            return stochastic_quantize_fixed(
                value,
                UPDATE_PRECISION,
                generator=generator,
                rng_sharing=rng_sharing,
            )
        return base_quantize(value, UPDATE_PRECISION)

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

        new_free: list[torch.Tensor] = []
        quantized_grads: list[torch.Tensor] = []
        requested_steps: list[torch.Tensor] = []
        realized_steps: list[torch.Tensor] = []
        for z, grad in zip(variables, raw_grads, strict=True):
            q_grad, saturated, total = q_update(grad, role in ("gradient", "both"))
            update_saturated += saturated
            update_total += total
            requested = -effective_lr * q_grad
            q_state, saturated, total = base_quantize(z.detach() + requested, STATE_PRECISION)
            state_saturated += saturated
            state_total += total
            realized = q_state - z.detach()
            quantized_grads.append(q_grad)
            requested_steps.append(requested)
            realized_steps.append(realized)
            new_free.append(q_state)
        free = new_free

        raw_residuals = constraint_residuals(model, x, free)
        update_residuals: list[torch.Tensor] = []
        for residual in raw_residuals:
            q_residual, saturated, total = q_update(residual, role in ("residual", "both"))
            update_saturated += saturated
            update_total += total
            update_residuals.append(q_residual)

        duals_after: list[torch.Tensor] = []
        for lam, residual in zip(duals, update_residuals, strict=True):
            q_dual, saturated, total = base_quantize(
                (1.0 - dual_leak) * lam + alpha * residual,
                DUAL_PRECISION,
            )
            dual_saturated += saturated
            dual_total += total
            duals_after.append(q_dual)

        if outer_ix + 1 >= max(1, budget - 31):
            late_requested_step += alignment.tensor_list_norm(requested_steps)
            late_realized_step += alignment.tensor_list_norm(realized_steps)
            late_update_zero += alignment.zero_fraction(quantized_grads)
            late_state_zero += alignment.zero_fraction(realized_steps)
            late_count += 1

        finite = finite and all(
            bool(torch.isfinite(tensor).all())
            for tensor in [*free, *raw_residuals, *update_residuals, *duals_after]
        )
        if outer_ix == budget - 1:
            credit_duals = duals
            break
        duals = duals_after
    else:
        raise AssertionError("unreachable")

    loss = al_energy_shifted(
        model,
        x,
        y,
        [z.detach() for z in free],
        [dual.detach() for dual in credit_duals],
        rho=rho,
    )
    grads = [grad.detach() for grad in torch.autograd.grad(loss, tuple(model.weights))]
    finite = finite and all(bool(torch.isfinite(grad).all()) for grad in grads)
    requested_mean = late_requested_step / late_count
    realized_mean = late_realized_step / late_count
    return grads, {
        "finite": finite,
        "late_realized_to_requested_ratio": (
            realized_mean / requested_mean if requested_mean else math.nan
        ),
        "late_update_zero_fraction": late_update_zero / late_count,
        "late_state_zero_step_fraction": late_state_zero / late_count,
        "state_saturation_rate": state_saturated / state_total if state_total else 0.0,
        "update_saturation_rate": update_saturated / update_total if update_total else 0.0,
        "dual_saturation_rate": dual_saturated / dual_total if dual_total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Disentangle stochastic rounding of PC-ALM state gradients and residuals."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--rounding-seed", type=int, required=True)
    parser.add_argument("--rng-sharing", choices=("element", "tensor"), default="tensor")
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

    bp_model = ResidualMLP(
        depth=depth,
        width=width,
        input_dim=8,
        output_dim=4,
        activation="relu",
        seed=model_seed,
    )
    bp_first = method_grad(
        bp_model,
        x,
        y,
        Schedule("bp", budget=0),
        state_lr=args.state_lr,
        rho=1.0,
    )[0]
    bp_norm = float(bp_first.norm())

    model = ResidualMLP(
        depth=depth,
        width=width,
        input_dim=8,
        output_dim=4,
        activation="relu",
        seed=model_seed,
    )
    rounding_generator = torch.Generator().manual_seed(args.rounding_seed)
    grads, stats = run_role_alignment(
        model,
        x,
        y,
        role=args.role,
        generator=rounding_generator,
        rng_sharing=args.rng_sharing,
        budget=args.budget,
        state_lr=args.state_lr,
        dual_leak=args.dual_leak,
    )
    first = grads[0]
    cosine = gradient_cosine([first], [bp_first])
    ratio = float(first.norm()) / bp_norm if bp_norm else math.nan
    rel = gradient_relative_error([first], [bp_first])
    finite = bool(stats["finite"])
    row = {
        "seed": args.seed,
        "role": args.role,
        "rounding_seed": args.rounding_seed,
        "rng_sharing": args.rng_sharing,
        **stats,
        "first_layer_cosine_to_bp": cosine,
        "first_layer_grad_norm_ratio_to_bp": ratio,
        "first_layer_relative_error_to_bp": rel,
        "useful_first_layer_credit": finite
        and cosine >= 0.9
        and 0.5 <= ratio <= 2.0
        and rel <= 0.6,
    }
    frame = pd.DataFrame([row])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
