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

UPDATE_CONFIGS = {
    "update12_i1": "fixed12_i1",
    "update13_i1": "fixed13_i1",
    "update14_i1": "fixed14_i1",
    "update15_i1": "fixed15_i1",
    "update16_i1": "fixed16_i1",
    "update_fp32": "fp32",
}


def fixed_lsb(precision: str) -> float | None:
    if not precision.startswith("fixed"):
        return None
    bits_text, integer_text = precision.removeprefix("fixed").split("_i", 1)
    bits = int(bits_text)
    integer_bits = int(integer_text)
    frac_bits = bits - 1 - integer_bits
    return 2.0 ** (-frac_bits)


def predicted_gradient_deadzone(
    update_precision: str,
    *,
    state_precision: str,
    effective_lr: float,
) -> tuple[float, int | None]:
    state_lsb = fixed_lsb(state_precision)
    assert state_lsb is not None
    half_state_lsb = 0.5 * state_lsb
    update_lsb = fixed_lsb(update_precision)
    if update_lsb is None:
        return half_state_lsb / effective_lr, None

    minimum_quanta = math.ceil(half_state_lsb / (effective_lr * update_lsb) - 1e-12)
    lower_bin_edge = max(0.0, (minimum_quanta - 0.5) * update_lsb)
    return lower_bin_edge, minimum_quanta


def tensor_list_norm(tensors: list[torch.Tensor]) -> float:
    if not tensors:
        return 0.0
    return float(torch.cat([tensor.reshape(-1) for tensor in tensors]).norm())


def zero_fraction(tensors: list[torch.Tensor]) -> float:
    total = sum(tensor.numel() for tensor in tensors)
    if total == 0:
        return 0.0
    zeros = sum(int((tensor == 0).sum()) for tensor in tensors)
    return zeros / total


def run_alignment(
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
    alpha: float = 0.925,
) -> tuple[list[torch.Tensor], dict[str, float | bool]]:
    rho = 1.0
    effective_lr = state_lr * x.shape[0]
    free = free_init(model, x)
    duals = zero_duals_like(constraint_residuals(model, x, free))

    state_saturated = state_total = 0
    update_saturated = update_total = 0
    dual_saturated = dual_total = 0
    dual_prequant_max_abs = 0.0
    dual_quantized_max_abs = 0.0
    dual_zero = 0
    late_requested_step = 0.0
    late_realized_step = 0.0
    late_update_zero = 0.0
    late_state_zero = 0.0
    late_count = 0
    finite = True

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
        requested_steps: list[torch.Tensor] = []
        realized_steps: list[torch.Tensor] = []
        quantized_grads: list[torch.Tensor] = []
        for z, grad in zip(variables, raw_grads, strict=True):
            q_grad, saturated, total = quantize(grad, update_precision)
            update_saturated += saturated
            update_total += total
            requested = -effective_lr * q_grad
            q_state, saturated, total = quantize(z.detach() + requested, state_precision)
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
            q_residual, saturated, total = quantize(residual, update_precision)
            update_saturated += saturated
            update_total += total
            update_residuals.append(q_residual)

        duals_after: list[torch.Tensor] = []
        for lam, residual in zip(duals, update_residuals, strict=True):
            dual_candidate = (1.0 - dual_leak) * lam + alpha * residual
            dual_prequant_max_abs = max(
                dual_prequant_max_abs, float(dual_candidate.abs().max())
            )
            q_dual, saturated, total = quantize(dual_candidate, dual_precision)
            dual_saturated += saturated
            dual_total += total
            dual_quantized_max_abs = max(
                dual_quantized_max_abs, float(q_dual.abs().max())
            )
            dual_zero += int((q_dual == 0).sum())
            duals_after.append(q_dual)

        if outer_ix + 1 >= max(1, budget - 31):
            late_requested_step += tensor_list_norm(requested_steps)
            late_realized_step += tensor_list_norm(realized_steps)
            late_update_zero += zero_fraction(quantized_grads)
            late_state_zero += zero_fraction(realized_steps)
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
    residual_total = tensor_list_norm(constraint_residuals(model, x, free))
    finite = finite and all(bool(torch.isfinite(grad).all()) for grad in grads)

    requested_mean = late_requested_step / late_count
    realized_mean = late_realized_step / late_count
    return grads, {
        "finite": finite,
        "residual_total": residual_total,
        "late_requested_state_step_norm": requested_mean,
        "late_realized_state_step_norm": realized_mean,
        "late_realized_to_requested_ratio": (
            realized_mean / requested_mean if requested_mean else math.nan
        ),
        "late_update_zero_fraction": late_update_zero / late_count,
        "late_state_zero_step_fraction": late_state_zero / late_count,
        "state_saturation_rate": state_saturated / state_total if state_total else 0.0,
        "update_saturation_rate": update_saturated / update_total if update_total else 0.0,
        "dual_saturation_rate": dual_saturated / dual_total if dual_total else 0.0,
        "dual_prequant_max_abs": dual_prequant_max_abs,
        "dual_quantized_max_abs": dual_quantized_max_abs,
        "dual_zero_fraction": dual_zero / dual_total if dual_total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test update/state fixed-point lattice alignment at width 64."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--state-lr", type=float, default=0.234285)
    parser.add_argument("--dual-leak", type=float, default=0.02)
    args = parser.parse_args()

    depth, width = 32, 64
    state_precision = "fixed15_i3"
    dual_precision = "fixed12_i1"
    effective_lr = args.state_lr * 4

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
    for config, update_precision in UPDATE_CONFIGS.items():
        model = ResidualMLP(
            depth=depth,
            width=width,
            input_dim=8,
            output_dim=4,
            activation="relu",
            seed=model_seed,
        )
        grads, stats = run_alignment(
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
        first = grads[0]
        ratio = float(first.norm()) / bp_norm if bp_norm else math.nan
        credit_cosine = gradient_cosine([first], [bp_first])
        relative_error = gradient_relative_error([first], [bp_first])
        finite = bool(stats["finite"])
        deadzone, minimum_quanta = predicted_gradient_deadzone(
            update_precision,
            state_precision=state_precision,
            effective_lr=effective_lr,
        )
        rows.append(
            {
                "seed": args.seed,
                "config": config,
                "budget": args.budget,
                "state_lr": args.state_lr,
                "effective_lr": effective_lr,
                "dual_leak": args.dual_leak,
                "update_precision": update_precision,
                "state_precision": state_precision,
                "dual_precision": dual_precision,
                "predicted_gradient_deadzone": deadzone,
                "minimum_update_quanta_to_move_state": minimum_quanta,
                **stats,
                "first_layer_cosine_to_bp": credit_cosine,
                "first_layer_grad_norm_ratio_to_bp": ratio,
                "first_layer_relative_error_to_bp": relative_error,
                "useful_first_layer_credit": finite
                and credit_cosine >= 0.9
                and 0.5 <= ratio <= 2.0
                and relative_error <= 0.6,
            }
        )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
