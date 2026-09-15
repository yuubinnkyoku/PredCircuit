from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from predcircuit.pcalm import (
    ResidualMLP,
    _solve_inner,
    constraint_residuals,
    free_init,
    zero_duals_like,
)

# SakanaAI/pc-alm tuned activity step sizes for the reference ResidualMLP.
REFERENCE_STATE_LR = {8: 0.209541, 16: 0.221921, 32: 0.234285, 64: 0.242954, 128: 0.247725}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    ix = round((len(ordered) - 1) * q)
    return ordered[ix]


def integer_bits_for_signed(max_abs: float) -> int:
    """Return sign + magnitude bits needed for a signed fixed-point value."""
    if max_abs <= 0.0:
        return 1
    return max(1, math.ceil(math.log2(max_abs)) + 1)


def format_stats(
    *, total_bits: int, max_abs_dual: float, nonzero_updates: list[float]
) -> dict[str, float | int]:
    integer_bits = integer_bits_for_signed(max_abs_dual)
    fractional_bits = total_bits - integer_bits
    if fractional_bits < 0:
        return {
            "integer_bits": integer_bits,
            "fractional_bits": fractional_bits,
            "lsb": float("inf"),
            "lost_nonzero_update_fraction": 1.0,
        }
    lsb = 2.0 ** (-fractional_bits)
    threshold = 0.5 * lsb
    lost = sum(value < threshold for value in nonzero_updates)
    return {
        "integer_bits": integer_bits,
        "fractional_bits": fractional_bits,
        "lsb": lsb,
        "lost_nonzero_update_fraction": lost / max(len(nonzero_updates), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure real PC-ALM lambda/update ranges.")
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--output-dim", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--state-lr", type=float, default=None)
    parser.add_argument("--activation", choices=["linear", "tanh", "relu"], default="relu")
    parser.add_argument("--seed", type=int, default=940)
    parser.add_argument("--out", type=Path, default=Path("results/generated/pcalm_numeric_range.json"))
    args = parser.parse_args()

    state_lr = args.state_lr
    if state_lr is None:
        if args.depth not in REFERENCE_STATE_LR:
            raise ValueError("provide --state-lr for depths without a reference value")
        state_lr = REFERENCE_STATE_LR[args.depth]

    model = ResidualMLP(
        depth=args.depth,
        width=args.width,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        activation=args.activation,
        seed=args.seed,
    )
    gen = torch.Generator().manual_seed(args.seed + 10_000)
    x = torch.randn(args.batch_size, args.input_dim, generator=gen)
    y = torch.randn(args.batch_size, args.output_dim, generator=gen)

    free = free_init(model, x)
    duals = zero_duals_like(constraint_residuals(model, x, free))
    layer_count = args.depth - 1
    per_layer_max_dual = [0.0] * layer_count
    per_layer_max_residual = [0.0] * layer_count
    per_layer_max_update = [0.0] * layer_count
    per_layer_nonzero_updates: list[list[float]] = [[] for _ in range(layer_count)]
    all_updates: list[float] = []
    all_nonzero_updates: list[float] = []
    finite = True

    for _ in range(args.budget):
        free = _solve_inner(
            model,
            x,
            y,
            free,
            duals,
            state_lr=state_lr,
            rho=args.rho,
            steps=1,
        )
        residuals = constraint_residuals(model, x, free)
        duals = [
            (dual + args.alpha * residual).detach()
            for dual, residual in zip(duals, residuals, strict=True)
        ]
        for layer, (residual, dual) in enumerate(zip(residuals, duals, strict=True)):
            update_tensor = (args.alpha * residual).abs().reshape(-1)
            update_values = [float(value) for value in update_tensor]
            nonzero = [value for value in update_values if value > 0.0]
            all_updates.extend(update_values)
            all_nonzero_updates.extend(nonzero)
            per_layer_nonzero_updates[layer].extend(nonzero)
            per_layer_max_dual[layer] = max(per_layer_max_dual[layer], float(dual.abs().max()))
            per_layer_max_residual[layer] = max(
                per_layer_max_residual[layer], float(residual.abs().max())
            )
            per_layer_max_update[layer] = max(
                per_layer_max_update[layer], float(update_tensor.max())
            )
            finite = finite and bool(torch.isfinite(residual).all()) and bool(torch.isfinite(dual).all())

    global_max_dual = max(per_layer_max_dual)
    fixed_point = {
        str(bits): format_stats(
            total_bits=bits,
            max_abs_dual=global_max_dual,
            nonzero_updates=all_nonzero_updates,
        )
        for bits in (16, 12, 8)
    }
    per_layer_fixed_point = {
        str(bits): [
            format_stats(
                total_bits=bits,
                max_abs_dual=per_layer_max_dual[layer],
                nonzero_updates=per_layer_nonzero_updates[layer],
            )
            for layer in range(layer_count)
        ]
        for bits in (16, 12, 8)
    }

    result = {
        "depth": args.depth,
        "width": args.width,
        "seed": args.seed,
        "budget": args.budget,
        "state_lr": state_lr,
        "rho": args.rho,
        "alpha": args.alpha,
        "finite": finite,
        "global_max_abs_dual": global_max_dual,
        "global_max_abs_residual": max(per_layer_max_residual),
        "global_max_abs_dual_update": max(per_layer_max_update),
        "dual_update_zero_fraction_fp32": 1.0 - len(all_nonzero_updates) / max(len(all_updates), 1),
        "nonzero_dual_update_p50": percentile(all_nonzero_updates, 0.50),
        "nonzero_dual_update_p10": percentile(all_nonzero_updates, 0.10),
        "nonzero_dual_update_p01": percentile(all_nonzero_updates, 0.01),
        "per_layer_max_abs_dual": per_layer_max_dual,
        "per_layer_max_abs_residual": per_layer_max_residual,
        "per_layer_max_abs_dual_update": per_layer_max_update,
        "global_fixed_point_from_fp32_trajectory": fixed_point,
        "per_layer_fixed_point_from_fp32_trajectory": per_layer_fixed_point,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
