from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from predcircuit.epc import error_energy, states_from_errors, zero_errors
from predcircuit.pcalm import ResidualMLP


def stationarity(
    model: ResidualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    errors: list[torch.Tensor],
) -> float:
    variables = [error.detach().requires_grad_(True) for error in errors]
    energy = error_energy(model, x, y, variables)
    grads = torch.autograd.grad(energy, variables)
    return float(torch.sqrt(sum(grad.square().sum() for grad in grads)))


def relu_masks(
    model: ResidualMLP,
    x: torch.Tensor,
    errors: list[torch.Tensor],
) -> list[torch.Tensor]:
    hidden, _ = states_from_errors(model, x, errors)
    return [(state > 0).detach() for state in hidden]


def main() -> None:
    p = argparse.ArgumentParser(description="Trace ReLU active-set switching during ePC inference.")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--depth", type=int, default=32)
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--error-lr", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    model = ResidualMLP(
        depth=a.depth,
        width=a.width,
        input_dim=8,
        output_dim=4,
        activation="relu",
        seed=a.seed + a.depth,
    )
    gen = torch.Generator().manual_seed(a.seed + 10_000 + a.depth)
    x = torch.randn(a.batch_size, 8, generator=gen)
    y = torch.randn(a.batch_size, 4, generator=gen)

    current = zero_errors(model, x)
    initial_stationarity = stationarity(model, x, y, current)
    previous_masks = relu_masks(model, x, current)
    rows: list[dict[str, float | int | bool]] = []

    for step in range(1, a.steps + 1):
        variables = [error.detach().requires_grad_(True) for error in current]
        energy = error_energy(model, x, y, variables)
        grads = torch.autograd.grad(energy, variables)
        current = [
            (error - a.error_lr * grad).detach()
            for error, grad in zip(variables, grads, strict=True)
        ]

        current_masks = relu_masks(model, x, current)
        flips = [
            int((before != after).sum())
            for before, after in zip(previous_masks, current_masks, strict=True)
        ]
        counts = [mask.numel() for mask in current_masks]
        total_flips = sum(flips)
        total_count = sum(counts)
        current_stationarity = stationarity(model, x, y, current)

        rows.append(
            {
                "seed": a.seed,
                "step": step,
                "error_lr": a.error_lr,
                "energy_before_update": float(energy.detach()),
                "stationarity_error_grad_norm": current_stationarity,
                "stationarity_relative_to_init": (
                    current_stationarity / initial_stationarity
                    if initial_stationarity > 0.0
                    else float("nan")
                ),
                "relu_mask_flips": total_flips,
                "relu_mask_count": total_count,
                "relu_mask_flip_rate": total_flips / total_count if total_count else 0.0,
                "any_relu_mask_flip": total_flips > 0,
                "max_layer_flip_rate": max(
                    (flip / count for flip, count in zip(flips, counts, strict=True)),
                    default=0.0,
                ),
                "finite": all(bool(torch.isfinite(error).all()) for error in current),
            }
        )
        previous_masks = current_masks

    frame = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(a.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
