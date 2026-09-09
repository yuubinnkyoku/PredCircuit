from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from run_flyvis_credit_residual_scaling import evaluate_metrics
from run_flyvis_exact_pc_gradient import differentiable_sequence, sync_model
from run_flyvis_retinotopic_contrastive import (
    DIRECTIONS,
    output_nodes,
    render_motion_batch,
    targets_for,
)

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def flatten_pair(edge: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.cat((edge.flatten(), bias.flatten()))


def split_pair(vector: torch.Tensor, edge_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    return vector[:edge_size], vector[edge_size:]


def perturb_to_cosine(
    gradient: torch.Tensor,
    *,
    target_cosine: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if not 0.0 < target_cosine <= 1.0:
        raise ValueError("target_cosine must be in (0, 1]")
    norm = torch.linalg.vector_norm(gradient)
    if float(norm) <= 1e-30 or target_cosine == 1.0:
        return gradient.clone()
    random_vector = torch.randn(
        gradient.shape,
        generator=generator,
        dtype=gradient.dtype,
        device=gradient.device,
    )
    unit_gradient = gradient / norm
    orthogonal = random_vector - torch.dot(random_vector, unit_gradient) * unit_gradient
    orthogonal_norm = torch.linalg.vector_norm(orthogonal).clamp_min(1e-30)
    orthogonal = orthogonal / orthogonal_norm
    sine = math.sqrt(max(1.0 - target_cosine**2, 0.0))
    return norm * (target_cosine * unit_gradient + sine * orthogonal)


def per_direction_ce_gradients(
    circuit,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    seed: int,
    epoch: int,
    frames: int,
    width: float,
    frame_steps: int,
    step_size: float,
    train_repeats: int,
) -> list[torch.Tensor]:
    gradients: list[torch.Tensor] = []
    for direction_index, direction in enumerate(DIRECTIONS):
        directions = [direction] * train_repeats
        stimulus = render_motion_batch(
            circuit,
            directions,
            frames=frames,
            width=width,
            jitter_seed=100_000 * seed + 100 * epoch + direction_index,
        )
        _, classes = targets_for(directions)
        state = differentiable_sequence(
            circuit,
            stimulus,
            weight,
            bias,
            frame_steps=frame_steps,
            step_size=step_size,
        )
        loss = F.cross_entropy(state[:, output_nodes(circuit)], classes)
        edge_grad, bias_grad = torch.autograd.grad(loss, (weight, bias))
        gradients.append(flatten_pair(edge_grad.detach(), bias_grad.detach()))
    return gradients


def run_one(
    *,
    seed: int,
    target_cosine: float,
    epochs: int,
    frames: int,
    width: float,
    frame_steps: int,
    step_size: float,
    learning_rate: float,
    train_repeats: int,
    test_repeats: int,
) -> dict[str, float | int | bool]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    weight = torch.nn.Parameter(model.weight.detach().clone())
    bias = torch.nn.Parameter(model.bias.detach().clone())
    optimizer = torch.optim.Adam([weight, bias], lr=learning_rate)
    generator = torch.Generator().manual_seed(7_000_000 + seed)

    before = evaluate_metrics(
        model,
        circuit,
        repeats=test_repeats,
        frames=frames,
        width=width,
        frame_steps=frame_steps,
        step_size=step_size,
        jitter_seed=900_000 + seed,
    )
    aggregate_cosine_sum = 0.0
    relative_noise_sum = 0.0
    exact_cancellation_sum = 0.0

    for epoch in range(epochs):
        exact_parts = per_direction_ce_gradients(
            circuit,
            weight,
            bias,
            seed=seed,
            epoch=epoch,
            frames=frames,
            width=width,
            frame_steps=frame_steps,
            step_size=step_size,
            train_repeats=train_repeats,
        )
        noisy_parts = [
            perturb_to_cosine(part, target_cosine=target_cosine, generator=generator)
            for part in exact_parts
        ]
        exact = torch.stack(exact_parts).mean(dim=0)
        noisy = torch.stack(noisy_parts).mean(dim=0)
        exact_norm = torch.linalg.vector_norm(exact).clamp_min(1e-30)
        noisy_norm = torch.linalg.vector_norm(noisy).clamp_min(1e-30)
        aggregate_cosine_sum += float(torch.dot(exact, noisy) / (exact_norm * noisy_norm))
        relative_noise_sum += float(torch.linalg.vector_norm(noisy - exact) / exact_norm)
        individual_norm_sum = sum(float(torch.linalg.vector_norm(part)) for part in exact_parts)
        exact_cancellation_sum += (
            float(exact_norm) * len(exact_parts) / max(individual_norm_sum, 1e-30)
        )

        edge_grad, bias_grad = split_pair(noisy, weight.numel())
        optimizer.zero_grad(set_to_none=True)
        weight.grad = edge_grad.reshape_as(weight).clone()
        bias.grad = bias_grad.reshape_as(bias).clone()
        torch.nn.utils.clip_grad_norm_([weight, bias], 10.0)
        optimizer.step()

    sync_model(model, weight, bias)
    after = evaluate_metrics(
        model,
        circuit,
        repeats=test_repeats,
        frames=frames,
        width=width,
        frame_steps=frame_steps,
        step_size=step_size,
        jitter_seed=900_000 + seed,
    )
    count = max(epochs, 1)
    return {
        "seed": seed,
        "target_individual_cosine": target_cosine,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "accuracy_before": before["accuracy"],
        "accuracy_after": after["accuracy"],
        "cross_entropy_before": before["cross_entropy"],
        "cross_entropy_after": after["cross_entropy"],
        "cross_entropy_improvement": before["cross_entropy"] - after["cross_entropy"],
        "margin_before": before["margin"],
        "margin_after": after["margin"],
        "mse_after": after["mse"],
        "mean_aggregate_cosine": aggregate_cosine_sum / count,
        "mean_noise_to_exact_aggregate_norm": relative_noise_sum / count,
        "mean_exact_cancellation_ratio": exact_cancellation_sum / count,
        "finite": bool(torch.isfinite(weight).all()) and math.isfinite(after["cross_entropy"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test cancellation sensitivity by perturbing exact per-direction CE gradients"
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--target-cosine", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--train-repeats", type=int, default=4)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_noisy_per_direction_ce_oracle.csv"),
    )
    args = parser.parse_args()

    rows = [
        run_one(
            seed=seed,
            target_cosine=args.target_cosine,
            epochs=args.epochs,
            frames=args.frames,
            width=args.bar_width,
            frame_steps=args.frame_steps,
            step_size=args.step_size,
            learning_rate=args.learning_rate,
            train_repeats=args.train_repeats,
            test_repeats=args.test_repeats,
        )
        for seed in range(args.seeds)
    ]
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    metrics = [
        "accuracy_after",
        "cross_entropy_after",
        "cross_entropy_improvement",
        "margin_after",
        "mean_aggregate_cosine",
        "mean_noise_to_exact_aggregate_norm",
        "mean_exact_cancellation_ratio",
    ]
    print(
        f"Noisy per-direction CE oracle: target cosine={args.target_cosine:g}, seeds={args.seeds}"
    )
    print(frame[metrics].agg(["mean", "median", "std"]).to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
