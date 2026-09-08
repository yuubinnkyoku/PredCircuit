from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import pandas as pd
import torch
from diagnose_local_gradient_trajectory import oracle_descent_directions
from run_flyvis_retinotopic_contrastive import (
    DIRECTIONS,
    evaluate,
    render_motion_batch,
    targets_for,
)
from run_flyvis_retinotopic_temporal_adam import temporal_local_direction

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import RetinotopicFlyVisCircuit, graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.float().flatten()
    right_flat = right.float().flatten()
    denom = torch.linalg.vector_norm(left_flat) * torch.linalg.vector_norm(right_flat)
    if float(denom) == 0.0:
        return float("nan")
    return float(torch.dot(left_flat, right_flat) / denom)


def combined_norm(edge: torch.Tensor, bias: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(torch.cat((edge.flatten(), bias.flatten())).float()))


@torch.no_grad()
def apply_mixed_credit(
    model: PredictiveCodingGraph,
    *,
    local_edge: torch.Tensor,
    local_bias: torch.Tensor,
    oracle_edge: torch.Tensor,
    oracle_bias: torch.Tensor,
    alpha: float,
    learning_rate: float,
    weight_decay: float,
    max_update: float,
) -> tuple[float, float]:
    mixed_edge = (1.0 - alpha) * local_edge + alpha * oracle_edge
    mixed_bias = (1.0 - alpha) * local_bias + alpha * oracle_bias
    edge_update = learning_rate * (mixed_edge - weight_decay * model.weight)
    bias_update = learning_rate * mixed_bias
    if max_update > 0.0:
        edge_update.clamp_(-max_update, max_update)
        bias_update.clamp_(-max_update, max_update)
    model.weight.add_(edge_update)
    model.bias.add_(bias_update)
    return float(edge_update.abs().mean()), float(edge_update.abs().max())


def cycle_directions(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    *,
    seed: int,
    epoch: int,
    frames: int,
    width: float,
    beta: float,
    frame_steps: int,
    step_size: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    local_edge_sum = torch.zeros_like(model.weight)
    local_bias_sum = torch.zeros_like(model.bias)
    oracle_edge_sum = torch.zeros_like(model.weight)
    oracle_bias_sum = torch.zeros_like(model.bias)
    directions = list(DIRECTIONS)
    random.Random(700_000 * seed + epoch).shuffle(directions)

    for sample_index, direction in enumerate(directions):
        stimulus = render_motion_batch(
            circuit,
            [direction],
            frames=frames,
            width=width,
            jitter_seed=100_000 * seed + 100 * epoch + sample_index,
        )
        targets, _ = targets_for([direction])
        local_edge, local_bias, _ = temporal_local_direction(
            model,
            circuit,
            stimulus,
            targets,
            beta=beta,
            frame_steps=frame_steps,
            step_size=step_size,
        )
        oracle_edge, oracle_bias, _ = oracle_descent_directions(
            model,
            circuit,
            stimulus,
            targets,
            frame_steps=frame_steps,
            step_size=step_size,
        )
        local_edge_sum += local_edge
        local_bias_sum += local_bias
        oracle_edge_sum += oracle_edge
        oracle_bias_sum += oracle_bias

    return local_edge_sum, local_bias_sum, oracle_edge_sum, oracle_bias_sum


def run_one(
    circuit: RetinotopicFlyVisCircuit,
    *,
    seed: int,
    alpha: float,
    epochs: int,
    frames: int,
    width: float,
    frame_steps: int,
    step_size: float,
    beta: float,
    learning_rate: float,
    weight_decay: float,
    max_update: float,
    test_repeats: int,
) -> dict[str, float | int | str | bool]:
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    mse_before, accuracy_before, margin_before = evaluate(
        model,
        circuit,
        repeats=test_repeats,
        frames=frames,
        width=width,
        frame_steps=frame_steps,
        step_size=step_size,
        jitter_seed=900_000 + seed,
    )

    initial_cosine = float("nan")
    initial_local_norm = float("nan")
    initial_oracle_norm = float("nan")
    mean_update_sum = 0.0
    max_abs_update = 0.0
    for epoch in range(epochs):
        local_edge, local_bias, oracle_edge, oracle_bias = cycle_directions(
            model,
            circuit,
            seed=seed,
            epoch=epoch,
            frames=frames,
            width=width,
            beta=beta,
            frame_steps=frame_steps,
            step_size=step_size,
        )
        if epoch == 0:
            local_combined = torch.cat((local_edge.flatten(), local_bias.flatten()))
            oracle_combined = torch.cat((oracle_edge.flatten(), oracle_bias.flatten()))
            initial_cosine = cosine(local_combined, oracle_combined)
            initial_local_norm = combined_norm(local_edge, local_bias)
            initial_oracle_norm = combined_norm(oracle_edge, oracle_bias)
        mean_update, step_max = apply_mixed_credit(
            model,
            local_edge=local_edge,
            local_bias=local_bias,
            oracle_edge=oracle_edge,
            oracle_bias=oracle_bias,
            alpha=alpha,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_update=max_update,
        )
        mean_update_sum += mean_update
        max_abs_update = max(max_abs_update, step_max)

    mse_after, accuracy_after, margin_after = evaluate(
        model,
        circuit,
        repeats=test_repeats,
        frames=frames,
        width=width,
        frame_steps=frame_steps,
        step_size=step_size,
        jitter_seed=900_000 + seed,
    )
    return {
        "learning_rule": "local_oracle_interpolation",
        "topology": "biological",
        "seed": seed,
        "alpha": alpha,
        "epochs": epochs,
        "beta": beta,
        "learning_rate": learning_rate,
        "mse_before": mse_before,
        "mse_after": mse_after,
        "mse_improvement": mse_before - mse_after,
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "margin_before": margin_before,
        "margin_after": margin_after,
        "initial_local_oracle_cosine": initial_cosine,
        "initial_local_norm": initial_local_norm,
        "initial_oracle_norm": initial_oracle_norm,
        "mean_abs_update": mean_update_sum / max(epochs, 1),
        "max_abs_update": max_abs_update,
        "mean_abs_weight": float(model.weight.abs().mean()),
        "nodes": circuit.graph.num_nodes,
        "edges": circuit.graph.num_edges,
        "finite": bool(torch.isfinite(model.weight).all()) and math.isfinite(mse_after),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate cycle-summed local predictive-coding credit with the exact "
            "descent direction to measure how much global credit information is needed"
        )
    )
    parser.add_argument("--extent", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--learning-rate", type=float, default=10.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_temporal_oracle_interpolation.csv"),
    )
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")

    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=args.extent)
    rows = [
        run_one(
            circuit,
            seed=seed,
            alpha=args.alpha,
            epochs=args.epochs,
            frames=args.frames,
            width=args.bar_width,
            frame_steps=args.frame_steps,
            step_size=args.step_size,
            beta=args.beta,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_update=args.max_update,
            test_repeats=args.test_repeats,
        )
        for seed in range(args.seeds)
    ]
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = frame[["mse_after", "accuracy_after", "margin_after"]].agg(
        ["mean", "median", "std"]
    )
    print(
        f"Local/oracle interpolation: extent={args.extent}, alpha={args.alpha:g}, "
        f"beta={args.beta:g}, lr={args.learning_rate:g}"
    )
    print(summary.to_string())
    print("\nPer-seed results:")
    print(
        frame[
            [
                "seed",
                "alpha",
                "mse_before",
                "mse_after",
                "accuracy_before",
                "accuracy_after",
                "margin_after",
                "initial_local_oracle_cosine",
                "initial_local_norm",
                "initial_oracle_norm",
                "mean_abs_update",
                "finite",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
