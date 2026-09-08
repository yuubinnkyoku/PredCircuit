from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from run_flyvis_retinotopic_contrastive import (
    DIRECTIONS,
    infer_sequence,
    output_nodes,
    render_motion_batch,
    targets_for,
)
from run_flyvis_temporal_credit_projection import apply_credit, projection_decomposition
from run_flyvis_temporal_oracle_interpolation import cycle_directions

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import RetinotopicFlyVisCircuit, graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


@torch.no_grad()
def evaluate_metrics(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    *,
    repeats: int,
    frames: int,
    width: float,
    frame_steps: int,
    step_size: float,
    jitter_seed: int,
) -> dict[str, float]:
    directions = list(DIRECTIONS) * repeats
    stimulus = render_motion_batch(
        circuit,
        directions,
        frames=frames,
        width=width,
        jitter_seed=jitter_seed,
    )
    targets, classes = targets_for(directions)
    state = infer_sequence(
        model,
        circuit,
        stimulus,
        frame_steps=frame_steps,
        step_size=step_size,
    )
    readout = state[:, output_nodes(circuit)]
    correct = readout.gather(1, classes[:, None]).squeeze(1)
    masked = readout.clone()
    masked[torch.arange(len(classes)), classes] = -torch.inf
    return {
        "mse": float((readout - targets).square().mean()),
        "accuracy": float((readout.argmax(dim=1) == classes).float().mean()),
        "margin": float((correct - masked.max(dim=1).values).mean()),
        "cross_entropy": float(F.cross_entropy(readout, classes)),
    }


def run_one(
    circuit: RetinotopicFlyVisCircuit,
    *,
    seed: int,
    residual_scale: float,
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
) -> dict[str, float | int | bool]:
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
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

    cosine_sum = 0.0
    residual_fraction_sum = 0.0
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
        local, parallel, residual, _, cosine = projection_decomposition(
            local_edge,
            local_bias,
            oracle_edge,
            oracle_bias,
        )
        direction = parallel + residual_scale * residual
        mean_update, step_max = apply_credit(
            model,
            direction,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_update=max_update,
        )
        local_norm = torch.linalg.vector_norm(local).clamp_min(1e-30)
        cosine_sum += cosine
        residual_fraction_sum += float(torch.linalg.vector_norm(residual) / local_norm)
        mean_update_sum += mean_update
        max_abs_update = max(max_abs_update, step_max)

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
        "residual_scale": residual_scale,
        "epochs": epochs,
        "beta": beta,
        "learning_rate": learning_rate,
        "mse_before": before["mse"],
        "mse_after": after["mse"],
        "accuracy_before": before["accuracy"],
        "accuracy_after": after["accuracy"],
        "margin_before": before["margin"],
        "margin_after": after["margin"],
        "cross_entropy_before": before["cross_entropy"],
        "cross_entropy_after": after["cross_entropy"],
        "cross_entropy_improvement": before["cross_entropy"] - after["cross_entropy"],
        "mean_local_oracle_cosine": cosine_sum / count,
        "mean_residual_norm_fraction": residual_fraction_sum / count,
        "mean_abs_update": mean_update_sum / count,
        "max_abs_update": max_abs_update,
        "nodes": circuit.graph.num_nodes,
        "edges": circuit.graph.num_edges,
        "finite": bool(torch.isfinite(model.weight).all()) and math.isfinite(after["mse"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure how orthogonal local-credit residuals scale with FlyVis crop extent"
    )
    parser.add_argument("--extent", type=int, required=True)
    parser.add_argument("--residual-scale", type=float, required=True)
    parser.add_argument("--seeds", type=int, default=10)
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
        default=Path("results/generated/flyvis_credit_residual_scaling.csv"),
    )
    args = parser.parse_args()
    if args.extent < 0:
        raise ValueError("--extent must be non-negative")
    if not 0.0 <= args.residual_scale <= 1.0:
        raise ValueError("--residual-scale must be between 0 and 1")

    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=args.extent)
    rows = [
        run_one(
            circuit,
            seed=seed,
            residual_scale=args.residual_scale,
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
    summary = frame[
        [
            "mse_after",
            "accuracy_after",
            "margin_after",
            "cross_entropy_after",
            "mean_local_oracle_cosine",
            "mean_residual_norm_fraction",
        ]
    ].agg(["mean", "median", "std"])
    print(
        f"Residual scaling: extent={args.extent}, residual_scale={args.residual_scale:g}, "
        f"{circuit.graph.num_nodes} nodes, {circuit.graph.num_edges} edges"
    )
    print(summary.to_string())
    print("\nPer-seed results:")
    print(
        frame[
            [
                "seed",
                "accuracy_after",
                "mse_after",
                "cross_entropy_after",
                "margin_after",
                "mean_local_oracle_cosine",
                "mean_residual_norm_fraction",
                "finite",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
