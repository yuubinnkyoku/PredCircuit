from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from run_flyvis_credit_residual_scaling import evaluate_metrics
from run_flyvis_temporal_coherence_gate import (
    apply_local_credit,
    coherence_gated_sum,
    cycle_local_credits,
)

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import RetinotopicFlyVisCircuit, graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def match_mean_absolute_norm(
    gated: torch.Tensor,
    reference: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> tuple[torch.Tensor, float]:
    """Rescale a gated credit vector to the reference mean-absolute magnitude.

    This keeps the coherence gate's relative per-synapse reweighting while removing the
    trivial explanation that gating merely reduces the global effective learning rate.
    """
    gated_mean = gated.abs().mean()
    reference_mean = reference.abs().mean()
    if float(reference_mean) <= epsilon:
        return torch.zeros_like(gated), 0.0
    if float(gated_mean) <= epsilon:
        return gated, 0.0
    scale = reference_mean / gated_mean
    return gated * scale, float(scale)


def run_one(
    circuit: RetinotopicFlyVisCircuit,
    *,
    seed: int,
    coherence_power: float,
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

    mean_update_sum = 0.0
    max_abs_update = 0.0
    edge_coherence_sum = 0.0
    edge_scale_sum = 0.0
    bias_scale_sum = 0.0
    pre_match_ratio_sum = 0.0

    for epoch in range(epochs):
        edge_credits, bias_credits = cycle_local_credits(
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
        reference_edge = edge_credits.sum(dim=0)
        reference_bias = bias_credits.sum(dim=0)
        gated_edge, edge_coherence = coherence_gated_sum(
            edge_credits,
            power=coherence_power,
        )
        gated_bias, _ = coherence_gated_sum(
            bias_credits,
            power=coherence_power,
        )

        reference_mean = reference_edge.abs().mean()
        gated_mean = gated_edge.abs().mean()
        if float(reference_mean) > 1e-12:
            pre_match_ratio_sum += float(gated_mean / reference_mean)

        edge_direction, edge_scale = match_mean_absolute_norm(gated_edge, reference_edge)
        bias_direction, bias_scale = match_mean_absolute_norm(gated_bias, reference_bias)
        mean_update, step_max = apply_local_credit(
            model,
            edge_direction,
            bias_direction,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_update=max_update,
        )
        mean_update_sum += mean_update
        max_abs_update = max(max_abs_update, step_max)
        edge_coherence_sum += float(edge_coherence.mean())
        edge_scale_sum += edge_scale
        bias_scale_sum += bias_scale

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
        "coherence_power": coherence_power,
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
        "mean_edge_coherence": edge_coherence_sum / count,
        "mean_pre_match_norm_ratio": pre_match_ratio_sum / count,
        "mean_edge_match_scale": edge_scale_sum / count,
        "mean_bias_match_scale": bias_scale_sum / count,
        "mean_abs_update": mean_update_sum / count,
        "max_abs_update": max_abs_update,
        "nodes": circuit.graph.num_nodes,
        "edges": circuit.graph.num_edges,
        "finite": bool(torch.isfinite(model.weight).all()) and math.isfinite(after["mse"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Control whether local coherence gating helps beyond merely shrinking updates by "
            "renormalizing every gated cycle to the ungated mean-absolute credit magnitude"
        )
    )
    parser.add_argument("--extent", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--coherence-power", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--learning-rate", type=float, default=30.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_coherence_norm_control.csv"),
    )
    args = parser.parse_args()
    if args.coherence_power < 0.0:
        raise ValueError("--coherence-power must be non-negative")

    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=args.extent)
    rows = [
        run_one(
            circuit,
            seed=seed,
            coherence_power=args.coherence_power,
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
            "mean_pre_match_norm_ratio",
            "mean_edge_match_scale",
            "mean_abs_update",
        ]
    ].agg(["mean", "median", "std"])
    print(
        f"Norm-matched coherence: extent={args.extent}, power={args.coherence_power:g}, "
        f"lr={args.learning_rate:g}, {circuit.graph.num_nodes} nodes, "
        f"{circuit.graph.num_edges} edges"
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
                "mean_pre_match_norm_ratio",
                "mean_edge_match_scale",
                "mean_abs_update",
                "finite",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
