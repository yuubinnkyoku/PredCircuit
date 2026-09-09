from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from run_flyvis_credit_residual_scaling import evaluate_metrics
from run_flyvis_retinotopic_temporal_adam import local_adam_step

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def run_one(
    *,
    extent: int,
    seed: int,
    epochs: int,
    frames: int,
    width: float,
    frame_steps: int,
    nudge_steps: int,
    step_size: float,
    beta: float,
    learning_rate: float,
    beta1: float,
    beta2: float,
    adam_epsilon: float,
    test_repeats: int,
) -> dict[str, float | int | bool]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=extent)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    edge_m = torch.zeros_like(model.weight)
    edge_v = torch.zeros_like(model.weight)
    bias_m = torch.zeros_like(model.bias)
    bias_v = torch.zeros_like(model.bias)

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
    raw_direction_sum = 0.0
    applied_update_sum = 0.0
    max_applied_update = 0.0

    for epoch in range(epochs):
        edge_direction, bias_direction = cycle_branched_credit(
            model,
            circuit,
            seed=seed,
            epoch=epoch,
            frames=frames,
            width=width,
            beta=beta,
            frame_steps=frame_steps,
            nudge_steps=nudge_steps,
            step_size=step_size,
            terminal_only=False,
        )
        edge_delta = local_adam_step(
            model.weight,
            edge_direction,
            edge_m,
            edge_v,
            step=epoch + 1,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=adam_epsilon,
        )
        local_adam_step(
            model.bias,
            bias_direction,
            bias_m,
            bias_v,
            step=epoch + 1,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=adam_epsilon,
        )
        raw_direction_sum += float(edge_direction.abs().mean())
        applied_update_sum += float(edge_delta.abs().mean())
        max_applied_update = max(max_applied_update, float(edge_delta.abs().max()))

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
    return {
        "seed": seed,
        "epochs": epochs,
        "frame_steps": frame_steps,
        "nudge_steps": nudge_steps,
        "beta": beta,
        "learning_rate": learning_rate,
        "beta1": beta1,
        "beta2": beta2,
        "adam_epsilon": adam_epsilon,
        "mse_before": before["mse"],
        "mse_after": after["mse"],
        "accuracy_before": before["accuracy"],
        "accuracy_after": after["accuracy"],
        "margin_before": before["margin"],
        "margin_after": after["margin"],
        "cross_entropy_before": before["cross_entropy"],
        "cross_entropy_after": after["cross_entropy"],
        "cross_entropy_improvement": before["cross_entropy"] - after["cross_entropy"],
        "mean_abs_raw_direction": raw_direction_sum / max(epochs, 1),
        "mean_abs_applied_update": applied_update_sum / max(epochs, 1),
        "max_abs_applied_update": max_applied_update,
        "mean_abs_weight": float(model.weight.abs().mean()),
        "nodes": circuit.graph.num_nodes,
        "edges": circuit.graph.num_edges,
        "finite": bool(torch.isfinite(model.weight).all())
        and math.isfinite(after["cross_entropy"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train per-frame branched classification credit with local Adam adaptation"
    )
    parser.add_argument("--extent", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--nudge-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_branched_classification_adam.csv"),
    )
    args = parser.parse_args()

    rows = [
        run_one(
            extent=args.extent,
            seed=seed,
            epochs=args.epochs,
            frames=args.frames,
            width=args.bar_width,
            frame_steps=args.frame_steps,
            nudge_steps=args.nudge_steps,
            step_size=args.step_size,
            beta=args.beta,
            learning_rate=args.learning_rate,
            beta1=args.beta1,
            beta2=args.beta2,
            adam_epsilon=args.adam_epsilon,
            test_repeats=args.test_repeats,
        )
        for seed in range(args.seeds)
    ]
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = frame[
        [
            "accuracy_after",
            "cross_entropy_after",
            "cross_entropy_improvement",
            "margin_after",
            "mse_after",
            "mean_abs_applied_update",
            "max_abs_applied_update",
        ]
    ].agg(["mean", "median", "std"])
    print(
        f"Branched classification Adam: lr={args.learning_rate:g}, seeds={args.seeds}, "
        f"nudge_steps={args.nudge_steps}"
    )
    print(summary.to_string())
    print("\nPer-seed results:")
    print(
        frame[
            [
                "seed",
                "accuracy_after",
                "cross_entropy_after",
                "cross_entropy_improvement",
                "margin_after",
                "mean_abs_applied_update",
                "max_abs_applied_update",
                "finite",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
