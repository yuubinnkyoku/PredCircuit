from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from diagnose_flyvis_checkpoint_component_geometry import component_geometry
from diagnose_flyvis_temporal_ce_credit_alignment import cycle_ce_oracles
from run_flyvis_credit_residual_scaling import evaluate_metrics

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def run_one(
    *,
    seed: int,
    bias_lr_factor: float,
    epochs: int,
    edge_learning_rate: float,
    nudge_steps: int,
    max_update: float,
    test_repeats: int,
) -> dict[str, float | int | bool]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    before = evaluate_metrics(
        model,
        circuit,
        repeats=test_repeats,
        frames=7,
        width=0.5,
        frame_steps=2,
        step_size=0.015,
        jitter_seed=900_000 + seed,
    )
    edge_clip_sum = 0.0
    bias_clip_sum = 0.0

    for epoch in range(epochs):
        edge_direction, bias_direction = cycle_branched_credit(
            model,
            circuit,
            seed=seed,
            epoch=epoch,
            frames=7,
            width=0.5,
            beta=0.03,
            frame_steps=2,
            nudge_steps=nudge_steps,
            step_size=0.015,
            terminal_only=False,
        )
        edge_update = edge_learning_rate * edge_direction
        bias_update = edge_learning_rate * bias_lr_factor * bias_direction
        if max_update > 0.0:
            edge_clip_sum += float((edge_update.abs() > max_update).float().mean())
            bias_clip_sum += float((bias_update.abs() > max_update).float().mean())
            edge_update.clamp_(-max_update, max_update)
            bias_update.clamp_(-max_update, max_update)
        with torch.no_grad():
            model.weight.add_(edge_update)
            model.bias.add_(bias_update)

    after = evaluate_metrics(
        model,
        circuit,
        repeats=test_repeats,
        frames=7,
        width=0.5,
        frame_steps=2,
        step_size=0.015,
        jitter_seed=900_000 + seed,
    )
    local_edge, local_bias = cycle_branched_credit(
        model,
        circuit,
        seed=seed,
        epoch=0,
        frames=7,
        width=0.5,
        beta=0.03,
        frame_steps=2,
        nudge_steps=nudge_steps,
        step_size=0.015,
        terminal_only=False,
    )
    oracle_edge, oracle_bias = cycle_ce_oracles(
        model,
        circuit,
        seed=seed,
        epoch=0,
        frames=7,
        width=0.5,
        frame_steps=2,
        step_size=0.015,
    )["final_ce_oracle"]
    final_geometry = component_geometry(
        local_edge,
        local_bias,
        oracle_edge,
        oracle_bias,
    )
    return {
        "seed": seed,
        "epochs": epochs,
        "nudge_steps": nudge_steps,
        "edge_learning_rate": edge_learning_rate,
        "bias_lr_factor": bias_lr_factor,
        "bias_learning_rate": edge_learning_rate * bias_lr_factor,
        "cross_entropy_before": before["cross_entropy"],
        "cross_entropy_after": after["cross_entropy"],
        "cross_entropy_improvement": before["cross_entropy"] - after["cross_entropy"],
        "accuracy_after": after["accuracy"],
        "margin_after": after["margin"],
        "weight_norm": float(torch.linalg.vector_norm(model.weight)),
        "bias_parameter_norm": float(torch.linalg.vector_norm(model.bias)),
        "mean_edge_clip_fraction": edge_clip_sum / max(epochs, 1),
        "mean_bias_clip_fraction": bias_clip_sum / max(epochs, 1),
        "final_full_cosine": final_geometry["full_cosine"],
        "final_edge_cosine": final_geometry["edge_cosine"],
        "final_bias_cosine": final_geometry["bias_cosine"],
        "finite": bool(torch.isfinite(model.weight).all())
        and bool(torch.isfinite(model.bias).all())
        and math.isfinite(after["cross_entropy"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep the bias learning-rate factor under branched local SGD"
    )
    parser.add_argument("--bias-lr-factor", type=float, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--edge-learning-rate", type=float, default=160.0)
    parser.add_argument("--nudge-steps", type=int, default=2)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_branched_bias_lr_sweep.csv"),
    )
    args = parser.parse_args()

    frame = pd.DataFrame(
        [
            run_one(
                seed=seed,
                bias_lr_factor=args.bias_lr_factor,
                epochs=args.epochs,
                edge_learning_rate=args.edge_learning_rate,
                nudge_steps=args.nudge_steps,
                max_update=args.max_update,
                test_repeats=args.test_repeats,
            )
            for seed in range(args.seeds)
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.select_dtypes(include="number").agg(["mean", "median", "std"]).to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
