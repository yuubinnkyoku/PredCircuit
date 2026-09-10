from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from run_flyvis_credit_residual_scaling import evaluate_metrics
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_consensus import type_pair_indices
from run_flyvis_type_pair_linear_mix import linear_mix_direction
from run_flyvis_type_pair_norm_matched_control import match_norm

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

CHECKPOINTS = (0, 1, 2, 5, 10, 20, 50, 100)


def group_constant_fraction(direction: torch.Tensor, groups: list[torch.Tensor]) -> float:
    projected = torch.zeros_like(direction)
    for indices in groups:
        projected[indices] = direction[indices].mean()
    norm = torch.linalg.vector_norm(direction).clamp_min(1e-30)
    return float(torch.linalg.vector_norm(projected) / norm)


def run_seed(*, seed: int, learning_rate: float, max_update: float) -> pd.DataFrame:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    local_model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    shared_model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    groups = type_pair_indices(circuit)
    rows: list[dict[str, float | int | bool | str]] = []
    running_geometry = {
        "local": {"relative_change": 0.0, "constant_fraction": 0.0, "count": 0},
        "type_pair_norm_matched": {
            "relative_change": 0.0,
            "constant_fraction": 0.0,
            "count": 0,
        },
    }

    def record(epoch: int) -> None:
        for rule, model in (
            ("local", local_model),
            ("type_pair_norm_matched", shared_model),
        ):
            metrics = evaluate_metrics(
                model,
                circuit,
                repeats=8,
                frames=7,
                width=0.5,
                frame_steps=2,
                step_size=0.015,
                jitter_seed=900_000 + seed,
            )
            geometry = running_geometry[rule]
            count = int(geometry["count"])
            rows.append(
                {
                    "seed": seed,
                    "rule": rule,
                    "epoch": epoch,
                    "cross_entropy": metrics["cross_entropy"],
                    "accuracy": metrics["accuracy"],
                    "margin": metrics["margin"],
                    "weight_norm": float(torch.linalg.vector_norm(model.weight)),
                    "bias_norm": float(torch.linalg.vector_norm(model.bias)),
                    "mean_relative_direction_change": (
                        float(geometry["relative_change"]) / count if count else 0.0
                    ),
                    "mean_group_constant_fraction": (
                        float(geometry["constant_fraction"]) / count if count else 0.0
                    ),
                    "finite": bool(torch.isfinite(model.weight).all())
                    and bool(torch.isfinite(model.bias).all())
                    and math.isfinite(metrics["cross_entropy"]),
                }
            )

    record(0)
    for epoch in range(100):
        for rule, model in (
            ("local", local_model),
            ("type_pair_norm_matched", shared_model),
        ):
            local_edge, local_bias = cycle_branched_credit(
                model,
                circuit,
                seed=seed,
                epoch=epoch,
                frames=7,
                width=0.5,
                beta=0.03,
                frame_steps=2,
                nudge_steps=2,
                step_size=0.015,
                terminal_only=False,
            )
            if rule == "local":
                edge_direction = local_edge
            else:
                mixed = linear_mix_direction(
                    local_edge,
                    groups,
                    coarse_gain=4.0,
                    residual_gain=1.0,
                )
                edge_direction = match_norm(mixed, local_edge)

            local_norm = torch.linalg.vector_norm(local_edge).clamp_min(1e-30)
            geometry = running_geometry[rule]
            geometry["relative_change"] = float(geometry["relative_change"]) + float(
                torch.linalg.vector_norm(edge_direction - local_edge) / local_norm
            )
            geometry["constant_fraction"] = float(geometry["constant_fraction"]) + (
                group_constant_fraction(edge_direction, groups)
            )
            geometry["count"] = int(geometry["count"]) + 1

            apply_local_credit(
                model,
                edge_direction,
                local_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )

        completed_epoch = epoch + 1
        if completed_epoch in CHECKPOINTS:
            record(completed_epoch)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace when norm-matched type-pair shared credit separates margin from CE"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frame = run_seed(
        seed=args.seed,
        learning_rate=args.learning_rate,
        max_update=args.max_update,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
