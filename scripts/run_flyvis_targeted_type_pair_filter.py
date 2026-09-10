from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from diagnose_flyvis_temporal_ce_credit_alignment import cycle_ce_oracles, geometry
from diagnose_flyvis_type_pair_credit_curvature import type_pair_indices
from run_flyvis_credit_residual_scaling import evaluate_metrics
from run_flyvis_strict_residual_controls import match_projection_and_norm
from run_flyvis_temporal_coherence_gate import apply_local_credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import RetinotopicFlyVisCircuit, graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

DISCOVERY_PAIRS = {
    "robust_t4a_t4c": {("T4a", "T4c")},
    "t4_mi9": {("T4a", "Mi9"), ("T4b", "Mi9"), ("T4c", "Mi9")},
    "top4": {
        ("T4a", "Mi9"),
        ("T4b", "Mi9"),
        ("T4c", "Mi9"),
        ("T4a", "T4c"),
    },
}


def targeted_direction(
    circuit: RetinotopicFlyVisCircuit,
    local_edge: torch.Tensor,
    local_bias: torch.Tensor,
    oracle_edge: torch.Tensor,
    oracle_bias: torch.Tensor,
    *,
    rule: str,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if rule == "local":
        return local_edge, local_bias, 0.0

    local = torch.cat((local_edge, local_bias))
    oracle = torch.cat((oracle_edge, oracle_bias))
    alpha = torch.dot(local, oracle) / torch.dot(oracle, oracle).clamp_min(1e-30)
    residual_edge = local_edge - alpha * oracle_edge
    residual_bias = local_bias - alpha * oracle_bias
    filtered_edge = residual_edge.clone()

    groups = type_pair_indices(circuit)
    selected = set(groups) if rule == "all_type_pair" else DISCOVERY_PAIRS[rule]
    for pair in selected:
        indices = groups.get(pair)
        if indices is None:
            continue
        target = residual_edge[indices]
        candidate = torch.ones_like(target) * target.mean()
        filtered_edge[indices] = match_projection_and_norm(
            candidate,
            oracle_edge[indices],
            target,
        )

    direction_edge = alpha * oracle_edge + filtered_edge
    direction_bias = alpha * oracle_bias + residual_bias
    relative_change = float(
        torch.linalg.vector_norm(direction_edge - local_edge)
        / torch.linalg.vector_norm(local_edge).clamp_min(1e-30)
    )
    return direction_edge, direction_bias, relative_change


def run_seed(
    *,
    seed: int,
    rule: str,
    epochs: int,
    start_epoch: int,
    nudge_steps: int,
    learning_rate: float,
    max_update: float,
    test_repeats: int,
) -> dict[str, float | int | str | bool]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    eval_kwargs = {
        "repeats": test_repeats,
        "frames": 7,
        "width": 0.5,
        "frame_steps": 2,
        "step_size": 0.015,
        "jitter_seed": 900_000 + seed,
    }
    before = evaluate_metrics(model, circuit, **eval_kwargs)
    cosine_error_sum = 0.0
    norm_error_sum = 0.0
    projection_error_sum = 0.0
    relative_change_sum = 0.0

    for epoch in range(epochs):
        local_edge, local_bias = cycle_branched_credit(
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
        oracle_edge, oracle_bias = cycle_ce_oracles(
            model,
            circuit,
            seed=seed,
            epoch=epoch,
            frames=7,
            width=0.5,
            frame_steps=2,
            step_size=0.015,
        )["final_ce_oracle"]
        active_rule = rule if epoch >= start_epoch else "local"
        edge_direction, bias_direction, relative_change = targeted_direction(
            circuit,
            local_edge,
            local_bias,
            oracle_edge,
            oracle_bias,
            rule=active_rule,
        )
        local_geometry = geometry(local_edge, local_bias, oracle_edge, oracle_bias)
        filtered_geometry = geometry(edge_direction, bias_direction, oracle_edge, oracle_bias)
        cosine_error_sum += abs(filtered_geometry["cosine"] - local_geometry["cosine"])
        norm_error_sum += abs(filtered_geometry["norm_ratio"] - local_geometry["norm_ratio"])
        projection_error_sum += abs(
            filtered_geometry["projection_coefficient"] - local_geometry["projection_coefficient"]
        )
        relative_change_sum += relative_change

        apply_local_credit(
            model,
            edge_direction,
            bias_direction,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )

    after = evaluate_metrics(model, circuit, **eval_kwargs)
    count = max(epochs, 1)
    return {
        "rule": rule,
        "seed": seed,
        "epochs": epochs,
        "start_epoch": start_epoch,
        "nudge_steps": nudge_steps,
        "learning_rate": learning_rate,
        "accuracy_before": before["accuracy"],
        "accuracy_after": after["accuracy"],
        "cross_entropy_before": before["cross_entropy"],
        "cross_entropy_after": after["cross_entropy"],
        "cross_entropy_improvement": before["cross_entropy"] - after["cross_entropy"],
        "margin_after": after["margin"],
        "mean_abs_cosine_geometry_error": cosine_error_sum / count,
        "mean_abs_norm_ratio_geometry_error": norm_error_sum / count,
        "mean_abs_projection_geometry_error": projection_error_sum / count,
        "mean_relative_edge_direction_change": relative_change_sum / count,
        "finite": bool(torch.isfinite(model.weight).all())
        and bool(torch.isfinite(model.bias).all())
        and math.isfinite(after["cross_entropy"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test discovery-set type-pair residual filters on held-out seeds"
    )
    parser.add_argument(
        "--rule",
        choices=("local", "robust_t4a_t4c", "t4_mi9", "top4", "all_type_pair"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--nudge-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_targeted_type_pair_filter.csv"),
    )
    args = parser.parse_args()
    if not 0 <= args.start_epoch <= args.epochs:
        raise ValueError("start-epoch must be between 0 and epochs")

    frame = pd.DataFrame(
        [
            run_seed(
                seed=args.seed,
                rule=args.rule,
                epochs=args.epochs,
                start_epoch=args.start_epoch,
                nudge_steps=args.nudge_steps,
                learning_rate=args.learning_rate,
                max_update=args.max_update,
                test_repeats=args.test_repeats,
            )
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
