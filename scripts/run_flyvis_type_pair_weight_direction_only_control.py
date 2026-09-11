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
from predcircuit.flyvis_retinotopy import RetinotopicFlyVisCircuit, graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def evaluate(
    model: PredictiveCodingGraph, circuit: RetinotopicFlyVisCircuit, seed: int
) -> dict[str, float]:
    return evaluate_metrics(
        model,
        circuit,
        repeats=8,
        frames=7,
        width=0.5,
        frame_steps=2,
        step_size=0.015,
        jitter_seed=900_000 + seed,
    )


def run_seed(
    *, seed: int, learning_rate: float, max_update: float
) -> list[dict[str, float | int | bool | str]]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    local_model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    shared_model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    groups = type_pair_indices(circuit)

    local_before = evaluate(local_model, circuit, seed)
    shared_before = evaluate(shared_model, circuit, seed)

    direction_change_sum = 0.0
    weight_norm_error_sum = 0.0
    bias_error_sum = 0.0
    pre_weight_norm_ratio_sum = 0.0
    clip_fraction_sum = 0.0

    for epoch in range(100):
        local_edge, local_bias = cycle_branched_credit(
            local_model,
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
        shared_edge, shared_bias = cycle_branched_credit(
            shared_model,
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
        mixed = linear_mix_direction(
            shared_edge,
            groups,
            coarse_gain=4.0,
            residual_gain=1.0,
        )
        shared_direction = match_norm(mixed, shared_edge)
        shared_edge_norm = torch.linalg.vector_norm(shared_edge).clamp_min(1e-30)
        direction_change_sum += float(
            torch.linalg.vector_norm(shared_direction - shared_edge) / shared_edge_norm
        )
        clip_fraction_sum += float(
            (learning_rate * shared_direction).abs().gt(max_update).float().mean()
        )

        apply_local_credit(
            local_model,
            local_edge,
            local_bias,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )
        apply_local_credit(
            shared_model,
            shared_direction,
            shared_bias,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )

        target_norm = torch.linalg.vector_norm(local_model.weight)
        treatment_norm = torch.linalg.vector_norm(shared_model.weight).clamp_min(1e-30)
        pre_weight_norm_ratio_sum += float(treatment_norm / target_norm.clamp_min(1e-30))
        shared_model.weight.mul_(target_norm / treatment_norm)
        shared_model.bias.copy_(local_model.bias)

        post_norm = torch.linalg.vector_norm(shared_model.weight)
        weight_norm_error_sum += float(
            (post_norm - target_norm).abs() / target_norm.clamp_min(1e-30)
        )
        bias_error_sum += float(torch.linalg.vector_norm(shared_model.bias - local_model.bias))

    local_after = evaluate(local_model, circuit, seed)
    shared_after = evaluate(shared_model, circuit, seed)

    common = {"seed": seed, "learning_rate": learning_rate, "max_update": max_update}
    rows: list[dict[str, float | int | bool | str]] = []
    for rule, model, before, after in (
        ("local", local_model, local_before, local_after),
        ("shared_weight_direction_only", shared_model, shared_before, shared_after),
    ):
        is_shared = rule != "local"
        rows.append(
            {
                **common,
                "rule": rule,
                "cross_entropy_before": before["cross_entropy"],
                "cross_entropy_after": after["cross_entropy"],
                "cross_entropy_improvement": before["cross_entropy"] - after["cross_entropy"],
                "accuracy_after": after["accuracy"],
                "margin_after": after["margin"],
                "weight_norm": float(torch.linalg.vector_norm(model.weight)),
                "bias_norm": float(torch.linalg.vector_norm(model.bias)),
                "mean_relative_direction_change": direction_change_sum / 100.0
                if is_shared
                else 0.0,
                "mean_pre_weight_norm_ratio": pre_weight_norm_ratio_sum / 100.0
                if is_shared
                else 1.0,
                "mean_post_weight_norm_error": weight_norm_error_sum / 100.0 if is_shared else 0.0,
                "mean_post_bias_vector_error": bias_error_sum / 100.0 if is_shared else 0.0,
                "mean_edge_clip_fraction": clip_fraction_sum / 100.0 if is_shared else float("nan"),
                "finite": bool(torch.isfinite(model.weight).all())
                and bool(torch.isfinite(model.bias).all())
                and math.isfinite(after["cross_entropy"]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Isolate type-pair shared-credit weight-direction effects by matching the paired "
            "local weight-norm trajectory and copying the full local bias vector after every epoch"
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.DataFrame(
        run_seed(seed=args.seed, learning_rate=args.learning_rate, max_update=args.max_update)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
