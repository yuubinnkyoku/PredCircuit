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

RULES = ("local", "shared_norm_matched", "shared_radial_matched")


def match_norm_and_radial(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    reference_norm = torch.linalg.vector_norm(reference)
    weight_norm = torch.linalg.vector_norm(weight)
    if float(weight_norm) < 1e-30:
        return match_norm(candidate, reference)

    unit_weight = weight / weight_norm
    radial = torch.dot(reference, unit_weight)
    candidate_tangent = candidate - torch.dot(candidate, unit_weight) * unit_weight
    candidate_tangent_norm = torch.linalg.vector_norm(candidate_tangent)
    target_tangent_sq = torch.clamp(reference_norm.square() - radial.square(), min=0.0)
    target_tangent_norm = torch.sqrt(target_tangent_sq)

    if float(candidate_tangent_norm) < 1e-30:
        reference_tangent = reference - radial * unit_weight
        reference_tangent_norm = torch.linalg.vector_norm(reference_tangent).clamp_min(1e-30)
        tangent = reference_tangent * (target_tangent_norm / reference_tangent_norm)
    else:
        tangent = candidate_tangent * (target_tangent_norm / candidate_tangent_norm)
    return radial * unit_weight + tangent


def radial_fraction(direction: torch.Tensor, weight: torch.Tensor) -> float:
    direction_norm = torch.linalg.vector_norm(direction).clamp_min(1e-30)
    weight_norm = torch.linalg.vector_norm(weight).clamp_min(1e-30)
    return float(torch.dot(direction, weight) / (direction_norm * weight_norm))


def run_seed(
    *,
    seed: int,
    rule: str,
    learning_rate: float,
    max_update: float,
) -> dict[str, float | int | bool | str]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    groups = type_pair_indices(circuit)
    before = evaluate_metrics(
        model,
        circuit,
        repeats=8,
        frames=7,
        width=0.5,
        frame_steps=2,
        step_size=0.015,
        jitter_seed=900_000 + seed,
    )
    norm_ratio_sum = 0.0
    radial_fraction_sum = 0.0
    local_radial_fraction_sum = 0.0
    relative_change_sum = 0.0

    for epoch in range(100):
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
        mixed = linear_mix_direction(
            local_edge,
            groups,
            coarse_gain=4.0,
            residual_gain=1.0,
        )
        if rule == "local":
            edge_direction = local_edge
        elif rule == "shared_norm_matched":
            edge_direction = match_norm(mixed, local_edge)
        elif rule == "shared_radial_matched":
            edge_direction = match_norm_and_radial(mixed, local_edge, model.weight)
        else:
            raise ValueError(f"unknown rule: {rule}")

        local_norm = torch.linalg.vector_norm(local_edge).clamp_min(1e-30)
        norm_ratio_sum += float(torch.linalg.vector_norm(edge_direction) / local_norm)
        radial_fraction_sum += radial_fraction(edge_direction, model.weight)
        local_radial_fraction_sum += radial_fraction(local_edge, model.weight)
        relative_change_sum += float(
            torch.linalg.vector_norm(edge_direction - local_edge) / local_norm
        )
        apply_local_credit(
            model,
            edge_direction,
            local_bias,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )

    after = evaluate_metrics(
        model,
        circuit,
        repeats=8,
        frames=7,
        width=0.5,
        frame_steps=2,
        step_size=0.015,
        jitter_seed=900_000 + seed,
    )
    return {
        "seed": seed,
        "rule": rule,
        "cross_entropy_before": before["cross_entropy"],
        "cross_entropy_after": after["cross_entropy"],
        "cross_entropy_improvement": before["cross_entropy"] - after["cross_entropy"],
        "accuracy_after": after["accuracy"],
        "margin_after": after["margin"],
        "mean_edge_norm_ratio": norm_ratio_sum / 100.0,
        "mean_radial_fraction": radial_fraction_sum / 100.0,
        "mean_local_radial_fraction": local_radial_fraction_sum / 100.0,
        "mean_relative_direction_change": relative_change_sum / 100.0,
        "weight_norm": float(torch.linalg.vector_norm(model.weight)),
        "bias_norm": float(torch.linalg.vector_norm(model.bias)),
        "finite": bool(torch.isfinite(model.weight).all())
        and bool(torch.isfinite(model.bias).all())
        and math.isfinite(after["cross_entropy"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Control shared credit for both update norm and radial projection on the weights"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rule", choices=RULES, required=True)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.DataFrame(
        [
            run_seed(
                seed=args.seed,
                rule=args.rule,
                learning_rate=args.learning_rate,
                max_update=args.max_update,
            )
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
