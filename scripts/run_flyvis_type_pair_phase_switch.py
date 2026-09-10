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

RULES = ("local", "shared_all", "shared_early20", "shared_late20")


def uses_shared(rule: str, epoch: int) -> bool:
    if rule == "local":
        return False
    if rule == "shared_all":
        return True
    if rule == "shared_early20":
        return epoch < 20
    if rule == "shared_late20":
        return epoch >= 80
    raise ValueError(f"unknown rule: {rule}")


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
    shared_steps = 0
    relative_change_sum = 0.0
    group_fraction_sum = 0.0

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
        edge_direction = local_edge
        if uses_shared(rule, epoch):
            mixed = linear_mix_direction(
                local_edge,
                groups,
                coarse_gain=4.0,
                residual_gain=1.0,
            )
            edge_direction = match_norm(mixed, local_edge)
            shared_steps += 1

        local_norm = torch.linalg.vector_norm(local_edge).clamp_min(1e-30)
        relative_change_sum += float(
            torch.linalg.vector_norm(edge_direction - local_edge) / local_norm
        )
        projected = torch.zeros_like(edge_direction)
        for indices in groups:
            projected[indices] = edge_direction[indices].mean()
        direction_norm = torch.linalg.vector_norm(edge_direction).clamp_min(1e-30)
        group_fraction_sum += float(torch.linalg.vector_norm(projected) / direction_norm)

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
        "shared_steps": shared_steps,
        "cross_entropy_before": before["cross_entropy"],
        "cross_entropy_after": after["cross_entropy"],
        "cross_entropy_improvement": before["cross_entropy"] - after["cross_entropy"],
        "accuracy_after": after["accuracy"],
        "margin_after": after["margin"],
        "mean_relative_direction_change": relative_change_sum / 100.0,
        "mean_group_constant_fraction": group_fraction_sum / 100.0,
        "weight_norm": float(torch.linalg.vector_norm(model.weight)),
        "bias_norm": float(torch.linalg.vector_norm(model.bias)),
        "finite": bool(torch.isfinite(model.weight).all())
        and bool(torch.isfinite(model.bias).all())
        and math.isfinite(after["cross_entropy"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test whether early or late norm-matched type-pair sharing drives the final gain"
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
