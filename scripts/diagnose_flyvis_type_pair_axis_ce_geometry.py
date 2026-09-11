from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_temporal_ce_credit_alignment import cycle_ce_oracles, geometry
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_angle_matched_control import (
    angle_match_along_axis,
    relative_direction_change,
)
from run_flyvis_type_pair_axis_identity_control import credit
from run_flyvis_type_pair_consensus import type_pair_indices
from run_flyvis_type_pair_linear_mix import linear_mix_direction
from run_flyvis_type_pair_norm_matched_control import match_norm
from run_flyvis_type_pair_shuffle_control import shuffled_groups_like

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def candidate_directions(
    edge: torch.Tensor,
    biological_groups: list[torch.Tensor],
    shuffled_groups: list[torch.Tensor],
) -> dict[str, tuple[torch.Tensor, float, float]]:
    candidates: dict[str, tuple[torch.Tensor, float, float]] = {"local": (edge.clone(), 0.0, 1.0)}
    target_changes: dict[str, float] = {}
    for suffix, gain in (("neg", -2.0), ("pos", 4.0)):
        mixed = linear_mix_direction(
            edge,
            biological_groups,
            coarse_gain=gain,
            residual_gain=1.0,
        )
        direction = match_norm(mixed, edge)
        change = relative_direction_change(direction, edge)
        candidates[f"type_{suffix}"] = (direction, change, gain)
        target_changes[suffix] = change

    for suffix, sign in (("neg", -1.0), ("pos", 1.0)):
        base_gain = 1.0 + 3.0 * sign
        base_mixed = linear_mix_direction(
            edge,
            shuffled_groups,
            coarse_gain=base_gain,
            residual_gain=1.0,
        )
        axis_step = base_mixed - edge
        direction, scale, change = angle_match_along_axis(
            edge,
            axis_step,
            target_changes[suffix],
        )
        effective_gain = 1.0 + 3.0 * sign * scale
        candidates[f"shuffle_angle_{suffix}"] = (direction, change, effective_gain)
    return candidates


def measure_checkpoint(
    model: PredictiveCodingGraph,
    *,
    circuit,
    seed: int,
    checkpoint: int,
    biological_groups: list[torch.Tensor],
    shuffled_groups: list[torch.Tensor],
) -> list[dict[str, float | int | str]]:
    edge, bias = credit(model, circuit, seed=seed, epoch=checkpoint)
    candidates = candidate_directions(edge, biological_groups, shuffled_groups)
    oracles = cycle_ce_oracles(
        model,
        circuit,
        seed=seed,
        epoch=checkpoint,
        frames=7,
        width=0.5,
        frame_steps=2,
        step_size=0.015,
    )
    zero_bias = torch.zeros_like(bias)
    rows: list[dict[str, float | int | str]] = []
    for condition, (direction, change, effective_gain) in candidates.items():
        for oracle_name, (oracle_edge, oracle_bias) in oracles.items():
            full = geometry(direction, bias, oracle_edge, oracle_bias)
            edge_only = geometry(direction, zero_bias, oracle_edge, torch.zeros_like(oracle_bias))
            rows.append(
                {
                    "seed": seed,
                    "checkpoint": checkpoint,
                    "condition": condition,
                    "oracle_objective": oracle_name,
                    "relative_direction_change": change,
                    "effective_gain": effective_gain,
                    "full_cosine": full["cosine"],
                    "full_projection_coefficient": full["projection_coefficient"],
                    "full_parallel_fraction": full["parallel_fraction"],
                    "full_residual_fraction": full["residual_fraction"],
                    "edge_cosine": edge_only["cosine"],
                    "edge_projection_coefficient": edge_only["projection_coefficient"],
                    "edge_parallel_fraction": edge_only["parallel_fraction"],
                    "edge_residual_fraction": edge_only["residual_fraction"],
                    "candidate_edge_norm": float(torch.linalg.vector_norm(direction)),
                    "oracle_edge_norm": float(torch.linalg.vector_norm(oracle_edge)),
                }
            )
    return rows


def run_seed(
    *,
    seed: int,
    shuffle_seed: int,
    learning_rate: float,
    max_update: float,
) -> list[dict[str, float | int | str]]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    biological_groups = type_pair_indices(circuit)
    shuffled_groups = shuffled_groups_like(
        biological_groups,
        edge_count=model.weight.numel(),
        shuffle_seed=shuffle_seed,
    )
    checkpoints = {0, 20, 100}
    rows: list[dict[str, float | int | str]] = []
    for epoch in range(101):
        if epoch in checkpoints:
            rows.extend(
                measure_checkpoint(
                    model,
                    circuit=circuit,
                    seed=seed,
                    checkpoint=epoch,
                    biological_groups=biological_groups,
                    shuffled_groups=shuffled_groups,
                )
            )
        if epoch == 100:
            break
        edge, bias = credit(model, circuit, seed=seed, epoch=epoch)
        apply_local_credit(
            model,
            edge,
            bias,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether type-pair axis perturbations align with exact CE descent better "
            "than angle-matched shuffled axes at identical parameter states"
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shuffle-seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.DataFrame(
        run_seed(
            seed=args.seed,
            shuffle_seed=args.shuffle_seed,
            learning_rate=args.learning_rate,
            max_update=args.max_update,
        )
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
