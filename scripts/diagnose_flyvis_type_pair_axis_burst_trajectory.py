from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_angle_matched_control import (
    angle_match_along_axis,
    relative_direction_change,
)
from run_flyvis_type_pair_axis_identity_control import credit, evaluate
from run_flyvis_type_pair_consensus import type_pair_indices
from run_flyvis_type_pair_linear_mix import linear_mix_direction
from run_flyvis_type_pair_norm_matched_control import match_norm
from run_flyvis_type_pair_shuffle_control import shuffled_groups_like

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

CHECKPOINTS = (0, 25, 26, 30, 35, 40, 50, 60, 70, 75, 80, 90, 100)
BURST_START = 25
BURST_END = 75


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return float(torch.dot(a.flatten(), b.flatten()) / denom.clamp_min(1e-30))


def _relative_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-30))


def _checkpoint_rows(
    *,
    epoch: int,
    seed: int,
    circuit,
    local_model: PredictiveCodingGraph,
    type_model: PredictiveCodingGraph,
    shuffle_model: PredictiveCodingGraph,
    pulse_count: int,
    direction_change_sum: dict[str, float],
    angle_match_error_sum: float,
    effective_gain_sum: float,
) -> list[dict[str, float | int | bool | str]]:
    models = {
        "local": local_model,
        "type": type_model,
        "shuffle": shuffle_model,
    }
    evaluations = {name: evaluate(model, circuit, seed) for name, model in models.items()}
    rows: list[dict[str, float | int | bool | str]] = []
    for name, model in models.items():
        evaluation = evaluations[name]
        rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "condition": name,
                "pulse_count": pulse_count if name != "local" else 0,
                "cross_entropy": evaluation["cross_entropy"],
                "accuracy": evaluation["accuracy"],
                "margin": evaluation["margin"],
                "weight_norm": float(torch.linalg.vector_norm(model.weight)),
                "bias_norm": float(torch.linalg.vector_norm(model.bias)),
                "weight_cosine_to_local": 1.0
                if name == "local"
                else _cosine(model.weight, local_model.weight),
                "weight_relative_distance_to_local": 0.0
                if name == "local"
                else _relative_distance(model.weight, local_model.weight),
                "weight_cosine_type_shuffle": _cosine(type_model.weight, shuffle_model.weight),
                "mean_pulse_relative_direction_change": 0.0
                if name == "local" or pulse_count == 0
                else direction_change_sum[name] / pulse_count,
                "mean_pulse_angle_match_error": 0.0
                if name != "shuffle" or pulse_count == 0
                else angle_match_error_sum / pulse_count,
                "mean_shuffle_effective_gain": 4.0
                if name != "shuffle" or pulse_count == 0
                else effective_gain_sum / pulse_count,
                "finite": bool(torch.isfinite(model.weight).all())
                and bool(torch.isfinite(model.bias).all())
                and all(math.isfinite(float(evaluation[key])) for key in ("cross_entropy", "accuracy", "margin")),
            }
        )
    return rows


def run_seed(
    *,
    seed: int,
    shuffle_seed: int,
    learning_rate: float,
    max_update: float,
) -> list[dict[str, float | int | bool | str]]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    local_model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    type_model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    shuffle_model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    biological_groups = type_pair_indices(circuit)
    shuffled_groups = shuffled_groups_like(
        biological_groups,
        edge_count=local_model.weight.numel(),
        shuffle_seed=shuffle_seed,
    )

    pulse_count = 0
    direction_change_sum = {"type": 0.0, "shuffle": 0.0}
    angle_match_error_sum = 0.0
    effective_gain_sum = 0.0
    rows = _checkpoint_rows(
        epoch=0,
        seed=seed,
        circuit=circuit,
        local_model=local_model,
        type_model=type_model,
        shuffle_model=shuffle_model,
        pulse_count=pulse_count,
        direction_change_sum=direction_change_sum,
        angle_match_error_sum=angle_match_error_sum,
        effective_gain_sum=effective_gain_sum,
    )

    for epoch in range(100):
        local_edge, local_bias = credit(local_model, circuit, seed=seed, epoch=epoch)
        type_edge, _ = credit(type_model, circuit, seed=seed, epoch=epoch)
        shuffle_edge, _ = credit(shuffle_model, circuit, seed=seed, epoch=epoch)

        if BURST_START <= epoch < BURST_END:
            type_mixed = linear_mix_direction(
                type_edge,
                biological_groups,
                coarse_gain=4.0,
                residual_gain=1.0,
            )
            type_direction = match_norm(type_mixed, type_edge)
            target_change = relative_direction_change(type_direction, type_edge)

            shuffle_mixed = linear_mix_direction(
                shuffle_edge,
                shuffled_groups,
                coarse_gain=4.0,
                residual_gain=1.0,
            )
            axis_step = shuffle_mixed - shuffle_edge
            shuffle_direction, scale, shuffle_change = angle_match_along_axis(
                shuffle_edge,
                axis_step,
                target_change,
            )
            pulse_count += 1
            direction_change_sum["type"] += target_change
            direction_change_sum["shuffle"] += shuffle_change
            angle_match_error_sum += abs(shuffle_change - target_change)
            effective_gain_sum += 1.0 + 3.0 * scale
        else:
            type_direction = type_edge
            shuffle_direction = shuffle_edge

        apply_local_credit(
            local_model,
            local_edge,
            local_bias,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )
        target_norm = torch.linalg.vector_norm(local_model.weight)
        for model, edge in ((type_model, type_direction), (shuffle_model, shuffle_direction)):
            apply_local_credit(
                model,
                edge,
                local_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )
            model.weight.mul_(target_norm / torch.linalg.vector_norm(model.weight).clamp_min(1e-30))
            model.bias.copy_(local_model.bias)

        completed_epoch = epoch + 1
        if completed_epoch in CHECKPOINTS:
            rows.extend(
                _checkpoint_rows(
                    epoch=completed_epoch,
                    seed=seed,
                    circuit=circuit,
                    local_model=local_model,
                    type_model=type_model,
                    shuffle_model=shuffle_model,
                    pulse_count=pulse_count,
                    direction_change_sum=direction_change_sum,
                    angle_match_error_sum=angle_match_error_sum,
                    effective_gain_sum=effective_gain_sum,
                )
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Track when a centered 50-epoch type-pair burst separates from an angle-matched "
            "shuffled axis while weight norm and bias trajectories are controlled"
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
