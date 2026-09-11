from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_axis_burst_trajectory import (
    BURST_END,
    BURST_START,
    _cosine,
    _relative_distance,
)
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

CHECKPOINTS = (75, 80, 90, 100)


def run_seed(
    *,
    seed: int,
    shuffle_seed: int,
    learning_rate: float,
    max_update: float,
) -> list[dict[str, float | int | bool | str]]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    local_model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    names = (
        "type_donor",
        "shuffle_donor",
        "type_receives_shuffle",
        "shuffle_receives_type",
    )
    models = {
        name: PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08) for name in names
    }
    biological_groups = type_pair_indices(circuit)
    shuffled_groups = shuffled_groups_like(
        biological_groups,
        edge_count=local_model.weight.numel(),
        shuffle_seed=shuffle_seed,
    )
    angle_error_sum = {name: 0.0 for name in names if name.startswith("shuffle")}
    own_credit_cosine_sum = {name: 0.0 for name in names}
    own_credit_distance_sum = {name: 0.0 for name in names}
    applied_credit_cosine_sum = {name: 0.0 for name in names}
    applied_credit_distance_sum = {name: 0.0 for name in names}
    postburst_steps = 0
    rows: list[dict[str, float | int | bool | str]] = []

    for epoch in range(100):
        local_edge, local_bias = credit(local_model, circuit, seed=seed, epoch=epoch)
        own_edges = {
            name: credit(model, circuit, seed=seed, epoch=epoch)[0]
            for name, model in models.items()
        }
        directions: dict[str, torch.Tensor]
        if BURST_START <= epoch < BURST_END:
            directions = {}
            for type_name, shuffle_name in (
                ("type_donor", "shuffle_donor"),
                ("type_receives_shuffle", "shuffle_receives_type"),
            ):
                type_edge = own_edges[type_name]
                shuffle_edge = own_edges[shuffle_name]
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
                shuffle_direction, _, shuffle_change = angle_match_along_axis(
                    shuffle_edge,
                    shuffle_mixed - shuffle_edge,
                    target_change,
                )
                directions[type_name] = type_direction
                directions[shuffle_name] = shuffle_direction
                angle_error_sum[shuffle_name] += abs(shuffle_change - target_change)
        elif epoch >= BURST_END:
            postburst_steps += 1
            directions = {
                "type_donor": own_edges["type_donor"],
                "shuffle_donor": own_edges["shuffle_donor"],
                "type_receives_shuffle": own_edges["shuffle_donor"],
                "shuffle_receives_type": own_edges["type_donor"],
            }
            for name, edge in own_edges.items():
                own_credit_cosine_sum[name] += _cosine(edge, local_edge)
                own_credit_distance_sum[name] += _relative_distance(edge, local_edge)
            for name, edge in directions.items():
                applied_credit_cosine_sum[name] += _cosine(edge, local_edge)
                applied_credit_distance_sum[name] += _relative_distance(edge, local_edge)
        else:
            directions = own_edges

        apply_local_credit(
            local_model,
            local_edge,
            local_bias,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )
        target_norm = torch.linalg.vector_norm(local_model.weight)
        for name, model in models.items():
            apply_local_credit(
                model,
                directions[name],
                local_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )
            model.weight.mul_(target_norm / torch.linalg.vector_norm(model.weight).clamp_min(1e-30))
            model.bias.copy_(local_model.bias)

        completed_epoch = epoch + 1
        if completed_epoch not in CHECKPOINTS:
            continue
        local_eval = evaluate(local_model, circuit, seed)
        rows.append(
            {
                "seed": seed,
                "epoch": completed_epoch,
                "condition": "local",
                "cross_entropy": local_eval["cross_entropy"],
                "accuracy": local_eval["accuracy"],
                "margin": local_eval["margin"],
                "weight_cosine_to_local": 1.0,
                "weight_relative_distance_to_local": 0.0,
                "mean_postburst_own_credit_cosine_to_local": 1.0,
                "mean_postburst_own_credit_relative_distance_to_local": 0.0,
                "mean_postburst_applied_credit_cosine_to_local": 1.0,
                "mean_postburst_applied_credit_relative_distance_to_local": 0.0,
                "mean_burst_angle_match_error": 0.0,
                "finite": bool(torch.isfinite(local_model.weight).all())
                and bool(torch.isfinite(local_model.bias).all())
                and math.isfinite(float(local_eval["cross_entropy"])),
            }
        )
        denom = max(postburst_steps, 1)
        for name, model in models.items():
            evaluation = evaluate(model, circuit, seed)
            rows.append(
                {
                    "seed": seed,
                    "epoch": completed_epoch,
                    "condition": name,
                    "cross_entropy": evaluation["cross_entropy"],
                    "accuracy": evaluation["accuracy"],
                    "margin": evaluation["margin"],
                    "weight_cosine_to_local": _cosine(model.weight, local_model.weight),
                    "weight_relative_distance_to_local": _relative_distance(
                        model.weight, local_model.weight
                    ),
                    "mean_postburst_own_credit_cosine_to_local": own_credit_cosine_sum[name]
                    / denom,
                    "mean_postburst_own_credit_relative_distance_to_local": own_credit_distance_sum[
                        name
                    ]
                    / denom,
                    "mean_postburst_applied_credit_cosine_to_local": applied_credit_cosine_sum[
                        name
                    ]
                    / denom,
                    "mean_postburst_applied_credit_relative_distance_to_local": applied_credit_distance_sum[
                        name
                    ]
                    / denom,
                    "mean_burst_angle_match_error": angle_error_sum.get(name, 0.0) / 50.0,
                    "finite": bool(torch.isfinite(model.weight).all())
                    and bool(torch.isfinite(model.bias).all())
                    and math.isfinite(float(evaluation["cross_entropy"])),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer post-burst credit from independent endogenous type/shuffle donor trajectories "
            "to matched recipient states"
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
