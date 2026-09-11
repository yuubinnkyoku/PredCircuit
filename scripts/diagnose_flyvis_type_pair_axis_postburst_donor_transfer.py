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

CHECKPOINTS: tuple[int, ...] = (75, 80, 90, 100)


def run_seed(
    *,
    seed: int,
    shuffle_seed: int,
    learning_rate: float,
    max_update: float,
) -> list[dict[str, float | int | bool | str]]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    local_model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    biological_groups = type_pair_indices(circuit)
    shuffled_groups = shuffled_groups_like(
        biological_groups,
        edge_count=local_model.weight.numel(),
        shuffle_seed=shuffle_seed,
    )
    names = (
        "type_donor",
        "shuffle_donor",
        "type_receives_shuffle",
        "shuffle_receives_type",
    )
    models = {
        name: PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08) for name in names
    }
    direction_change = {name: 0.0 for name in names}
    angle_match_error = {name: 0.0 for name in names}
    effective_gain = {name: 0.0 for name in names}
    clip_fraction = {name: 0.0 for name in names}
    post_weight_error = {name: 0.0 for name in names}
    post_bias_error = {name: 0.0 for name in names}
    own_credit_cosine = {name: 0.0 for name in names}
    own_credit_relative_distance = {name: 0.0 for name in names}
    applied_credit_cosine = {name: 0.0 for name in names}
    applied_credit_relative_distance = {name: 0.0 for name in names}
    postburst_credit_count = {name: 0 for name in names}

    before_local = evaluate(local_model, circuit, seed)
    rows: list[dict[str, float | int | bool | str]] = []

    for epoch in range(100):
        local_edge, local_bias = credit(local_model, circuit, seed=seed, epoch=epoch)
        treatment_edges: dict[str, torch.Tensor] = {}

        donor_edges: dict[str, torch.Tensor] = {}
        for donor_name in ("type_donor", "shuffle_donor"):
            donor_edge, _ = credit(models[donor_name], circuit, seed=seed, epoch=epoch)
            donor_edges[donor_name] = donor_edge

        if BURST_START <= epoch < BURST_END:
            type_mixed = linear_mix_direction(
                donor_edges["type_donor"],
                biological_groups,
                coarse_gain=4.0,
                residual_gain=1.0,
            )
            type_direction = match_norm(type_mixed, donor_edges["type_donor"])
            target_change = relative_direction_change(
                type_direction,
                donor_edges["type_donor"],
            )
            for name in ("type_donor", "type_receives_shuffle"):
                current_edge, _ = credit(models[name], circuit, seed=seed, epoch=epoch)
                current_mixed = linear_mix_direction(
                    current_edge,
                    biological_groups,
                    coarse_gain=4.0,
                    residual_gain=1.0,
                )
                current_direction = match_norm(current_mixed, current_edge)
                treatment_edges[name] = current_direction
                direction_change[name] += relative_direction_change(
                    current_direction,
                    current_edge,
                )
                effective_gain[name] += 4.0
                clip_fraction[name] += float(
                    (learning_rate * current_direction).abs().gt(max_update).float().mean()
                )
            for name in ("shuffle_donor", "shuffle_receives_type"):
                current_edge, _ = credit(models[name], circuit, seed=seed, epoch=epoch)
                shuffle_mixed = linear_mix_direction(
                    current_edge,
                    shuffled_groups,
                    coarse_gain=4.0,
                    residual_gain=1.0,
                )
                axis_step = shuffle_mixed - current_edge
                shuffle_direction, scale, shuffle_change = angle_match_along_axis(
                    current_edge,
                    axis_step,
                    target_change,
                )
                treatment_edges[name] = shuffle_direction
                direction_change[name] += shuffle_change
                angle_match_error[name] += abs(shuffle_change - target_change)
                effective_gain[name] += 1.0 + 3.0 * scale
                clip_fraction[name] += float(
                    (learning_rate * shuffle_direction).abs().gt(max_update).float().mean()
                )
        else:
            own_edges = {}
            for name in names:
                own_edge, _ = credit(models[name], circuit, seed=seed, epoch=epoch)
                own_edges[name] = own_edge
            if epoch >= BURST_END:
                treatment_edges["type_donor"] = own_edges["type_donor"]
                treatment_edges["shuffle_donor"] = own_edges["shuffle_donor"]
                treatment_edges["type_receives_shuffle"] = donor_edges["shuffle_donor"]
                treatment_edges["shuffle_receives_type"] = donor_edges["type_donor"]
                for name in names:
                    postburst_credit_count[name] += 1
                    own_credit_cosine[name] += _cosine(own_edges[name], local_edge)
                    own_credit_relative_distance[name] += _relative_distance(
                        own_edges[name],
                        local_edge,
                    )
                    applied_edge = treatment_edges[name]
                    applied_credit_cosine[name] += _cosine(applied_edge, local_edge)
                    applied_credit_relative_distance[name] += _relative_distance(
                        applied_edge,
                        local_edge,
                    )
            else:
                treatment_edges = own_edges

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
                treatment_edges[name],
                local_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )
            model.weight.mul_(target_norm / torch.linalg.vector_norm(model.weight).clamp_min(1e-30))
            model.bias.copy_(local_model.bias)
            post_weight_error[name] += float(
                (torch.linalg.vector_norm(model.weight) - target_norm).abs()
                / target_norm.clamp_min(1e-30)
            )
            post_bias_error[name] += float(torch.linalg.vector_norm(model.bias - local_model.bias))

        completed_epoch = epoch + 1
        if completed_epoch in CHECKPOINTS:
            local_eval = evaluate(local_model, circuit, seed)
            rows.append(
                {
                    "seed": seed,
                    "epoch": completed_epoch,
                    "condition": "local",
                    "cross_entropy_before": before_local["cross_entropy"],
                    "cross_entropy": local_eval["cross_entropy"],
                    "accuracy": local_eval["accuracy"],
                    "margin": local_eval["margin"],
                    "weight_cosine_to_local": 1.0,
                    "weight_relative_distance_to_local": 0.0,
                    "mean_postburst_own_credit_cosine_to_local": 1.0,
                    "mean_postburst_own_credit_relative_distance_to_local": 0.0,
                    "mean_postburst_applied_credit_cosine_to_local": 1.0,
                    "mean_postburst_applied_credit_relative_distance_to_local": 0.0,
                    "mean_burst_relative_direction_change": 0.0,
                    "mean_burst_angle_match_error": 0.0,
                    "mean_burst_effective_gain": 1.0,
                    "mean_burst_edge_clip_fraction": 0.0,
                    "mean_post_weight_norm_error": 0.0,
                    "mean_post_bias_vector_error": 0.0,
                    "finite": bool(torch.isfinite(local_model.weight).all())
                    and bool(torch.isfinite(local_model.bias).all())
                    and math.isfinite(local_eval["cross_entropy"]),
                }
            )
            for name, model in models.items():
                after = evaluate(model, circuit, seed)
                burst_count = max(1, min(completed_epoch, BURST_END) - BURST_START)
                credit_count = postburst_credit_count[name]
                rows.append(
                    {
                        "seed": seed,
                        "epoch": completed_epoch,
                        "condition": name,
                        "cross_entropy_before": before_local["cross_entropy"],
                        "cross_entropy": after["cross_entropy"],
                        "accuracy": after["accuracy"],
                        "margin": after["margin"],
                        "weight_cosine_to_local": _cosine(
                            model.weight,
                            local_model.weight,
                        ),
                        "weight_relative_distance_to_local": _relative_distance(
                            model.weight,
                            local_model.weight,
                        ),
                        "mean_postburst_own_credit_cosine_to_local": (
                            own_credit_cosine[name] / credit_count if credit_count else 0.0
                        ),
                        "mean_postburst_own_credit_relative_distance_to_local": (
                            own_credit_relative_distance[name] / credit_count
                            if credit_count
                            else 0.0
                        ),
                        "mean_postburst_applied_credit_cosine_to_local": (
                            applied_credit_cosine[name] / credit_count if credit_count else 0.0
                        ),
                        "mean_postburst_applied_credit_relative_distance_to_local": (
                            applied_credit_relative_distance[name] / credit_count
                            if credit_count
                            else 0.0
                        ),
                        "mean_burst_relative_direction_change": direction_change[name]
                        / burst_count,
                        "mean_burst_angle_match_error": angle_match_error[name] / burst_count,
                        "mean_burst_effective_gain": effective_gain[name] / burst_count,
                        "mean_burst_edge_clip_fraction": clip_fraction[name] / burst_count,
                        "mean_post_weight_norm_error": post_weight_error[name] / completed_epoch,
                        "mean_post_bias_vector_error": post_bias_error[name] / completed_epoch,
                        "finite": bool(torch.isfinite(model.weight).all())
                        and bool(torch.isfinite(model.bias).all())
                        and math.isfinite(after["cross_entropy"]),
                    }
                )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer post-burst credit from independent type/shuffle donor trajectories into "
            "opposite-state recipients without feedback from recipients into donors"
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
