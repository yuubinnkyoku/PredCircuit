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

HORIZONS = (0, 1, 2, 5, 10, 25)
CHECKPOINTS = tuple(BURST_END + horizon for horizon in HORIZONS)
MODES = ("recompute", "frozen")
STATE_CREDIT_CONDITIONS = (
    "type_donor",
    "shuffle_donor",
    "type_receives_shuffle",
    "shuffle_receives_type",
)


def _condition_name(mode: str, condition: str) -> str:
    return f"{mode}_{condition}"


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
    names = tuple(
        _condition_name(mode, condition)
        for mode in MODES
        for condition in STATE_CREDIT_CONDITIONS
    )
    models = {
        name: PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
        for name in names
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
    source_cosine_to_current_donor = {name: 0.0 for name in names}
    postburst_credit_count = {name: 0 for name in names}

    frozen_type_source: torch.Tensor | None = None
    frozen_shuffle_source: torch.Tensor | None = None
    before_local = evaluate(local_model, circuit, seed)
    rows: list[dict[str, float | int | bool | str]] = []

    for epoch in range(100):
        local_edge, local_bias = credit(local_model, circuit, seed=seed, epoch=epoch)
        treatment_edges: dict[str, torch.Tensor] = {}

        if BURST_START <= epoch < BURST_END:
            reference_type_name = _condition_name("recompute", "type_donor")
            reference_type_edge, _ = credit(
                models[reference_type_name], circuit, seed=seed, epoch=epoch
            )
            type_mixed = linear_mix_direction(
                reference_type_edge,
                biological_groups,
                coarse_gain=4.0,
                residual_gain=1.0,
            )
            type_direction = match_norm(type_mixed, reference_type_edge)
            target_change = relative_direction_change(type_direction, reference_type_edge)

            for mode in MODES:
                for condition in STATE_CREDIT_CONDITIONS:
                    name = _condition_name(mode, condition)
                    current_edge, _ = credit(models[name], circuit, seed=seed, epoch=epoch)
                    if condition in ("type_donor", "type_receives_shuffle"):
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
                    else:
                        shuffle_mixed = linear_mix_direction(
                            current_edge,
                            shuffled_groups,
                            coarse_gain=4.0,
                            residual_gain=1.0,
                        )
                        axis_step = shuffle_mixed - current_edge
                        current_direction, scale, shuffle_change = angle_match_along_axis(
                            current_edge,
                            axis_step,
                            target_change,
                        )
                        treatment_edges[name] = current_direction
                        direction_change[name] += shuffle_change
                        angle_match_error[name] += abs(shuffle_change - target_change)
                        effective_gain[name] += 1.0 + 3.0 * scale
                    clip_fraction[name] += float(
                        (learning_rate * treatment_edges[name])
                        .abs()
                        .gt(max_update)
                        .float()
                        .mean()
                    )
        else:
            own_edges = {
                name: credit(models[name], circuit, seed=seed, epoch=epoch)[0]
                for name in names
            }
            if epoch < BURST_START:
                treatment_edges = own_edges
            else:
                dynamic_type = own_edges[_condition_name("recompute", "type_donor")]
                dynamic_shuffle = own_edges[_condition_name("recompute", "shuffle_donor")]
                if frozen_type_source is None:
                    frozen_type_source = dynamic_type.detach().clone()
                    frozen_shuffle_source = dynamic_shuffle.detach().clone()
                assert frozen_shuffle_source is not None

                frozen_type = match_norm(frozen_type_source, dynamic_type)
                frozen_shuffle = match_norm(frozen_shuffle_source, dynamic_shuffle)

                source_edges = {
                    _condition_name("recompute", "type_donor"): dynamic_type,
                    _condition_name("recompute", "shuffle_donor"): dynamic_shuffle,
                    _condition_name("recompute", "type_receives_shuffle"): dynamic_shuffle,
                    _condition_name("recompute", "shuffle_receives_type"): dynamic_type,
                    _condition_name("frozen", "type_donor"): frozen_type,
                    _condition_name("frozen", "shuffle_donor"): frozen_shuffle,
                    _condition_name("frozen", "type_receives_shuffle"): frozen_shuffle,
                    _condition_name("frozen", "shuffle_receives_type"): frozen_type,
                }
                treatment_edges.update(source_edges)

                for name in names:
                    postburst_credit_count[name] += 1
                    own_credit_cosine[name] += _cosine(own_edges[name], local_edge)
                    own_credit_relative_distance[name] += _relative_distance(
                        own_edges[name], local_edge
                    )
                    applied_edge = treatment_edges[name]
                    applied_credit_cosine[name] += _cosine(applied_edge, local_edge)
                    applied_credit_relative_distance[name] += _relative_distance(
                        applied_edge, local_edge
                    )
                    source_reference = (
                        dynamic_type
                        if name.endswith("type_donor") or name.endswith("shuffle_receives_type")
                        else dynamic_shuffle
                    )
                    source_cosine_to_current_donor[name] += _cosine(
                        applied_edge, source_reference
                    )

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
            model.weight.mul_(
                target_norm / torch.linalg.vector_norm(model.weight).clamp_min(1e-30)
            )
            model.bias.copy_(local_model.bias)
            post_weight_error[name] += float(
                (torch.linalg.vector_norm(model.weight) - target_norm).abs()
                / target_norm.clamp_min(1e-30)
            )
            post_bias_error[name] += float(
                torch.linalg.vector_norm(model.bias - local_model.bias)
            )

        completed_epoch = epoch + 1
        if completed_epoch not in CHECKPOINTS:
            continue

        local_eval = evaluate(local_model, circuit, seed)
        rows.append(
            {
                "seed": seed,
                "epoch": completed_epoch,
                "horizon": completed_epoch - BURST_END,
                "mode": "local",
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
                "mean_postburst_source_cosine_to_current_donor": 1.0,
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
            mode, condition = name.split("_", 1)
            burst_count = max(1, min(completed_epoch, BURST_END) - BURST_START)
            credit_count = postburst_credit_count[name]
            rows.append(
                {
                    "seed": seed,
                    "epoch": completed_epoch,
                    "horizon": completed_epoch - BURST_END,
                    "mode": mode,
                    "condition": condition,
                    "cross_entropy_before": before_local["cross_entropy"],
                    "cross_entropy": after["cross_entropy"],
                    "accuracy": after["accuracy"],
                    "margin": after["margin"],
                    "weight_cosine_to_local": _cosine(model.weight, local_model.weight),
                    "weight_relative_distance_to_local": _relative_distance(
                        model.weight, local_model.weight
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
                    "mean_postburst_source_cosine_to_current_donor": (
                        source_cosine_to_current_donor[name] / credit_count
                        if credit_count
                        else 1.0
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
            "Compare post-burst recomputed credit against epoch-75 frozen credit directions, "
            "while matching frozen source norms to the corresponding live donor each step"
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
