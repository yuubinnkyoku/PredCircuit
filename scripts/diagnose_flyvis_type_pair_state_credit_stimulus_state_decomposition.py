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
MODES = ("live", "fixed_sample", "frozen_vector")
CONDITIONS = (
    "type_donor",
    "shuffle_donor",
    "type_receives_shuffle",
    "shuffle_receives_type",
)


def _name(mode: str, condition: str) -> str:
    return f"{mode}:{condition}"


def _is_type_state(condition: str) -> bool:
    return condition in ("type_donor", "type_receives_shuffle")


def _source_kind(condition: str) -> str:
    return "type" if condition in ("type_donor", "shuffle_receives_type") else "shuffle"


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
    names = tuple(_name(mode, condition) for mode in MODES for condition in CONDITIONS)
    models = {
        name: PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08) for name in names
    }

    direction_change = {name: 0.0 for name in names}
    angle_match_error = {name: 0.0 for name in names}
    clip_fraction = {name: 0.0 for name in names}
    post_weight_error = {name: 0.0 for name in names}
    post_bias_error = {name: 0.0 for name in names}
    applied_credit_cosine = {name: 0.0 for name in names}
    applied_credit_relative_distance = {name: 0.0 for name in names}
    source_cosine_to_live = {name: 0.0 for name in names}
    source_cosine_to_fixed_sample = {name: 0.0 for name in names}
    postburst_count = {name: 0 for name in names}

    frozen_type: torch.Tensor | None = None
    frozen_shuffle: torch.Tensor | None = None
    before_local = evaluate(local_model, circuit, seed)
    rows: list[dict[str, float | int | bool | str]] = []

    for epoch in range(100):
        local_edge, local_bias = credit(local_model, circuit, seed=seed, epoch=epoch)
        treatment_edges: dict[str, torch.Tensor] = {}

        if BURST_START <= epoch < BURST_END:
            reference_name = _name("live", "type_donor")
            reference_edge, _ = credit(models[reference_name], circuit, seed=seed, epoch=epoch)
            reference_mixed = linear_mix_direction(
                reference_edge,
                biological_groups,
                coarse_gain=4.0,
                residual_gain=1.0,
            )
            reference_direction = match_norm(reference_mixed, reference_edge)
            target_change = relative_direction_change(reference_direction, reference_edge)

            for name in names:
                _, condition = name.split(":", 1)
                current_edge, _ = credit(models[name], circuit, seed=seed, epoch=epoch)
                if _is_type_state(condition):
                    mixed = linear_mix_direction(
                        current_edge,
                        biological_groups,
                        coarse_gain=4.0,
                        residual_gain=1.0,
                    )
                    direction = match_norm(mixed, current_edge)
                    treatment_edges[name] = direction
                    direction_change[name] += relative_direction_change(direction, current_edge)
                else:
                    mixed = linear_mix_direction(
                        current_edge,
                        shuffled_groups,
                        coarse_gain=4.0,
                        residual_gain=1.0,
                    )
                    direction, _, change = angle_match_along_axis(
                        current_edge,
                        mixed - current_edge,
                        target_change,
                    )
                    treatment_edges[name] = direction
                    direction_change[name] += change
                    angle_match_error[name] += abs(change - target_change)
                clip_fraction[name] += float(
                    (learning_rate * treatment_edges[name]).abs().gt(max_update).float().mean()
                )
        elif epoch < BURST_START:
            treatment_edges = {
                name: credit(models[name], circuit, seed=seed, epoch=epoch)[0] for name in names
            }
        else:
            live_type = credit(
                models[_name("live", "type_donor")], circuit, seed=seed, epoch=epoch
            )[0]
            live_shuffle = credit(
                models[_name("live", "shuffle_donor")], circuit, seed=seed, epoch=epoch
            )[0]
            fixed_type = credit(
                models[_name("fixed_sample", "type_donor")],
                circuit,
                seed=seed,
                epoch=BURST_END,
            )[0]
            fixed_shuffle = credit(
                models[_name("fixed_sample", "shuffle_donor")],
                circuit,
                seed=seed,
                epoch=BURST_END,
            )[0]

            if frozen_type is None:
                frozen_type = (
                    credit(
                        models[_name("frozen_vector", "type_donor")],
                        circuit,
                        seed=seed,
                        epoch=BURST_END,
                    )[0]
                    .detach()
                    .clone()
                )
                frozen_shuffle = (
                    credit(
                        models[_name("frozen_vector", "shuffle_donor")],
                        circuit,
                        seed=seed,
                        epoch=BURST_END,
                    )[0]
                    .detach()
                    .clone()
                )
            assert frozen_shuffle is not None

            frozen_type_step = match_norm(frozen_type, fixed_type)
            frozen_shuffle_step = match_norm(frozen_shuffle, fixed_shuffle)
            sources = {
                "live": {"type": live_type, "shuffle": live_shuffle},
                "fixed_sample": {"type": fixed_type, "shuffle": fixed_shuffle},
                "frozen_vector": {
                    "type": frozen_type_step,
                    "shuffle": frozen_shuffle_step,
                },
            }

            for name in names:
                mode, condition = name.split(":", 1)
                kind = _source_kind(condition)
                applied = sources[mode][kind]
                treatment_edges[name] = applied
                postburst_count[name] += 1
                applied_credit_cosine[name] += _cosine(applied, local_edge)
                applied_credit_relative_distance[name] += _relative_distance(applied, local_edge)
                live_reference = live_type if kind == "type" else live_shuffle
                fixed_reference = fixed_type if kind == "type" else fixed_shuffle
                source_cosine_to_live[name] += _cosine(applied, live_reference)
                source_cosine_to_fixed_sample[name] += _cosine(applied, fixed_reference)

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
                "mean_postburst_applied_credit_cosine_to_local": 1.0,
                "mean_postburst_applied_credit_relative_distance_to_local": 0.0,
                "mean_postburst_source_cosine_to_live": 1.0,
                "mean_postburst_source_cosine_to_fixed_sample": 1.0,
                "mean_burst_relative_direction_change": 0.0,
                "mean_burst_angle_match_error": 0.0,
                "mean_burst_edge_clip_fraction": 0.0,
                "mean_post_weight_norm_error": 0.0,
                "mean_post_bias_vector_error": 0.0,
                "finite": bool(torch.isfinite(local_model.weight).all())
                and bool(torch.isfinite(local_model.bias).all())
                and math.isfinite(local_eval["cross_entropy"]),
            }
        )
        for name, model in models.items():
            mode, condition = name.split(":", 1)
            after = evaluate(model, circuit, seed)
            burst_count = max(1, min(completed_epoch, BURST_END) - BURST_START)
            count = postburst_count[name]
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
                    "mean_postburst_applied_credit_cosine_to_local": (
                        applied_credit_cosine[name] / count if count else 0.0
                    ),
                    "mean_postburst_applied_credit_relative_distance_to_local": (
                        applied_credit_relative_distance[name] / count if count else 0.0
                    ),
                    "mean_postburst_source_cosine_to_live": (
                        source_cosine_to_live[name] / count if count else 1.0
                    ),
                    "mean_postburst_source_cosine_to_fixed_sample": (
                        source_cosine_to_fixed_sample[name] / count if count else 1.0
                    ),
                    "mean_burst_relative_direction_change": direction_change[name] / burst_count,
                    "mean_burst_angle_match_error": angle_match_error[name] / burst_count,
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
            "Decompose post-burst credit adaptation into changing model state, changing stimulus "
            "jitter, and a fully frozen epoch-75 credit direction"
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
