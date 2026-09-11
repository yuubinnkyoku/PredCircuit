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


PULSE_INTERVALS = (1, 5, 20, 100)
SIGNS = (("neg", -1.0), ("pos", 1.0))


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
        f"{grouping}_{suffix}_every{interval}"
        for interval in PULSE_INTERVALS
        for grouping in ("type", "shuffle")
        for suffix, _ in SIGNS
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
    pulse_count = {name: 0 for name in names}

    before_local = evaluate(local_model, circuit, seed)
    for epoch in range(100):
        local_edge, local_bias = credit(local_model, circuit, seed=seed, epoch=epoch)
        treatment_edges: dict[str, torch.Tensor] = {}

        for interval in PULSE_INTERVALS:
            pulsed = epoch % interval == 0
            for suffix, sign in SIGNS:
                type_name = f"type_{suffix}_every{interval}"
                shuffle_name = f"shuffle_{suffix}_every{interval}"
                type_model = models[type_name]
                shuffle_model = models[shuffle_name]
                type_edge, _ = credit(type_model, circuit, seed=seed, epoch=epoch)
                shuffle_edge, _ = credit(shuffle_model, circuit, seed=seed, epoch=epoch)

                if not pulsed:
                    treatment_edges[type_name] = type_edge
                    treatment_edges[shuffle_name] = shuffle_edge
                    continue

                type_gain = 1.0 + 3.0 * sign
                type_mixed = linear_mix_direction(
                    type_edge,
                    biological_groups,
                    coarse_gain=type_gain,
                    residual_gain=1.0,
                )
                type_direction = match_norm(type_mixed, type_edge)
                target_change = relative_direction_change(type_direction, type_edge)
                treatment_edges[type_name] = type_direction
                direction_change[type_name] += target_change
                effective_gain[type_name] += type_gain
                clip_fraction[type_name] += float(
                    (learning_rate * type_direction).abs().gt(max_update).float().mean()
                )
                pulse_count[type_name] += 1

                shuffle_base_mixed = linear_mix_direction(
                    shuffle_edge,
                    shuffled_groups,
                    coarse_gain=type_gain,
                    residual_gain=1.0,
                )
                axis_step = shuffle_base_mixed - shuffle_edge
                shuffle_direction, scale, shuffle_change = angle_match_along_axis(
                    shuffle_edge,
                    axis_step,
                    target_change,
                )
                treatment_edges[shuffle_name] = shuffle_direction
                direction_change[shuffle_name] += shuffle_change
                angle_match_error[shuffle_name] += abs(shuffle_change - target_change)
                effective_gain[shuffle_name] += 1.0 + 3.0 * sign * scale
                clip_fraction[shuffle_name] += float(
                    (learning_rate * shuffle_direction).abs().gt(max_update).float().mean()
                )
                pulse_count[shuffle_name] += 1

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
            treatment_norm = torch.linalg.vector_norm(model.weight).clamp_min(1e-30)
            model.weight.mul_(target_norm / treatment_norm)
            model.bias.copy_(local_model.bias)
            post_weight_error[name] += float(
                (torch.linalg.vector_norm(model.weight) - target_norm).abs()
                / target_norm.clamp_min(1e-30)
            )
            post_bias_error[name] += float(torch.linalg.vector_norm(model.bias - local_model.bias))

    after_local = evaluate(local_model, circuit, seed)
    rows: list[dict[str, float | int | bool | str]] = [
        {
            "seed": seed,
            "condition": "local",
            "grouping": "local",
            "sign": "local",
            "pulse_interval": 0,
            "pulse_count": 0,
            "cross_entropy_before": before_local["cross_entropy"],
            "cross_entropy_after": after_local["cross_entropy"],
            "accuracy_after": after_local["accuracy"],
            "margin_after": after_local["margin"],
            "mean_pulse_relative_direction_change": 0.0,
            "mean_pulse_angle_match_error": 0.0,
            "mean_pulse_effective_gain": 1.0,
            "mean_pulse_edge_clip_fraction": float("nan"),
            "mean_post_weight_norm_error": 0.0,
            "mean_post_bias_vector_error": 0.0,
            "finite": bool(torch.isfinite(local_model.weight).all())
            and bool(torch.isfinite(local_model.bias).all())
            and math.isfinite(after_local["cross_entropy"]),
        }
    ]

    for interval in PULSE_INTERVALS:
        for grouping in ("type", "shuffle"):
            for suffix, _ in SIGNS:
                name = f"{grouping}_{suffix}_every{interval}"
                model = models[name]
                after = evaluate(model, circuit, seed)
                count = pulse_count[name]
                rows.append(
                    {
                        "seed": seed,
                        "condition": name,
                        "grouping": grouping,
                        "sign": suffix,
                        "pulse_interval": interval,
                        "pulse_count": count,
                        "cross_entropy_before": before_local["cross_entropy"],
                        "cross_entropy_after": after["cross_entropy"],
                        "accuracy_after": after["accuracy"],
                        "margin_after": after["margin"],
                        "mean_pulse_relative_direction_change": direction_change[name] / count,
                        "mean_pulse_angle_match_error": angle_match_error[name] / count,
                        "mean_pulse_effective_gain": effective_gain[name] / count,
                        "mean_pulse_edge_clip_fraction": clip_fraction[name] / count,
                        "mean_post_weight_norm_error": post_weight_error[name] / 100.0,
                        "mean_post_bias_vector_error": post_bias_error[name] / 100.0,
                        "finite": bool(torch.isfinite(model.weight).all())
                        and bool(torch.isfinite(model.bias).all())
                        and math.isfinite(after["cross_entropy"]),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how often type-pair credit must be re-injected for its CE effect to appear, "
            "using angle-matched shuffled controls and matched weight/bias trajectories"
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
