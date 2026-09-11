from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit, evaluate
from run_flyvis_type_pair_consensus import type_pair_indices
from run_flyvis_type_pair_linear_mix import linear_mix_direction
from run_flyvis_type_pair_norm_matched_control import match_norm
from run_flyvis_type_pair_shuffle_control import shuffled_groups_like

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def relative_direction_change(direction: torch.Tensor, edge: torch.Tensor) -> float:
    edge_norm = torch.linalg.vector_norm(edge).clamp_min(1e-30)
    return float(torch.linalg.vector_norm(direction - edge) / edge_norm)


def angle_match_along_axis(
    edge: torch.Tensor,
    axis_step: torch.Tensor,
    target_change: float,
    *,
    max_scale: float = 64.0,
    iterations: int = 40,
) -> tuple[torch.Tensor, float, float]:
    if target_change <= 0.0:
        return edge.clone(), 0.0, 0.0

    def candidate(scale: float) -> tuple[torch.Tensor, float]:
        direction = match_norm(edge + scale * axis_step, edge)
        return direction, relative_direction_change(direction, edge)

    low = 0.0
    high = 1.0
    high_direction, high_change = candidate(high)
    while high_change < target_change and high < max_scale:
        low = high
        high = min(2.0 * high, max_scale)
        high_direction, high_change = candidate(high)

    if high_change < target_change:
        return high_direction, high, high_change

    for _ in range(iterations):
        mid = 0.5 * (low + high)
        _, mid_change = candidate(mid)
        if mid_change < target_change:
            low = mid
        else:
            high = mid

    direction, change = candidate(high)
    return direction, high, change


def run_seed(
    *,
    seed: int,
    shuffle_seed: int,
    learning_rate: float,
    max_update: float,
) -> list[dict[str, float | int | bool | str]]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    local_model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    names = ("type_neg", "type_pos", "shuffle_angle_neg", "shuffle_angle_pos")
    models = {
        name: PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08) for name in names
    }
    biological_groups = type_pair_indices(circuit)
    shuffled_groups = shuffled_groups_like(
        biological_groups,
        edge_count=local_model.weight.numel(),
        shuffle_seed=shuffle_seed,
    )
    before_local = evaluate(local_model, circuit, seed)
    before = {name: evaluate(model, circuit, seed) for name, model in models.items()}
    direction_change = {name: 0.0 for name in names}
    clip_fraction = {name: 0.0 for name in names}
    post_weight_error = {name: 0.0 for name in names}
    post_bias_error = {name: 0.0 for name in names}
    angle_match_error = {name: 0.0 for name in names}
    effective_gain = {name: 0.0 for name in names}

    for epoch in range(100):
        local_edge, local_bias = credit(local_model, circuit, seed=seed, epoch=epoch)
        treatment_edges: dict[str, torch.Tensor] = {}
        target_changes: dict[str, float] = {}

        for name, gain in (("type_neg", -2.0), ("type_pos", 4.0)):
            model = models[name]
            edge, _ = credit(model, circuit, seed=seed, epoch=epoch)
            mixed = linear_mix_direction(
                edge,
                biological_groups,
                coarse_gain=gain,
                residual_gain=1.0,
            )
            direction = match_norm(mixed, edge)
            treatment_edges[name] = direction
            change = relative_direction_change(direction, edge)
            target_changes[name] = change
            direction_change[name] += change
            effective_gain[name] += gain
            clip_fraction[name] += float(
                (learning_rate * direction).abs().gt(max_update).float().mean()
            )

        for suffix, sign in (("neg", -1.0), ("pos", 1.0)):
            name = f"shuffle_angle_{suffix}"
            model = models[name]
            edge, _ = credit(model, circuit, seed=seed, epoch=epoch)
            base_gain = 1.0 + 3.0 * sign
            base_mixed = linear_mix_direction(
                edge,
                shuffled_groups,
                coarse_gain=base_gain,
                residual_gain=1.0,
            )
            axis_step = base_mixed - edge
            target = target_changes[f"type_{suffix}"]
            direction, scale, change = angle_match_along_axis(edge, axis_step, target)
            treatment_edges[name] = direction
            direction_change[name] += change
            angle_match_error[name] += abs(change - target)
            effective_gain[name] += 1.0 + 3.0 * sign * scale
            clip_fraction[name] += float(
                (learning_rate * direction).abs().gt(max_update).float().mean()
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
            "cross_entropy_before": before_local["cross_entropy"],
            "cross_entropy_after": after_local["cross_entropy"],
            "accuracy_after": after_local["accuracy"],
            "margin_after": after_local["margin"],
            "mean_relative_direction_change": 0.0,
            "mean_angle_match_error": 0.0,
            "mean_effective_gain": 1.0,
            "mean_edge_clip_fraction": float("nan"),
            "mean_post_weight_norm_error": 0.0,
            "mean_post_bias_vector_error": 0.0,
            "finite": bool(torch.isfinite(local_model.weight).all())
            and bool(torch.isfinite(local_model.bias).all())
            and math.isfinite(after_local["cross_entropy"]),
        }
    ]
    for name, model in models.items():
        after = evaluate(model, circuit, seed)
        grouping = "type_pair" if name.startswith("type_") else "shuffled_angle_matched"
        sign = "neg" if name.endswith("neg") else "pos"
        rows.append(
            {
                "seed": seed,
                "condition": name,
                "grouping": grouping,
                "sign": sign,
                "cross_entropy_before": before[name]["cross_entropy"],
                "cross_entropy_after": after["cross_entropy"],
                "accuracy_after": after["accuracy"],
                "margin_after": after["margin"],
                "mean_relative_direction_change": direction_change[name] / 100.0,
                "mean_angle_match_error": angle_match_error[name] / 100.0,
                "mean_effective_gain": effective_gain[name] / 100.0,
                "mean_edge_clip_fraction": clip_fraction[name] / 100.0,
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
            "Match shuffled-axis angular displacement to the biological type-pair axis while "
            "holding update norm, weight-norm trajectory, and bias trajectory fixed"
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
