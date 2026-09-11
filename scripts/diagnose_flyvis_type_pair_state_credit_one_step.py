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


def _clone_model(model: PredictiveCodingGraph, circuit, seed: int) -> PredictiveCodingGraph:
    clone = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    clone.weight.copy_(model.weight)
    clone.bias.copy_(model.bias)
    return clone


def _finite(model: PredictiveCodingGraph, evaluation: dict[str, float]) -> bool:
    return (
        bool(torch.isfinite(model.weight).all())
        and bool(torch.isfinite(model.bias).all())
        and math.isfinite(float(evaluation["cross_entropy"]))
    )


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
    burst_angle_error = 0.0
    rows: list[dict[str, float | int | bool | str]] = []

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
            shuffle_direction, _, shuffle_change = angle_match_along_axis(
                shuffle_edge,
                shuffle_mixed - shuffle_edge,
                target_change,
            )
            burst_angle_error += abs(shuffle_change - target_change)
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
        for model, direction in (
            (type_model, type_direction),
            (shuffle_model, shuffle_direction),
        ):
            apply_local_credit(
                model,
                direction,
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

        probe_epoch = completed_epoch
        local_probe_edge, local_probe_bias = credit(
            local_model,
            circuit,
            seed=seed,
            epoch=probe_epoch,
        )
        type_probe_edge, _ = credit(
            type_model,
            circuit,
            seed=seed,
            epoch=probe_epoch,
        )
        shuffle_probe_edge, _ = credit(
            shuffle_model,
            circuit,
            seed=seed,
            epoch=probe_epoch,
        )
        local_probe = _clone_model(local_model, circuit, seed)
        apply_local_credit(
            local_probe,
            local_probe_edge,
            local_probe_bias,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )
        probe_target_norm = torch.linalg.vector_norm(local_probe.weight)

        state_models = {"type": type_model, "shuffle": shuffle_model}
        raw_credits = {"type": type_probe_edge, "shuffle": shuffle_probe_edge}
        credit_modes = {
            "raw": raw_credits,
            "norm_matched": {
                name: match_norm(edge, local_probe_edge) for name, edge in raw_credits.items()
            },
        }
        state_before = {
            state: evaluate(model, circuit, seed) for state, model in state_models.items()
        }

        for mode, credits in credit_modes.items():
            for state_name, state_model in state_models.items():
                start_weight = state_model.weight.detach().clone()
                for credit_name, applied_edge in credits.items():
                    probe = _clone_model(state_model, circuit, seed)
                    clip_fraction = float(
                        (learning_rate * applied_edge).abs().gt(max_update).float().mean()
                    )
                    apply_local_credit(
                        probe,
                        applied_edge,
                        local_probe_bias,
                        learning_rate=learning_rate,
                        weight_decay=0.0,
                        max_update=max_update,
                    )
                    probe.weight.mul_(
                        probe_target_norm / torch.linalg.vector_norm(probe.weight).clamp_min(1e-30)
                    )
                    probe.bias.copy_(local_probe.bias)
                    after = evaluate(probe, circuit, seed)
                    rows.append(
                        {
                            "seed": seed,
                            "epoch": completed_epoch,
                            "mode": mode,
                            "state": state_name,
                            "credit_source": credit_name,
                            "cross_entropy_before": state_before[state_name]["cross_entropy"],
                            "cross_entropy_after": after["cross_entropy"],
                            "accuracy_before": state_before[state_name]["accuracy"],
                            "accuracy_after": after["accuracy"],
                            "margin_before": state_before[state_name]["margin"],
                            "margin_after": after["margin"],
                            "credit_norm": float(torch.linalg.vector_norm(applied_edge)),
                            "credit_cosine_to_local": _cosine(applied_edge, local_probe_edge),
                            "credit_relative_distance_to_local": _relative_distance(
                                applied_edge,
                                local_probe_edge,
                            ),
                            "credit_type_shuffle_cosine": _cosine(
                                type_probe_edge,
                                shuffle_probe_edge,
                            ),
                            "edge_clip_fraction": clip_fraction,
                            "weight_step_relative_distance": _relative_distance(
                                probe.weight,
                                start_weight,
                            ),
                            "post_weight_norm_error": float(
                                (torch.linalg.vector_norm(probe.weight) - probe_target_norm).abs()
                                / probe_target_norm.clamp_min(1e-30)
                            ),
                            "post_bias_vector_error": float(
                                torch.linalg.vector_norm(probe.bias - local_probe.bias)
                            ),
                            "state_weight_cosine_to_local": _cosine(
                                state_model.weight,
                                local_model.weight,
                            ),
                            "state_weight_relative_distance_to_local": _relative_distance(
                                state_model.weight,
                                local_model.weight,
                            ),
                            "mean_burst_angle_match_error": burst_angle_error / 50.0,
                            "finite": _finite(probe, after),
                        }
                    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the immediate one-step compatibility between type/shuffle states and "
            "independent type/shuffle local-credit fields"
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
