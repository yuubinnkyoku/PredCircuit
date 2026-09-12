from __future__ import annotations

import argparse
import copy
import math
import random
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_horizon_causal import _match_local_state
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from diagnose_flyvis_type_pair_multi_eval_holdout import heldout_metrics
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit
from run_flyvis_type_pair_norm_matched_control import match_norm

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

HORIZONS = (60, 80, 100)
EVAL_REPS = 4
EVAL_JITTER_BASE = 3_200_000
RULES = ("local", "r7r8_mi9", "angle_matched_sparse")


def _r7r8_direction(
    raw_edge: torch.Tensor,
    named_groups: list[tuple[str, str, torch.Tensor]],
) -> torch.Tensor:
    result = raw_edge.clone()
    for source_type, target_type, indices in named_groups:
        if target_type == "Mi9" and source_type in {"R7", "R8"}:
            result[indices] = raw_edge[indices] + 3.0 * raw_edge[indices].mean()
    return match_norm(result, raw_edge)


def _angle_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = (torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)).clamp_min(
        1e-30
    )
    return float((torch.dot(left, right) / denominator).clamp(-1.0, 1.0))


def _angle_matched_sparse_direction(
    raw_edge: torch.Tensor,
    named_groups: list[tuple[str, str, torch.Tensor]],
    selected_ids: set[int],
    *,
    target_cosine: float,
) -> torch.Tensor:
    norm = torch.linalg.vector_norm(raw_edge).clamp_min(1e-30)
    q = torch.zeros_like(raw_edge)
    for group_id in selected_ids:
        _, _, indices = named_groups[group_id]
        mean = raw_edge[indices].mean()
        if float(mean.abs()) < 1e-12:
            mean = torch.tensor(1.0, dtype=raw_edge.dtype, device=raw_edge.device)
        q[indices] = mean
    tangent = q - raw_edge * (torch.dot(q, raw_edge) / norm.square())
    tangent_norm = torch.linalg.vector_norm(tangent)
    if float(tangent_norm) < 1e-12:
        raise RuntimeError("degenerate sparse tangent control direction")
    cosine = min(1.0, max(-1.0, target_cosine))
    sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    direction = cosine * raw_edge + sine * norm * tangent / tangent_norm
    return match_norm(direction, raw_edge)


def run_seed(
    *,
    seed: int,
    control_seed: int,
    learning_rate: float,
    max_update: float,
) -> pd.DataFrame:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    base = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    models = {rule: copy.deepcopy(base) for rule in RULES}
    named_groups = _named_groups(circuit)
    candidates = [
        group_id
        for group_id, (_, target_type, indices) in enumerate(named_groups)
        if target_type != "Mi9" and int(indices.numel()) == 19
    ]
    rng = random.Random(control_seed)
    selected_ids = set(rng.sample(candidates, 2))

    rows: list[dict[str, float | int | bool | str]] = []
    direction_change_sum = {rule: 0.0 for rule in RULES if rule != "local"}
    angle_error_sum = 0.0
    weight_error_sum = {rule: 0.0 for rule in RULES if rule != "local"}
    bias_error_sum = {rule: 0.0 for rule in RULES if rule != "local"}

    for step in range(1, max(HORIZONS) + 1):
        epoch = step - 1
        local_edge, local_bias = credit(models["local"], circuit, seed=seed, epoch=epoch)
        apply_local_credit(
            models["local"],
            local_edge,
            local_bias,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )

        r7_raw, r7_bias = credit(models["r7r8_mi9"], circuit, seed=seed, epoch=epoch)
        r7_direction = _r7r8_direction(r7_raw, named_groups)
        target_cosine = _angle_cosine(r7_raw, r7_direction)

        control_raw, control_bias = credit(
            models["angle_matched_sparse"], circuit, seed=seed, epoch=epoch
        )
        control_direction = _angle_matched_sparse_direction(
            control_raw,
            named_groups,
            selected_ids,
            target_cosine=target_cosine,
        )
        angle_error_sum += abs(_angle_cosine(control_raw, control_direction) - target_cosine)

        for rule, raw_edge, raw_bias, direction in (
            ("r7r8_mi9", r7_raw, r7_bias, r7_direction),
            ("angle_matched_sparse", control_raw, control_bias, control_direction),
        ):
            direction_change_sum[rule] += float(
                torch.linalg.vector_norm(direction - raw_edge)
                / torch.linalg.vector_norm(raw_edge).clamp_min(1e-30)
            )
            apply_local_credit(
                models[rule],
                direction,
                raw_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )
            weight_error, bias_error = _match_local_state(models[rule], models["local"])
            weight_error_sum[rule] += weight_error
            bias_error_sum[rule] += bias_error

        if step not in HORIZONS:
            continue

        for eval_rep in range(EVAL_REPS):
            for rule in RULES:
                metrics = heldout_metrics(
                    models[rule],
                    circuit,
                    jitter_seed=EVAL_JITTER_BASE + eval_rep,
                )
                if rule == "local":
                    mean_direction_change = 0.0
                    mean_weight_error = 0.0
                    mean_bias_error = 0.0
                    mean_angle_error = 0.0
                else:
                    mean_direction_change = direction_change_sum[rule] / step
                    mean_weight_error = weight_error_sum[rule] / step
                    mean_bias_error = bias_error_sum[rule] / step
                    mean_angle_error = (
                        angle_error_sum / step if rule == "angle_matched_sparse" else 0.0
                    )
                finite = (
                    bool(torch.isfinite(models[rule].weight).all())
                    and bool(torch.isfinite(models[rule].bias).all())
                    and all(math.isfinite(value) for value in metrics.values())
                    and math.isfinite(mean_direction_change)
                    and math.isfinite(mean_weight_error)
                    and math.isfinite(mean_bias_error)
                    and math.isfinite(mean_angle_error)
                )
                rows.append(
                    {
                        "seed": seed,
                        "control_seed": control_seed,
                        "horizon": step,
                        "eval_rep": eval_rep,
                        "rule": rule,
                        "selected_group_count": 2 if rule != "local" else 0,
                        "mean_relative_direction_change": mean_direction_change,
                        "mean_angle_cosine_error": mean_angle_error,
                        "mean_weight_norm_error": mean_weight_error,
                        "mean_bias_error": mean_bias_error,
                        **metrics,
                        "finite": finite,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare R7/R8->Mi9 shared credit with a non-Mi9 two-group sparse control that "
            "matches the per-epoch update-vector angle and norm"
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--control-seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = run_seed(
        seed=args.seed,
        control_seed=args.control_seed,
        learning_rate=args.learning_rate,
        max_update=args.max_update,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
