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
EVAL_JITTER_BASE = 2_900_000
RULES = ("local", "mi9_only", "r7r8_mi9", "mi9_without_r7r8", "matched_two19")
R7R8_PAIRS = {("R7", "Mi9"), ("R8", "Mi9")}


def _selected_direction(
    raw_edge: torch.Tensor,
    named_groups: list[tuple[str, str, torch.Tensor]],
    selected_group_ids: set[int],
) -> torch.Tensor:
    result = raw_edge.clone()
    for group_id, (_, _, indices) in enumerate(named_groups):
        if group_id in selected_group_ids:
            mean = raw_edge[indices].mean()
            result[indices] = raw_edge[indices] + 3.0 * mean
    return match_norm(result, raw_edge)


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

    mi9_ids = {
        group_id
        for group_id, (_, target_type, _) in enumerate(named_groups)
        if target_type == "Mi9"
    }
    r7r8_ids = {
        group_id
        for group_id, (source_type, target_type, _) in enumerate(named_groups)
        if (source_type, target_type) in R7R8_PAIRS
    }
    if len(r7r8_ids) != 2:
        raise RuntimeError(f"expected two R7/R8->Mi9 groups, got {len(r7r8_ids)}")

    matched_candidates = [
        group_id
        for group_id, (_, target_type, indices) in enumerate(named_groups)
        if int(indices.numel()) == 19 and target_type != "Mi9"
    ]
    rng = random.Random(control_seed)
    matched_ids = set(rng.sample(matched_candidates, 2))
    selections = {
        "mi9_only": mi9_ids,
        "r7r8_mi9": r7r8_ids,
        "mi9_without_r7r8": mi9_ids - r7r8_ids,
        "matched_two19": matched_ids,
    }

    rows: list[dict[str, float | int | bool | str]] = []
    direction_change_sum = {rule: 0.0 for rule in RULES if rule != "local"}
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

        for rule, selected_ids in selections.items():
            model = models[rule]
            raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
            direction = _selected_direction(raw_edge, named_groups, selected_ids)
            direction_change_sum[rule] += float(
                torch.linalg.vector_norm(direction - raw_edge)
                / torch.linalg.vector_norm(raw_edge).clamp_min(1e-30)
            )
            apply_local_credit(
                model,
                direction,
                raw_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )
            weight_error, bias_error = _match_local_state(model, models["local"])
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
                    selected_count = 0
                    mean_direction_change = 0.0
                    mean_weight_error = 0.0
                    mean_bias_error = 0.0
                else:
                    selected_count = len(selections[rule])
                    mean_direction_change = direction_change_sum[rule] / step
                    mean_weight_error = weight_error_sum[rule] / step
                    mean_bias_error = bias_error_sum[rule] / step
                finite = (
                    bool(torch.isfinite(models[rule].weight).all())
                    and bool(torch.isfinite(models[rule].bias).all())
                    and all(math.isfinite(value) for value in metrics.values())
                    and math.isfinite(mean_direction_change)
                    and math.isfinite(mean_weight_error)
                    and math.isfinite(mean_bias_error)
                )
                rows.append(
                    {
                        "seed": seed,
                        "control_seed": control_seed,
                        "horizon": step,
                        "eval_rep": eval_rep,
                        "rule": rule,
                        "selected_group_count": selected_count,
                        "mean_relative_direction_change": mean_direction_change,
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
            "Causally test whether the R7->Mi9 and R8->Mi9 type-pair groups account for "
            "the Mi9-localized shared-credit benefit"
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
