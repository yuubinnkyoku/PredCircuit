from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_horizon_causal import _match_local_state, _mixed_direction
from diagnose_flyvis_type_pair_multi_eval_holdout import heldout_metrics
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit
from run_flyvis_type_pair_consensus import type_pair_indices
from run_flyvis_type_pair_shuffle_control import shuffled_groups_like

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

HORIZONS = (20, 40, 60, 70, 80, 90, 100)
EVAL_REPS = 4
EVAL_JITTER_BASE = 2_100_000
RULES = ("local", "type", "bio_clamp", "shuffle_clamp")


def _group_constant_projection(vector: torch.Tensor, groups: list[torch.Tensor]) -> torch.Tensor:
    projection = torch.zeros_like(vector)
    for group in groups:
        projection[group] = vector[group].mean()
    return projection


def _clamp_group_constant_state(
    treatment: PredictiveCodingGraph,
    local: PredictiveCodingGraph,
    groups: list[torch.Tensor],
) -> tuple[float, float]:
    local_projection = _group_constant_projection(local.weight, groups)
    treatment_projection = _group_constant_projection(treatment.weight, groups)
    treatment_residual = treatment.weight - treatment_projection
    local_residual = local.weight - local_projection

    target_residual_norm = torch.linalg.vector_norm(local_residual).clamp_min(1e-30)
    residual_norm = torch.linalg.vector_norm(treatment_residual).clamp_min(1e-30)
    treatment.weight.copy_(
        local_projection + treatment_residual * (target_residual_norm / residual_norm)
    )
    treatment.bias.copy_(local.bias)

    displacement = treatment.weight - local.weight
    projection_error = float(
        torch.linalg.vector_norm(_group_constant_projection(displacement, groups))
        / torch.linalg.vector_norm(local.weight).clamp_min(1e-30)
    )
    weight_norm_error = float(
        (
            torch.linalg.vector_norm(treatment.weight)
            - torch.linalg.vector_norm(local.weight)
        ).abs()
        / torch.linalg.vector_norm(local.weight).clamp_min(1e-30)
    )
    return projection_error, weight_norm_error


def run_seed(
    *,
    seed: int,
    shuffle_seed: int,
    learning_rate: float,
    max_update: float,
) -> pd.DataFrame:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    base = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    biological_groups = type_pair_indices(circuit)
    shuffled_groups = shuffled_groups_like(
        biological_groups,
        edge_count=base.weight.numel(),
        shuffle_seed=shuffle_seed,
    )
    models = {rule: copy.deepcopy(base) for rule in RULES}
    projection_error_sum = {"bio_clamp": 0.0, "shuffle_clamp": 0.0}
    clamp_weight_error_sum = {"bio_clamp": 0.0, "shuffle_clamp": 0.0}
    strict_weight_error_sum = {"type": 0.0}
    rows: list[dict[str, float | int | bool | str]] = []

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

        for rule in ("type", "bio_clamp", "shuffle_clamp"):
            raw_edge, raw_bias = credit(models[rule], circuit, seed=seed, epoch=epoch)
            direction = _mixed_direction(raw_edge, biological_groups)
            apply_local_credit(
                models[rule],
                direction,
                raw_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )

        weight_error, _ = _match_local_state(models["type"], models["local"])
        strict_weight_error_sum["type"] += weight_error

        for rule, groups in (
            ("bio_clamp", biological_groups),
            ("shuffle_clamp", shuffled_groups),
        ):
            projection_error, weight_norm_error = _clamp_group_constant_state(
                models[rule], models["local"], groups
            )
            projection_error_sum[rule] += projection_error
            clamp_weight_error_sum[rule] += weight_norm_error

        if step not in HORIZONS:
            continue

        for eval_rep in range(EVAL_REPS):
            for rule in RULES:
                metrics = heldout_metrics(
                    models[rule],
                    circuit,
                    jitter_seed=EVAL_JITTER_BASE + eval_rep,
                )
                finite = (
                    bool(torch.isfinite(models[rule].weight).all())
                    and bool(torch.isfinite(models[rule].bias).all())
                    and all(math.isfinite(value) for value in metrics.values())
                )
                rows.append(
                    {
                        "seed": seed,
                        "horizon": step,
                        "eval_rep": eval_rep,
                        "rule": rule,
                        "mean_projection_clamp_error": (
                            projection_error_sum[rule] / step
                            if rule in projection_error_sum
                            else 0.0
                        ),
                        "mean_weight_norm_error": (
                            clamp_weight_error_sum[rule] / step
                            if rule in clamp_weight_error_sum
                            else strict_weight_error_sum.get(rule, 0.0) / step
                        ),
                        **metrics,
                        "finite": finite,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether accumulation of the biological type-pair group-constant weight "
            "state is necessary for the long-horizon 4m+r crossover"
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shuffle-seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frame = run_seed(
        seed=args.seed,
        shuffle_seed=args.shuffle_seed,
        learning_rate=args.learning_rate,
        max_update=args.max_update,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
