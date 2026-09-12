from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_horizon_causal import _match_local_state, _mixed_direction
from diagnose_flyvis_type_pair_multi_eval_holdout import heldout_metrics
from diagnose_flyvis_type_pair_state_geometry import _group_constant_projection
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit
from run_flyvis_type_pair_consensus import type_pair_indices
from run_flyvis_type_pair_shuffle_control import shuffled_groups_like

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

HORIZONS = (60, 70, 80, 90, 100)
EVAL_REPS = 4
EVAL_JITTER_BASE = 2_300_000
RULES = (
    "local",
    "type",
    "bio_type_projection_local_residual",
    "bio_local_projection_type_residual",
    "shuffle_type_projection_local_residual",
    "shuffle_local_projection_type_residual",
)


def _recombine_state(
    projection_source: PredictiveCodingGraph,
    residual_source: PredictiveCodingGraph,
    groups: list[torch.Tensor],
    target_norm: torch.Tensor,
    bias_source: PredictiveCodingGraph,
) -> tuple[PredictiveCodingGraph, float, float]:
    result = copy.deepcopy(residual_source)
    projection = _group_constant_projection(projection_source.weight, groups)
    residual_projection = _group_constant_projection(residual_source.weight, groups)
    residual = residual_source.weight - residual_projection

    target_residual_sq = (target_norm.square() - torch.dot(projection, projection)).clamp_min(0.0)
    target_residual_norm = torch.sqrt(target_residual_sq)
    residual_norm = torch.linalg.vector_norm(residual).clamp_min(1e-30)
    result.weight.copy_(projection + residual * (target_residual_norm / residual_norm))
    result.bias.copy_(bias_source.bias)

    projection_error = float(
        torch.linalg.vector_norm(_group_constant_projection(result.weight, groups) - projection)
        / target_norm.clamp_min(1e-30)
    )
    norm_error = float(
        (torch.linalg.vector_norm(result.weight) - target_norm).abs()
        / target_norm.clamp_min(1e-30)
    )
    return result, projection_error, norm_error


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
    local = copy.deepcopy(base)
    type_ = copy.deepcopy(base)
    rows: list[dict[str, float | int | bool | str]] = []

    for step in range(1, max(HORIZONS) + 1):
        epoch = step - 1
        local_edge, local_bias = credit(local, circuit, seed=seed, epoch=epoch)
        apply_local_credit(
            local,
            local_edge,
            local_bias,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )

        raw_edge, raw_bias = credit(type_, circuit, seed=seed, epoch=epoch)
        direction = _mixed_direction(raw_edge, biological_groups)
        apply_local_credit(
            type_,
            direction,
            raw_bias,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )
        _match_local_state(type_, local)

        if step not in HORIZONS:
            continue

        target_norm = torch.linalg.vector_norm(local.weight)
        models: dict[str, PredictiveCodingGraph] = {"local": local, "type": type_}
        errors: dict[str, tuple[float, float]] = {"local": (0.0, 0.0), "type": (0.0, 0.0)}

        for prefix, groups in (("bio", biological_groups), ("shuffle", shuffled_groups)):
            projection_type, projection_error, norm_error = _recombine_state(
                type_, local, groups, target_norm, local
            )
            name = f"{prefix}_type_projection_local_residual"
            models[name] = projection_type
            errors[name] = (projection_error, norm_error)

            residual_type, projection_error, norm_error = _recombine_state(
                local, type_, groups, target_norm, local
            )
            name = f"{prefix}_local_projection_type_residual"
            models[name] = residual_type
            errors[name] = (projection_error, norm_error)

        for eval_rep in range(EVAL_REPS):
            for rule in RULES:
                metrics = heldout_metrics(
                    models[rule],
                    circuit,
                    jitter_seed=EVAL_JITTER_BASE + eval_rep,
                )
                projection_error, norm_error = errors[rule]
                finite = (
                    bool(torch.isfinite(models[rule].weight).all())
                    and bool(torch.isfinite(models[rule].bias).all())
                    and math.isfinite(projection_error)
                    and math.isfinite(norm_error)
                    and all(math.isfinite(value) for value in metrics.values())
                )
                rows.append(
                    {
                        "seed": seed,
                        "horizon": step,
                        "eval_rep": eval_rep,
                        "rule": rule,
                        "projection_error": projection_error,
                        "weight_norm_error": norm_error,
                        **metrics,
                        "finite": finite,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Factor the long-horizon type-pair state into group-constant projection and "
            "within-group residual components, then recombine them across local/type checkpoints"
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
