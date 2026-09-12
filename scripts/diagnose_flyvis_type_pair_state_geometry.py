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
EVAL_JITTER_BASE = 2_000_000
RULES = ("local", "type", "shuffle")


def _group_constant_projection(vector: torch.Tensor, groups: list[torch.Tensor]) -> torch.Tensor:
    projection = torch.zeros_like(vector)
    for group in groups:
        projection[group] = vector[group].mean()
    return projection


def _projection_fraction(vector: torch.Tensor, groups: list[torch.Tensor]) -> float:
    denominator = torch.dot(vector, vector).clamp_min(1e-30)
    projection = _group_constant_projection(vector, groups)
    return float(torch.dot(projection, projection) / denominator)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = (
        torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    ).clamp_min(1e-30)
    return float(torch.dot(left, right) / denominator)


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

        for rule, groups in (
            ("type", biological_groups),
            ("shuffle", shuffled_groups),
        ):
            raw_edge, raw_bias = credit(models[rule], circuit, seed=seed, epoch=epoch)
            direction = _mixed_direction(raw_edge, groups)
            apply_local_credit(
                models[rule],
                direction,
                raw_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )
            _match_local_state(models[rule], models["local"])

        if step not in HORIZONS:
            continue

        local_weight = models["local"].weight
        geometry: dict[str, dict[str, float]] = {}
        for rule in RULES:
            weight = models[rule].weight
            displacement = weight - local_weight
            geometry[rule] = {
                "weight_biological_projection_fraction": _projection_fraction(
                    weight, biological_groups
                ),
                "weight_shuffle_projection_fraction": _projection_fraction(
                    weight, shuffled_groups
                ),
                "weight_cosine_to_local": (
                    1.0 if rule == "local" else _cosine(weight, local_weight)
                ),
                "displacement_norm": float(torch.linalg.vector_norm(displacement)),
                "displacement_biological_projection_fraction": (
                    0.0
                    if rule == "local"
                    else _projection_fraction(displacement, biological_groups)
                ),
                "displacement_shuffle_projection_fraction": (
                    0.0
                    if rule == "local"
                    else _projection_fraction(displacement, shuffled_groups)
                ),
            }

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
                    and all(math.isfinite(value) for value in geometry[rule].values())
                )
                rows.append(
                    {
                        "seed": seed,
                        "horizon": step,
                        "eval_rep": eval_rep,
                        "rule": rule,
                        **metrics,
                        **geometry[rule],
                        "finite": finite,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether the long-horizon biological type-pair advantage coincides "
            "with persistent weight-state alignment to the biological group-constant subspace"
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
