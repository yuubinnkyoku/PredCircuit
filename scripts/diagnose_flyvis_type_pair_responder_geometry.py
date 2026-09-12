from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_multi_eval_holdout import heldout_metrics
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit
from run_flyvis_type_pair_consensus import type_pair_indices
from run_flyvis_type_pair_linear_mix import linear_mix_direction
from run_flyvis_type_pair_norm_matched_control import match_norm
from run_flyvis_type_pair_shuffle_control import shuffled_groups_like

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

FROZEN_EPOCH = 75
POST_STEPS = 25
SAMPLE_EPOCHS = (75, 80, 85, 90, 95)
EVAL_REPS = 8
EVAL_JITTER_BASE = 1_600_000
RULES = ("local", "type", "shuffle")


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float(torch.dot(left, right) / denom.clamp_min(1e-30))


def frozen_geometry(
    model: PredictiveCodingGraph,
    circuit,
    *,
    seed: int,
    biological_groups: list[torch.Tensor],
    shuffled_groups: list[torch.Tensor],
    learning_rate: float,
    max_update: float,
) -> dict[str, float]:
    accum: dict[str, float] = {}
    for epoch in SAMPLE_EPOCHS:
        raw_edge, _ = credit(model, circuit, seed=seed, epoch=epoch)
        raw_norm = torch.linalg.vector_norm(raw_edge).clamp_min(1e-30)
        for grouping, groups in (
            ("type", biological_groups),
            ("shuffle", shuffled_groups),
        ):
            coarse = linear_mix_direction(
                raw_edge,
                groups,
                coarse_gain=1.0,
                residual_gain=0.0,
            )
            mixed = linear_mix_direction(
                raw_edge,
                groups,
                coarse_gain=4.0,
                residual_gain=1.0,
            )
            matched = match_norm(mixed, raw_edge)
            values = {
                "coarse_norm_fraction": float(torch.linalg.vector_norm(coarse) / raw_norm),
                "mixed_relative_change": float(
                    torch.linalg.vector_norm(matched - raw_edge) / raw_norm
                ),
                "mixed_cosine_to_raw": _cosine(matched, raw_edge),
                "matched_clip_fraction": float(
                    (learning_rate * matched).abs().gt(max_update).float().mean()
                ),
            }
            for metric, value in values.items():
                key = f"{grouping}_{metric}"
                accum[key] = accum.get(key, 0.0) + value

    count = float(len(SAMPLE_EPOCHS))
    averaged = {key: value / count for key, value in accum.items()}
    for metric in (
        "coarse_norm_fraction",
        "mixed_relative_change",
        "mixed_cosine_to_raw",
        "matched_clip_fraction",
    ):
        averaged[f"type_minus_shuffle_{metric}"] = (
            averaged[f"type_{metric}"] - averaged[f"shuffle_{metric}"]
        )
    return averaged


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

    for epoch in range(FROZEN_EPOCH):
        edge, bias = credit(base, circuit, seed=seed, epoch=epoch)
        apply_local_credit(
            base,
            edge,
            bias,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )

    geometry = frozen_geometry(
        base,
        circuit,
        seed=seed,
        biological_groups=biological_groups,
        shuffled_groups=shuffled_groups,
        learning_rate=learning_rate,
        max_update=max_update,
    )

    base_eval = []
    for eval_rep in range(EVAL_REPS):
        base_eval.append(
            heldout_metrics(
                base,
                circuit,
                jitter_seed=EVAL_JITTER_BASE + eval_rep,
            )
        )
    base_hard_margin = sum(row["hard_margin"] for row in base_eval) / EVAL_REPS
    base_soft_margin = sum(row["soft_margin"] for row in base_eval) / EVAL_REPS
    base_ce = sum(row["cross_entropy"] for row in base_eval) / EVAL_REPS

    models = {rule: copy.deepcopy(base) for rule in RULES}
    for step in range(POST_STEPS):
        epoch = FROZEN_EPOCH + step
        for rule in RULES:
            model = models[rule]
            raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
            if rule == "local":
                direction = raw_edge
            else:
                groups = biological_groups if rule == "type" else shuffled_groups
                direction = match_norm(
                    linear_mix_direction(
                        raw_edge,
                        groups,
                        coarse_gain=4.0,
                        residual_gain=1.0,
                    ),
                    raw_edge,
                )
            apply_local_credit(
                model,
                direction,
                raw_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )

    rows: list[dict[str, float | int | bool | str]] = []
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
                and all(math.isfinite(value) for value in geometry.values())
            )
            rows.append(
                {
                    "seed": seed,
                    "eval_rep": eval_rep,
                    "rule": rule,
                    **geometry,
                    "base_hard_margin": base_hard_margin,
                    "base_soft_margin": base_soft_margin,
                    "base_cross_entropy": base_ce,
                    **metrics,
                    "finite": finite,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether preregistered epoch-75 type-pair credit geometry predicts "
            "which independent training seeds benefit from 25 steps of 4m+r"
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
