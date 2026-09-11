from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
from run_flyvis_credit_residual_scaling import evaluate_metrics
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
HORIZONS = (1, 2, 5, 10, 25)
RULES = ("local", "type", "shuffle")


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom) == 0.0:
        return float("nan")
    return float(torch.dot(a, b) / denom)


def _relative_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(b).clamp_min(1e-30)
    return float(torch.linalg.vector_norm(a - b) / denom)


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

    models = {rule: copy.deepcopy(base) for rule in RULES}
    rows: list[dict[str, float | int | bool | str]] = []
    target_horizons = set(HORIZONS)

    for step in range(1, max(HORIZONS) + 1):
        epoch = FROZEN_EPOCH + step - 1
        for rule in RULES:
            model = models[rule]
            raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
            if rule == "local":
                direction = raw_edge
            else:
                groups = biological_groups if rule == "type" else shuffled_groups
                mixed = linear_mix_direction(
                    raw_edge,
                    groups,
                    coarse_gain=4.0,
                    residual_gain=1.0,
                )
                direction = match_norm(mixed, raw_edge)
            apply_local_credit(
                model,
                direction,
                raw_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )

        if step not in target_horizons:
            continue

        local_weight = models["local"].weight.detach()
        for rule in RULES:
            model = models[rule]
            metrics = evaluate_metrics(
                model,
                circuit,
                repeats=12,
                frames=7,
                width=0.5,
                frame_steps=2,
                step_size=0.015,
                jitter_seed=1_300_000 + seed,
            )
            finite = (
                bool(torch.isfinite(model.weight).all())
                and bool(torch.isfinite(model.bias).all())
                and all(
                    math.isfinite(float(metrics[key]))
                    for key in ("cross_entropy", "margin", "accuracy")
                )
            )
            rows.append(
                {
                    "seed": seed,
                    "rule": rule,
                    "horizon": step,
                    "cross_entropy": metrics["cross_entropy"],
                    "margin": metrics["margin"],
                    "accuracy": metrics["accuracy"],
                    "weight_norm": float(torch.linalg.vector_norm(model.weight)),
                    "weight_cosine_to_local": _cosine(
                        model.weight.detach(), local_weight
                    ),
                    "weight_relative_distance_to_local": _relative_distance(
                        model.weight.detach(), local_weight
                    ),
                    "finite": finite,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "From a shared epoch-75 local state, compare local, biological type-pair 4m+r, "
            "and matched shuffled 4m+r trajectories on a fixed held-out stimulus bank"
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
