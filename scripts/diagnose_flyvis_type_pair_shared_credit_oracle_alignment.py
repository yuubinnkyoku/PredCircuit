from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_temporal_ce_credit_alignment import cycle_ce_oracles, geometry
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
SAMPLE_EPOCHS = (75, 80, 85, 90, 95)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float(torch.dot(left, right) / denom.clamp_min(1e-30))


def run_seed(
    *,
    seed: int,
    shuffle_seed: int,
    learning_rate: float,
    max_update: float,
) -> list[dict[str, float | int | bool | str]]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    biological_groups = type_pair_indices(circuit)
    shuffled_groups = shuffled_groups_like(
        biological_groups,
        edge_count=model.weight.numel(),
        shuffle_seed=shuffle_seed,
    )

    for epoch in range(FROZEN_EPOCH):
        edge, bias = credit(model, circuit, seed=seed, epoch=epoch)
        apply_local_credit(
            model,
            edge,
            bias,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )

    sums: dict[str, dict[str, float]] = {
        "type": {},
        "shuffle": {},
    }
    finite = bool(torch.isfinite(model.weight).all()) and bool(
        torch.isfinite(model.bias).all()
    )

    for epoch in SAMPLE_EPOCHS:
        raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
        oracle_edge, oracle_bias = cycle_ce_oracles(
            model,
            circuit,
            seed=seed,
            epoch=epoch,
            frames=7,
            width=0.5,
            frame_steps=2,
            step_size=0.015,
        )["final_ce_oracle"]
        raw_geometry = geometry(raw_edge, raw_bias, oracle_edge, oracle_bias)
        raw_edge_norm = torch.linalg.vector_norm(raw_edge).clamp_min(1e-30)

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
            residual = linear_mix_direction(
                raw_edge,
                groups,
                coarse_gain=0.0,
                residual_gain=1.0,
            )
            mixed = linear_mix_direction(
                raw_edge,
                groups,
                coarse_gain=4.0,
                residual_gain=1.0,
            )
            matched = match_norm(mixed, raw_edge)

            coarse_geometry = geometry(
                coarse,
                torch.zeros_like(raw_bias),
                oracle_edge,
                oracle_bias,
            )
            mixed_geometry = geometry(matched, raw_bias, oracle_edge, oracle_bias)
            values = {
                "raw_oracle_cosine": raw_geometry["cosine"],
                "coarse_oracle_cosine": coarse_geometry["cosine"],
                "coarse_oracle_projection_coefficient": coarse_geometry[
                    "projection_coefficient"
                ],
                "residual_oracle_edge_cosine": _cosine(residual, oracle_edge),
                "mixed_oracle_cosine": mixed_geometry["cosine"],
                "mixed_oracle_projection_coefficient": mixed_geometry[
                    "projection_coefficient"
                ],
                "mixed_cosine_gain_vs_raw": (
                    mixed_geometry["cosine"] - raw_geometry["cosine"]
                ),
                "coarse_norm_fraction": float(
                    torch.linalg.vector_norm(coarse) / raw_edge_norm
                ),
                "residual_norm_fraction": float(
                    torch.linalg.vector_norm(residual) / raw_edge_norm
                ),
                "mixed_relative_change": float(
                    torch.linalg.vector_norm(matched - raw_edge) / raw_edge_norm
                ),
            }
            target = sums[grouping]
            for key, value in values.items():
                target[key] = target.get(key, 0.0) + value
            finite = finite and all(math.isfinite(value) for value in values.values())

    rows: list[dict[str, float | int | bool | str]] = []
    count = float(len(SAMPLE_EPOCHS))
    for grouping in ("type", "shuffle"):
        rows.append(
            {
                "seed": seed,
                "grouping": grouping,
                "sample_count": len(SAMPLE_EPOCHS),
                **{key: value / count for key, value in sums[grouping].items()},
                "finite": finite,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "At a frozen post-local state, test whether biological type-pair shared credit "
            "and norm-matched 4m+r align better with the exact final-frame CE descent than "
            "a matched shuffled grouping"
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
