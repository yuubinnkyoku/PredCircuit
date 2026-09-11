from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
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
SAMPLE_EPOCHS = tuple(range(75, 100))


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float(torch.dot(left, right) / denom.clamp_min(1e-30))


def _pairwise_mean_cosine(vectors: list[torch.Tensor]) -> float:
    values = [
        _cosine(vectors[left], vectors[right])
        for left in range(len(vectors))
        for right in range(left + 1, len(vectors))
    ]
    return float(sum(values) / len(values))


def _leave_one_out_cosine(vectors: list[torch.Tensor], raw: list[torch.Tensor]) -> float:
    total = torch.stack(raw).sum(dim=0)
    values = []
    for index, vector in enumerate(vectors):
        target = (total - raw[index]) / (len(raw) - 1)
        values.append(_cosine(vector, target))
    return float(sum(values) / len(values))


def _mean_relative_distance_to_mean(vectors: list[torch.Tensor]) -> float:
    stacked = torch.stack(vectors)
    mean = stacked.mean(dim=0)
    mean_norm = torch.linalg.vector_norm(mean).clamp_min(1e-30)
    return float(torch.linalg.vector_norm(stacked - mean, dim=1).mean() / mean_norm)


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

    raw_vectors = [
        credit(model, circuit, seed=seed, epoch=epoch)[0].detach().clone()
        for epoch in SAMPLE_EPOCHS
    ]
    rows: list[dict[str, float | int | bool | str]] = []
    groupings = {
        "type": biological_groups,
        "shuffle": shuffled_groups,
    }

    raw_pairwise = _pairwise_mean_cosine(raw_vectors)
    raw_loo = _leave_one_out_cosine(raw_vectors, raw_vectors)
    raw_dispersion = _mean_relative_distance_to_mean(raw_vectors)
    raw_norm = float(torch.stack([torch.linalg.vector_norm(v) for v in raw_vectors]).mean())
    rows.append(
        {
            "seed": seed,
            "grouping": "raw",
            "sample_count": len(raw_vectors),
            "mean_pairwise_cosine": raw_pairwise,
            "mean_leave_one_out_cosine_to_raw_mean": raw_loo,
            "mean_relative_distance_to_own_mean": raw_dispersion,
            "mean_coarse_norm_fraction": 0.0,
            "mean_transformed_relative_change": 0.0,
            "mean_vector_norm": raw_norm,
            "finite": bool(torch.isfinite(model.weight).all())
            and bool(torch.isfinite(model.bias).all()),
        }
    )

    for grouping, groups in groupings.items():
        coarse_vectors = [
            linear_mix_direction(
                vector,
                groups,
                coarse_gain=1.0,
                residual_gain=0.0,
            )
            for vector in raw_vectors
        ]
        transformed = []
        coarse_fraction = []
        relative_change = []
        for raw, coarse in zip(raw_vectors, coarse_vectors, strict=True):
            mixed = linear_mix_direction(
                raw,
                groups,
                coarse_gain=4.0,
                residual_gain=1.0,
            )
            matched = match_norm(mixed, raw)
            transformed.append(matched)
            raw_norm_tensor = torch.linalg.vector_norm(raw).clamp_min(1e-30)
            coarse_fraction.append(float(torch.linalg.vector_norm(coarse) / raw_norm_tensor))
            relative_change.append(
                float(torch.linalg.vector_norm(matched - raw) / raw_norm_tensor)
            )

        finite = all(bool(torch.isfinite(vector).all()) for vector in transformed)
        rows.append(
            {
                "seed": seed,
                "grouping": grouping,
                "sample_count": len(raw_vectors),
                "mean_pairwise_cosine": _pairwise_mean_cosine(transformed),
                "mean_leave_one_out_cosine_to_raw_mean": _leave_one_out_cosine(
                    transformed, raw_vectors
                ),
                "mean_relative_distance_to_own_mean": _mean_relative_distance_to_mean(
                    transformed
                ),
                "mean_coarse_norm_fraction": float(sum(coarse_fraction) / len(coarse_fraction)),
                "mean_transformed_relative_change": float(
                    sum(relative_change) / len(relative_change)
                ),
                "mean_vector_norm": float(
                    torch.stack([torch.linalg.vector_norm(v) for v in transformed]).mean()
                ),
                "finite": finite and math.isfinite(_pairwise_mean_cosine(transformed)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "At a common frozen local state, test whether biological type-pair shared credit "
            "is more coherent across changing stimuli than matched shuffled grouping"
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
