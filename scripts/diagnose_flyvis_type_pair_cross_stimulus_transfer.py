from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_strict_curvature import exact_cycle_loss
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
TRAIN_EPOCHS = (75, 80, 85, 90, 95)
TRANSFER_OFFSETS = (1, 2, 3, 4)


def _loss(model: PredictiveCodingGraph, circuit, *, seed: int, epoch: int) -> float:
    return float(
        exact_cycle_loss(
            circuit,
            model.weight,
            model.bias,
            seed=seed,
            epoch=epoch,
        ).detach()
    )


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

    sums = {
        "type": {"self_ce_delta": 0.0, "transfer_ce_delta": 0.0, "transfer_minus_self": 0.0},
        "shuffle": {"self_ce_delta": 0.0, "transfer_ce_delta": 0.0, "transfer_minus_self": 0.0},
    }
    finite = bool(torch.isfinite(model.weight).all()) and bool(torch.isfinite(model.bias).all())

    for train_epoch in TRAIN_EPOCHS:
        raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=train_epoch)
        baseline_self = _loss(model, circuit, seed=seed, epoch=train_epoch)
        transfer_epochs = tuple(train_epoch + offset for offset in TRANSFER_OFFSETS)
        baseline_transfer = sum(
            _loss(model, circuit, seed=seed, epoch=epoch) for epoch in transfer_epochs
        ) / len(transfer_epochs)

        for grouping, groups in (
            ("type", biological_groups),
            ("shuffle", shuffled_groups),
        ):
            mixed = linear_mix_direction(
                raw_edge,
                groups,
                coarse_gain=4.0,
                residual_gain=1.0,
            )
            matched = match_norm(mixed, raw_edge)
            updated = copy.deepcopy(model)
            apply_local_credit(
                updated,
                matched,
                raw_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )

            self_delta = _loss(updated, circuit, seed=seed, epoch=train_epoch) - baseline_self
            transfer_delta = (
                sum(_loss(updated, circuit, seed=seed, epoch=epoch) for epoch in transfer_epochs)
                / len(transfer_epochs)
                - baseline_transfer
            )
            target = sums[grouping]
            target["self_ce_delta"] += self_delta
            target["transfer_ce_delta"] += transfer_delta
            target["transfer_minus_self"] += transfer_delta - self_delta
            finite = finite and math.isfinite(self_delta) and math.isfinite(transfer_delta)

    count = float(len(TRAIN_EPOCHS))
    rows: list[dict[str, float | int | bool | str]] = []
    for grouping in ("type", "shuffle"):
        rows.append(
            {
                "seed": seed,
                "grouping": grouping,
                "train_epoch_count": len(TRAIN_EPOCHS),
                "transfer_offset_count": len(TRANSFER_OFFSETS),
                **{key: value / count for key, value in sums[grouping].items()},
                "finite": finite,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "At a frozen post-local state, compare whether biological type-pair versus "
            "matched shuffled norm-matched 4m+r updates transfer CE improvement from the "
            "stimulus used to compute credit to unseen stimulus jitters from subsequent epochs"
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
