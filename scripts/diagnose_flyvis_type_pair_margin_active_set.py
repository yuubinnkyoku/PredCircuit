from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from run_flyvis_retinotopic_contrastive import (
    DIRECTIONS,
    infer_sequence,
    output_nodes,
    render_motion_batch,
    targets_for,
)
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
HORIZONS = tuple(range(0, 26))
RULES = ("local", "type", "shuffle")


@torch.no_grad()
def heldout_state_metrics(
    model: PredictiveCodingGraph,
    circuit,
    *,
    seed: int,
) -> tuple[dict[str, float], torch.Tensor]:
    directions = list(DIRECTIONS) * 12
    stimulus = render_motion_batch(
        circuit,
        directions,
        frames=7,
        width=0.5,
        jitter_seed=1_300_000 + seed,
    )
    _, classes = targets_for(directions)
    state = infer_sequence(
        model,
        circuit,
        stimulus,
        frame_steps=2,
        step_size=0.015,
    )
    readout = state[:, output_nodes(circuit)]
    correct = readout.gather(1, classes[:, None]).squeeze(1)
    masked = readout.clone()
    masked[torch.arange(len(classes)), classes] = -torch.inf
    wrong_sorted, wrong_indices = masked.sort(dim=1, descending=True)
    hard_margin = correct - wrong_sorted[:, 0]

    wrong_for_soft = readout.clone()
    wrong_for_soft[torch.arange(len(classes)), classes] = -torch.inf
    soft_margin = correct - torch.logsumexp(wrong_for_soft, dim=1)
    active_gap = wrong_sorted[:, 0] - wrong_sorted[:, 1]
    metrics = {
        "cross_entropy": float(F.cross_entropy(readout, classes)),
        "hard_margin": float(hard_margin.mean()),
        "soft_margin": float(soft_margin.mean()),
        "accuracy": float((readout.argmax(dim=1) == classes).float().mean()),
        "active_gap": float(active_gap.mean()),
    }
    return metrics, wrong_indices[:, 0]


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
    base_weight = base.weight.detach().clone()
    base_metrics, base_runner_up = heldout_state_metrics(base, circuit, seed=seed)
    previous_runner_up = {rule: base_runner_up.clone() for rule in RULES}
    rows: list[dict[str, float | int | bool | str]] = []

    for horizon in HORIZONS:
        if horizon > 0:
            epoch = FROZEN_EPOCH + horizon - 1
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

        for rule in RULES:
            model = models[rule]
            metrics, runner_up = heldout_state_metrics(model, circuit, seed=seed)
            finite = (
                bool(torch.isfinite(model.weight).all())
                and bool(torch.isfinite(model.bias).all())
                and all(math.isfinite(value) for value in metrics.values())
            )
            previous_switch = float((runner_up != previous_runner_up[rule]).float().mean())
            base_switch = float((runner_up != base_runner_up).float().mean())
            weight_distance = float(
                torch.linalg.vector_norm(model.weight.detach() - base_weight)
                / torch.linalg.vector_norm(base_weight).clamp_min(1e-30)
            )
            rows.append(
                {
                    "seed": seed,
                    "rule": rule,
                    "horizon": horizon,
                    **metrics,
                    "hard_margin_gain_from_base": metrics["hard_margin"]
                    - base_metrics["hard_margin"],
                    "soft_margin_gain_from_base": metrics["soft_margin"]
                    - base_metrics["soft_margin"],
                    "runner_up_switch_from_previous": previous_switch,
                    "runner_up_switch_from_base": base_switch,
                    "weight_relative_distance_from_base": weight_distance,
                    "finite": finite,
                }
            )
            previous_runner_up[rule] = runner_up

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Track held-out hard/soft margin and runner-up active-set changes across the "
            "local, biological type-pair 4m+r, and matched shuffled trajectories"
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
