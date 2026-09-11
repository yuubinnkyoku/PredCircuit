from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from diagnose_flyvis_temporal_ce_credit_alignment import differentiable_frame_states, geometry
from run_flyvis_retinotopic_contrastive import (
    DIRECTIONS,
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
SAMPLE_EPOCHS = (75, 80, 85, 90, 95)


def heldout_oracles(
    model: PredictiveCodingGraph,
    circuit,
    *,
    seed: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    directions = list(DIRECTIONS) * 12
    stimulus = render_motion_batch(
        circuit,
        directions,
        frames=7,
        width=0.5,
        jitter_seed=1_300_000 + seed,
    )
    _, classes = targets_for(directions)
    weight = model.weight.detach().clone().requires_grad_(True)
    bias = model.bias.detach().clone().requires_grad_(True)
    states = differentiable_frame_states(
        circuit,
        stimulus,
        weight,
        bias,
        frame_steps=2,
        step_size=0.015,
    )
    readout = states[-1][:, output_nodes(circuit)]
    correct = readout.gather(1, classes[:, None]).squeeze(1)
    masked = readout.clone()
    masked[torch.arange(len(classes)), classes] = -torch.inf
    margin = (correct - masked.max(dim=1).values).mean()
    cross_entropy = F.cross_entropy(readout, classes)

    margin_edge, margin_bias = torch.autograd.grad(
        margin,
        (weight, bias),
        retain_graph=True,
    )
    ce_edge, ce_bias = torch.autograd.grad(cross_entropy, (weight, bias))
    return {
        "margin_ascent": (margin_edge.detach(), margin_bias.detach()),
        "ce_descent": (-ce_edge.detach(), -ce_bias.detach()),
    }


def run_seed(
    *,
    seed: int,
    shuffle_seed: int,
    learning_rate: float,
    max_update: float,
) -> pd.DataFrame:
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

    oracles = heldout_oracles(model, circuit, seed=seed)
    finite = bool(torch.isfinite(model.weight).all()) and bool(torch.isfinite(model.bias).all())
    rows: list[dict[str, float | int | bool | str]] = []

    for epoch in SAMPLE_EPOCHS:
        raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
        candidates: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
            "local": (raw_edge, raw_bias),
        }
        for name, groups in (("type", biological_groups), ("shuffle", shuffled_groups)):
            mixed = linear_mix_direction(
                raw_edge,
                groups,
                coarse_gain=4.0,
                residual_gain=1.0,
            )
            candidates[name] = (match_norm(mixed, raw_edge), raw_bias)

        for rule, (edge_direction, bias_direction) in candidates.items():
            margin_geometry = geometry(
                edge_direction,
                bias_direction,
                *oracles["margin_ascent"],
            )
            ce_geometry = geometry(
                edge_direction,
                bias_direction,
                *oracles["ce_descent"],
            )
            values = {
                "margin_oracle_cosine": margin_geometry["cosine"],
                "margin_oracle_projection_coefficient": margin_geometry["projection_coefficient"],
                "ce_oracle_cosine": ce_geometry["cosine"],
                "ce_oracle_projection_coefficient": ce_geometry["projection_coefficient"],
            }
            finite = finite and all(math.isfinite(value) for value in values.values())
            rows.append(
                {
                    "seed": seed,
                    "sample_epoch": epoch,
                    "rule": rule,
                    **values,
                    "finite": finite,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "At a frozen epoch-75 local state, compare local, biological type-pair 4m+r, "
            "and matched shuffled 4m+r against exact held-out margin-ascent and CE-descent "
            "directions"
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
