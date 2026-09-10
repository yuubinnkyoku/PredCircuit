from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from diagnose_flyvis_temporal_ce_credit_alignment import cycle_ce_oracles, geometry
from run_flyvis_type_pair_consensus import type_pair_indices
from run_flyvis_type_pair_linear_mix import linear_mix_direction
from run_flyvis_type_pair_shuffle_control import shuffled_groups_like

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def measure(seed: int, *, shuffle_seed: int) -> list[dict[str, float | int | str]]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    biological_groups = type_pair_indices(circuit)
    shuffled_groups = shuffled_groups_like(
        biological_groups,
        edge_count=model.weight.numel(),
        shuffle_seed=shuffle_seed,
    )
    local_edge, local_bias = cycle_branched_credit(
        model,
        circuit,
        seed=seed,
        epoch=0,
        frames=7,
        width=0.5,
        beta=0.03,
        frame_steps=2,
        nudge_steps=2,
        step_size=0.015,
        terminal_only=False,
    )
    candidates = {
        "local": local_edge,
        "type_pair_4m_plus_r": linear_mix_direction(
            local_edge,
            biological_groups,
            coarse_gain=4.0,
            residual_gain=1.0,
        ),
        "shuffled_4m_plus_r": linear_mix_direction(
            local_edge,
            shuffled_groups,
            coarse_gain=4.0,
            residual_gain=1.0,
        ),
    }
    oracles = cycle_ce_oracles(
        model,
        circuit,
        seed=seed,
        epoch=0,
        frames=7,
        width=0.5,
        frame_steps=2,
        step_size=0.015,
    )

    rows: list[dict[str, float | int | str]] = []
    zero_bias = torch.zeros_like(local_bias)
    for rule, edge in candidates.items():
        for oracle_name, (oracle_edge, oracle_bias) in oracles.items():
            for component, candidate_bias, target_bias in (
                ("edge", zero_bias, zero_bias),
                ("combined", local_bias, oracle_bias),
            ):
                rows.append(
                    {
                        "seed": seed,
                        "shuffle_seed": shuffle_seed,
                        "rule": rule,
                        "oracle_objective": oracle_name,
                        "component": component,
                        **geometry(edge, candidate_bias, oracle_edge, target_bias),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare local, biological type-pair 4m+r, and size-matched shuffled 4m+r "
            "credit geometry against exact CE descent"
        )
    )
    parser.add_argument("--seed-start", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--shuffle-offset", type=int, default=10_000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, float | int | str]] = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        rows.extend(measure(seed, shuffle_seed=args.shuffle_offset + seed))

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = frame.groupby(["oracle_objective", "component", "rule"])[
        ["cosine", "projection_coefficient", "residual_fraction", "norm_ratio", "sign_agreement"]
    ].agg(["mean", "median", "std"])
    print(summary.to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
