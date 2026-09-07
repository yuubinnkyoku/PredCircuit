from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from run_flyvis_retinotopic_temporal_contrastive import run_one

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import (
    graph_from_flyvis_retinotopy,
    type_pair_preserving_rewire,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare increasingly constrained FlyVis topology nulls under local PC"
    )
    parser.add_argument("--extent", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--weight-lr", type=float, default=10.0)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_temporal_topology_nulls.csv"),
    )
    args = parser.parse_args()

    spec = load_flyvis_spec()
    biological = graph_from_flyvis_retinotopy(spec, extent=args.extent)
    rows: list[dict[str, float | int | str | bool]] = []
    for seed in range(args.seeds):
        type_pair = type_pair_preserving_rewire(
            biological,
            swaps=max(biological.graph.num_edges * 2, 1),
            seed=10_000 + seed,
        )
        pair_rotation = graph_from_flyvis_retinotopy(
            spec,
            extent=args.extent,
            pair_rotation_seed=20_000 + seed,
        )
        circuits = (
            ("biological", biological),
            ("pair_rotation", pair_rotation),
            ("type_pair_rewire", type_pair),
        )
        for topology, circuit in circuits:
            row = run_one(
                circuit,
                topology=topology,
                seed=seed,
                epochs=args.epochs,
                frames=args.frames,
                width=args.bar_width,
                frame_steps=args.frame_steps,
                step_size=args.step_size,
                beta=args.beta,
                weight_lr=args.weight_lr,
                test_repeats=args.test_repeats,
            )
            row["edge_count_delta"] = circuit.graph.num_edges - biological.graph.num_edges
            rows.append(row)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = (
        frame.groupby("topology")[
            [
                "mse_after",
                "mse_improvement",
                "accuracy_after",
                "margin_after",
                "edge_count_delta",
            ]
        ]
        .agg(["mean", "median", "std"])
        .sort_index()
    )
    print(
        f"Temporal local PC topology nulls: lr={args.weight_lr:g}, "
        f"seeds={args.seeds}, extent={args.extent}"
    )
    print(summary.to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
