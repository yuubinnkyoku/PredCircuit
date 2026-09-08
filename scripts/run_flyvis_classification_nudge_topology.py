from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from run_flyvis_temporal_classification_nudge import run_one

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import (
    RetinotopicFlyVisCircuit,
    graph_from_flyvis_retinotopy,
    type_pair_preserving_rewire,
)


def circuit_for(
    topology: str,
    base: RetinotopicFlyVisCircuit,
    spec: dict[str, object],
    *,
    extent: int,
    seed: int,
) -> RetinotopicFlyVisCircuit:
    if topology == "biological":
        return base
    if topology == "type_pair_rewire":
        return type_pair_preserving_rewire(
            base,
            swaps=max(base.graph.num_edges * 2, 1),
            seed=10_000 + seed,
        )
    if topology == "pair_rotation":
        return graph_from_flyvis_retinotopy(
            spec,
            extent=extent,
            pair_rotation_seed=30_000 + seed,
        )
    raise ValueError(f"unsupported topology: {topology}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare classification-aligned local PC learning across FlyVis topology nulls"
    )
    parser.add_argument(
        "--topology",
        choices=("biological", "type_pair_rewire", "pair_rotation"),
        required=True,
    )
    parser.add_argument("--extent", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--learning-rate", type=float, default=30.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_classification_nudge_topology.csv"),
    )
    args = parser.parse_args()

    spec = load_flyvis_spec()
    base = graph_from_flyvis_retinotopy(spec, extent=args.extent)
    rows: list[dict[str, float | int | bool | str]] = []
    for seed in range(args.seeds):
        circuit = circuit_for(
            args.topology,
            base,
            spec,
            extent=args.extent,
            seed=seed,
        )
        row = {
            **run_one(
                circuit,
                seed=seed,
                epochs=args.epochs,
                frames=args.frames,
                width=args.bar_width,
                frame_steps=args.frame_steps,
                step_size=args.step_size,
                beta=args.beta,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                max_update=args.max_update,
                test_repeats=args.test_repeats,
            ),
            "topology": args.topology,
        }
        rows.append(row)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = frame[
        [
            "mse_after",
            "accuracy_after",
            "margin_after",
            "cross_entropy_after",
            "cross_entropy_improvement",
            "mean_abs_update",
        ]
    ].agg(["mean", "median", "std"])
    print(
        f"Classification topology: topology={args.topology}, extent={args.extent}, "
        f"lr={args.learning_rate:g}, seeds={args.seeds}"
    )
    print(summary.to_string())
    print("\nPer-seed results:")
    print(
        frame[
            [
                "seed",
                "accuracy_after",
                "cross_entropy_after",
                "mse_after",
                "margin_after",
                "mean_abs_update",
                "finite",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
