from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from run_flyvis_branched_classification_adam import run_one
from run_flyvis_classification_nudge_topology import circuit_for

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare per-frame branched local Adam learning across FlyVis topology nulls"
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
    parser.add_argument("--nudge-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_branched_classification_adam_topology.csv"),
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
                extent=args.extent,
                seed=seed,
                epochs=args.epochs,
                frames=args.frames,
                width=args.bar_width,
                frame_steps=args.frame_steps,
                nudge_steps=args.nudge_steps,
                step_size=args.step_size,
                beta=args.beta,
                learning_rate=args.learning_rate,
                beta1=args.beta1,
                beta2=args.beta2,
                adam_epsilon=args.adam_epsilon,
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
            "accuracy_after",
            "cross_entropy_after",
            "cross_entropy_improvement",
            "margin_after",
            "mse_after",
            "mean_abs_applied_update",
        ]
    ].agg(["mean", "median", "std"])
    print(
        f"Branched Adam topology: topology={args.topology}, extent={args.extent}, "
        f"lr={args.learning_rate:g}, seeds={args.seeds}"
    )
    print(summary.to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
