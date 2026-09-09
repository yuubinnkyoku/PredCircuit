from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from diagnose_flyvis_topology_credit_geometry import null_circuits
from run_flyvis_branched_classification_adam import run_circuit
from run_flyvis_exact_ce_gradient import run_one as run_exact

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy


def degree_rewire_for_seed(spec, base, *, extent: int, seed: int):
    for topology, circuit in null_circuits(spec, base, extent=extent, seed=seed):
        if topology == "degree_rewire":
            return circuit
    raise RuntimeError("degree_rewire null was not constructed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test degree-preserving FlyVis rewires with exact CE or corrected branched Adam"
    )
    parser.add_argument("--rule", choices=("exact_ce", "branched_adam"), required=True)
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
    parser.add_argument("--train-repeats", type=int, default=4)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_degree_rewire_control.csv"),
    )
    args = parser.parse_args()

    spec = load_flyvis_spec()
    base = graph_from_flyvis_retinotopy(spec, extent=args.extent)
    rows: list[dict[str, float | int | str | bool]] = []
    for seed in range(args.seeds):
        circuit = degree_rewire_for_seed(spec, base, extent=args.extent, seed=seed)
        if args.rule == "exact_ce":
            row = run_exact(
                circuit,
                topology="degree_rewire",
                seed=seed,
                epochs=args.epochs,
                frames=args.frames,
                width=args.bar_width,
                frame_steps=args.frame_steps,
                step_size=args.step_size,
                learning_rate=args.learning_rate,
                train_repeats=args.train_repeats,
                test_repeats=args.test_repeats,
            )
        else:
            row = {
                **run_circuit(
                    circuit,
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
                "topology": "degree_rewire",
                "learning_rule": "branched_local_adam",
            }
        rows.append(row)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    metrics = [
        "accuracy_after",
        "cross_entropy_after",
        "cross_entropy_improvement",
        "margin_after",
        "mse_after",
    ]
    print(f"Degree rewire control: rule={args.rule}, lr={args.learning_rate:g}, seeds={args.seeds}")
    print(frame[metrics].agg(["mean", "median", "std"]).to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
