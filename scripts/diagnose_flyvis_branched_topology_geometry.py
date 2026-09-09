from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from diagnose_flyvis_temporal_ce_credit_alignment import cycle_ce_oracles, geometry
from diagnose_flyvis_topology_credit_geometry import null_circuits

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def measure(
    circuit,
    *,
    topology: str,
    seed: int,
    repeat: int,
    frames: int,
    width: float,
    beta: float,
    frame_steps: int,
    nudge_steps: int,
    step_size: float,
) -> list[dict[str, float | int | str]]:
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    local_edge, local_bias = cycle_branched_credit(
        model,
        circuit,
        seed=seed,
        epoch=repeat,
        frames=frames,
        width=width,
        beta=beta,
        frame_steps=frame_steps,
        nudge_steps=nudge_steps,
        step_size=step_size,
        terminal_only=False,
    )
    oracles = cycle_ce_oracles(
        model,
        circuit,
        seed=seed,
        epoch=repeat,
        frames=frames,
        width=width,
        frame_steps=frame_steps,
        step_size=step_size,
    )

    rows: list[dict[str, float | int | str]] = []
    for oracle_name, (oracle_edge, oracle_bias) in oracles.items():
        rows.append(
            {
                "topology": topology,
                "oracle_objective": oracle_name,
                "seed": seed,
                "repeat": repeat,
                "frame_steps": frame_steps,
                "nudge_steps": nudge_steps,
                "nodes": circuit.graph.num_nodes,
                "edges": circuit.graph.num_edges,
                **geometry(local_edge, local_bias, oracle_edge, oracle_bias),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare corrected per-frame branched local credit against exact CE gradients "
            "across FlyVis topology nulls"
        )
    )
    parser.add_argument("--extent", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--nudge-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_branched_topology_geometry.csv"),
    )
    args = parser.parse_args()

    spec = load_flyvis_spec()
    base = graph_from_flyvis_retinotopy(spec, extent=args.extent)
    rows: list[dict[str, float | int | str]] = []
    for seed in range(args.seeds):
        for topology, circuit in null_circuits(
            spec,
            base,
            extent=args.extent,
            seed=seed,
        ):
            for repeat in range(args.repeats):
                rows.extend(
                    measure(
                        circuit,
                        topology=topology,
                        seed=seed,
                        repeat=repeat,
                        frames=args.frames,
                        width=args.bar_width,
                        beta=args.beta,
                        frame_steps=args.frame_steps,
                        nudge_steps=args.nudge_steps,
                        step_size=args.step_size,
                    )
                )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    metrics = [
        "cosine",
        "sign_agreement",
        "projection_coefficient",
        "residual_fraction",
        "parallel_fraction",
        "norm_ratio",
    ]
    summary = frame.groupby(["oracle_objective", "topology"])[metrics].agg(
        ["mean", "median", "std"]
    )
    print(
        f"Branched topology geometry: extent={args.extent}, seeds={args.seeds}, "
        f"repeats={args.repeats}, nudge_steps={args.nudge_steps}"
    )
    print(summary.to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
