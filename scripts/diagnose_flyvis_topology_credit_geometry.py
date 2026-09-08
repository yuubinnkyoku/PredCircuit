from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch
from run_flyvis_temporal_credit_projection import flatten_credit, projection_decomposition
from run_flyvis_temporal_oracle_interpolation import cycle_directions

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import (
    RetinotopicFlyVisCircuit,
    graph_from_flyvis_retinotopy,
    type_pair_preserving_rewire,
)
from predcircuit.model import PredictiveCodingGraph


def measure_cycle(
    circuit: RetinotopicFlyVisCircuit,
    *,
    topology: str,
    seed: int,
    repeat: int,
    frames: int,
    width: float,
    beta: float,
    frame_steps: int,
    step_size: float,
) -> dict[str, float | int | str]:
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    local_edge, local_bias, oracle_edge, oracle_bias = cycle_directions(
        model,
        circuit,
        seed=seed,
        epoch=repeat,
        frames=frames,
        width=width,
        beta=beta,
        frame_steps=frame_steps,
        step_size=step_size,
    )
    local, parallel, residual, coefficient, cosine = projection_decomposition(
        local_edge,
        local_bias,
        oracle_edge,
        oracle_bias,
    )
    oracle = flatten_credit(oracle_edge, oracle_bias)
    local_norm = torch.linalg.vector_norm(local)
    oracle_norm = torch.linalg.vector_norm(oracle)
    safe_local_norm = local_norm.clamp_min(1e-30)
    return {
        "topology": topology,
        "seed": seed,
        "repeat": repeat,
        "local_oracle_cosine": cosine,
        "projection_coefficient": coefficient,
        "parallel_norm_fraction": float(torch.linalg.vector_norm(parallel) / safe_local_norm),
        "residual_norm_fraction": float(torch.linalg.vector_norm(residual) / safe_local_norm),
        "local_norm": float(local_norm),
        "oracle_norm": float(oracle_norm),
        "nodes": circuit.graph.num_nodes,
        "edges": circuit.graph.num_edges,
    }


def null_circuits(
    spec: dict[str, object],
    base: RetinotopicFlyVisCircuit,
    *,
    extent: int,
    seed: int,
) -> tuple[tuple[str, RetinotopicFlyVisCircuit], ...]:
    swaps = max(base.graph.num_edges * 2, 1)
    type_rewire = type_pair_preserving_rewire(
        base,
        swaps=swaps,
        seed=10_000 + seed,
    )
    degree_rewire = replace(
        base,
        graph=base.graph.degree_preserving_rewire(
            swaps=swaps,
            seed=20_000 + seed,
        ),
    )
    pair_rotation = graph_from_flyvis_retinotopy(
        spec,
        extent=extent,
        pair_rotation_seed=30_000 + seed,
    )
    return (
        ("biological", base),
        ("type_pair_rewire", type_rewire),
        ("pair_rotation", pair_rotation),
        ("degree_rewire", degree_rewire),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare local-credit geometry across biological FlyVis wiring, fine-wiring "
            "nulls, and a global degree-preserving null"
        )
    )
    parser.add_argument("--extent", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_topology_credit_geometry.csv"),
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
                rows.append(
                    measure_cycle(
                        circuit,
                        topology=topology,
                        seed=seed,
                        repeat=repeat,
                        frames=args.frames,
                        width=args.bar_width,
                        beta=args.beta,
                        frame_steps=args.frame_steps,
                        step_size=args.step_size,
                    )
                )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    metrics = [
        "local_oracle_cosine",
        "parallel_norm_fraction",
        "residual_norm_fraction",
        "local_norm",
        "oracle_norm",
    ]
    summary = frame.groupby("topology")[metrics].agg(["mean", "median", "std"])
    per_seed = frame.groupby(["topology", "seed"])[metrics].mean().reset_index()
    paired = per_seed.pivot(index="seed", columns="topology", values=metrics)

    print(
        f"Topology credit geometry: extent={args.extent}, seeds={args.seeds}, "
        f"repeats={args.repeats}, {base.graph.num_nodes} nodes, {base.graph.num_edges} edges"
    )
    print(summary.to_string())
    print("\nPaired biological - null deltas across seeds:")
    for null in ("type_pair_rewire", "pair_rotation", "degree_rewire"):
        cosine_delta = (
            paired[("local_oracle_cosine", "biological")]
            - paired[("local_oracle_cosine", null)]
        )
        residual_delta = (
            paired[("residual_norm_fraction", "biological")]
            - paired[("residual_norm_fraction", null)]
        )
        print(
            f"{null}: cosine mean={cosine_delta.mean():.6f}, "
            f"median={cosine_delta.median():.6f}; residual fraction "
            f"mean={residual_delta.mean():.6f}, median={residual_delta.median():.6f}"
        )
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
