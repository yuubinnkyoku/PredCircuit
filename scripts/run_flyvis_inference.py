from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from predcircuit.flyvis import (
    graph_from_flyvis_spec,
    load_flyvis_spec,
    with_sensory_distance_rank,
)
from predcircuit.metrics import graph_metrics
from predcircuit.model import PredictiveCodingGraph
from predcircuit.topology import CircuitGraph


def evaluate_inference(
    graph: CircuitGraph,
    *,
    input_nodes: tuple[int, ...],
    output_nodes: tuple[int, ...],
    seed: int,
    samples: int,
    steps: int,
    step_size: float,
    use_biological_strength: bool,
) -> dict[str, float | int | bool]:
    model = PredictiveCodingGraph(
        graph,
        seed=seed,
        use_biological_strength=use_biological_strength,
    )
    generator = torch.Generator().manual_seed(100_000 + seed)
    inputs = 0.5 * torch.randn(samples, len(input_nodes), generator=generator)
    state = torch.zeros(samples, graph.num_nodes, dtype=torch.float32)
    clamp_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
    clamp_mask[list(input_nodes)] = True
    clamp_values = torch.zeros_like(state)
    clamp_values[:, list(input_nodes)] = inputs
    state[:, clamp_mask] = clamp_values[:, clamp_mask]

    initial_energy = float(model.energy(state))
    inferred, trace = model.infer(
        state,
        clamp_mask=clamp_mask,
        clamp_values=clamp_values,
        steps=steps,
        step_size=step_size,
        record_trace=True,
    )
    if trace is None or not trace.energy:
        raise RuntimeError("inference trace was not recorded")
    final_energy = trace.energy[-1]
    decreases = [later <= earlier + 1e-8 for earlier, later in zip(trace.energy, trace.energy[1:])]
    monotone_fraction = sum(decreases) / max(len(decreases), 1)
    output_rms = (
        float(inferred[:, list(output_nodes)].square().mean().sqrt())
        if output_nodes
        else float("nan")
    )
    metrics = graph_metrics(graph)
    return {
        "seed": seed,
        "nodes": graph.num_nodes,
        "edges": graph.num_edges,
        "inputs": len(input_nodes),
        "outputs": len(output_nodes),
        "initial_energy_per_node": initial_energy / graph.num_nodes,
        "final_energy_per_node": final_energy / graph.num_nodes,
        "energy_ratio": final_energy / max(initial_energy, 1e-12),
        "monotone_step_fraction": monotone_fraction,
        "final_state_rms": float(inferred.square().mean().sqrt()),
        "output_rms": output_rms,
        "reciprocal_fraction": metrics.reciprocal_fraction,
        "finite": bool(torch.isfinite(inferred).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a topology/inference pilot on the published FlyVis type-level connectome"
    )
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--step-size", type=float, default=0.03)
    parser.add_argument(
        "--out", type=Path, default=Path("results/generated/flyvis_type_inference.csv")
    )
    args = parser.parse_args()

    circuit = with_sensory_distance_rank(graph_from_flyvis_spec(load_flyvis_spec()))
    base = circuit.graph
    rows: list[dict[str, float | int | bool | str]] = []

    for seed in range(args.seeds):
        degree_rewire = base.degree_preserving_rewire(
            max(base.num_edges * 5, 1), seed=10_000 + seed
        )
        rank_pair_rewire = base.rank_pair_preserving_rewire(
            max(base.num_edges * 5, 1), seed=20_000 + seed
        )
        weight_shuffle = base.shuffle_edge_weights(seed=30_000 + seed)
        configs = (
            ("topology_only", "biological", base, False),
            ("topology_only", "degree_rewire", degree_rewire, False),
            ("topology_only", "rank_pair_rewire", rank_pair_rewire, False),
            ("measured_strength", "biological", base, True),
            ("measured_strength", "weight_shuffle", weight_shuffle, True),
            ("measured_strength", "rank_pair_rewire", rank_pair_rewire, True),
        )
        for family, topology, graph, biological_strength in configs:
            row: dict[str, float | int | bool | str] = {
                "family": family,
                "topology": topology,
            }
            row.update(
                evaluate_inference(
                    graph,
                    input_nodes=circuit.input_nodes,
                    output_nodes=circuit.output_nodes,
                    seed=seed,
                    samples=args.samples,
                    steps=args.steps,
                    step_size=args.step_size,
                    use_biological_strength=biological_strength,
                )
            )
            rows.append(row)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = (
        frame.groupby(["family", "topology"])[
            ["final_energy_per_node", "energy_ratio", "monotone_step_fraction"]
        ]
        .agg(["mean", "median", "std"])
        .sort_index()
    )
    print(
        f"FlyVis type graph: {base.num_nodes} nodes, {base.num_edges} type-level edges, "
        f"{len(circuit.input_nodes)} input types, {len(circuit.output_nodes)} output types"
    )
    print(summary.to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
