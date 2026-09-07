from __future__ import annotations

import argparse
import math
from pathlib import Path

import networkx as nx
import pandas as pd

from predcircuit.topology import CircuitGraph, erdos_renyi_matched, layered_graph


def task_path_stats(
    graph: CircuitGraph, input_nodes: list[int], output_nodes: list[int]
) -> dict[str, float | int]:
    """Measure directed input-to-output path structure for a supervised graph task."""
    nx_graph = graph.to_networkx()
    lengths: list[int] = []
    direct_pairs = 0
    total_pairs = len(input_nodes) * len(output_nodes)
    for src in input_nodes:
        for dst in output_nodes:
            if nx_graph.has_edge(src, dst):
                direct_pairs += 1
            try:
                lengths.append(nx.shortest_path_length(nx_graph, src, dst))
            except nx.NetworkXNoPath:
                pass
    return {
        "reachable_pair_fraction": len(lengths) / max(total_pairs, 1),
        "direct_pair_fraction": direct_pairs / max(total_pairs, 1),
        "min_path_length": min(lengths) if lengths else math.inf,
        "mean_path_length": sum(lengths) / len(lengths) if lengths else math.inf,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose whether synthetic topology nulls create task shortcuts"
    )
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument(
        "--out", type=Path, default=Path("results/generated/synthetic_null_paths.csv")
    )
    args = parser.parse_args()

    rows: list[dict[str, float | int | str]] = []
    for seed in range(args.seeds):
        base = layered_graph(
            [2, 8, 1], recurrent_probability=0.2, feedback_probability=0.08, seed=seed
        )
        variants = {
            "base": base,
            "degree_preserving_rewire": base.degree_preserving_rewire(
                max(base.num_edges * 3, 1), seed=10_000 + seed
            ),
            "rank_pair_preserving_rewire": base.rank_pair_preserving_rewire(
                max(base.num_edges * 3, 1), seed=15_000 + seed
            ),
            "er_matched": erdos_renyi_matched(base, seed=20_000 + seed),
            "no_feedback": base.remove_feedback_edges(),
            "no_reciprocal": base.remove_reciprocal_edges(),
        }
        for topology, graph in variants.items():
            row: dict[str, float | int | str] = {
                "seed": seed,
                "topology": topology,
                "nodes": graph.num_nodes,
                "edges": graph.num_edges,
            }
            row.update(task_path_stats(graph, [0, 1], [graph.num_nodes - 1]))
            rows.append(row)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = frame.groupby("topology")[
        ["reachable_pair_fraction", "direct_pair_fraction", "min_path_length", "mean_path_length"]
    ].agg(["mean", "median", "min", "max"])
    print(summary.to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
