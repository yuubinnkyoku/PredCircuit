from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from diagnose_local_gradient_trajectory import extended_metrics, oracle_descent_directions
from run_flyvis_retinotopic_contrastive import DIRECTIONS, render_motion_batch, targets_for
from run_flyvis_retinotopic_temporal_adam import temporal_local_direction

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import (
    RetinotopicFlyVisCircuit,
    graph_from_flyvis_retinotopy,
    type_pair_preserving_rewire,
)
from predcircuit.model import PredictiveCodingGraph


def _norm(vector: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(vector.float().flatten()))


def _top_l2_fraction(vector: torch.Tensor, fraction: float) -> float:
    flat = vector.float().flatten().square()
    total = float(flat.sum())
    if total <= 0.0:
        return float("nan")
    count = max(int(flat.numel() * fraction), 1)
    top = torch.topk(flat, count, sorted=False).values
    return float(top.sum()) / total


def _edge_distribution(vectors: list[torch.Tensor], prefix: str) -> dict[str, float]:
    absolute = torch.cat([vector.float().flatten().abs() for vector in vectors])
    mean_abs = float(absolute.mean())
    quantiles = torch.quantile(
        absolute,
        torch.tensor([0.5, 0.9, 0.99, 0.999], dtype=absolute.dtype),
    )
    stacked = torch.cat([vector.float().flatten() for vector in vectors])
    return {
        f"{prefix}_mean_abs": mean_abs,
        f"{prefix}_abs_q50": float(quantiles[0]),
        f"{prefix}_abs_q90": float(quantiles[1]),
        f"{prefix}_abs_q99": float(quantiles[2]),
        f"{prefix}_abs_q999": float(quantiles[3]),
        f"{prefix}_max_abs": float(absolute.max()),
        f"{prefix}_max_to_mean_abs": float(absolute.max()) / mean_abs
        if mean_abs > 0.0
        else float("nan"),
        f"{prefix}_top_0p1pct_l2_fraction": _top_l2_fraction(stacked, 0.001),
        f"{prefix}_top_1pct_l2_fraction": _top_l2_fraction(stacked, 0.01),
    }


def component_row(
    *,
    topology: str,
    seed: int,
    extent: int,
    component: str,
    local_vectors: list[torch.Tensor],
    oracle_vectors: list[torch.Tensor],
    nodes: int,
    edges: int,
) -> dict[str, float | int | str]:
    individual_cosines = [
        extended_metrics(local, oracle)["cosine"]
        for local, oracle in zip(local_vectors, oracle_vectors, strict=True)
    ]
    local_sum = torch.stack(local_vectors).sum(dim=0)
    oracle_sum = torch.stack(oracle_vectors).sum(dim=0)
    residual_sum = local_sum - oracle_sum
    local_norm_sum = sum(_norm(vector) for vector in local_vectors)
    oracle_norm_sum = sum(_norm(vector) for vector in oracle_vectors)
    local_sum_norm = _norm(local_sum)
    oracle_sum_norm = _norm(oracle_sum)
    summed = extended_metrics(local_sum, oracle_sum)
    row: dict[str, float | int | str] = {
        "topology": topology,
        "seed": seed,
        "extent": extent,
        "component": component,
        "mean_individual_cosine": sum(individual_cosines) / len(individual_cosines),
        "min_individual_cosine": min(individual_cosines),
        "summed_cosine": summed["cosine"],
        "summed_norm_ratio": summed["norm_ratio"],
        "local_cancellation_ratio": local_sum_norm / local_norm_sum,
        "oracle_cancellation_ratio": oracle_sum_norm / oracle_norm_sum,
        "residual_to_oracle_sum_norm": _norm(residual_sum) / oracle_sum_norm,
        "local_sum_norm": local_sum_norm,
        "oracle_sum_norm": oracle_sum_norm,
        "nodes": nodes,
        "edges": edges,
    }
    if component == "edge":
        row.update(_edge_distribution(local_vectors, "local"))
        row.update(_edge_distribution(oracle_vectors, "oracle"))
    return row


def run_one(
    circuit: RetinotopicFlyVisCircuit,
    *,
    topology: str,
    seed: int,
    extent: int,
    frames: int,
    width: float,
    frame_steps: int,
    step_size: float,
    beta: float,
) -> list[dict[str, float | int | str]]:
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    local_by_component: dict[str, list[torch.Tensor]] = {
        "edge": [],
        "bias": [],
        "combined": [],
    }
    oracle_by_component: dict[str, list[torch.Tensor]] = {
        "edge": [],
        "bias": [],
        "combined": [],
    }

    for direction in DIRECTIONS:
        stimulus = render_motion_batch(
            circuit,
            [direction],
            frames=frames,
            width=width,
            jitter_seed=800_000 + 10_000 * seed + int(direction),
        )
        targets, _ = targets_for([direction])
        oracle_edge, oracle_bias, _ = oracle_descent_directions(
            model,
            circuit,
            stimulus,
            targets,
            frame_steps=frame_steps,
            step_size=step_size,
        )
        local_edge, local_bias, _ = temporal_local_direction(
            model,
            circuit,
            stimulus,
            targets,
            beta=beta,
            frame_steps=frame_steps,
            step_size=step_size,
        )
        local_by_component["edge"].append(local_edge.flatten())
        oracle_by_component["edge"].append(oracle_edge.flatten())
        local_by_component["bias"].append(local_bias.flatten())
        oracle_by_component["bias"].append(oracle_bias.flatten())
        local_by_component["combined"].append(
            torch.cat((local_edge.flatten(), local_bias.flatten()))
        )
        oracle_by_component["combined"].append(
            torch.cat((oracle_edge.flatten(), oracle_bias.flatten()))
        )

    return [
        component_row(
            topology=topology,
            seed=seed,
            extent=extent,
            component=component,
            local_vectors=local_by_component[component],
            oracle_vectors=oracle_by_component[component],
            nodes=circuit.graph.num_nodes,
            edges=circuit.graph.num_edges,
        )
        for component in ("edge", "bias", "combined")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure cancellation of local and exact credit across motion directions"
    )
    parser.add_argument("--extent", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_direction_cancellation.csv"),
    )
    args = parser.parse_args()

    biological = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=args.extent)
    rows: list[dict[str, float | int | str]] = []
    for seed in range(args.seeds):
        rewired = type_pair_preserving_rewire(
            biological,
            swaps=max(biological.graph.num_edges * 2, 1),
            seed=10_000 + seed,
        )
        for topology, circuit in (
            ("biological", biological),
            ("type_pair_rewire", rewired),
        ):
            rows.extend(
                run_one(
                    circuit,
                    topology=topology,
                    seed=seed,
                    extent=args.extent,
                    frames=args.frames,
                    width=args.bar_width,
                    frame_steps=args.frame_steps,
                    step_size=args.step_size,
                    beta=args.beta,
                )
            )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = (
        frame.groupby(["topology", "component"])[
            [
                "mean_individual_cosine",
                "summed_cosine",
                "local_cancellation_ratio",
                "oracle_cancellation_ratio",
                "residual_to_oracle_sum_norm",
                "summed_norm_ratio",
            ]
        ]
        .agg(["mean", "median", "std"])
        .sort_index()
    )
    print(f"Direction cancellation diagnostic: extent={args.extent}, seeds={args.seeds}")
    print(summary.to_string())

    edge = frame[frame["component"] == "edge"]
    distribution_summary = (
        edge.groupby("topology")[
            [
                "local_mean_abs",
                "local_abs_q99",
                "local_abs_q999",
                "local_max_abs",
                "local_max_to_mean_abs",
                "local_top_0p1pct_l2_fraction",
                "local_top_1pct_l2_fraction",
                "oracle_max_to_mean_abs",
                "oracle_top_1pct_l2_fraction",
            ]
        ]
        .mean()
        .sort_index()
    )
    print("\nEdge-distribution means across seeds:")
    print(distribution_summary.to_string())
    print("\nPer-seed/component results:")
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
