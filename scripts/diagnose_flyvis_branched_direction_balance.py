from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_branch_nudge_alignment import branched_local_direction
from diagnose_flyvis_temporal_ce_credit_alignment import ce_oracle_descents
from diagnose_flyvis_topology_credit_geometry import null_circuits
from run_flyvis_retinotopic_contrastive import DIRECTIONS, render_motion_batch, targets_for

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def vector(edge: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.cat((edge.flatten(), bias.flatten()))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = torch.linalg.vector_norm(a).clamp_min(1e-30)
    b_norm = torch.linalg.vector_norm(b).clamp_min(1e-30)
    return float(torch.dot(a, b) / (a_norm * b_norm))


def measure(
    circuit,
    *,
    topology: str,
    seed: int,
    frames: int,
    width: float,
    beta: float,
    frame_steps: int,
    nudge_steps: int,
    step_size: float,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    local_vectors: list[torch.Tensor] = []
    oracle_vectors: list[torch.Tensor] = []
    direction_rows: list[dict[str, float | int | str]] = []

    for sample_index, direction in enumerate(DIRECTIONS):
        stimulus = render_motion_batch(
            circuit,
            [direction],
            frames=frames,
            width=width,
            jitter_seed=100_000 * seed + sample_index,
        )
        _, classes = targets_for([direction])
        local_edge, local_bias = branched_local_direction(
            model,
            circuit,
            stimulus,
            classes,
            beta=beta,
            frame_steps=frame_steps,
            nudge_steps=nudge_steps,
            step_size=step_size,
            terminal_only=False,
        )
        oracle_edge, oracle_bias = ce_oracle_descents(
            model,
            circuit,
            stimulus,
            classes,
            frame_steps=frame_steps,
            step_size=step_size,
        )["final_ce_oracle"]
        local = vector(local_edge, local_bias)
        oracle = vector(oracle_edge, oracle_bias)
        local_norm = torch.linalg.vector_norm(local).clamp_min(1e-30)
        oracle_norm = torch.linalg.vector_norm(oracle).clamp_min(1e-30)
        local_vectors.append(local)
        oracle_vectors.append(oracle)
        direction_rows.append(
            {
                "topology": topology,
                "seed": seed,
                "direction": float(direction),
                "local_norm": float(local_norm),
                "oracle_norm": float(oracle_norm),
                "norm_ratio": float(local_norm / oracle_norm),
                "cosine": cosine(local, oracle),
            }
        )

    local_norms = torch.tensor([torch.linalg.vector_norm(item) for item in local_vectors])
    oracle_norms = torch.tensor([torch.linalg.vector_norm(item) for item in oracle_vectors])
    local_sum = torch.stack(local_vectors).sum(dim=0)
    oracle_sum = torch.stack(oracle_vectors).sum(dim=0)
    local_norm_sum = local_norms.sum().clamp_min(1e-30)
    oracle_norm_sum = oracle_norms.sum().clamp_min(1e-30)

    mean_local_norm = local_norms.mean()
    equalized = torch.stack(
        [
            item * (mean_local_norm / norm.clamp_min(1e-30))
            for item, norm in zip(local_vectors, local_norms, strict=True)
        ]
    ).sum(dim=0)
    oracle_matched = torch.stack(
        [
            local * (oracle_norm / local_norm.clamp_min(1e-30))
            for local, local_norm, oracle_norm in zip(
                local_vectors,
                local_norms,
                oracle_norms,
                strict=True,
            )
        ]
    ).sum(dim=0)

    local_cv = float(local_norms.std(unbiased=False) / mean_local_norm.clamp_min(1e-30))
    oracle_cv = float(
        oracle_norms.std(unbiased=False) / oracle_norms.mean().clamp_min(1e-30)
    )
    individual_cosines = [float(row["cosine"]) for row in direction_rows]
    summary = {
        "topology": topology,
        "seed": seed,
        "raw_summed_cosine": cosine(local_sum, oracle_sum),
        "equalized_summed_cosine": cosine(equalized, oracle_sum),
        "oracle_norm_matched_summed_cosine": cosine(oracle_matched, oracle_sum),
        "mean_individual_cosine": sum(individual_cosines) / len(DIRECTIONS),
        "min_individual_cosine": min(individual_cosines),
        "local_norm_cv": local_cv,
        "oracle_norm_cv": oracle_cv,
        "local_max_to_min_norm": float(local_norms.max() / local_norms.min().clamp_min(1e-30)),
        "oracle_max_to_min_norm": float(oracle_norms.max() / oracle_norms.min().clamp_min(1e-30)),
        "local_cancellation_ratio": float(torch.linalg.vector_norm(local_sum) / local_norm_sum),
        "oracle_cancellation_ratio": float(torch.linalg.vector_norm(oracle_sum) / oracle_norm_sum),
        "nodes": circuit.graph.num_nodes,
        "edges": circuit.graph.num_edges,
    }
    return summary, direction_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure direction-wise imbalance and cancellation in corrected branched credit"
    )
    parser.add_argument("--extent", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--nudge-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("results/generated/flyvis_branched_direction_balance.csv"),
    )
    parser.add_argument(
        "--direction-out",
        type=Path,
        default=Path("results/generated/flyvis_branched_direction_components.csv"),
    )
    args = parser.parse_args()

    spec = load_flyvis_spec()
    base = graph_from_flyvis_retinotopy(spec, extent=args.extent)
    summaries: list[dict[str, float | int | str]] = []
    direction_rows: list[dict[str, float | int | str]] = []
    for seed in range(args.seeds):
        for topology, circuit in null_circuits(
            spec,
            base,
            extent=args.extent,
            seed=seed,
        ):
            summary, components = measure(
                circuit,
                topology=topology,
                seed=seed,
                frames=args.frames,
                width=args.bar_width,
                beta=args.beta,
                frame_steps=args.frame_steps,
                nudge_steps=args.nudge_steps,
                step_size=args.step_size,
            )
            summaries.append(summary)
            direction_rows.extend(components)

    summary_frame = pd.DataFrame(summaries)
    direction_frame = pd.DataFrame(direction_rows)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(args.summary_out, index=False)
    direction_frame.to_csv(args.direction_out, index=False)
    metrics = [
        "raw_summed_cosine",
        "equalized_summed_cosine",
        "oracle_norm_matched_summed_cosine",
        "mean_individual_cosine",
        "local_norm_cv",
        "oracle_norm_cv",
        "local_cancellation_ratio",
        "oracle_cancellation_ratio",
    ]
    print(summary_frame.groupby("topology")[metrics].agg(["mean", "median", "std"]).to_string())
    print(f"\nSaved: {args.summary_out}")
    print(f"Saved: {args.direction_out}")


if __name__ == "__main__":
    main()
