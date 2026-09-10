from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from diagnose_flyvis_temporal_ce_credit_alignment import cycle_ce_oracles
from run_flyvis_credit_residual_scaling import evaluate_metrics
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_consensus import consensus_direction, type_pair_indices

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def pair_decompose(
    vector: torch.Tensor,
    groups: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    coarse = torch.zeros_like(vector)
    for indices in groups:
        coarse[indices] = vector[indices].mean()
    residual = vector - coarse
    return coarse, residual


def vector_geometry(candidate: torch.Tensor, oracle: torch.Tensor) -> dict[str, float]:
    candidate_norm = torch.linalg.vector_norm(candidate)
    oracle_norm = torch.linalg.vector_norm(oracle)
    denom = candidate_norm * oracle_norm
    cosine = float(torch.dot(candidate, oracle) / denom) if float(denom) > 1e-30 else float("nan")
    projection = float(
        torch.dot(candidate, oracle) / oracle_norm.square().clamp_min(1e-30)
    )
    return {
        "cosine": cosine,
        "candidate_norm": float(candidate_norm),
        "oracle_norm": float(oracle_norm),
        "projection": projection,
    }


def checkpoint_row(
    model: PredictiveCodingGraph,
    circuit,
    groups: list[torch.Tensor],
    initial_weight: torch.Tensor,
    initial_bias: torch.Tensor,
    *,
    seed: int,
    trajectory_rule: str,
    checkpoint: int,
    gamma: float,
) -> dict[str, float | int | str | bool]:
    local_edge, local_bias = cycle_branched_credit(
        model,
        circuit,
        seed=seed,
        epoch=0,
        frames=7,
        width=0.5,
        beta=0.03,
        frame_steps=2,
        nudge_steps=2,
        step_size=0.015,
        terminal_only=False,
    )
    oracle_edge, oracle_bias = cycle_ce_oracles(
        model,
        circuit,
        seed=seed,
        epoch=0,
        frames=7,
        width=0.5,
        frame_steps=2,
        step_size=0.015,
    )["final_ce_oracle"]
    consensus_edge = consensus_direction(local_edge, groups, gamma=gamma)

    local_coarse, local_residual = pair_decompose(local_edge, groups)
    oracle_coarse, oracle_residual = pair_decompose(oracle_edge, groups)
    consensus_coarse, consensus_residual = pair_decompose(consensus_edge, groups)

    weight_delta = model.weight.detach() - initial_weight
    weight_delta_coarse, weight_delta_residual = pair_decompose(weight_delta, groups)
    bias_delta = model.bias.detach() - initial_bias

    edge_local_geometry = vector_geometry(local_edge, oracle_edge)
    edge_consensus_geometry = vector_geometry(consensus_edge, oracle_edge)
    coarse_geometry = vector_geometry(local_coarse, oracle_coarse)
    residual_geometry = vector_geometry(local_residual, oracle_residual)
    consensus_coarse_geometry = vector_geometry(consensus_coarse, oracle_coarse)
    consensus_residual_geometry = vector_geometry(consensus_residual, oracle_residual)

    eval_metrics = evaluate_metrics(
        model,
        circuit,
        repeats=8,
        frames=7,
        width=0.5,
        frame_steps=2,
        step_size=0.015,
        jitter_seed=900_000 + seed,
    )

    local_norm = torch.linalg.vector_norm(local_edge).clamp_min(1e-30)
    oracle_norm = torch.linalg.vector_norm(oracle_edge).clamp_min(1e-30)
    weight_delta_norm = torch.linalg.vector_norm(weight_delta).clamp_min(1e-30)
    return {
        "seed": seed,
        "trajectory_rule": trajectory_rule,
        "checkpoint": checkpoint,
        "cross_entropy": eval_metrics["cross_entropy"],
        "margin": eval_metrics["margin"],
        "edge_local_cosine": edge_local_geometry["cosine"],
        "edge_consensus_cosine": edge_consensus_geometry["cosine"],
        "coarse_local_cosine": coarse_geometry["cosine"],
        "residual_local_cosine": residual_geometry["cosine"],
        "coarse_consensus_cosine": consensus_coarse_geometry["cosine"],
        "residual_consensus_cosine": consensus_residual_geometry["cosine"],
        "coarse_local_projection": coarse_geometry["projection"],
        "residual_local_projection": residual_geometry["projection"],
        "coarse_consensus_projection": consensus_coarse_geometry["projection"],
        "residual_consensus_projection": consensus_residual_geometry["projection"],
        "local_coarse_norm_fraction": float(torch.linalg.vector_norm(local_coarse) / local_norm),
        "local_residual_norm_fraction": float(
            torch.linalg.vector_norm(local_residual) / local_norm
        ),
        "oracle_coarse_norm_fraction": float(
            torch.linalg.vector_norm(oracle_coarse) / oracle_norm
        ),
        "oracle_residual_norm_fraction": float(
            torch.linalg.vector_norm(oracle_residual) / oracle_norm
        ),
        "consensus_coarse_norm_fraction": float(
            torch.linalg.vector_norm(consensus_coarse)
            / torch.linalg.vector_norm(consensus_edge).clamp_min(1e-30)
        ),
        "consensus_residual_norm_fraction": float(
            torch.linalg.vector_norm(consensus_residual)
            / torch.linalg.vector_norm(consensus_edge).clamp_min(1e-30)
        ),
        "weight_delta_norm": float(torch.linalg.vector_norm(weight_delta)),
        "weight_delta_coarse_norm": float(torch.linalg.vector_norm(weight_delta_coarse)),
        "weight_delta_residual_norm": float(torch.linalg.vector_norm(weight_delta_residual)),
        "weight_delta_coarse_fraction": float(
            torch.linalg.vector_norm(weight_delta_coarse) / weight_delta_norm
        ) if checkpoint > 0 else 0.0,
        "weight_delta_residual_fraction": float(
            torch.linalg.vector_norm(weight_delta_residual) / weight_delta_norm
        ) if checkpoint > 0 else 0.0,
        "bias_delta_norm": float(torch.linalg.vector_norm(bias_delta)),
        "oracle_bias_norm": float(torch.linalg.vector_norm(oracle_bias)),
        "local_bias_norm": float(torch.linalg.vector_norm(local_bias)),
        "finite": bool(torch.isfinite(model.weight).all())
        and bool(torch.isfinite(model.bias).all())
        and math.isfinite(eval_metrics["cross_entropy"]),
    }


def run_trajectory(
    *,
    seed: int,
    trajectory_rule: str,
    gamma: float,
    checkpoints: list[int],
    learning_rate: float,
    max_update: float,
) -> list[dict[str, float | int | str | bool]]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    groups = type_pair_indices(circuit)
    initial_weight = model.weight.detach().clone()
    initial_bias = model.bias.detach().clone()
    rows: list[dict[str, float | int | str | bool]] = []
    previous = 0
    for checkpoint in checkpoints:
        for epoch in range(previous, checkpoint):
            edge_direction, bias_direction = cycle_branched_credit(
                model,
                circuit,
                seed=seed,
                epoch=epoch,
                frames=7,
                width=0.5,
                beta=0.03,
                frame_steps=2,
                nudge_steps=2,
                step_size=0.015,
                terminal_only=False,
            )
            if trajectory_rule == "consensus":
                edge_direction = consensus_direction(edge_direction, groups, gamma=gamma)
            apply_local_credit(
                model,
                edge_direction,
                bias_direction,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )
        rows.append(
            checkpoint_row(
                model,
                circuit,
                groups,
                initial_weight,
                initial_bias,
                seed=seed,
                trajectory_rule=trajectory_rule,
                checkpoint=checkpoint,
                gamma=gamma,
            )
        )
        previous = checkpoint
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose weak-consensus training into type-pair mean and within-pair residual modes"
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, float | int | str | bool]] = []
    for trajectory_rule in ("local", "consensus"):
        rows.extend(
            run_trajectory(
                seed=args.seed,
                trajectory_rule=trajectory_rule,
                gamma=args.gamma,
                checkpoints=[0, 25, 50, 100],
                learning_rate=args.learning_rate,
                max_update=args.max_update,
            )
        )
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
