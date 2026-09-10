from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from diagnose_flyvis_strict_curvature import exact_cycle_loss
from diagnose_flyvis_temporal_ce_credit_alignment import cycle_ce_oracles, geometry
from diagnose_flyvis_type_pair_intervention_loss import clipped_update
from run_flyvis_credit_residual_scaling import evaluate_metrics
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_consensus import consensus_direction, type_pair_indices

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def direction_curvature(
    *,
    grad_edge: torch.Tensor,
    grad_bias: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    direction_edge: torch.Tensor,
    direction_bias: torch.Tensor,
    oracle_edge: torch.Tensor,
    oracle_bias: torch.Tensor,
) -> dict[str, float]:
    oracle_norm_sq = torch.dot(oracle_edge, oracle_edge) + torch.dot(
        oracle_bias, oracle_bias
    )
    alpha = (
        torch.dot(direction_edge, oracle_edge)
        + torch.dot(direction_bias, oracle_bias)
    ) / oracle_norm_sq.clamp_min(1e-30)
    parallel_edge = alpha * oracle_edge
    parallel_bias = alpha * oracle_bias
    residual_edge = direction_edge - parallel_edge
    residual_bias = direction_bias - parallel_bias
    residual_derivative = torch.dot(grad_edge, residual_edge) + torch.dot(
        grad_bias, residual_bias
    )
    hvp_edge, hvp_bias = torch.autograd.grad(
        residual_derivative,
        (weight, bias),
        retain_graph=True,
    )
    residual_curvature = torch.dot(residual_edge, hvp_edge) + torch.dot(
        residual_bias, hvp_bias
    )
    cross_curvature = 2.0 * (
        torch.dot(parallel_edge, hvp_edge) + torch.dot(parallel_bias, hvp_bias)
    )
    return {
        "alpha": float(alpha),
        "residual_norm": float(
            torch.linalg.vector_norm(torch.cat((residual_edge, residual_bias)))
        ),
        "residual_curvature": float(residual_curvature),
        "cross_curvature": float(cross_curvature),
    }


def checkpoint_rows(
    model: PredictiveCodingGraph,
    circuit,
    *,
    seed: int,
    checkpoint: int,
    trajectory_rule: str,
    gamma: float,
    learning_rate: float,
    max_update: float,
) -> list[dict[str, float | int | str | bool]]:
    local_edge, local_bias = cycle_branched_credit(
        model,
        circuit,
        seed=seed,
        epoch=checkpoint,
        frames=7,
        width=0.5,
        beta=0.03,
        frame_steps=2,
        nudge_steps=2,
        step_size=0.015,
        terminal_only=False,
    )
    groups = type_pair_indices(circuit)
    consensus_edge = consensus_direction(local_edge, groups, gamma=gamma)
    oracle_edge, oracle_bias = cycle_ce_oracles(
        model,
        circuit,
        seed=seed,
        epoch=checkpoint,
        frames=7,
        width=0.5,
        frame_steps=2,
        step_size=0.015,
    )["final_ce_oracle"]

    weight = model.weight.detach().clone().requires_grad_(True)
    bias = model.bias.detach().clone().requires_grad_(True)
    exact_before = exact_cycle_loss(
        circuit,
        weight,
        bias,
        seed=seed,
        epoch=checkpoint,
    )
    grad_edge, grad_bias = torch.autograd.grad(
        exact_before,
        (weight, bias),
        create_graph=True,
    )
    task = evaluate_metrics(
        model,
        circuit,
        repeats=8,
        frames=7,
        width=0.5,
        frame_steps=2,
        step_size=0.015,
        jitter_seed=900_000 + seed,
    )

    rows: list[dict[str, float | int | str | bool]] = []
    post_losses: dict[str, float] = {}
    for candidate_rule, edge_direction in (
        ("local", local_edge),
        ("consensus", consensus_edge),
    ):
        geom = geometry(
            edge_direction,
            local_bias,
            oracle_edge,
            oracle_bias,
        )
        curvature = direction_curvature(
            grad_edge=grad_edge,
            grad_bias=grad_bias,
            weight=weight,
            bias=bias,
            direction_edge=edge_direction,
            direction_bias=local_bias,
            oracle_edge=oracle_edge,
            oracle_bias=oracle_bias,
        )
        edge_update = clipped_update(
            edge_direction,
            learning_rate=learning_rate,
            max_update=max_update,
        )
        bias_update = clipped_update(
            local_bias,
            learning_rate=learning_rate,
            max_update=max_update,
        )
        update_geom = geometry(
            edge_update,
            bias_update,
            oracle_edge,
            oracle_bias,
        )
        with torch.no_grad():
            post_loss = exact_cycle_loss(
                circuit,
                model.weight + edge_update,
                model.bias + bias_update,
                seed=seed,
                epoch=checkpoint,
            )
        post_losses[candidate_rule] = float(post_loss)
        rows.append(
            {
                "seed": seed,
                "trajectory_rule": trajectory_rule,
                "checkpoint": checkpoint,
                "candidate_rule": candidate_rule,
                "gamma": gamma,
                "cross_entropy": task["cross_entropy"],
                "accuracy": task["accuracy"],
                "margin": task["margin"],
                "exact_cycle_loss": float(exact_before.detach()),
                "post_step_exact_loss": float(post_loss),
                "cosine": geom["cosine"],
                "norm_ratio": geom["norm_ratio"],
                "projection_coefficient": geom["projection_coefficient"],
                "sign_agreement": geom["sign_agreement"],
                "alpha": curvature["alpha"],
                "residual_norm": curvature["residual_norm"],
                "residual_curvature": curvature["residual_curvature"],
                "cross_curvature": curvature["cross_curvature"],
                "applied_update_cosine": update_geom["cosine"],
                "applied_update_norm_ratio": update_geom["norm_ratio"],
                "edge_clip_fraction": float(
                    (learning_rate * edge_direction)
                    .abs()
                    .gt(max_update)
                    .float()
                    .mean()
                ),
                "weight_norm": float(torch.linalg.vector_norm(model.weight)),
                "bias_norm": float(torch.linalg.vector_norm(model.bias)),
                "finite": bool(torch.isfinite(model.weight).all())
                and bool(torch.isfinite(model.bias).all()),
            }
        )

    consensus_gain = post_losses["local"] - post_losses["consensus"]
    for row in rows:
        row["consensus_one_step_gain_vs_local"] = consensus_gain
        row["consensus_edge_direction_change"] = float(
            torch.linalg.vector_norm(consensus_edge - local_edge)
            / torch.linalg.vector_norm(local_edge).clamp_min(1e-30)
        )
    return rows


def run_trajectory(
    *,
    seed: int,
    trajectory_rule: str,
    checkpoints: list[int],
    gamma: float,
    learning_rate: float,
    max_update: float,
) -> list[dict[str, float | int | str | bool]]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    groups = type_pair_indices(circuit)
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
                edge_direction = consensus_direction(
                    edge_direction,
                    groups,
                    gamma=gamma,
                )
            apply_local_credit(
                model,
                edge_direction,
                bias_direction,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )
        rows.extend(
            checkpoint_rows(
                model,
                circuit,
                seed=seed,
                checkpoint=checkpoint,
                trajectory_rule=trajectory_rule,
                gamma=gamma,
                learning_rate=learning_rate,
                max_update=max_update,
            )
        )
        previous = checkpoint
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace exact-gradient geometry and curvature under type-pair consensus"
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
                checkpoints=[0, 25, 50, 100],
                gamma=args.gamma,
                learning_rate=args.learning_rate,
                max_update=args.max_update,
            )
        )
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(
        frame[
            [
                "trajectory_rule",
                "checkpoint",
                "candidate_rule",
                "cross_entropy",
                "cosine",
                "residual_curvature",
                "cross_curvature",
                "post_step_exact_loss",
                "consensus_one_step_gain_vs_local",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
