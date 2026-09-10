from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from diagnose_flyvis_strict_curvature import exact_cycle_loss
from diagnose_flyvis_temporal_ce_credit_alignment import cycle_ce_oracles
from diagnose_flyvis_type_pair_credit_curvature import type_pair_indices
from diagnose_flyvis_type_pair_intervention_loss import (
    clipped_update,
    filtered_edge_residual,
    train_to_checkpoint,
)

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

CANDIDATES = (
    ("R7", "Mi9"),
    ("R8", "Mi9"),
    ("L4", "Mi9"),
    ("T4a", "Mi9"),
    ("T4b", "Mi9"),
    ("T4c", "Mi9"),
    ("T4a", "T4c"),
)


def run_seed(
    *,
    seed: int,
    checkpoint: int,
    probe_learning_rate: float,
    max_update: float,
) -> list[dict[str, float | int | str]]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    train_to_checkpoint(
        model,
        circuit,
        seed=seed,
        checkpoint=checkpoint,
        learning_rate=160.0,
        max_update=max_update,
    )

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

    oracle_norm_sq = torch.dot(oracle_edge, oracle_edge) + torch.dot(oracle_bias, oracle_bias)
    alpha = (
        torch.dot(local_edge, oracle_edge) + torch.dot(local_bias, oracle_bias)
    ) / oracle_norm_sq.clamp_min(1e-30)
    parallel_edge = alpha * oracle_edge
    parallel_bias = alpha * oracle_bias
    residual_edge = local_edge - parallel_edge
    residual_bias = local_bias - parallel_bias

    weight = model.weight.detach().clone().requires_grad_(True)
    bias = model.bias.detach().clone().requires_grad_(True)
    exact_before = exact_cycle_loss(circuit, weight, bias, seed=seed, epoch=0)
    grad_edge, grad_bias = torch.autograd.grad(
        exact_before,
        (weight, bias),
        create_graph=True,
    )
    residual_derivative = torch.dot(grad_edge, residual_edge) + torch.dot(grad_bias, residual_bias)
    hvp_edge, _ = torch.autograd.grad(residual_derivative, (weight, bias))

    local_edge_update = clipped_update(
        local_edge,
        learning_rate=probe_learning_rate,
        max_update=max_update,
    )
    local_bias_update = clipped_update(
        local_bias,
        learning_rate=probe_learning_rate,
        max_update=max_update,
    )
    with torch.no_grad():
        local_post_loss = exact_cycle_loss(
            circuit,
            model.weight + local_edge_update,
            model.bias + local_bias_update,
            seed=seed,
            epoch=0,
        )

    groups = type_pair_indices(circuit)
    rows: list[dict[str, float | int | str]] = []
    for pair in CANDIDATES:
        indices = groups[pair]
        filtered_residual = filtered_edge_residual(
            residual_edge,
            oracle_edge,
            groups,
            {pair},
        )
        direction_edge = parallel_edge + filtered_residual
        direction_bias = parallel_bias + residual_bias
        edge_update = clipped_update(
            direction_edge,
            learning_rate=probe_learning_rate,
            max_update=max_update,
        )
        bias_update = clipped_update(
            direction_bias,
            learning_rate=probe_learning_rate,
            max_update=max_update,
        )
        with torch.no_grad():
            post_loss = exact_cycle_loss(
                circuit,
                model.weight + edge_update,
                model.bias + bias_update,
                seed=seed,
                epoch=0,
            )
        rows.append(
            {
                "seed": seed,
                "checkpoint": checkpoint,
                "source_type": pair[0],
                "target_type": pair[1],
                "edge_count": int(indices.numel()),
                "local_post_loss": float(local_post_loss),
                "filtered_post_loss": float(post_loss),
                "filtered_loss_gain_vs_local": float(local_post_loss - post_loss),
                "residual_curvature_contribution": float(
                    torch.dot(residual_edge[indices], hvp_edge[indices])
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replicate intervention signs for high-signal FlyVis type pairs"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=int, default=100)
    parser.add_argument("--probe-learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.DataFrame(
        run_seed(
            seed=args.seed,
            checkpoint=args.checkpoint,
            probe_learning_rate=args.probe_learning_rate,
            max_update=args.max_update,
        )
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
