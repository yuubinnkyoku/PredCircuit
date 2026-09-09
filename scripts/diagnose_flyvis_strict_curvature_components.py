from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from diagnose_flyvis_strict_curvature import exact_cycle_loss, transformed_direction
from diagnose_flyvis_temporal_ce_credit_alignment import cycle_ce_oracles, geometry
from run_flyvis_strict_residual_controls import (
    bias_blocks,
    edge_blocks,
    permutation_from_blocks,
)

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def run_one(
    *, seed: int, rule: str, nudge_steps: int
) -> dict[str, float | int | str | bool]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    generator = torch.Generator().manual_seed(12_000_000 + seed)
    edge_permutation = torch.randperm(model.weight.numel(), generator=generator)
    type_pair_permutation = permutation_from_blocks(
        model.weight.numel(), edge_blocks(circuit, mode="type_pair"), generator
    )
    offset_permutation = permutation_from_blocks(
        model.weight.numel(), edge_blocks(circuit, mode="offset"), generator
    )
    source_position_permutation = permutation_from_blocks(
        model.weight.numel(), edge_blocks(circuit, mode="source_position"), generator
    )
    bias_permutation = torch.randperm(model.bias.numel(), generator=generator)
    node_type_bias_permutation = permutation_from_blocks(
        model.bias.numel(), bias_blocks(circuit), generator
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
        nudge_steps=nudge_steps,
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
    edge_direction, bias_direction = transformed_direction(
        rule=rule,
        circuit=circuit,
        local_edge=local_edge,
        local_bias=local_bias,
        oracle_edge=oracle_edge,
        oracle_bias=oracle_bias,
        edge_permutation=edge_permutation,
        type_pair_permutation=type_pair_permutation,
        offset_permutation=offset_permutation,
        source_position_permutation=source_position_permutation,
        bias_permutation=bias_permutation,
        node_type_bias_permutation=node_type_bias_permutation,
    )

    oracle_norm_sq = torch.dot(oracle_edge, oracle_edge) + torch.dot(oracle_bias, oracle_bias)
    alpha = (
        torch.dot(edge_direction, oracle_edge) + torch.dot(bias_direction, oracle_bias)
    ) / oracle_norm_sq.clamp_min(1e-30)
    parallel_edge = alpha * oracle_edge
    parallel_bias = alpha * oracle_bias
    residual_edge = edge_direction - parallel_edge
    residual_bias = bias_direction - parallel_bias

    weight = model.weight.detach().clone().requires_grad_(True)
    bias = model.bias.detach().clone().requires_grad_(True)
    loss = exact_cycle_loss(circuit, weight, bias, seed=seed, epoch=0)
    grad_edge, grad_bias = torch.autograd.grad(loss, (weight, bias), create_graph=True)

    directional_derivative = torch.dot(grad_edge, edge_direction) + torch.dot(
        grad_bias, bias_direction
    )
    hvp_edge, hvp_bias = torch.autograd.grad(
        directional_derivative,
        (weight, bias),
        retain_graph=True,
    )
    total_quadratic = torch.dot(edge_direction, hvp_edge) + torch.dot(
        bias_direction, hvp_bias
    )

    residual_derivative = torch.dot(grad_edge, residual_edge) + torch.dot(
        grad_bias, residual_bias
    )
    residual_hvp_edge, residual_hvp_bias = torch.autograd.grad(
        residual_derivative,
        (weight, bias),
    )
    residual_quadratic = torch.dot(residual_edge, residual_hvp_edge) + torch.dot(
        residual_bias, residual_hvp_bias
    )
    cross_quadratic = 2.0 * (
        torch.dot(parallel_edge, residual_hvp_edge)
        + torch.dot(parallel_bias, residual_hvp_bias)
    )
    parallel_quadratic = total_quadratic - residual_quadratic - cross_quadratic

    direction_norm_sq = torch.dot(edge_direction, edge_direction) + torch.dot(
        bias_direction, bias_direction
    )
    residual_norm_sq = torch.dot(residual_edge, residual_edge) + torch.dot(
        residual_bias, residual_bias
    )
    parallel_norm_sq = torch.dot(parallel_edge, parallel_edge) + torch.dot(
        parallel_bias, parallel_bias
    )
    scale = direction_norm_sq.clamp_min(1e-30)
    geometry_values = geometry(edge_direction, bias_direction, oracle_edge, oracle_bias)

    reconstruction_error = abs(
        float(total_quadratic - (parallel_quadratic + cross_quadratic + residual_quadratic))
    )
    return {
        "rule": rule,
        "seed": seed,
        "nudge_steps": nudge_steps,
        "alpha": float(alpha),
        "raw_cosine": geometry_values["cosine"],
        "raw_norm_ratio": geometry_values["norm_ratio"],
        "direction_norm": float(torch.sqrt(direction_norm_sq)),
        "parallel_norm": float(torch.sqrt(parallel_norm_sq)),
        "residual_norm": float(torch.sqrt(residual_norm_sq)),
        "total_quadratic": float(total_quadratic),
        "parallel_quadratic": float(parallel_quadratic),
        "cross_quadratic": float(cross_quadratic),
        "residual_quadratic": float(residual_quadratic),
        "total_normalized": float(total_quadratic / scale),
        "parallel_normalized": float(parallel_quadratic / scale),
        "cross_normalized": float(cross_quadratic / scale),
        "residual_normalized": float(residual_quadratic / scale),
        "quadratic_reconstruction_abs_error": reconstruction_error,
        "finite": bool(torch.isfinite(total_quadratic))
        and bool(torch.isfinite(residual_quadratic)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decompose strict matched credit curvature into parallel, cross, and residual terms"
    )
    parser.add_argument(
        "--rule",
        choices=(
            "local",
            "type_pair_edge_permuted",
            "source_position_edge_permuted",
            "bias_permuted",
            "type_pair_mean",
        ),
        required=True,
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--nudge-steps", type=int, default=16)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_strict_curvature_components.csv"),
    )
    args = parser.parse_args()

    rows = [
        run_one(seed=seed, rule=args.rule, nudge_steps=args.nudge_steps)
        for seed in range(args.seeds)
    ]
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.select_dtypes(include="number").agg(["mean", "median", "std"]).to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
