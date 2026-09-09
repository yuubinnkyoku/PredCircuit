from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from diagnose_flyvis_temporal_ce_credit_alignment import (
    cycle_ce_oracles,
    differentiable_frame_states,
    geometry,
)
from run_flyvis_retinotopic_contrastive import DIRECTIONS, output_nodes, render_motion_batch, targets_for
from run_flyvis_strict_residual_controls import (
    bias_blocks,
    edge_blocks,
    permutation_from_blocks,
    strict_direction,
)
from run_flyvis_strict_residual_filters import filtered_direction

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def exact_cycle_loss(
    circuit,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    seed: int,
    epoch: int,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    directions = list(DIRECTIONS)
    random.Random(700_000 * seed + epoch).shuffle(directions)
    outs = output_nodes(circuit)
    for sample_index, direction in enumerate(directions):
        stimulus = render_motion_batch(
            circuit,
            [direction],
            frames=7,
            width=0.5,
            jitter_seed=100_000 * seed + 100 * epoch + sample_index,
        )
        _, classes = targets_for([direction])
        states = differentiable_frame_states(
            circuit,
            stimulus,
            weight,
            bias,
            frame_steps=2,
            step_size=0.015,
        )
        losses.append(F.cross_entropy(states[-1][:, outs], classes))
    return torch.stack(losses).sum()


def transformed_direction(
    *,
    rule: str,
    circuit,
    local_edge: torch.Tensor,
    local_bias: torch.Tensor,
    oracle_edge: torch.Tensor,
    oracle_bias: torch.Tensor,
    edge_permutation: torch.Tensor,
    type_pair_permutation: torch.Tensor,
    offset_permutation: torch.Tensor,
    source_position_permutation: torch.Tensor,
    bias_permutation: torch.Tensor,
    node_type_bias_permutation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rule == "local":
        return local_edge, local_bias
    if rule in {
        "type_pair_edge_permuted",
        "source_position_edge_permuted",
        "bias_permuted",
    }:
        return strict_direction(
            local_edge,
            local_bias,
            oracle_edge,
            oracle_bias,
            rule=rule,
            edge_permutation=edge_permutation,
            type_pair_permutation=type_pair_permutation,
            offset_permutation=offset_permutation,
            source_position_permutation=source_position_permutation,
            bias_permutation=bias_permutation,
            node_type_bias_permutation=node_type_bias_permutation,
        )
    if rule == "type_pair_mean":
        return filtered_direction(
            circuit,
            local_edge,
            local_bias,
            oracle_edge,
            oracle_bias,
            rule=rule,
        )
    raise ValueError(f"unsupported rule: {rule}")


def run_one(
    *,
    seed: int,
    rule: str,
    nudge_steps: int,
    learning_rate: float,
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

    weight = model.weight.detach().clone().requires_grad_(True)
    bias = model.bias.detach().clone().requires_grad_(True)
    loss = exact_cycle_loss(circuit, weight, bias, seed=seed, epoch=0)
    grad_edge, grad_bias = torch.autograd.grad(loss, (weight, bias), create_graph=True)
    oracle_error = torch.linalg.vector_norm(
        torch.cat((grad_edge + oracle_edge, grad_bias + oracle_bias))
    ) / torch.linalg.vector_norm(torch.cat((oracle_edge, oracle_bias))).clamp_min(1e-30)

    directional_derivative = torch.dot(grad_edge, edge_direction) + torch.dot(
        grad_bias, bias_direction
    )
    hvp_edge, hvp_bias = torch.autograd.grad(
        directional_derivative,
        (weight, bias),
    )
    quadratic_form = torch.dot(edge_direction, hvp_edge) + torch.dot(
        bias_direction, hvp_bias
    )
    direction_norm_sq = torch.dot(edge_direction, edge_direction) + torch.dot(
        bias_direction, bias_direction
    )
    normalized_curvature = quadratic_form / direction_norm_sq.clamp_min(1e-30)

    with torch.no_grad():
        first_order_change = learning_rate * directional_derivative
        second_order_change = 0.5 * learning_rate**2 * quadratic_form
        second_order_prediction = first_order_change + second_order_change

    after_weight = (model.weight + learning_rate * edge_direction).detach().requires_grad_(False)
    after_bias = (model.bias + learning_rate * bias_direction).detach().requires_grad_(False)
    with torch.no_grad():
        actual_after_loss = exact_cycle_loss(
            circuit,
            after_weight,
            after_bias,
            seed=seed,
            epoch=0,
        )
    raw_geometry = geometry(
        edge_direction,
        bias_direction,
        oracle_edge,
        oracle_bias,
    )
    local_geometry = geometry(local_edge, local_bias, oracle_edge, oracle_bias)
    return {
        "rule": rule,
        "seed": seed,
        "nudge_steps": nudge_steps,
        "learning_rate": learning_rate,
        "loss_before": float(loss.detach()),
        "loss_after_one_step": float(actual_after_loss),
        "actual_loss_change": float(actual_after_loss - loss.detach()),
        "first_order_loss_change": float(first_order_change),
        "second_order_loss_change": float(second_order_change),
        "second_order_predicted_loss_change": float(second_order_prediction),
        "quadratic_form": float(quadratic_form),
        "normalized_curvature": float(normalized_curvature),
        "direction_norm": float(torch.sqrt(direction_norm_sq)),
        "raw_cosine": raw_geometry["cosine"],
        "raw_projection_coefficient": raw_geometry["projection_coefficient"],
        "raw_norm_ratio": raw_geometry["norm_ratio"],
        "raw_cosine_match_error": abs(raw_geometry["cosine"] - local_geometry["cosine"]),
        "raw_norm_ratio_match_error": abs(
            raw_geometry["norm_ratio"] - local_geometry["norm_ratio"]
        ),
        "oracle_reconstruction_relative_error": float(oracle_error),
        "finite": bool(torch.isfinite(actual_after_loss)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure exact loss curvature along strict matched local-credit directions"
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
    parser.add_argument("--learning-rate", type=float, default=20.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_strict_curvature.csv"),
    )
    args = parser.parse_args()

    rows = [
        run_one(
            seed=seed,
            rule=args.rule,
            nudge_steps=args.nudge_steps,
            learning_rate=args.learning_rate,
        )
        for seed in range(args.seeds)
    ]
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.select_dtypes(include="number").agg(["mean", "median", "std"]).to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
