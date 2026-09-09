from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from diagnose_flyvis_temporal_ce_credit_alignment import cycle_ce_oracles, geometry
from run_flyvis_credit_residual_scaling import evaluate_metrics
from run_flyvis_retinotopic_temporal_adam import local_adam_step

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import RetinotopicFlyVisCircuit, graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def permutation_from_blocks(
    size: int,
    blocks: list[list[int]],
    generator: torch.Generator,
) -> torch.Tensor:
    permutation = torch.arange(size)
    for indices in blocks:
        block = torch.tensor(indices, dtype=torch.long)
        permutation[block] = block[torch.randperm(len(indices), generator=generator)]
    return permutation


def edge_blocks(
    circuit: RetinotopicFlyVisCircuit,
    *,
    mode: str,
) -> list[list[int]]:
    blocks: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for index, (source, target) in enumerate(circuit.graph.edge_index.t().tolist()):
        source_type = circuit.node_types[source]
        target_type = circuit.node_types[target]
        if mode == "type_pair":
            key: tuple[object, ...] = (source_type, target_type)
        elif mode == "offset":
            key = (
                source_type,
                target_type,
                int(circuit.node_u[target] - circuit.node_u[source]),
                int(circuit.node_v[target] - circuit.node_v[source]),
            )
        elif mode == "source_position":
            key = (
                source_type,
                target_type,
                int(circuit.node_u[source]),
                int(circuit.node_v[source]),
            )
        else:
            raise ValueError(f"unsupported edge block mode: {mode}")
        blocks[key].append(index)
    return list(blocks.values())


def bias_blocks(circuit: RetinotopicFlyVisCircuit) -> list[list[int]]:
    blocks: dict[str, list[int]] = defaultdict(list)
    for index, cell_type in enumerate(circuit.node_types):
        blocks[cell_type].append(index)
    return list(blocks.values())


def match_projection_and_norm(
    candidate: torch.Tensor,
    oracle: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Match target projection onto oracle and target norm without touching other groups."""
    oracle_norm_sq = torch.dot(oracle, oracle)
    target_norm_sq = torch.dot(target, target)
    if float(oracle_norm_sq) <= 1e-30:
        candidate_norm = torch.linalg.vector_norm(candidate)
        if float(candidate_norm) <= 1e-30 or float(target_norm_sq) <= 1e-30:
            return torch.zeros_like(candidate)
        return candidate * (torch.sqrt(target_norm_sq) / candidate_norm)

    target_dot = torch.dot(target, oracle)
    parallel = target_dot / oracle_norm_sq * oracle
    parallel_norm_sq = torch.dot(parallel, parallel)
    orthogonal_target_norm_sq = (target_norm_sq - parallel_norm_sq).clamp_min(0.0)

    candidate_orthogonal = candidate - torch.dot(candidate, oracle) / oracle_norm_sq * oracle
    candidate_orthogonal_norm = torch.linalg.vector_norm(candidate_orthogonal)
    desired_orthogonal_norm = torch.sqrt(orthogonal_target_norm_sq)
    if float(desired_orthogonal_norm) <= 1e-30:
        return parallel
    if float(candidate_orthogonal_norm) <= 1e-30:
        target_orthogonal = target - target_dot / oracle_norm_sq * oracle
        target_orthogonal_norm = torch.linalg.vector_norm(target_orthogonal)
        if float(target_orthogonal_norm) <= 1e-30:
            return parallel
        candidate_orthogonal = target_orthogonal
        candidate_orthogonal_norm = target_orthogonal_norm
    return parallel + candidate_orthogonal * (desired_orthogonal_norm / candidate_orthogonal_norm)


def strict_direction(
    local_edge: torch.Tensor,
    local_bias: torch.Tensor,
    oracle_edge: torch.Tensor,
    oracle_bias: torch.Tensor,
    *,
    rule: str,
    edge_permutation: torch.Tensor,
    type_pair_permutation: torch.Tensor,
    offset_permutation: torch.Tensor,
    source_position_permutation: torch.Tensor,
    bias_permutation: torch.Tensor,
    node_type_bias_permutation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    local = torch.cat((local_edge, local_bias))
    oracle = torch.cat((oracle_edge, oracle_bias))
    alpha = torch.dot(local, oracle) / torch.dot(oracle, oracle).clamp_min(1e-30)
    residual_edge = local_edge - alpha * oracle_edge
    residual_bias = local_bias - alpha * oracle_bias

    edge_candidate = residual_edge
    bias_candidate = residual_bias
    if rule == "local":
        return local_edge, local_bias
    if rule == "edge_permuted":
        edge_candidate = residual_edge[edge_permutation]
    elif rule == "type_pair_edge_permuted":
        edge_candidate = residual_edge[type_pair_permutation]
    elif rule == "offset_edge_permuted":
        edge_candidate = residual_edge[offset_permutation]
    elif rule == "source_position_edge_permuted":
        edge_candidate = residual_edge[source_position_permutation]
    elif rule == "bias_permuted":
        bias_candidate = residual_bias[bias_permutation]
    elif rule == "node_type_bias_permuted":
        bias_candidate = residual_bias[node_type_bias_permutation]
    else:
        raise ValueError(f"unsupported rule: {rule}")

    if rule in {
        "edge_permuted",
        "type_pair_edge_permuted",
        "offset_edge_permuted",
        "source_position_edge_permuted",
    }:
        residual_edge = match_projection_and_norm(edge_candidate, oracle_edge, residual_edge)
    else:
        residual_bias = match_projection_and_norm(bias_candidate, oracle_bias, residual_bias)

    return alpha * oracle_edge + residual_edge, alpha * oracle_bias + residual_bias


def run_one(
    *,
    seed: int,
    rule: str,
    epochs: int,
    nudge_steps: int,
    learning_rate: float,
    test_repeats: int,
) -> dict[str, float | int | str | bool]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    edge_m = torch.zeros_like(model.weight)
    edge_v = torch.zeros_like(model.weight)
    bias_m = torch.zeros_like(model.bias)
    bias_v = torch.zeros_like(model.bias)
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

    eval_kwargs = {
        "repeats": test_repeats,
        "frames": 7,
        "width": 0.5,
        "frame_steps": 2,
        "step_size": 0.015,
        "jitter_seed": 900_000 + seed,
    }
    before = evaluate_metrics(model, circuit, **eval_kwargs)
    cosine_difference_sum = 0.0
    norm_ratio_difference_sum = 0.0

    for epoch in range(epochs):
        local_edge, local_bias = cycle_branched_credit(
            model,
            circuit,
            seed=seed,
            epoch=epoch,
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
            epoch=epoch,
            frames=7,
            width=0.5,
            frame_steps=2,
            step_size=0.015,
        )["final_ce_oracle"]
        edge_direction, bias_direction = strict_direction(
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
        local_geometry = geometry(local_edge, local_bias, oracle_edge, oracle_bias)
        strict_geometry = geometry(edge_direction, bias_direction, oracle_edge, oracle_bias)
        cosine_difference_sum += abs(strict_geometry["cosine"] - local_geometry["cosine"])
        norm_ratio_difference_sum += abs(
            strict_geometry["norm_ratio"] - local_geometry["norm_ratio"]
        )

        local_adam_step(
            model.weight,
            edge_direction,
            edge_m,
            edge_v,
            step=epoch + 1,
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
        )
        local_adam_step(
            model.bias,
            bias_direction,
            bias_m,
            bias_v,
            step=epoch + 1,
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
        )

    after = evaluate_metrics(model, circuit, **eval_kwargs)
    count = max(epochs, 1)
    return {
        "rule": rule,
        "seed": seed,
        "accuracy_after": after["accuracy"],
        "cross_entropy_after": after["cross_entropy"],
        "cross_entropy_improvement": before["cross_entropy"] - after["cross_entropy"],
        "margin_after": after["margin"],
        "mean_abs_cosine_geometry_error": cosine_difference_sum / count,
        "mean_abs_norm_ratio_geometry_error": norm_ratio_difference_sum / count,
        "finite": bool(torch.isfinite(model.weight).all())
        and math.isfinite(after["cross_entropy"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict residual controls that preserve group projection/norm independently"
    )
    parser.add_argument(
        "--rule",
        choices=(
            "local",
            "edge_permuted",
            "type_pair_edge_permuted",
            "offset_edge_permuted",
            "source_position_edge_permuted",
            "bias_permuted",
            "node_type_bias_permuted",
        ),
        required=True,
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--nudge-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_strict_residual_control.csv"),
    )
    args = parser.parse_args()

    rows = [
        run_one(
            seed=seed,
            rule=args.rule,
            epochs=args.epochs,
            nudge_steps=args.nudge_steps,
            learning_rate=args.learning_rate,
            test_repeats=args.test_repeats,
        )
        for seed in range(args.seeds)
    ]
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.agg(["mean", "median", "std"], numeric_only=True).to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
