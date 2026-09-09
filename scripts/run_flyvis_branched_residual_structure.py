from __future__ import annotations

import argparse
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from diagnose_flyvis_temporal_ce_credit_alignment import cycle_ce_oracles
from run_flyvis_credit_residual_scaling import evaluate_metrics
from run_flyvis_retinotopic_temporal_adam import local_adam_step
from run_flyvis_temporal_credit_projection import flatten_credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import RetinotopicFlyVisCircuit, graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def split_credit(
    vector: torch.Tensor,
    edge_shape: torch.Size,
    bias_shape: torch.Size,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_size = math.prod(edge_shape)
    return vector[:edge_size].reshape(edge_shape), vector[edge_size:].reshape(bias_shape)


def orthogonalize_and_match(
    candidate: torch.Tensor,
    oracle: torch.Tensor,
    target_norm: torch.Tensor,
) -> torch.Tensor:
    oracle_norm_sq = torch.dot(oracle, oracle).clamp_min(1e-30)
    orthogonal = candidate - torch.dot(candidate, oracle) / oracle_norm_sq * oracle
    orthogonal_norm = torch.linalg.vector_norm(orthogonal)
    if float(orthogonal_norm) <= 1e-30 or float(target_norm) <= 1e-30:
        return torch.zeros_like(candidate)
    return orthogonal * (target_norm / orthogonal_norm)


def permute_residual_groups(
    residual: torch.Tensor,
    *,
    edge_size: int,
    edge_permutation: torch.Tensor,
    bias_permutation: torch.Tensor,
    permute_edge: bool,
    permute_bias: bool,
) -> torch.Tensor:
    edge = residual[:edge_size]
    bias = residual[edge_size:]
    if permute_edge:
        edge = edge[edge_permutation]
    if permute_bias:
        bias = bias[bias_permutation]
    return torch.cat((edge, bias))


def _permutation_from_blocks(
    size: int,
    blocks: Iterable[list[int]],
    generator: torch.Generator,
) -> torch.Tensor:
    permutation = torch.arange(size)
    for indices in blocks:
        block = torch.tensor(indices, dtype=torch.long)
        permutation[block] = block[torch.randperm(len(indices), generator=generator)]
    return permutation


def permutation_within_edge_type_pairs(
    circuit: RetinotopicFlyVisCircuit,
    generator: torch.Generator,
) -> torch.Tensor:
    blocks: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (source, target) in enumerate(circuit.graph.edge_index.t().tolist()):
        blocks[(circuit.node_types[source], circuit.node_types[target])].append(index)
    return _permutation_from_blocks(circuit.graph.num_edges, blocks.values(), generator)


def permutation_within_edge_offsets(
    circuit: RetinotopicFlyVisCircuit,
    generator: torch.Generator,
) -> torch.Tensor:
    """Shuffle translations while preserving type pair and relative retinotopic offset."""
    blocks: dict[tuple[str, str, int, int], list[int]] = defaultdict(list)
    for index, (source, target) in enumerate(circuit.graph.edge_index.t().tolist()):
        du = int(circuit.node_u[target] - circuit.node_u[source])
        dv = int(circuit.node_v[target] - circuit.node_v[source])
        key = (circuit.node_types[source], circuit.node_types[target], du, dv)
        blocks[key].append(index)
    return _permutation_from_blocks(circuit.graph.num_edges, blocks.values(), generator)


def permutation_within_source_positions(
    circuit: RetinotopicFlyVisCircuit,
    generator: torch.Generator,
) -> torch.Tensor:
    """Shuffle offsets while preserving type pair and the absolute source location."""
    blocks: dict[tuple[str, str, int, int], list[int]] = defaultdict(list)
    for index, (source, target) in enumerate(circuit.graph.edge_index.t().tolist()):
        key = (
            circuit.node_types[source],
            circuit.node_types[target],
            int(circuit.node_u[source]),
            int(circuit.node_v[source]),
        )
        blocks[key].append(index)
    return _permutation_from_blocks(circuit.graph.num_edges, blocks.values(), generator)


def permutation_within_node_types(
    circuit: RetinotopicFlyVisCircuit,
    generator: torch.Generator,
) -> torch.Tensor:
    blocks: dict[str, list[int]] = defaultdict(list)
    for index, cell_type in enumerate(circuit.node_types):
        blocks[cell_type].append(index)
    return _permutation_from_blocks(circuit.graph.num_nodes, blocks.values(), generator)


def synthetic_direction(
    local: torch.Tensor,
    oracle: torch.Tensor,
    *,
    rule: str,
    generator: torch.Generator,
    permutation: torch.Tensor,
    edge_size: int,
    edge_permutation: torch.Tensor,
    bias_permutation: torch.Tensor,
    type_pair_edge_permutation: torch.Tensor,
    offset_edge_permutation: torch.Tensor,
    source_position_edge_permutation: torch.Tensor,
    node_type_bias_permutation: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    oracle_norm_sq = torch.dot(oracle, oracle).clamp_min(1e-30)
    alpha = torch.dot(local, oracle) / oracle_norm_sq
    parallel = alpha * oracle
    residual = local - parallel
    residual_norm = torch.linalg.vector_norm(residual)

    if rule == "local":
        direction = local
    elif rule == "parallel_only":
        direction = parallel
    elif rule == "random_residual":
        random_vector = torch.randn(
            local.shape,
            generator=generator,
            dtype=local.dtype,
            device=local.device,
        )
        matched = orthogonalize_and_match(random_vector, oracle, residual_norm)
        direction = parallel + matched
    elif rule == "permuted_residual":
        matched = orthogonalize_and_match(residual[permutation], oracle, residual_norm)
        direction = parallel + matched
    elif rule in {
        "group_permuted_residual",
        "edge_permuted_residual",
        "bias_permuted_residual",
    }:
        grouped = permute_residual_groups(
            residual,
            edge_size=edge_size,
            edge_permutation=edge_permutation,
            bias_permutation=bias_permutation,
            permute_edge=rule != "bias_permuted_residual",
            permute_bias=rule != "edge_permuted_residual",
        )
        matched = orthogonalize_and_match(grouped, oracle, residual_norm)
        direction = parallel + matched
    elif rule in {
        "type_pair_edge_permuted_residual",
        "offset_edge_permuted_residual",
        "source_position_edge_permuted_residual",
    }:
        if rule == "type_pair_edge_permuted_residual":
            spatial_permutation = type_pair_edge_permutation
        elif rule == "offset_edge_permuted_residual":
            spatial_permutation = offset_edge_permutation
        else:
            spatial_permutation = source_position_edge_permutation
        grouped = torch.cat(
            (
                residual[:edge_size][spatial_permutation],
                residual[edge_size:],
            )
        )
        matched = orthogonalize_and_match(grouped, oracle, residual_norm)
        direction = parallel + matched
    elif rule == "node_type_bias_permuted_residual":
        grouped = torch.cat(
            (
                residual[:edge_size],
                residual[edge_size:][node_type_bias_permutation],
            )
        )
        matched = orthogonalize_and_match(grouped, oracle, residual_norm)
        direction = parallel + matched
    else:
        raise ValueError(f"unsupported rule: {rule}")

    denominator = (
        torch.linalg.vector_norm(direction) * torch.linalg.vector_norm(oracle)
    ).clamp_min(1e-30)
    cosine = float(torch.dot(direction, oracle) / denominator)
    norm_ratio = float(
        torch.linalg.vector_norm(direction) / torch.linalg.vector_norm(oracle).clamp_min(1e-30)
    )
    return direction, cosine, norm_ratio


def run_one(
    *,
    seed: int,
    rule: str,
    epochs: int,
    frames: int,
    width: float,
    frame_steps: int,
    nudge_steps: int,
    step_size: float,
    beta: float,
    learning_rate: float,
    beta1: float,
    beta2: float,
    adam_epsilon: float,
    test_repeats: int,
) -> dict[str, float | int | str | bool]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    edge_m = torch.zeros_like(model.weight)
    edge_v = torch.zeros_like(model.weight)
    bias_m = torch.zeros_like(model.bias)
    bias_v = torch.zeros_like(model.bias)
    generator = torch.Generator().manual_seed(9_000_000 + seed)
    edge_size = model.weight.numel()
    bias_size = model.bias.numel()
    permutation = torch.randperm(edge_size + bias_size, generator=generator)
    edge_permutation = torch.randperm(edge_size, generator=generator)
    bias_permutation = torch.randperm(bias_size, generator=generator)
    type_pair_edge_permutation = permutation_within_edge_type_pairs(circuit, generator)
    offset_edge_permutation = permutation_within_edge_offsets(circuit, generator)
    source_position_edge_permutation = permutation_within_source_positions(circuit, generator)
    node_type_bias_permutation = permutation_within_node_types(circuit, generator)

    before = evaluate_metrics(
        model,
        circuit,
        repeats=test_repeats,
        frames=frames,
        width=width,
        frame_steps=frame_steps,
        step_size=step_size,
        jitter_seed=900_000 + seed,
    )
    cosine_sum = 0.0
    norm_ratio_sum = 0.0
    applied_update_sum = 0.0

    for epoch in range(epochs):
        local_edge, local_bias = cycle_branched_credit(
            model,
            circuit,
            seed=seed,
            epoch=epoch,
            frames=frames,
            width=width,
            beta=beta,
            frame_steps=frame_steps,
            nudge_steps=nudge_steps,
            step_size=step_size,
            terminal_only=False,
        )
        oracle_edge, oracle_bias = cycle_ce_oracles(
            model,
            circuit,
            seed=seed,
            epoch=epoch,
            frames=frames,
            width=width,
            frame_steps=frame_steps,
            step_size=step_size,
        )["final_ce_oracle"]
        local = flatten_credit(local_edge, local_bias)
        oracle = flatten_credit(oracle_edge, oracle_bias)
        direction, direction_cosine, norm_ratio = synthetic_direction(
            local,
            oracle,
            rule=rule,
            generator=generator,
            permutation=permutation,
            edge_size=edge_size,
            edge_permutation=edge_permutation,
            bias_permutation=bias_permutation,
            type_pair_edge_permutation=type_pair_edge_permutation,
            offset_edge_permutation=offset_edge_permutation,
            source_position_edge_permutation=source_position_edge_permutation,
            node_type_bias_permutation=node_type_bias_permutation,
        )
        edge_direction, bias_direction = split_credit(
            direction,
            model.weight.shape,
            model.bias.shape,
        )
        edge_delta = local_adam_step(
            model.weight,
            edge_direction,
            edge_m,
            edge_v,
            step=epoch + 1,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=adam_epsilon,
        )
        local_adam_step(
            model.bias,
            bias_direction,
            bias_m,
            bias_v,
            step=epoch + 1,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=adam_epsilon,
        )
        cosine_sum += direction_cosine
        norm_ratio_sum += norm_ratio
        applied_update_sum += float(edge_delta.abs().mean())

    after = evaluate_metrics(
        model,
        circuit,
        repeats=test_repeats,
        frames=frames,
        width=width,
        frame_steps=frame_steps,
        step_size=step_size,
        jitter_seed=900_000 + seed,
    )
    count = max(epochs, 1)
    return {
        "rule": rule,
        "seed": seed,
        "epochs": epochs,
        "nudge_steps": nudge_steps,
        "beta": beta,
        "learning_rate": learning_rate,
        "adam_epsilon": adam_epsilon,
        "accuracy_before": before["accuracy"],
        "accuracy_after": after["accuracy"],
        "cross_entropy_before": before["cross_entropy"],
        "cross_entropy_after": after["cross_entropy"],
        "cross_entropy_improvement": before["cross_entropy"] - after["cross_entropy"],
        "margin_after": after["margin"],
        "mse_after": after["mse"],
        "mean_direction_cosine": cosine_sum / count,
        "mean_direction_norm_ratio": norm_ratio_sum / count,
        "mean_abs_applied_update": applied_update_sum / count,
        "finite": bool(torch.isfinite(model.weight).all())
        and math.isfinite(after["cross_entropy"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare actual branched local residual with norm- and projection-matched synthetic "
            "residuals under the same local Adam optimizer"
        )
    )
    parser.add_argument(
        "--rule",
        choices=(
            "local",
            "parallel_only",
            "random_residual",
            "permuted_residual",
            "group_permuted_residual",
            "edge_permuted_residual",
            "bias_permuted_residual",
            "type_pair_edge_permuted_residual",
            "offset_edge_permuted_residual",
            "source_position_edge_permuted_residual",
            "node_type_bias_permuted_residual",
        ),
        required=True,
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--nudge-steps", type=int, default=16)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_branched_residual_structure.csv"),
    )
    args = parser.parse_args()

    rows = [
        run_one(
            seed=seed,
            rule=args.rule,
            epochs=args.epochs,
            frames=args.frames,
            width=args.bar_width,
            frame_steps=args.frame_steps,
            nudge_steps=args.nudge_steps,
            step_size=args.step_size,
            beta=args.beta,
            learning_rate=args.learning_rate,
            beta1=args.beta1,
            beta2=args.beta2,
            adam_epsilon=args.adam_epsilon,
            test_repeats=args.test_repeats,
        )
        for seed in range(args.seeds)
    ]
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    metrics = [
        "accuracy_after",
        "cross_entropy_after",
        "cross_entropy_improvement",
        "margin_after",
        "mean_direction_cosine",
        "mean_direction_norm_ratio",
        "mean_abs_applied_update",
    ]
    print(f"Residual structure control: rule={args.rule}, nudge_steps={args.nudge_steps}")
    print(frame[metrics].agg(["mean", "median", "std"]).to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
