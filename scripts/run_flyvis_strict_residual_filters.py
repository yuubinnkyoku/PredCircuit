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
from run_flyvis_strict_residual_controls import edge_blocks, match_projection_and_norm

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import RetinotopicFlyVisCircuit, graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def target_position_blocks(circuit: RetinotopicFlyVisCircuit) -> list[list[int]]:
    blocks: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for index, (source, target) in enumerate(circuit.graph.edge_index.t().tolist()):
        key = (
            circuit.node_types[source],
            circuit.node_types[target],
            int(circuit.node_u[target]),
            int(circuit.node_v[target]),
        )
        blocks[key].append(index)
    return list(blocks.values())


def block_mean_candidate(residual: torch.Tensor, blocks: list[list[int]]) -> torch.Tensor:
    candidate = torch.empty_like(residual)
    for indices in blocks:
        block = torch.tensor(indices, dtype=torch.long, device=residual.device)
        candidate[block] = residual[block].mean()
    return candidate


def filtered_direction(
    circuit: RetinotopicFlyVisCircuit,
    local_edge: torch.Tensor,
    local_bias: torch.Tensor,
    oracle_edge: torch.Tensor,
    oracle_bias: torch.Tensor,
    *,
    rule: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rule == "local":
        return local_edge, local_bias

    local = torch.cat((local_edge, local_bias))
    oracle = torch.cat((oracle_edge, oracle_bias))
    alpha = torch.dot(local, oracle) / torch.dot(oracle, oracle).clamp_min(1e-30)
    residual_edge = local_edge - alpha * oracle_edge
    residual_bias = local_bias - alpha * oracle_bias

    if rule == "type_pair_mean":
        blocks = edge_blocks(circuit, mode="type_pair")
    elif rule == "offset_mean":
        blocks = edge_blocks(circuit, mode="offset")
    elif rule == "source_position_mean":
        blocks = edge_blocks(circuit, mode="source_position")
    elif rule == "target_position_mean":
        blocks = target_position_blocks(circuit)
    else:
        raise ValueError(f"unsupported rule: {rule}")

    candidate = block_mean_candidate(residual_edge, blocks)
    filtered_residual_edge = match_projection_and_norm(candidate, oracle_edge, residual_edge)
    return alpha * oracle_edge + filtered_residual_edge, alpha * oracle_bias + residual_bias


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
        edge_direction, bias_direction = filtered_direction(
            circuit,
            local_edge,
            local_bias,
            oracle_edge,
            oracle_bias,
            rule=rule,
        )
        local_geometry = geometry(local_edge, local_bias, oracle_edge, oracle_bias)
        filtered_geometry = geometry(edge_direction, bias_direction, oracle_edge, oracle_bias)
        cosine_difference_sum += abs(filtered_geometry["cosine"] - local_geometry["cosine"])
        norm_ratio_difference_sum += abs(
            filtered_geometry["norm_ratio"] - local_geometry["norm_ratio"]
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
        description="Strict spatial filtering of branched local-credit edge residuals"
    )
    parser.add_argument(
        "--rule",
        choices=(
            "local",
            "type_pair_mean",
            "offset_mean",
            "source_position_mean",
            "target_position_mean",
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
        default=Path("results/generated/flyvis_strict_residual_filter.csv"),
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
    numeric = frame.select_dtypes(include="number")
    print(numeric.agg(["mean", "median", "std"]).to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
