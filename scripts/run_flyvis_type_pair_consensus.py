from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from run_flyvis_credit_residual_scaling import evaluate_metrics
from run_flyvis_temporal_coherence_gate import apply_local_credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import RetinotopicFlyVisCircuit, graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def type_pair_indices(circuit: RetinotopicFlyVisCircuit) -> list[torch.Tensor]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (source, target) in enumerate(circuit.graph.edge_index.t().tolist()):
        groups[(circuit.node_types[source], circuit.node_types[target])].append(index)
    return [torch.tensor(indices, dtype=torch.long) for indices in groups.values()]


def consensus_direction(
    direction: torch.Tensor,
    groups: list[torch.Tensor],
    *,
    gamma: float,
) -> torch.Tensor:
    if gamma >= 1.0:
        return direction
    result = direction.clone()
    for indices in groups:
        raw = direction[indices]
        mean = raw.mean()
        candidate = mean + gamma * (raw - mean)
        raw_norm = torch.linalg.vector_norm(raw)
        candidate_norm = torch.linalg.vector_norm(candidate)
        if float(raw_norm) <= 1e-30:
            result[indices] = 0.0
        elif float(candidate_norm) <= 1e-30:
            result[indices] = raw
        else:
            result[indices] = candidate * (raw_norm / candidate_norm)
    return result


def run_seed(
    *,
    seed: int,
    gamma: float,
    start_epoch: int,
    epochs: int,
    learning_rate: float,
    max_update: float,
    test_repeats: int,
) -> dict[str, float | int | bool]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    groups = type_pair_indices(circuit)
    eval_kwargs = {
        "repeats": test_repeats,
        "frames": 7,
        "width": 0.5,
        "frame_steps": 2,
        "step_size": 0.015,
        "jitter_seed": 900_000 + seed,
    }
    before = evaluate_metrics(model, circuit, **eval_kwargs)
    relative_change_sum = 0.0
    norm_error_sum = 0.0

    for epoch in range(epochs):
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
        original_norm = torch.linalg.vector_norm(edge_direction)
        if epoch >= start_epoch:
            filtered_edge = consensus_direction(edge_direction, groups, gamma=gamma)
        else:
            filtered_edge = edge_direction
        filtered_norm = torch.linalg.vector_norm(filtered_edge)
        relative_change_sum += float(
            torch.linalg.vector_norm(filtered_edge - edge_direction)
            / original_norm.clamp_min(1e-30)
        )
        norm_error_sum += float(
            torch.abs(filtered_norm - original_norm) / original_norm.clamp_min(1e-30)
        )
        apply_local_credit(
            model,
            filtered_edge,
            bias_direction,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )

    after = evaluate_metrics(model, circuit, **eval_kwargs)
    count = max(epochs, 1)
    return {
        "seed": seed,
        "gamma": gamma,
        "start_epoch": start_epoch,
        "accuracy_before": before["accuracy"],
        "accuracy_after": after["accuracy"],
        "cross_entropy_before": before["cross_entropy"],
        "cross_entropy_after": after["cross_entropy"],
        "cross_entropy_improvement": before["cross_entropy"] - after["cross_entropy"],
        "margin_after": after["margin"],
        "mean_relative_edge_direction_change": relative_change_sum / count,
        "mean_relative_edge_norm_error": norm_error_sum / count,
        "finite": bool(torch.isfinite(model.weight).all())
        and bool(torch.isfinite(model.bias).all())
        and math.isfinite(after["cross_entropy"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train with oracle-free type-pair credit consensus")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--start-epoch", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    frame = pd.DataFrame(
        [
            run_seed(
                seed=args.seed,
                gamma=args.gamma,
                start_epoch=args.start_epoch,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                max_update=args.max_update,
                test_repeats=args.test_repeats,
            )
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
