from __future__ import annotations

import argparse
import hashlib
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from run_flyvis_credit_residual_scaling import evaluate_metrics
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_consensus import type_pair_indices
from run_flyvis_type_pair_linear_mix import linear_mix_direction

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import RetinotopicFlyVisCircuit, graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def grouped_edge_indices(circuit: RetinotopicFlyVisCircuit, grouping: str) -> list[torch.Tensor]:
    if grouping == "type_pair":
        return type_pair_indices(circuit)

    edges = circuit.graph.edge_index.t().tolist()
    if grouping in {"source_type", "target_type"}:
        groups: dict[str, list[int]] = defaultdict(list)
        endpoint = 0 if grouping == "source_type" else 1
        for index, edge in enumerate(edges):
            node = edge[endpoint]
            groups[circuit.node_types[node]].append(index)
        return [torch.tensor(indices, dtype=torch.long) for indices in groups.values()]

    if grouping in {"random_a", "random_b"}:
        biological = type_pair_indices(circuit)
        sizes = [int(indices.numel()) for indices in biological]
        salt = grouping.encode("utf-8")
        digest = hashlib.sha256(salt).digest()
        seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(len(edges), generator=generator)
        groups = []
        offset = 0
        for size in sizes:
            groups.append(permutation[offset : offset + size])
            offset += size
        if offset != len(edges):
            raise RuntimeError("random grouping did not cover all edges")
        return groups

    if grouping == "global":
        return [torch.arange(len(edges), dtype=torch.long)]
    raise ValueError(grouping)


def apply_group_boost(
    edge_direction: torch.Tensor,
    groups: list[torch.Tensor],
    *,
    target_norm: torch.Tensor | None,
) -> torch.Tensor:
    boosted = linear_mix_direction(
        edge_direction,
        groups,
        coarse_gain=4.0,
        residual_gain=1.0,
    )
    if target_norm is None:
        return boosted
    norm = torch.linalg.vector_norm(boosted)
    if float(norm) <= 1e-30:
        return boosted
    return boosted * (target_norm / norm)


def run_seed(
    *,
    seed: int,
    grouping: str,
    epochs: int,
    learning_rate: float,
    max_update: float,
    test_repeats: int,
) -> dict[str, float | int | str | bool]:
    choices = {"local", "type_pair", "random_a", "random_b", "source_type", "target_type", "global"}
    if grouping not in choices:
        raise ValueError(grouping)

    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    biological_groups = type_pair_indices(circuit)
    candidate_groups = (
        biological_groups if grouping == "local" else grouped_edge_indices(circuit, grouping)
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
    edge_norm_ratio_sum = 0.0
    target_norm_ratio_sum = 0.0
    relative_change_sum = 0.0
    clip_fraction_sum = 0.0

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
            nudge_steps=2,
            step_size=0.015,
            terminal_only=False,
        )
        local_norm = torch.linalg.vector_norm(local_edge).clamp_min(1e-30)
        biological_boost = apply_group_boost(
            local_edge,
            biological_groups,
            target_norm=None,
        )
        target_norm = torch.linalg.vector_norm(biological_boost)
        if grouping == "local":
            edge_direction = local_edge
        elif grouping == "type_pair":
            edge_direction = biological_boost
        else:
            edge_direction = apply_group_boost(
                local_edge,
                candidate_groups,
                target_norm=target_norm,
            )
        edge_norm = torch.linalg.vector_norm(edge_direction)
        edge_norm_ratio_sum += float(edge_norm / local_norm)
        target_norm_ratio_sum += float(edge_norm / target_norm.clamp_min(1e-30))
        relative_change_sum += float(
            torch.linalg.vector_norm(edge_direction - local_edge) / local_norm
        )
        clip_fraction_sum += float(
            (learning_rate * edge_direction).abs().gt(max_update).float().mean()
        )
        apply_local_credit(
            model,
            edge_direction,
            local_bias,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )

    after = evaluate_metrics(model, circuit, **eval_kwargs)
    count = max(epochs, 1)
    return {
        "seed": seed,
        "grouping": grouping,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "cross_entropy_before": before["cross_entropy"],
        "cross_entropy_after": after["cross_entropy"],
        "cross_entropy_improvement": before["cross_entropy"] - after["cross_entropy"],
        "accuracy_after": after["accuracy"],
        "margin_after": after["margin"],
        "mean_edge_norm_ratio_vs_local": edge_norm_ratio_sum / count,
        "mean_edge_norm_ratio_vs_bio_target": target_norm_ratio_sum / count,
        "mean_relative_edge_direction_change": relative_change_sum / count,
        "mean_edge_clip_fraction": clip_fraction_sum / count,
        "weight_norm": float(torch.linalg.vector_norm(model.weight)),
        "bias_norm": float(torch.linalg.vector_norm(model.bias)),
        "finite": bool(torch.isfinite(model.weight).all())
        and bool(torch.isfinite(model.bias).all())
        and math.isfinite(after["cross_entropy"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare biological and control groupings for the 4x shared-credit mode boost"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--grouping",
        choices=[
            "local",
            "type_pair",
            "random_a",
            "random_b",
            "source_type",
            "target_type",
            "global",
        ],
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.DataFrame(
        [
            run_seed(
                seed=args.seed,
                grouping=args.grouping,
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
