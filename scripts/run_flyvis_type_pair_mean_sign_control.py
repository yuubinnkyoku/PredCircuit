from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from run_flyvis_credit_residual_scaling import evaluate_metrics
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_consensus import type_pair_indices
from run_flyvis_type_pair_linear_mix import linear_mix_direction
from run_flyvis_type_pair_norm_matched_control import match_norm

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import RetinotopicFlyVisCircuit, graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def evaluate(
    model: PredictiveCodingGraph, circuit: RetinotopicFlyVisCircuit, seed: int
) -> dict[str, float]:
    return evaluate_metrics(
        model,
        circuit,
        repeats=8,
        frames=7,
        width=0.5,
        frame_steps=2,
        step_size=0.015,
        jitter_seed=900_000 + seed,
    )


def credit(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    *,
    seed: int,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return cycle_branched_credit(
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


def run_seed(
    *, seed: int, learning_rate: float, max_update: float
) -> list[dict[str, float | int | bool | str]]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    local_model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    positive_model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    negative_model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    groups = type_pair_indices(circuit)
    models = {
        "local": local_model,
        "positive_shared": positive_model,
        "negative_shared": negative_model,
    }
    before = {name: evaluate(model, circuit, seed) for name, model in models.items()}
    direction_change = {"positive_shared": 0.0, "negative_shared": 0.0}
    clip_fraction = {"positive_shared": 0.0, "negative_shared": 0.0}
    post_weight_error = {"positive_shared": 0.0, "negative_shared": 0.0}
    post_bias_error = {"positive_shared": 0.0, "negative_shared": 0.0}

    for epoch in range(100):
        local_edge, local_bias = credit(local_model, circuit, seed=seed, epoch=epoch)
        treatment_edges: dict[str, torch.Tensor] = {}
        for name, model, coarse_gain in (
            ("positive_shared", positive_model, 4.0),
            ("negative_shared", negative_model, -2.0),
        ):
            edge, _ = credit(model, circuit, seed=seed, epoch=epoch)
            mixed = linear_mix_direction(
                edge,
                groups,
                coarse_gain=coarse_gain,
                residual_gain=1.0,
            )
            direction = match_norm(mixed, edge)
            treatment_edges[name] = direction
            edge_norm = torch.linalg.vector_norm(edge).clamp_min(1e-30)
            direction_change[name] += float(
                torch.linalg.vector_norm(direction - edge) / edge_norm
            )
            clip_fraction[name] += float(
                (learning_rate * direction).abs().gt(max_update).float().mean()
            )

        apply_local_credit(
            local_model,
            local_edge,
            local_bias,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )
        for name, model in (
            ("positive_shared", positive_model),
            ("negative_shared", negative_model),
        ):
            apply_local_credit(
                model,
                treatment_edges[name],
                local_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )
            target_norm = torch.linalg.vector_norm(local_model.weight)
            treatment_norm = torch.linalg.vector_norm(model.weight).clamp_min(1e-30)
            model.weight.mul_(target_norm / treatment_norm)
            model.bias.copy_(local_model.bias)
            post_weight_error[name] += float(
                (torch.linalg.vector_norm(model.weight) - target_norm).abs()
                / target_norm.clamp_min(1e-30)
            )
            post_bias_error[name] += float(torch.linalg.vector_norm(model.bias - local_model.bias))

    after = {name: evaluate(model, circuit, seed) for name, model in models.items()}
    rows: list[dict[str, float | int | bool | str]] = []
    for name, model in models.items():
        is_local = name == "local"
        rows.append(
            {
                "seed": seed,
                "learning_rate": learning_rate,
                "max_update": max_update,
                "rule": name,
                "cross_entropy_before": before[name]["cross_entropy"],
                "cross_entropy_after": after[name]["cross_entropy"],
                "cross_entropy_improvement": before[name]["cross_entropy"]
                - after[name]["cross_entropy"],
                "accuracy_after": after[name]["accuracy"],
                "margin_after": after[name]["margin"],
                "weight_norm": float(torch.linalg.vector_norm(model.weight)),
                "bias_norm": float(torch.linalg.vector_norm(model.bias)),
                "mean_relative_direction_change": 0.0
                if is_local
                else direction_change[name] / 100.0,
                "mean_post_weight_norm_error": 0.0
                if is_local
                else post_weight_error[name] / 100.0,
                "mean_post_bias_vector_error": 0.0
                if is_local
                else post_bias_error[name] / 100.0,
                "mean_edge_clip_fraction": float("nan")
                if is_local
                else clip_fraction[name] / 100.0,
                "finite": bool(torch.isfinite(model.weight).all())
                and bool(torch.isfinite(model.bias).all())
                and math.isfinite(after[name]["cross_entropy"]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether the sign of type-pair common-mode credit matters while matching "
            "the paired local weight-norm trajectory and bias vector"
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.DataFrame(
        run_seed(seed=args.seed, learning_rate=args.learning_rate, max_update=args.max_update)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
