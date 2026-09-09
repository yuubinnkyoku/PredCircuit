from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from diagnose_flyvis_temporal_ce_credit_alignment import cycle_ce_oracles, geometry
from run_flyvis_credit_residual_scaling import evaluate_metrics
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


STRICT_RULES = {
    "type_pair_edge_permuted",
    "source_position_edge_permuted",
    "bias_permuted",
}


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

    before = evaluate_metrics(
        model,
        circuit,
        repeats=test_repeats,
        frames=7,
        width=0.5,
        frame_steps=2,
        step_size=0.015,
        jitter_seed=900_000 + seed,
    )
    cosine_match_error_sum = 0.0
    norm_match_error_sum = 0.0
    max_abs_delta = 0.0
    completed_epochs = 0
    finite = True

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
        if rule == "local":
            edge_direction, bias_direction = local_edge, local_bias
        elif rule in STRICT_RULES:
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
        elif rule == "type_pair_mean":
            edge_direction, bias_direction = filtered_direction(
                circuit,
                local_edge,
                local_bias,
                oracle_edge,
                oracle_bias,
                rule=rule,
            )
        else:
            raise ValueError(f"unsupported rule: {rule}")

        local_geo = geometry(local_edge, local_bias, oracle_edge, oracle_bias)
        transformed_geo = geometry(edge_direction, bias_direction, oracle_edge, oracle_bias)
        cosine_match_error_sum += abs(transformed_geo["cosine"] - local_geo["cosine"])
        norm_match_error_sum += abs(transformed_geo["norm_ratio"] - local_geo["norm_ratio"])

        edge_delta = learning_rate * edge_direction
        bias_delta = learning_rate * bias_direction
        max_abs_delta = max(
            max_abs_delta,
            float(edge_delta.abs().max()),
            float(bias_delta.abs().max()),
        )
        with torch.no_grad():
            model.weight.add_(edge_delta)
            model.bias.add_(bias_delta)
        completed_epochs += 1
        finite = bool(torch.isfinite(model.weight).all()) and bool(torch.isfinite(model.bias).all())
        if not finite:
            break

    if finite:
        after = evaluate_metrics(
            model,
            circuit,
            repeats=test_repeats,
            frames=7,
            width=0.5,
            frame_steps=2,
            step_size=0.015,
            jitter_seed=900_000 + seed,
        )
    else:
        after = {
            "accuracy": float("nan"),
            "cross_entropy": float("nan"),
            "margin": float("nan"),
        }
    count = max(completed_epochs, 1)
    return {
        "rule": rule,
        "seed": seed,
        "epochs_requested": epochs,
        "epochs_completed": completed_epochs,
        "nudge_steps": nudge_steps,
        "learning_rate": learning_rate,
        "accuracy_after": after["accuracy"],
        "cross_entropy_after": after["cross_entropy"],
        "cross_entropy_improvement": before["cross_entropy"] - after["cross_entropy"],
        "margin_after": after["margin"],
        "mean_abs_cosine_geometry_error": cosine_match_error_sum / count,
        "mean_abs_norm_ratio_geometry_error": norm_match_error_sum / count,
        "max_abs_delta": max_abs_delta,
        "finite": finite and math.isfinite(after["cross_entropy"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict residual-placement controls with plain SGD rather than Adam"
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
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--nudge-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_strict_residual_sgd.csv"),
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
    print(frame.select_dtypes(include="number").agg(["mean", "median", "std"]).to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
