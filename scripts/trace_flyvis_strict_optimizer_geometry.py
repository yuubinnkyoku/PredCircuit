from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from diagnose_flyvis_temporal_ce_credit_alignment import cycle_ce_oracles, geometry
from run_flyvis_credit_residual_scaling import evaluate_metrics
from run_flyvis_retinotopic_temporal_adam import local_adam_step
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
FILTER_RULES = {"type_pair_mean"}


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
    if rule in STRICT_RULES:
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
    if rule in FILTER_RULES:
        return filtered_direction(
            circuit,
            local_edge,
            local_bias,
            oracle_edge,
            oracle_bias,
            rule=rule,
        )
    raise ValueError(f"unsupported rule: {rule}")


def run_seed(
    *,
    seed: int,
    rule: str,
    epochs: int,
    checkpoint_every: int,
    nudge_steps: int,
    learning_rate: float,
    test_repeats: int,
) -> list[dict[str, float | int | str | bool]]:
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

    rows: list[dict[str, float | int | str | bool]] = []
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

        local_geo = geometry(local_edge, local_bias, oracle_edge, oracle_bias)
        raw_geo = geometry(edge_direction, bias_direction, oracle_edge, oracle_bias)
        edge_delta = local_adam_step(
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
        bias_delta = local_adam_step(
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
        delta_geo = geometry(edge_delta, bias_delta, oracle_edge, oracle_bias)

        should_evaluate = epoch == 0 or (epoch + 1) % checkpoint_every == 0 or epoch == epochs - 1
        task = None
        if should_evaluate:
            task = evaluate_metrics(
                model,
                circuit,
                repeats=test_repeats,
                frames=7,
                width=0.5,
                frame_steps=2,
                step_size=0.015,
                jitter_seed=900_000 + seed,
            )

        rows.append(
            {
                "rule": rule,
                "seed": seed,
                "epoch": epoch,
                "nudge_steps": nudge_steps,
                "learning_rate": learning_rate,
                "raw_cosine": raw_geo["cosine"],
                "raw_projection_coefficient": raw_geo["projection_coefficient"],
                "raw_norm_ratio": raw_geo["norm_ratio"],
                "raw_residual_fraction": raw_geo["residual_fraction"],
                "local_cosine": local_geo["cosine"],
                "local_projection_coefficient": local_geo["projection_coefficient"],
                "local_norm_ratio": local_geo["norm_ratio"],
                "raw_cosine_match_error": abs(raw_geo["cosine"] - local_geo["cosine"]),
                "raw_norm_ratio_match_error": abs(raw_geo["norm_ratio"] - local_geo["norm_ratio"]),
                "delta_cosine": delta_geo["cosine"],
                "delta_projection_coefficient": delta_geo["projection_coefficient"],
                "delta_norm_ratio": delta_geo["norm_ratio"],
                "delta_residual_fraction": delta_geo["residual_fraction"],
                "mean_abs_edge_delta": float(edge_delta.abs().mean()),
                "mean_abs_bias_delta": float(bias_delta.abs().mean()),
                "accuracy": float(task["accuracy"]) if task is not None else float("nan"),
                "cross_entropy": float(task["cross_entropy"]) if task is not None else float("nan"),
                "margin": float(task["margin"]) if task is not None else float("nan"),
                "finite": bool(torch.isfinite(model.weight).all())
                and bool(torch.isfinite(model.bias).all()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace raw local-credit and actual Adam-delta geometry under strict controls"
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
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--nudge-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--test-repeats", type=int, default=4)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_strict_optimizer_geometry.csv"),
    )
    args = parser.parse_args()

    rows: list[dict[str, float | int | str | bool]] = []
    for seed in range(args.seeds):
        rows.extend(
            run_seed(
                seed=seed,
                rule=args.rule,
                epochs=args.epochs,
                checkpoint_every=args.checkpoint_every,
                nudge_steps=args.nudge_steps,
                learning_rate=args.learning_rate,
                test_repeats=args.test_repeats,
            )
        )
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    checkpoints = frame[frame["cross_entropy"].notna()]
    print(
        checkpoints.groupby("epoch")[["raw_cosine", "delta_cosine", "cross_entropy", "accuracy"]]
        .mean()
        .to_string()
    )
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
