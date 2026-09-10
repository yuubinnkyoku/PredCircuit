from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from diagnose_flyvis_temporal_ce_credit_alignment import cycle_ce_oracles
from run_flyvis_credit_residual_scaling import evaluate_metrics
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_consensus import type_pair_indices
from run_flyvis_type_pair_linear_mix import linear_mix_direction

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def parse_int_list(value: str) -> list[int]:
    values = sorted({int(item) for item in value.split(",")})
    if not values or values[0] < 0:
        raise ValueError("values must be non-negative integers")
    return values


def split_type_pair_modes(
    direction: torch.Tensor, groups: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    coarse = torch.zeros_like(direction)
    for indices in groups:
        coarse[indices] = direction[indices].mean()
    return coarse, direction - coarse


def vector_geometry(candidate: torch.Tensor, oracle: torch.Tensor) -> dict[str, float]:
    candidate_norm = torch.linalg.vector_norm(candidate)
    oracle_norm = torch.linalg.vector_norm(oracle)
    denom = candidate_norm * oracle_norm
    oracle_norm_sq = oracle_norm.square().clamp_min(1e-30)
    return {
        "cosine": float(torch.dot(candidate, oracle) / denom)
        if float(denom) > 1e-30
        else float("nan"),
        "projection": float(torch.dot(candidate, oracle) / oracle_norm_sq),
        "norm_ratio": float(candidate_norm / oracle_norm.clamp_min(1e-30)),
        "candidate_norm": float(candidate_norm),
        "oracle_norm": float(oracle_norm),
    }


def run_seed(
    *,
    seed: int,
    train_rule: str,
    checkpoints: list[int],
    learning_rate: float,
    max_update: float,
) -> list[dict[str, float | int | str | bool]]:
    if train_rule not in {"local", "4m+r"}:
        raise ValueError(train_rule)

    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    groups = type_pair_indices(circuit)
    rows: list[dict[str, float | int | str | bool]] = []
    previous_epoch = 0

    for checkpoint in checkpoints:
        for epoch in range(previous_epoch, checkpoint):
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
            edge_direction = (
                local_edge
                if train_rule == "local"
                else linear_mix_direction(
                    local_edge,
                    groups,
                    coarse_gain=4.0,
                    residual_gain=1.0,
                )
            )
            apply_local_credit(
                model,
                edge_direction,
                local_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )

        task = evaluate_metrics(
            model,
            circuit,
            repeats=8,
            frames=7,
            width=0.5,
            frame_steps=2,
            step_size=0.015,
            jitter_seed=900_000 + seed,
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
            nudge_steps=2,
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

        local_coarse, local_residual = split_type_pair_modes(local_edge, groups)
        oracle_coarse, oracle_residual = split_type_pair_modes(oracle_edge, groups)
        boosted_edge = linear_mix_direction(
            local_edge,
            groups,
            coarse_gain=4.0,
            residual_gain=1.0,
        )
        boosted_coarse, boosted_residual = split_type_pair_modes(boosted_edge, groups)

        local_full = vector_geometry(
            torch.cat((local_edge, local_bias)),
            torch.cat((oracle_edge, oracle_bias)),
        )
        boosted_full = vector_geometry(
            torch.cat((boosted_edge, local_bias)),
            torch.cat((oracle_edge, oracle_bias)),
        )
        coarse = vector_geometry(local_coarse, oracle_coarse)
        residual = vector_geometry(local_residual, oracle_residual)
        boosted_coarse_geometry = vector_geometry(boosted_coarse, oracle_coarse)
        boosted_residual_geometry = vector_geometry(boosted_residual, oracle_residual)
        bias = vector_geometry(local_bias, oracle_bias)

        oracle_edge_norm_sq = torch.dot(oracle_edge, oracle_edge).clamp_min(1e-30)
        rows.append(
            {
                "seed": seed,
                "train_rule": train_rule,
                "epoch": checkpoint,
                "cross_entropy": task["cross_entropy"],
                "accuracy": task["accuracy"],
                "margin": task["margin"],
                "full_local_cosine": local_full["cosine"],
                "full_local_projection": local_full["projection"],
                "full_local_norm_ratio": local_full["norm_ratio"],
                "full_boosted_cosine": boosted_full["cosine"],
                "full_boosted_projection": boosted_full["projection"],
                "full_boosted_norm_ratio": boosted_full["norm_ratio"],
                "coarse_cosine": coarse["cosine"],
                "coarse_projection": coarse["projection"],
                "coarse_norm_ratio": coarse["norm_ratio"],
                "boosted_coarse_projection": boosted_coarse_geometry["projection"],
                "boosted_coarse_norm_ratio": boosted_coarse_geometry["norm_ratio"],
                "residual_cosine": residual["cosine"],
                "residual_projection": residual["projection"],
                "residual_norm_ratio": residual["norm_ratio"],
                "boosted_residual_projection": boosted_residual_geometry["projection"],
                "boosted_residual_norm_ratio": boosted_residual_geometry["norm_ratio"],
                "bias_cosine": bias["cosine"],
                "bias_projection": bias["projection"],
                "bias_norm_ratio": bias["norm_ratio"],
                "oracle_coarse_edge_energy_fraction": float(
                    torch.dot(oracle_coarse, oracle_coarse) / oracle_edge_norm_sq
                ),
                "local_coarse_edge_energy_fraction": float(
                    torch.dot(local_coarse, local_coarse)
                    / torch.dot(local_edge, local_edge).clamp_min(1e-30)
                ),
                "weight_norm": float(torch.linalg.vector_norm(model.weight)),
                "bias_norm": float(torch.linalg.vector_norm(model.bias)),
                "finite": bool(torch.isfinite(model.weight).all())
                and bool(torch.isfinite(model.bias).all()),
            }
        )
        previous_epoch = checkpoint

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decompose local and exact credit into type-pair mean and within-pair residual modes"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-rule", choices=["local", "4m+r"], required=True)
    parser.add_argument("--checkpoints", default="0,25,50,100")
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.DataFrame(
        run_seed(
            seed=args.seed,
            train_rule=args.train_rule,
            checkpoints=parse_int_list(args.checkpoints),
            learning_rate=args.learning_rate,
            max_update=args.max_update,
        )
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
