from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from diagnose_flyvis_temporal_ce_credit_alignment import cycle_ce_oracles, geometry
from run_flyvis_credit_residual_scaling import evaluate_metrics
from run_flyvis_temporal_coherence_gate import apply_local_credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def component_geometry(
    local_edge: torch.Tensor,
    local_bias: torch.Tensor,
    oracle_edge: torch.Tensor,
    oracle_bias: torch.Tensor,
) -> dict[str, float]:
    zero_edge = torch.zeros_like(local_edge)
    zero_bias = torch.zeros_like(local_bias)
    full = geometry(local_edge, local_bias, oracle_edge, oracle_bias)
    edge = geometry(local_edge, zero_bias, oracle_edge, zero_bias)
    bias = geometry(zero_edge, local_bias, zero_edge, oracle_bias)
    edge_dot = torch.dot(local_edge, oracle_edge)
    bias_dot = torch.dot(local_bias, oracle_bias)
    return {
        "full_cosine": full["cosine"],
        "full_projection_coefficient": full["projection_coefficient"],
        "full_norm_ratio": full["norm_ratio"],
        "edge_cosine": edge["cosine"],
        "edge_projection_coefficient": edge["projection_coefficient"],
        "edge_norm_ratio": edge["norm_ratio"],
        "bias_cosine": bias["cosine"],
        "bias_projection_coefficient": bias["projection_coefficient"],
        "bias_norm_ratio": bias["norm_ratio"],
        "edge_alignment_dot": float(edge_dot),
        "bias_alignment_dot": float(bias_dot),
        "local_edge_norm": float(torch.linalg.vector_norm(local_edge)),
        "local_bias_norm": float(torch.linalg.vector_norm(local_bias)),
        "oracle_edge_norm": float(torch.linalg.vector_norm(oracle_edge)),
        "oracle_bias_norm": float(torch.linalg.vector_norm(oracle_bias)),
    }


def run_seed(
    *,
    seed: int,
    checkpoints: list[int],
    eval_horizons: tuple[int, ...],
    train_nudge_steps: int,
    learning_rate: float,
    max_update: float,
    test_repeats: int,
) -> list[dict[str, float | int | bool]]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    rows: list[dict[str, float | int | bool]] = []
    previous_epoch = 0

    for checkpoint in checkpoints:
        for epoch in range(previous_epoch, checkpoint):
            edge_direction, bias_direction = cycle_branched_credit(
                model,
                circuit,
                seed=seed,
                epoch=epoch,
                frames=7,
                width=0.5,
                beta=0.03,
                frame_steps=2,
                nudge_steps=train_nudge_steps,
                step_size=0.015,
                terminal_only=False,
            )
            apply_local_credit(
                model,
                edge_direction,
                bias_direction,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )

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

        for horizon in eval_horizons:
            local_edge, local_bias = cycle_branched_credit(
                model,
                circuit,
                seed=seed,
                epoch=0,
                frames=7,
                width=0.5,
                beta=0.03,
                frame_steps=2,
                nudge_steps=horizon,
                step_size=0.015,
                terminal_only=False,
            )
            rows.append(
                {
                    "seed": seed,
                    "epoch": checkpoint,
                    "train_nudge_steps": train_nudge_steps,
                    "eval_nudge_steps": horizon,
                    "learning_rate": learning_rate,
                    "cross_entropy": task["cross_entropy"],
                    "accuracy": task["accuracy"],
                    "margin": task["margin"],
                    "weight_norm": float(torch.linalg.vector_norm(model.weight)),
                    "bias_parameter_norm": float(torch.linalg.vector_norm(model.bias)),
                    **component_geometry(
                        local_edge,
                        local_bias,
                        oracle_edge,
                        oracle_bias,
                    ),
                    "finite": bool(torch.isfinite(model.weight).all())
                    and bool(torch.isfinite(model.bias).all()),
                }
            )
        previous_epoch = checkpoint

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split local-credit alignment into edge-weight and bias components"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--train-nudge-steps", type=int, default=2)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_checkpoint_component_geometry.csv"),
    )
    args = parser.parse_args()

    frame = pd.DataFrame(
        run_seed(
            seed=args.seed,
            checkpoints=[0, 25, 50, 100],
            eval_horizons=(1, 2),
            train_nudge_steps=args.train_nudge_steps,
            learning_rate=args.learning_rate,
            max_update=args.max_update,
            test_repeats=args.test_repeats,
        )
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(
        frame[
            [
                "epoch",
                "eval_nudge_steps",
                "full_cosine",
                "edge_cosine",
                "bias_cosine",
                "local_edge_norm",
                "local_bias_norm",
                "oracle_edge_norm",
                "oracle_bias_norm",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
