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


def parse_int_list(value: str) -> list[int]:
    values = sorted({int(item) for item in value.split(",")})
    if not values or values[0] < 0:
        raise ValueError("values must be non-negative integers")
    return values


def run_seed(
    *,
    seed: int,
    checkpoints: list[int],
    horizons: list[int],
    train_nudge_steps: int,
    learning_rate: float,
    weight_decay: float,
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
                weight_decay=weight_decay,
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
        oracles = cycle_ce_oracles(
            model,
            circuit,
            seed=seed,
            epoch=0,
            frames=7,
            width=0.5,
            frame_steps=2,
            step_size=0.015,
        )
        final_edge, final_bias = oracles["final_ce_oracle"]
        temporal_edge, temporal_bias = oracles["temporal_ce_oracle"]

        for horizon in horizons:
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
            final = geometry(local_edge, local_bias, final_edge, final_bias)
            temporal = geometry(local_edge, local_bias, temporal_edge, temporal_bias)
            rows.append(
                {
                    "seed": seed,
                    "epoch": checkpoint,
                    "train_nudge_steps": train_nudge_steps,
                    "eval_nudge_steps": horizon,
                    "learning_rate": learning_rate,
                    "accuracy": task["accuracy"],
                    "cross_entropy": task["cross_entropy"],
                    "margin": task["margin"],
                    "weight_norm": float(torch.linalg.vector_norm(model.weight)),
                    "bias_norm": float(torch.linalg.vector_norm(model.bias)),
                    "final_cosine": final["cosine"],
                    "final_projection_coefficient": final["projection_coefficient"],
                    "final_residual_fraction": final["residual_fraction"],
                    "final_norm_ratio": final["norm_ratio"],
                    "temporal_cosine": temporal["cosine"],
                    "temporal_projection_coefficient": temporal["projection_coefficient"],
                    "temporal_residual_fraction": temporal["residual_fraction"],
                    "temporal_norm_ratio": temporal["norm_ratio"],
                    "finite": bool(torch.isfinite(model.weight).all())
                    and bool(torch.isfinite(model.bias).all()),
                }
            )
        previous_epoch = checkpoint

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep branch horizon at fixed checkpoints of one local-learning trajectory"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoints", type=str, default="0,25,50,100")
    parser.add_argument("--horizons", type=str, default="1,2,4,8,16,32")
    parser.add_argument("--train-nudge-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_checkpoint_horizon_sweep.csv"),
    )
    args = parser.parse_args()
    checkpoints = parse_int_list(args.checkpoints)
    horizons = parse_int_list(args.horizons)

    frame = pd.DataFrame(
        run_seed(
            seed=args.seed,
            checkpoints=checkpoints,
            horizons=horizons,
            train_nudge_steps=args.train_nudge_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_update=args.max_update,
            test_repeats=args.test_repeats,
        )
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = frame[
        [
            "epoch",
            "eval_nudge_steps",
            "final_cosine",
            "final_projection_coefficient",
            "final_norm_ratio",
            "temporal_cosine",
            "cross_entropy",
        ]
    ]
    print(summary.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
