from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from diagnose_flyvis_temporal_ce_credit_alignment import cycle_ce_oracles, geometry
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from run_flyvis_credit_residual_scaling import evaluate_metrics
from run_flyvis_temporal_coherence_gate import apply_local_credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def alignment_metrics(
    model: PredictiveCodingGraph,
    circuit,
    *,
    seed: int,
    frames: int,
    width: float,
    beta: float,
    frame_steps: int,
    nudge_steps: int,
    step_size: float,
) -> dict[str, float]:
    edge, bias = cycle_branched_credit(
        model,
        circuit,
        seed=seed,
        epoch=0,
        frames=frames,
        width=width,
        beta=beta,
        frame_steps=frame_steps,
        nudge_steps=nudge_steps,
        step_size=step_size,
        terminal_only=False,
    )
    oracles = cycle_ce_oracles(
        model,
        circuit,
        seed=seed,
        epoch=0,
        frames=frames,
        width=width,
        frame_steps=frame_steps,
        step_size=step_size,
    )
    final_edge, final_bias = oracles["final_ce_oracle"]
    temporal_edge, temporal_bias = oracles["temporal_ce_oracle"]
    final = geometry(edge, bias, final_edge, final_bias)
    temporal = geometry(edge, bias, temporal_edge, temporal_bias)
    return {
        "final_cosine": final["cosine"],
        "final_residual_fraction": final["residual_fraction"],
        "final_projection_coefficient": final["projection_coefficient"],
        "final_norm_ratio": final["norm_ratio"],
        "temporal_cosine": temporal["cosine"],
        "temporal_residual_fraction": temporal["residual_fraction"],
        "temporal_projection_coefficient": temporal["projection_coefficient"],
        "temporal_norm_ratio": temporal["norm_ratio"],
    }


def run_one(
    *,
    extent: int,
    seed: int,
    checkpoints: list[int],
    frames: int,
    width: float,
    frame_steps: int,
    nudge_steps: int,
    step_size: float,
    beta: float,
    learning_rate: float,
    weight_decay: float,
    max_update: float,
    test_repeats: int,
) -> list[dict[str, float | int | bool]]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=extent)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    rows: list[dict[str, float | int | bool]] = []
    previous_epoch = 0
    clip_fraction_sum = 0.0
    update_mean_sum = 0.0
    update_count = 0

    for checkpoint in checkpoints:
        for epoch in range(previous_epoch, checkpoint):
            edge_direction, bias_direction = cycle_branched_credit(
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
            raw_edge_update = learning_rate * (
                edge_direction - weight_decay * model.weight.detach()
            )
            if max_update > 0.0:
                clip_fraction_sum += float((raw_edge_update.abs() > max_update).float().mean())
            mean_update, _ = apply_local_credit(
                model,
                edge_direction,
                bias_direction,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                max_update=max_update,
            )
            update_mean_sum += mean_update
            update_count += 1

        metrics = evaluate_metrics(
            model,
            circuit,
            repeats=test_repeats,
            frames=frames,
            width=width,
            frame_steps=frame_steps,
            step_size=step_size,
            jitter_seed=900_000 + seed,
        )
        align = alignment_metrics(
            model,
            circuit,
            seed=seed,
            frames=frames,
            width=width,
            beta=beta,
            frame_steps=frame_steps,
            nudge_steps=nudge_steps,
            step_size=step_size,
        )
        rows.append(
            {
                "seed": seed,
                "epoch": checkpoint,
                "learning_rate": learning_rate,
                "accuracy": metrics["accuracy"],
                "cross_entropy": metrics["cross_entropy"],
                "margin": metrics["margin"],
                "mse": metrics["mse"],
                "weight_norm": float(torch.linalg.vector_norm(model.weight)),
                "bias_norm": float(torch.linalg.vector_norm(model.bias)),
                "mean_abs_update_since_previous": (
                    update_mean_sum / update_count if update_count else 0.0
                ),
                "mean_edge_clip_fraction_since_previous": (
                    clip_fraction_sum / update_count if update_count else 0.0
                ),
                **align,
                "finite": bool(torch.isfinite(model.weight).all()),
            }
        )
        previous_epoch = checkpoint
        clip_fraction_sum = 0.0
        update_mean_sum = 0.0
        update_count = 0
    return rows


def parse_checkpoints(value: str) -> list[int]:
    checkpoints = sorted({int(item) for item in value.split(",")})
    if not checkpoints or checkpoints[0] < 0:
        raise ValueError("checkpoints must be non-negative integers")
    return checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace task metrics and exact-gradient alignment during branched local learning"
    )
    parser.add_argument("--extent", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--checkpoints", type=str, default="0,10,25,50,100,200")
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--nudge-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_branched_training_trace.csv"),
    )
    args = parser.parse_args()
    checkpoints = parse_checkpoints(args.checkpoints)

    rows: list[dict[str, float | int | bool]] = []
    for seed in range(args.seeds):
        rows.extend(
            run_one(
                extent=args.extent,
                seed=seed,
                checkpoints=checkpoints,
                frames=args.frames,
                width=args.bar_width,
                frame_steps=args.frame_steps,
                nudge_steps=args.nudge_steps,
                step_size=args.step_size,
                beta=args.beta,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                max_update=args.max_update,
                test_repeats=args.test_repeats,
            )
        )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = frame.groupby("epoch")[[
        "accuracy",
        "cross_entropy",
        "margin",
        "final_cosine",
        "final_residual_fraction",
        "final_norm_ratio",
        "temporal_cosine",
        "mean_abs_update_since_previous",
        "mean_edge_clip_fraction_since_previous",
    ]].agg(["mean", "median", "std"])
    print(
        f"Branched training trace: lr={args.learning_rate:g}, seeds={args.seeds}, "
        f"checkpoints={checkpoints}"
    )
    print(summary.to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
