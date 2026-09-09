from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from diagnose_flyvis_temporal_ce_credit_alignment import cycle_ce_oracles, geometry
from run_flyvis_credit_residual_scaling import evaluate_metrics
from run_flyvis_retinotopic_temporal_adam import local_adam_step

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


@torch.no_grad()
def preview_adam_delta(
    direction: torch.Tensor,
    first_moment: torch.Tensor,
    second_moment: torch.Tensor,
    *,
    step: int,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
) -> torch.Tensor:
    first = beta1 * first_moment + (1.0 - beta1) * direction
    second = beta2 * second_moment + (1.0 - beta2) * direction.square()
    first_hat = first / (1.0 - beta1**step)
    second_hat = second / (1.0 - beta2**step)
    return learning_rate * first_hat / (second_hat.sqrt() + epsilon)


def run_seed(
    *,
    seed: int,
    epochs: int,
    checkpoint_every: int,
    frames: int,
    width: float,
    frame_steps: int,
    nudge_steps: int,
    step_size: float,
    beta: float,
    learning_rate: float,
    beta1: float,
    beta2: float,
    adam_epsilon: float,
    test_repeats: int,
) -> list[dict[str, float | int]]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    edge_m = torch.zeros_like(model.weight)
    edge_v = torch.zeros_like(model.weight)
    bias_m = torch.zeros_like(model.bias)
    bias_v = torch.zeros_like(model.bias)
    rows: list[dict[str, float | int]] = []

    for epoch in range(epochs):
        local_edge, local_bias = cycle_branched_credit(
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
        should_measure = epoch % checkpoint_every == 0 or epoch == epochs - 1
        if should_measure:
            oracle_edge, oracle_bias = cycle_ce_oracles(
                model,
                circuit,
                seed=seed,
                epoch=epoch,
                frames=frames,
                width=width,
                frame_steps=frame_steps,
                step_size=step_size,
            )["final_ce_oracle"]
            raw = geometry(local_edge, local_bias, oracle_edge, oracle_bias)
            preview_edge = preview_adam_delta(
                local_edge,
                edge_m,
                edge_v,
                step=epoch + 1,
                learning_rate=learning_rate,
                beta1=beta1,
                beta2=beta2,
                epsilon=adam_epsilon,
            )
            preview_bias = preview_adam_delta(
                local_bias,
                bias_m,
                bias_v,
                step=epoch + 1,
                learning_rate=learning_rate,
                beta1=beta1,
                beta2=beta2,
                epsilon=adam_epsilon,
            )
            applied = geometry(
                preview_edge,
                preview_bias,
                oracle_edge,
                oracle_bias,
            )
            task = evaluate_metrics(
                model,
                circuit,
                repeats=test_repeats,
                frames=frames,
                width=width,
                frame_steps=frame_steps,
                step_size=step_size,
                jitter_seed=900_000 + seed,
            )
            rows.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "nudge_steps": nudge_steps,
                    "cross_entropy": task["cross_entropy"],
                    "accuracy": task["accuracy"],
                    "margin": task["margin"],
                    "raw_cosine": raw["cosine"],
                    "raw_projection_coefficient": raw["projection_coefficient"],
                    "raw_residual_fraction": raw["residual_fraction"],
                    "raw_norm_ratio": raw["norm_ratio"],
                    "raw_sign_agreement": raw["sign_agreement"],
                    "applied_cosine": applied["cosine"],
                    "applied_projection_coefficient": applied["projection_coefficient"],
                    "applied_residual_fraction": applied["residual_fraction"],
                    "applied_norm_ratio": applied["norm_ratio"],
                    "applied_sign_agreement": applied["sign_agreement"],
                    "mean_abs_local_edge": float(local_edge.abs().mean()),
                    "mean_abs_preview_edge": float(preview_edge.abs().mean()),
                }
            )

        local_adam_step(
            model.weight,
            local_edge,
            edge_m,
            edge_v,
            step=epoch + 1,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=adam_epsilon,
        )
        local_adam_step(
            model.bias,
            local_bias,
            bias_m,
            bias_v,
            step=epoch + 1,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=adam_epsilon,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track local and Adam-applied credit geometry during branched FlyVis training"
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--nudge-steps", type=int, required=True)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--test-repeats", type=int, default=4)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_branched_training_trajectory.csv"),
    )
    args = parser.parse_args()

    rows: list[dict[str, float | int]] = []
    for seed in range(args.seeds):
        rows.extend(
            run_seed(
                seed=seed,
                epochs=args.epochs,
                checkpoint_every=args.checkpoint_every,
                frames=args.frames,
                width=args.bar_width,
                frame_steps=args.frame_steps,
                nudge_steps=args.nudge_steps,
                step_size=args.step_size,
                beta=args.beta,
                learning_rate=args.learning_rate,
                beta1=args.beta1,
                beta2=args.beta2,
                adam_epsilon=args.adam_epsilon,
                test_repeats=args.test_repeats,
            )
        )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = frame.groupby("epoch")[[
        "cross_entropy",
        "accuracy",
        "raw_cosine",
        "raw_projection_coefficient",
        "applied_cosine",
        "applied_projection_coefficient",
    ]].mean()
    print(f"Training trajectory: nudge_steps={args.nudge_steps}, seeds={args.seeds}")
    print(summary.to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
