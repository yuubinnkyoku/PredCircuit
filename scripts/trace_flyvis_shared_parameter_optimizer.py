from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from diagnose_flyvis_temporal_ce_credit_alignment import cycle_ce_oracles
from run_flyvis_credit_residual_scaling import evaluate_metrics
from run_flyvis_retinotopic_temporal_adam import local_adam_step
from run_flyvis_shared_parameter_training import SharedFlyVisParameters, vector_geometry

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def run_seed(
    *,
    seed: int,
    mode: str,
    epochs: int,
    checkpoint_every: int,
    nudge_steps: int,
    learning_rate: float,
    test_repeats: int,
) -> list[dict[str, float | int | str | bool]]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    shared = SharedFlyVisParameters(circuit, gain_init=0.08)
    shared.write_to_model(model)

    gain_m = torch.zeros_like(shared.gain)
    gain_v = torch.zeros_like(shared.gain)
    bias_m = torch.zeros_like(shared.bias)
    bias_v = torch.zeros_like(shared.bias)
    rows: list[dict[str, float | int | str | bool]] = []

    for epoch in range(epochs):
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
        oracle_gain, oracle_shared_bias = shared.project_direction(oracle_edge, oracle_bias)

        if mode == "local":
            edge_direction, bias_direction = cycle_branched_credit(
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
            gain_direction, shared_bias_direction = shared.project_direction(
                edge_direction, bias_direction
            )
        elif mode == "exact":
            gain_direction = oracle_gain
            shared_bias_direction = oracle_shared_bias
        else:
            raise ValueError(f"unsupported mode: {mode}")

        raw = torch.cat((gain_direction, shared_bias_direction))
        oracle = torch.cat((oracle_gain, oracle_shared_bias))
        raw_cosine, raw_projection, raw_norm_ratio = vector_geometry(raw, oracle)

        gain_before = shared.gain.clone()
        bias_before = shared.bias.clone()
        local_adam_step(
            shared.gain,
            gain_direction,
            gain_m,
            gain_v,
            step=epoch + 1,
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
        )
        shared.gain.clamp_(min=0.0)
        local_adam_step(
            shared.bias,
            shared_bias_direction,
            bias_m,
            bias_v,
            step=epoch + 1,
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
        )
        actual_gain_delta = shared.gain - gain_before
        actual_bias_delta = shared.bias - bias_before
        actual_delta = torch.cat((actual_gain_delta, actual_bias_delta))
        delta_cosine, delta_projection, delta_norm_ratio = vector_geometry(actual_delta, oracle)
        shared.write_to_model(model)

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
                "mode": mode,
                "seed": seed,
                "epoch": epoch,
                "nudge_steps": nudge_steps,
                "learning_rate": learning_rate,
                "raw_cosine": raw_cosine,
                "raw_projection_coefficient": raw_projection,
                "raw_norm_ratio": raw_norm_ratio,
                "delta_cosine": delta_cosine,
                "delta_projection_coefficient": delta_projection,
                "delta_norm_ratio": delta_norm_ratio,
                "mean_abs_gain_delta": float(actual_gain_delta.abs().mean()),
                "mean_abs_bias_delta": float(actual_bias_delta.abs().mean()),
                "zero_gain_fraction": float((shared.gain == 0).float().mean()),
                "gain_mean": float(shared.gain.mean()),
                "bias_norm": float(torch.linalg.vector_norm(shared.bias)),
                "accuracy": float(task["accuracy"]) if task is not None else float("nan"),
                "cross_entropy": float(task["cross_entropy"]) if task is not None else float("nan"),
                "margin": float(task["margin"]) if task is not None else float("nan"),
                "finite": bool(torch.isfinite(shared.gain).all())
                and bool(torch.isfinite(shared.bias).all()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace raw shared credit and clamp-aware Adam parameter deltas"
    )
    parser.add_argument("--mode", choices=("local", "exact"), required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--nudge-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--test-repeats", type=int, default=4)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_shared_parameter_optimizer_trace.csv"),
    )
    args = parser.parse_args()

    rows: list[dict[str, float | int | str | bool]] = []
    for seed in range(args.seeds):
        rows.extend(
            run_seed(
                seed=seed,
                mode=args.mode,
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
