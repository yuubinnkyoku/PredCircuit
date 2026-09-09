from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from diagnose_flyvis_temporal_ce_credit_alignment import cycle_ce_oracles
from run_flyvis_temporal_credit_projection import flatten_credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import RetinotopicFlyVisCircuit, graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float(torch.dot(left, right) / denominator.clamp_min(1e-30))


def first_adam_direction(direction: torch.Tensor, epsilon: float) -> torch.Tensor:
    return direction / (direction.abs() + epsilon)


def weighted_sign_agreement(
    candidate: torch.Tensor,
    oracle: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> float:
    weights = oracle.abs()
    if mask is not None:
        weights = weights * mask
    same_sign = (candidate * oracle > 0).to(weights.dtype)
    return float((weights * same_sign).sum() / weights.sum().clamp_min(1e-30))


def top_fraction_mask(oracle: torch.Tensor, fraction: float) -> torch.Tensor:
    magnitudes = oracle.abs()
    nonzero = magnitudes[magnitudes > 0]
    if nonzero.numel() == 0:
        return torch.ones_like(magnitudes)
    threshold = torch.quantile(nonzero, max(1.0 - fraction, 0.0))
    return (magnitudes >= threshold).to(magnitudes.dtype)


def measure(
    circuit: RetinotopicFlyVisCircuit,
    *,
    seed: int,
    beta: float,
    frame_steps: int,
    nudge_steps: int,
    frames: int,
    width: float,
    step_size: float,
    adam_epsilon: float,
) -> dict[str, float | int]:
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    local_edge, local_bias = cycle_branched_credit(
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
    oracle_edge, oracle_bias = cycle_ce_oracles(
        model,
        circuit,
        seed=seed,
        epoch=0,
        frames=frames,
        width=width,
        frame_steps=frame_steps,
        step_size=step_size,
    )["final_ce_oracle"]

    local = flatten_credit(local_edge, local_bias)
    oracle = flatten_credit(oracle_edge, oracle_bias)
    oracle_norm_sq = torch.dot(oracle, oracle).clamp_min(1e-30)
    alpha = torch.dot(local, oracle) / oracle_norm_sq
    parallel = alpha * oracle
    residual = local - parallel

    adam_local = first_adam_direction(local, adam_epsilon)
    adam_oracle = first_adam_direction(oracle, adam_epsilon)
    adam_noise = adam_local - adam_oracle

    residual_dominates = residual.abs() > parallel.abs()
    oracle_weights = oracle.abs()
    weighted_residual_dominates = float(
        (oracle_weights * residual_dominates.to(oracle_weights.dtype)).sum()
        / oracle_weights.sum().clamp_min(1e-30)
    )
    top10 = top_fraction_mask(oracle, 0.10)
    top50 = top_fraction_mask(oracle, 0.50)

    return {
        "seed": seed,
        "beta": beta,
        "frame_steps": frame_steps,
        "nudge_steps": nudge_steps,
        "adam_epsilon": adam_epsilon,
        "raw_cosine": cosine(local, oracle),
        "projection_coefficient": float(alpha),
        "residual_to_parallel_norm": float(
            torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(parallel).clamp_min(1e-30)
        ),
        "coordinate_residual_dominates_fraction": float(residual_dominates.float().mean()),
        "oracle_weighted_residual_dominates_fraction": weighted_residual_dominates,
        "oracle_weighted_sign_agreement": weighted_sign_agreement(local, oracle),
        "top10_oracle_weighted_sign_agreement": weighted_sign_agreement(
            local, oracle, mask=top10
        ),
        "top50_oracle_weighted_sign_agreement": weighted_sign_agreement(
            local, oracle, mask=top50
        ),
        "adam_local_vs_adam_oracle_cosine": cosine(adam_local, adam_oracle),
        "adam_local_vs_raw_oracle_cosine": cosine(adam_local, oracle),
        "adam_oracle_vs_raw_oracle_cosine": cosine(adam_oracle, oracle),
        "adam_noise_to_oracle_adam_norm": float(
            torch.linalg.vector_norm(adam_noise)
            / torch.linalg.vector_norm(adam_oracle).clamp_min(1e-30)
        ),
        "local_below_epsilon_fraction": float((local.abs() <= adam_epsilon).float().mean()),
        "local_below_10epsilon_fraction": float((local.abs() <= 10 * adam_epsilon).float().mean()),
        "local_median_abs": float(local.abs().median()),
        "oracle_median_abs": float(oracle.abs().median()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure coordinate-level signal/noise and first-step Adam distortion for branched "
            "FlyVis local credit against the exact final-frame CE descent"
        )
    )
    parser.add_argument("--extent", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--nudge-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_coordinate_snr.csv"),
    )
    args = parser.parse_args()

    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=args.extent)
    rows = [
        measure(
            circuit,
            seed=seed,
            beta=args.beta,
            frame_steps=args.frame_steps,
            nudge_steps=args.nudge_steps,
            frames=args.frames,
            width=args.bar_width,
            step_size=args.step_size,
            adam_epsilon=args.adam_epsilon,
        )
        for seed in range(args.seeds)
    ]
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    metrics = [
        "raw_cosine",
        "projection_coefficient",
        "residual_to_parallel_norm",
        "oracle_weighted_residual_dominates_fraction",
        "oracle_weighted_sign_agreement",
        "top10_oracle_weighted_sign_agreement",
        "adam_local_vs_adam_oracle_cosine",
        "adam_local_vs_raw_oracle_cosine",
        "adam_noise_to_oracle_adam_norm",
        "local_below_10epsilon_fraction",
    ]
    print(
        f"Coordinate SNR: nudge_steps={args.nudge_steps}, beta={args.beta:g}, "
        f"seeds={args.seeds}"
    )
    print(frame[metrics].agg(["mean", "median", "std"]).to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
