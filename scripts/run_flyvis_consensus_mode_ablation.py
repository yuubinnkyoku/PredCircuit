from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_branch_nudge_alignment import cycle_branched_credit
from run_flyvis_credit_residual_scaling import evaluate_metrics
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_consensus import consensus_direction, type_pair_indices

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def globally_match_norm(candidate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    candidate_norm = torch.linalg.vector_norm(candidate)
    reference_norm = torch.linalg.vector_norm(reference)
    if float(candidate_norm) <= 1e-30 or float(reference_norm) <= 1e-30:
        return candidate
    return candidate * (reference_norm / candidate_norm)


def mode_ablation_direction(
    direction: torch.Tensor,
    groups: list[torch.Tensor],
    *,
    gamma: float,
    rule: str,
) -> torch.Tensor:
    if rule == "local":
        return direction
    if rule == "pair_consensus":
        return consensus_direction(direction, groups, gamma=gamma)

    result = direction.clone()
    for indices in groups:
        raw = direction[indices]
        mean = raw.mean()
        residual = raw - mean
        consensus_base = mean + gamma * residual
        raw_norm = torch.linalg.vector_norm(raw)
        base_norm = torch.linalg.vector_norm(consensus_base)
        scale = (
            raw_norm / base_norm
            if float(raw_norm) > 1e-30 and float(base_norm) > 1e-30
            else torch.ones((), dtype=raw.dtype, device=raw.device)
        )
        if rule == "coarse_boost":
            result[indices] = scale * mean + residual
        elif rule == "residual_shrink":
            result[indices] = mean + scale * gamma * residual
        elif rule == "plain_residual_shrink":
            result[indices] = mean + gamma * residual
        else:
            raise ValueError(f"unknown rule: {rule}")
    return globally_match_norm(result, direction)


def run_seed(
    *,
    seed: int,
    rule: str,
    gamma: float,
    epochs: int,
    learning_rate: float,
    max_update: float,
    test_repeats: int,
) -> dict[str, float | int | str | bool]:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    groups = type_pair_indices(circuit)
    eval_kwargs = {
        "repeats": test_repeats,
        "frames": 7,
        "width": 0.5,
        "frame_steps": 2,
        "step_size": 0.015,
        "jitter_seed": 900_000 + seed,
    }
    before = evaluate_metrics(model, circuit, **eval_kwargs)
    direction_change_sum = 0.0
    norm_error_sum = 0.0
    clip_fraction_sum = 0.0

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
            nudge_steps=2,
            step_size=0.015,
            terminal_only=False,
        )
        edge_direction = mode_ablation_direction(
            local_edge,
            groups,
            gamma=gamma,
            rule=rule,
        )
        local_norm = torch.linalg.vector_norm(local_edge).clamp_min(1e-30)
        direction_change_sum += float(
            torch.linalg.vector_norm(edge_direction - local_edge) / local_norm
        )
        norm_error_sum += float(
            torch.abs(torch.linalg.vector_norm(edge_direction) - local_norm) / local_norm
        )
        clip_fraction_sum += float(
            (learning_rate * edge_direction).abs().gt(max_update).float().mean()
        )
        apply_local_credit(
            model,
            edge_direction,
            local_bias,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )

    after = evaluate_metrics(model, circuit, **eval_kwargs)
    count = max(epochs, 1)
    return {
        "seed": seed,
        "rule": rule,
        "gamma": gamma,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "cross_entropy_before": before["cross_entropy"],
        "cross_entropy_after": after["cross_entropy"],
        "cross_entropy_improvement": before["cross_entropy"] - after["cross_entropy"],
        "accuracy_after": after["accuracy"],
        "margin_after": after["margin"],
        "mean_relative_edge_direction_change": direction_change_sum / count,
        "mean_relative_edge_norm_error": norm_error_sum / count,
        "mean_edge_clip_fraction": clip_fraction_sum / count,
        "weight_norm": float(torch.linalg.vector_norm(model.weight)),
        "bias_norm": float(torch.linalg.vector_norm(model.bias)),
        "finite": bool(torch.isfinite(model.weight).all())
        and bool(torch.isfinite(model.bias).all())
        and math.isfinite(after["cross_entropy"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablate coarse amplification versus within-pair residual attenuation"
    )
    parser.add_argument(
        "--rule",
        choices=(
            "local",
            "pair_consensus",
            "coarse_boost",
            "residual_shrink",
            "plain_residual_shrink",
        ),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.DataFrame(
        [
            run_seed(
                seed=args.seed,
                rule=args.rule,
                gamma=args.gamma,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                max_update=args.max_update,
                test_repeats=args.test_repeats,
            )
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
