from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_l1_gain_geometry import _heldout_objectives
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from diagnose_flyvis_type_pair_signed_band_holdout import (
    HIGH_EXTRA_GAIN,
    HIGH_RHO_THRESHOLD,
    LOW_EXTRA_GAIN,
    SIGNED_BAND_THRESHOLD,
)
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

HORIZONS = (80, 100)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = (torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)).clamp_min(
        1e-30
    )
    return float(torch.dot(left, right) / denominator)


def _clipped(values: torch.Tensor, learning_rate: float, max_update: float) -> torch.Tensor:
    update = learning_rate * values
    return update.clamp(-max_update, max_update) if max_update > 0.0 else update


def run_seed(*, seed: int, learning_rate: float, max_update: float) -> pd.DataFrame:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    named_groups = _named_groups(circuit)
    model = PredictiveCodingGraph(
        circuit.graph,
        seed=seed,
        init_scale=0.08,
        use_biological_strength=True,
    )
    rows: list[dict[str, float | int | bool | str]] = []
    trained = 0

    for horizon in HORIZONS:
        for epoch in range(trained, horizon):
            raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
            apply_local_credit(
                model,
                raw_edge,
                raw_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )
        trained = horizon

        raw_edge, _ = credit(model, circuit, seed=seed, epoch=horizon)
        weight = model.weight.detach().clone().requires_grad_(True)
        bias = model.bias.detach().clone()
        ce, hard_margin, soft_margin = _heldout_objectives(circuit, weight, bias)
        grad_ce = torch.autograd.grad(ce, weight, retain_graph=True)[0]
        grad_hard = torch.autograd.grad(hard_margin, weight, retain_graph=True)[0]
        grad_soft = torch.autograd.grad(soft_margin, weight)[0]

        for source_type, target_type, indices in named_groups:
            values = raw_edge[indices]
            mean = values.mean()
            residual = values - mean
            mean_component = mean.expand_as(values)
            mean_norm = torch.linalg.vector_norm(mean_component)
            residual_norm = torch.linalg.vector_norm(residual)
            rho = float(mean_norm / residual_norm.clamp_min(1e-30))
            if not (SIGNED_BAND_THRESHOLD <= rho < HIGH_RHO_THRESHOLD):
                continue

            group_weight = model.weight.detach()[indices]
            high_direction = values + HIGH_EXTRA_GAIN * mean
            low_direction = values + LOW_EXTRA_GAIN * mean
            high_update = _clipped(high_direction, learning_rate, max_update)
            low_update = _clipped(low_direction, learning_rate, max_update)
            suppression_delta = low_update - high_update

            high_clip = (
                (learning_rate * high_direction).abs() >= max_update
                if max_update > 0.0
                else torch.zeros_like(values, dtype=torch.bool)
            )
            low_clip = (
                (learning_rate * low_direction).abs() >= max_update
                if max_update > 0.0
                else torch.zeros_like(values, dtype=torch.bool)
            )

            weight_norm = torch.linalg.vector_norm(group_weight)
            credit_norm = torch.linalg.vector_norm(values)
            delta_norm = torch.linalg.vector_norm(suppression_delta)
            local_weight_credit_dot = torch.dot(group_weight, values)
            local_weight_delta_dot = torch.dot(group_weight, suppression_delta)
            mean_weight_credit = (group_weight * values).mean()
            proxy = group_weight.mean() * mean

            values_out = {
                "rho": rho,
                "mean_credit": float(mean),
                "mean_credit_abs": float(mean.abs()),
                "residual_norm": float(residual_norm),
                "credit_norm": float(credit_norm),
                "weight_mean": float(group_weight.mean()),
                "weight_std": float(group_weight.std(unbiased=False)),
                "weight_norm": float(weight_norm),
                "weight_mean_times_credit_mean": float(proxy),
                "mean_weight_times_credit": float(mean_weight_credit),
                "weight_credit_dot": float(local_weight_credit_dot),
                "weight_credit_cosine": _cosine(group_weight, values),
                "suppression_delta_norm": float(delta_norm),
                "weight_suppression_delta_dot": float(local_weight_delta_dot),
                "weight_suppression_delta_cosine": _cosine(group_weight, suppression_delta),
                "high_clip_fraction": float(high_clip.float().mean()),
                "low_clip_fraction": float(low_clip.float().mean()),
                "clip_fraction_drop": float(high_clip.float().mean() - low_clip.float().mean()),
                "ce_grad_dot_suppression_delta": float(
                    torch.dot(grad_ce[indices], suppression_delta)
                ),
                "hard_margin_grad_dot_suppression_delta": float(
                    torch.dot(grad_hard[indices], suppression_delta)
                ),
                "soft_margin_grad_dot_suppression_delta": float(
                    torch.dot(grad_soft[indices], suppression_delta)
                ),
            }
            finite = (
                bool(torch.isfinite(model.weight).all())
                and bool(torch.isfinite(model.bias).all())
                and all(math.isfinite(value) for value in values_out.values())
            )
            rows.append(
                {
                    "seed": seed,
                    "horizon": horizon,
                    "source_type": source_type,
                    "target_type": target_type,
                    "edge_count": len(indices),
                    **values_out,
                    "finite": finite,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure held-out gradient geometry of the actual clipped gain3-to-gain1 "
            "suppression update for boundary type-pair groups, alongside fully local proxies."
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = run_seed(seed=args.seed, learning_rate=args.learning_rate, max_update=args.max_update)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
