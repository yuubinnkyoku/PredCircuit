from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_boundary_actual_update_geometry import _clipped, _cosine
from diagnose_flyvis_type_pair_l1_gain_geometry import _heldout_objectives
from diagnose_flyvis_type_pair_local_power_holdout import LOCAL_POWER_THRESHOLD
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

HORIZONS = (0, 20, 40, 60, 80, 100)


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

            group_weight = model.weight.detach()[indices]
            high_direction = values + HIGH_EXTRA_GAIN * mean
            low_direction = values + LOW_EXTRA_GAIN * mean
            high_update = _clipped(high_direction, learning_rate, max_update)
            low_update = _clipped(low_direction, learning_rate, max_update)
            # actual clip-post update difference when gain 3 -> 1
            delta_u = low_update - high_update

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

            credit_std = float(values.std(unbiased=False))
            weight_std = float(group_weight.std(unbiased=False))
            cov_w_c = float(((group_weight - group_weight.mean()) * (values - mean)).mean())
            mean_w_c = float((group_weight * values).mean())
            mean_w_times_mean_c = float(group_weight.mean() * mean)
            delta_norm = float(torch.linalg.vector_norm(delta_u))

            # oracle: positive means suppressing (gain3->1) improves hard margin
            oracle_hard = float(torch.dot(grad_hard[indices], delta_u))
            oracle_ce = float(torch.dot(grad_ce[indices], delta_u))
            oracle_soft = float(torch.dot(grad_soft[indices], delta_u))

            values_out = {
                "rho": rho,
                "mean_credit": float(mean),
                "credit_mean_abs": float(mean.abs()),
                "credit_std": credit_std,
                "credit_norm": float(torch.linalg.vector_norm(values)),
                "weight_mean": float(group_weight.mean()),
                "weight_std": weight_std,
                "weight_norm": float(torch.linalg.vector_norm(group_weight)),
                "mean_w_c": mean_w_c,
                "mean_w_times_mean_c": mean_w_times_mean_c,
                "cov_w_c": cov_w_c,
                "weight_credit_cosine": _cosine(group_weight, values),
                "delta_u_norm": delta_norm,
                "weight_delta_dot": float(torch.dot(group_weight, delta_u)),
                "weight_delta_cosine": _cosine(group_weight, delta_u),
                "high_clip_fraction": float(high_clip.float().mean()),
                "low_clip_fraction": float(low_clip.float().mean()),
                "clip_fraction_drop": float(high_clip.float().mean() - low_clip.float().mean()),
                "oracle_hard_margin": oracle_hard,
                "oracle_ce": oracle_ce,
                "oracle_soft_margin": oracle_soft,
                "harmful_hard": oracle_hard > 0.0,
                "harmful_ce": oracle_ce < 0.0,  # suppressing reduces CE
                "harmful_soft": oracle_soft > 0.0,
                "in_rho05_band": rho >= HIGH_RHO_THRESHOLD,
                "in_power_band": (
                    SIGNED_BAND_THRESHOLD <= rho < HIGH_RHO_THRESHOLD
                    and mean_w_c > LOCAL_POWER_THRESHOLD
                ),
                "in_boundary_band": SIGNED_BAND_THRESHOLD <= rho < HIGH_RHO_THRESHOLD,
            }
            finite = (
                bool(torch.isfinite(model.weight).all())
                and bool(torch.isfinite(model.bias).all())
                and all(
                    math.isfinite(float(v)) for v in values_out.values() if not isinstance(v, bool)
                )
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
            "Mechanism discovery: for every type-pair group, measure the oracle "
            "hard-margin effect of gain3->1 suppression and compare local proxies. "
            "Oracle is diagnostic only and is never used as a learning rule."
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
