from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_cross_curvature import _heldout_ce
from diagnose_flyvis_type_pair_horizon_causal import _match_local_state, _mixed_direction
from diagnose_flyvis_type_pair_state_geometry import _group_constant_projection
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit
from run_flyvis_type_pair_consensus import type_pair_indices

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

HORIZONS = (60, 80, 100)
EVAL_REPS = 2
EVAL_JITTER_BASE = 2_600_000


def run_seed(
    *,
    seed: int,
    learning_rate: float,
    max_update: float,
) -> pd.DataFrame:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    base = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    local = copy.deepcopy(base)
    type_ = copy.deepcopy(base)
    groups = type_pair_indices(circuit)
    rows: list[dict[str, float | int | bool]] = []

    for step in range(1, max(HORIZONS) + 1):
        epoch = step - 1
        local_edge, local_bias = credit(local, circuit, seed=seed, epoch=epoch)
        apply_local_credit(
            local,
            local_edge,
            local_bias,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )

        raw_edge, raw_bias = credit(type_, circuit, seed=seed, epoch=epoch)
        direction = _mixed_direction(raw_edge, groups)
        apply_local_credit(
            type_,
            direction,
            raw_bias,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )
        _match_local_state(type_, local)

        if step not in HORIZONS:
            continue

        local_weight = local.weight.detach()
        delta = type_.weight.detach() - local_weight
        projection = _group_constant_projection(delta, groups).detach()
        residual = (delta - projection).detach()
        bias = local.bias.detach().clone()

        for eval_rep in range(EVAL_REPS):
            jitter_seed = EVAL_JITTER_BASE + eval_rep
            weight = local_weight.clone().requires_grad_(True)
            ce00 = _heldout_ce(circuit, weight, bias, jitter_seed=jitter_seed)
            (gradient,) = torch.autograd.grad(ce00, weight, create_graph=True)
            grad_residual = torch.dot(gradient, residual)
            residual_hvp = torch.autograd.grad(grad_residual, weight)[0]
            cross_curvature = torch.dot(projection, residual_hvp)
            predicted_synergy = -cross_curvature

            with torch.no_grad():
                ce10 = _heldout_ce(
                    circuit,
                    local_weight + projection,
                    bias,
                    jitter_seed=jitter_seed,
                )
                ce01 = _heldout_ce(
                    circuit,
                    local_weight + residual,
                    bias,
                    jitter_seed=jitter_seed,
                )
                ce11 = _heldout_ce(
                    circuit,
                    local_weight + projection + residual,
                    bias,
                    jitter_seed=jitter_seed,
                )
            observed_synergy = ce10 + ce01 - ce11 - ce00.detach()
            prediction_error = predicted_synergy.detach() - observed_synergy
            values = {
                "ce_local": float(ce00.detach()),
                "ce_projection": float(ce10),
                "ce_residual": float(ce01),
                "ce_type": float(ce11),
                "predicted_synergy": float(predicted_synergy.detach()),
                "observed_synergy": float(observed_synergy),
                "prediction_error": float(prediction_error),
            }
            finite = (
                bool(torch.isfinite(local.weight).all())
                and bool(torch.isfinite(type_.weight).all())
                and all(math.isfinite(value) for value in values.values())
            )
            rows.append(
                {
                    "seed": seed,
                    "horizon": step,
                    "eval_rep": eval_rep,
                    **values,
                    "finite": finite,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate whether the held-out CE Hessian projection-residual cross term "
            "predicts the exact finite type-pair state interaction"
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frame = run_seed(
        seed=args.seed,
        learning_rate=args.learning_rate,
        max_update=args.max_update,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
