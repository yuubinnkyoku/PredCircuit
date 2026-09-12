from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_biological_logit_balance import _heldout_readout, _metrics
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from diagnose_flyvis_type_pair_signed_band_holdout import (
    HIGH_EXTRA_GAIN,
    HIGH_RHO_THRESHOLD,
    LOW_EXTRA_GAIN,
    SIGNED_BAND_THRESHOLD,
)
from diagnose_flyvis_type_pair_threshold_mean_gain_holdout import _threshold_direction
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

HORIZONS = (80, 100)
EVAL_REPS = 4
EVAL_JITTER_BASE = 6_300_000
LOCAL_POWER_THRESHOLD = 5e-8
DELTA_NORM_THRESHOLD = 0.002


def _clipped(values: torch.Tensor, learning_rate: float, max_update: float) -> torch.Tensor:
    update = learning_rate * values
    return update.clamp(-max_update, max_update) if max_update > 0.0 else update


def _candidate_direction(
    raw_edge: torch.Tensor,
    weight: torch.Tensor,
    named_groups,
    *,
    learning_rate: float,
    max_update: float,
    mode: str,
):
    result = raw_edge.clone()
    added_groups = 0
    suppressed_groups = 0
    for _, _, indices in named_groups:
        values = raw_edge[indices]
        weights = weight[indices]
        mean = values.mean()
        residual = values - mean
        rho = float(
            torch.linalg.vector_norm(mean.expand_as(values))
            / torch.linalg.vector_norm(residual).clamp_min(1e-30)
        )
        boundary = SIGNED_BAND_THRESHOLD <= rho < HIGH_RHO_THRESHOLD
        if mode == "local_power":
            local_score = float((weights * values).mean())
            added = boundary and local_score > LOCAL_POWER_THRESHOLD
        elif mode == "delta_norm":
            high_direction = values + HIGH_EXTRA_GAIN * mean
            low_direction = values + LOW_EXTRA_GAIN * mean
            delta_norm = float(
                torch.linalg.vector_norm(
                    _clipped(low_direction, learning_rate, max_update)
                    - _clipped(high_direction, learning_rate, max_update)
                )
            )
            added = boundary and delta_norm > DELTA_NORM_THRESHOLD
        else:
            raise ValueError(f"unknown mode: {mode}")
        suppress = rho >= HIGH_RHO_THRESHOLD or added
        extra_gain = LOW_EXTRA_GAIN if suppress else HIGH_EXTRA_GAIN
        result[indices] = values + extra_gain * mean
        suppressed_groups += int(suppress)
        added_groups += int(added)
    group_count = max(len(named_groups), 1)
    return result, {
        f"{mode}_suppressed_group_fraction": suppressed_groups / group_count,
        f"{mode}_added_group_fraction": added_groups / group_count,
    }


def _apply(model, direction, bias, learning_rate: float, max_update: float):
    apply_local_credit(
        model,
        direction,
        bias,
        learning_rate=learning_rate,
        weight_decay=0.0,
        max_update=max_update,
    )


def run_seed(*, seed: int, learning_rate: float, max_update: float) -> pd.DataFrame:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    named_groups = _named_groups(circuit)
    base = PredictiveCodingGraph(
        circuit.graph,
        seed=seed,
        init_scale=0.08,
        use_biological_strength=True,
    )
    rows = []
    trained = 0
    for horizon in HORIZONS:
        for epoch in range(trained, horizon):
            raw_edge, raw_bias = credit(base, circuit, seed=seed, epoch=epoch)
            _apply(base, raw_edge, raw_bias, learning_rate, max_update)
        trained = horizon

        raw_edge, raw_bias = credit(base, circuit, seed=seed, epoch=horizon)
        threshold_direction, _ = _threshold_direction(raw_edge, named_groups)
        power_direction, power_diag = _candidate_direction(
            raw_edge,
            base.weight.detach(),
            named_groups,
            learning_rate=learning_rate,
            max_update=max_update,
            mode="local_power",
        )
        norm_direction, norm_diag = _candidate_direction(
            raw_edge,
            base.weight.detach(),
            named_groups,
            learning_rate=learning_rate,
            max_update=max_update,
            mode="delta_norm",
        )

        models = {
            "threshold": copy.deepcopy(base),
            "local_power": copy.deepcopy(base),
            "delta_norm": copy.deepcopy(base),
        }
        _apply(models["threshold"], threshold_direction, raw_bias, learning_rate, max_update)
        _apply(models["local_power"], power_direction, raw_bias, learning_rate, max_update)
        _apply(models["delta_norm"], norm_direction, raw_bias, learning_rate, max_update)

        for eval_rep in range(EVAL_REPS):
            jitter_seed = EVAL_JITTER_BASE + eval_rep
            base_readout, classes = _heldout_readout(base, circuit, jitter_seed=jitter_seed)
            outputs = {}
            for name, model in models.items():
                readout, paired_classes = _heldout_readout(
                    model,
                    circuit,
                    jitter_seed=jitter_seed,
                )
                if not torch.equal(classes, paired_classes):
                    raise RuntimeError("held-out class mismatch")
                outputs[name] = _metrics(readout, classes, local_readout=base_readout)

            values = {}
            for candidate in ("local_power", "delta_norm"):
                for metric in ("cross_entropy", "accuracy", "hard_margin", "soft_margin"):
                    values[f"{candidate}_minus_threshold_{metric}"] = (
                        outputs[candidate][metric] - outputs["threshold"][metric]
                    )
            finite = (
                all(torch.isfinite(model.weight).all() for model in models.values())
                and all(torch.isfinite(model.bias).all() for model in models.values())
                and all(math.isfinite(float(value)) for value in values.values())
            )
            rows.append(
                {
                    "seed": seed,
                    "horizon": horizon,
                    "eval_rep": eval_rep,
                    **values,
                    **power_diag,
                    **norm_diag,
                    "finite": finite,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fresh one-step holdout of a locked boundary gate using mean(weight*credit) "
            "against the rho>=0.5 threshold rule and an update-norm-only control."
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
