from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_adaptive_mean_gain_holdout import _standard_direction
from diagnose_flyvis_type_pair_biological_logit_balance import _heldout_readout, _metrics
from diagnose_flyvis_type_pair_local_power_holdout import _candidate_direction
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from diagnose_flyvis_type_pair_threshold_mean_gain_holdout import _threshold_direction
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

HORIZONS = (20, 40, 60, 80, 100)
EVAL_REPS = 4
EVAL_JITTER_BASE = 8_000_000
INIT_MODES = ("random", "biological_strength")
RULES = ("local", "standard", "rho05", "local_power", "delta_norm")
LOCKED_RHO_HIGH = 0.5
LOCKED_RHO_LOW = 0.35
LOCKED_POWER = 5e-8
LOCKED_DELTA_NORM = 0.002


def _apply_rule(
    model: PredictiveCodingGraph,
    raw_edge: torch.Tensor,
    raw_bias: torch.Tensor,
    named_groups,
    *,
    rule: str,
    learning_rate: float,
    max_update: float,
) -> dict[str, float]:
    if rule == "local":
        direction = raw_edge
        diag: dict[str, float] = {}
    elif rule == "standard":
        direction = _standard_direction(raw_edge, named_groups)
        diag = {}
    elif rule == "rho05":
        direction, diag = _threshold_direction(raw_edge, named_groups)
    elif rule == "local_power":
        direction, diag = _candidate_direction(
            raw_edge,
            model.weight.detach(),
            named_groups,
            learning_rate=learning_rate,
            max_update=max_update,
            mode="local_power",
        )
    elif rule == "delta_norm":
        direction, diag = _candidate_direction(
            raw_edge,
            model.weight.detach(),
            named_groups,
            learning_rate=learning_rate,
            max_update=max_update,
            mode="delta_norm",
        )
    else:
        raise ValueError(f"unknown rule: {rule}")
    apply_local_credit(
        model,
        direction,
        raw_bias,
        learning_rate=learning_rate,
        weight_decay=0.0,
        max_update=max_update,
    )
    return diag


def run_seed(*, seed: int, learning_rate: float, max_update: float) -> pd.DataFrame:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    named_groups = _named_groups(circuit)
    bases = {
        "random": PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08),
        "biological_strength": PredictiveCodingGraph(
            circuit.graph,
            seed=seed,
            init_scale=0.08,
            use_biological_strength=True,
        ),
    }
    models = {mode: {rule: copy.deepcopy(base) for rule in RULES} for mode, base in bases.items()}
    diag_sum = {
        mode: {
            rule: {
                f"{rule}_suppressed_group_fraction": 0.0,
                f"{rule}_added_group_fraction": 0.0,
            }
            for rule in ("rho05", "local_power", "delta_norm")
        }
        for mode in INIT_MODES
    }
    # rho05 uses different diagnostic keys from _threshold_direction
    for mode in INIT_MODES:
        diag_sum[mode]["rho05"] = {
            "threshold_suppressed_group_fraction": 0.0,
            "threshold_suppressed_edge_fraction": 0.0,
            "threshold_mean_mean_to_residual_norm": 0.0,
        }

    rows: list[dict[str, float | int | bool | str]] = []

    for step in range(1, max(HORIZONS) + 1):
        epoch = step - 1
        for mode in INIT_MODES:
            for rule in RULES:
                model = models[mode][rule]
                raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
                diag = _apply_rule(
                    model,
                    raw_edge,
                    raw_bias,
                    named_groups,
                    rule=rule,
                    learning_rate=learning_rate,
                    max_update=max_update,
                )
                if rule in diag_sum[mode]:
                    for key, value in diag.items():
                        diag_sum[mode][rule][key] = (
                            diag_sum[mode][rule].get(key, 0.0) * (step - 1) + value
                        ) / step

        if step not in HORIZONS:
            continue

        for mode in INIT_MODES:
            for eval_rep in range(EVAL_REPS):
                jitter_seed = EVAL_JITTER_BASE + eval_rep
                local_readout, classes = _heldout_readout(
                    models[mode]["local"], circuit, jitter_seed=jitter_seed
                )
                outputs: dict[str, dict[str, float]] = {}
                for rule in RULES:
                    model = models[mode][rule]
                    if rule == "local":
                        readout = local_readout
                    else:
                        readout, paired_classes = _heldout_readout(
                            model, circuit, jitter_seed=jitter_seed
                        )
                        if not torch.equal(classes, paired_classes):
                            raise RuntimeError("held-out class mismatch")
                    outputs[rule] = _metrics(readout, classes, local_readout=local_readout)

                metric_names = ("cross_entropy", "accuracy", "hard_margin", "soft_margin")
                values: dict[str, float] = {}
                for rule in RULES:
                    for metric in metric_names:
                        values[f"{rule}_{metric}"] = outputs[rule][metric]
                for candidate in ("standard", "rho05", "local_power", "delta_norm"):
                    for metric in metric_names:
                        values[f"{candidate}_minus_local_{metric}"] = (
                            outputs[candidate][metric] - outputs["local"][metric]
                        )
                for metric in metric_names:
                    values[f"local_power_minus_rho05_{metric}"] = (
                        outputs["local_power"][metric] - outputs["rho05"][metric]
                    )
                    values[f"local_power_minus_delta_norm_{metric}"] = (
                        outputs["local_power"][metric] - outputs["delta_norm"][metric]
                    )
                    values[f"rho05_minus_standard_{metric}"] = (
                        outputs["rho05"][metric] - outputs["standard"][metric]
                    )

                for rule, diags in diag_sum[mode].items():
                    for key, value in diags.items():
                        values[f"cum_{key}"] = float(value)

                finite = all(
                    bool(torch.isfinite(models[mode][rule].weight).all())
                    and bool(torch.isfinite(models[mode][rule].bias).all())
                    for rule in RULES
                ) and all(math.isfinite(float(value)) for value in values.values())
                rows.append(
                    {
                        "seed": seed,
                        "init_mode": mode,
                        "horizon": step,
                        "eval_rep": eval_rep,
                        "locked_rho_high": LOCKED_RHO_HIGH,
                        "locked_rho_low": LOCKED_RHO_LOW,
                        "locked_power": LOCKED_POWER,
                        "locked_delta_norm": LOCKED_DELTA_NORM,
                        **values,
                        "finite": finite,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Full-horizon epoch-0-to-100 paired hold-out of local, standard 4m+r, "
            "rho05, local_power, and delta_norm rules with locked thresholds."
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
