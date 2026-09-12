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
EVAL_JITTER_BASE = 6_900_000
INIT_MODES = ("random", "biological_strength")
RULES = ("local", "standard", "threshold", "local_power", "delta_norm")
METRICS = ("cross_entropy", "accuracy", "hard_margin", "soft_margin")
COMPARISONS = (
    ("standard", "local"),
    ("threshold", "standard"),
    ("threshold", "local"),
    ("local_power", "threshold"),
    ("delta_norm", "threshold"),
    ("local_power", "delta_norm"),
)


def _direction_for_rule(
    rule: str,
    raw_edge: torch.Tensor,
    weight: torch.Tensor,
    named_groups,
    *,
    learning_rate: float,
    max_update: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if rule == "local":
        return raw_edge, {}
    if rule == "standard":
        return _standard_direction(raw_edge, named_groups), {}
    if rule == "threshold":
        return _threshold_direction(raw_edge, named_groups)
    if rule in ("local_power", "delta_norm"):
        return _candidate_direction(
            raw_edge,
            weight,
            named_groups,
            learning_rate=learning_rate,
            max_update=max_update,
            mode=rule,
        )
    raise ValueError(f"unknown rule: {rule}")


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
    diag_sums: dict[str, dict[str, dict[str, float]]] = {
        mode: {rule: {} for rule in RULES} for mode in INIT_MODES
    }
    rows: list[dict[str, float | int | bool | str]] = []

    for step in range(1, max(HORIZONS) + 1):
        epoch = step - 1
        for mode in INIT_MODES:
            for rule in RULES:
                model = models[mode][rule]
                raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
                direction, diagnostics = _direction_for_rule(
                    rule,
                    raw_edge,
                    model.weight.detach(),
                    named_groups,
                    learning_rate=learning_rate,
                    max_update=max_update,
                )
                apply_local_credit(
                    model,
                    direction,
                    raw_bias,
                    learning_rate=learning_rate,
                    weight_decay=0.0,
                    max_update=max_update,
                )
                for key, value in diagnostics.items():
                    diag_sums[mode][rule][key] = diag_sums[mode][rule].get(key, 0.0) + float(value)

        if step not in HORIZONS:
            continue

        for mode in INIT_MODES:
            local = models[mode]["local"]
            diagnostics = {
                f"{rule}_mean_{key}": value / step
                for rule in RULES
                for key, value in diag_sums[mode][rule].items()
            }
            for eval_rep in range(EVAL_REPS):
                jitter_seed = EVAL_JITTER_BASE + eval_rep
                local_readout, classes = _heldout_readout(local, circuit, jitter_seed=jitter_seed)
                outputs: dict[str, dict[str, float]] = {
                    "local": _metrics(local_readout, classes, local_readout=local_readout)
                }
                for rule in RULES:
                    if rule == "local":
                        continue
                    readout, paired_classes = _heldout_readout(
                        models[mode][rule], circuit, jitter_seed=jitter_seed
                    )
                    if not torch.equal(classes, paired_classes):
                        raise RuntimeError("held-out class mismatch")
                    outputs[rule] = _metrics(readout, classes, local_readout=local_readout)

                values: dict[str, float] = {}
                for rule in RULES:
                    for metric in METRICS:
                        values[f"{rule}_{metric}"] = outputs[rule][metric]
                for left, right in COMPARISONS:
                    for metric in METRICS:
                        values[f"{left}_minus_{right}_{metric}"] = (
                            outputs[left][metric] - outputs[right][metric]
                        )
                values.update(diagnostics)
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
                        **values,
                        "finite": finite,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fresh epoch-0-to-100 paired holdout of local m+r, fixed 4m+r, rho>=0.5, "
            "locked mean(weight*credit), and update-norm type-pair shared-credit rules."
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
