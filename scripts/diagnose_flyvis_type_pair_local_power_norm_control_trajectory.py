from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_biological_logit_balance import _heldout_readout, _metrics
from diagnose_flyvis_type_pair_local_power_holdout import _candidate_direction
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from diagnose_flyvis_type_pair_threshold_mean_gain_holdout import _threshold_direction
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

BRANCH_HORIZON = 80
BRANCH_STEPS = (0, 1, 5, 10, 20)
EVAL_REPS = 4
EVAL_JITTER_BASE = 6_700_000
INIT_MODES = ("random", "biological_strength")
RULES = ("threshold", "local_power", "delta_norm")


def _apply_rule_step(
    model: PredictiveCodingGraph,
    circuit,
    named_groups,
    *,
    rule: str,
    seed: int,
    epoch: int,
    learning_rate: float,
    max_update: float,
) -> dict[str, float]:
    raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
    if rule == "threshold":
        direction, diag = _threshold_direction(raw_edge, named_groups)
        diag = {"threshold_suppressed_group_fraction": diag["suppressed_group_fraction"]}
    else:
        direction, diag = _candidate_direction(
            raw_edge,
            model.weight.detach(),
            named_groups,
            learning_rate=learning_rate,
            max_update=max_update,
            mode=rule,
        )
    apply_local_credit(
        model,
        direction,
        raw_bias,
        learning_rate=learning_rate,
        weight_decay=0.0,
        max_update=max_update,
    )
    return diag


def _current_diag(
    model: PredictiveCodingGraph,
    circuit,
    named_groups,
    *,
    rule: str,
    seed: int,
    epoch: int,
    learning_rate: float,
    max_update: float,
) -> dict[str, float]:
    raw_edge, _ = credit(model, circuit, seed=seed, epoch=epoch)
    if rule == "threshold":
        _, diag = _threshold_direction(raw_edge, named_groups)
        return {"threshold_suppressed_group_fraction": diag["suppressed_group_fraction"]}
    _, diag = _candidate_direction(
        raw_edge,
        model.weight.detach(),
        named_groups,
        learning_rate=learning_rate,
        max_update=max_update,
        mode=rule,
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
    for epoch in range(BRANCH_HORIZON):
        for base in bases.values():
            raw_edge, raw_bias = credit(base, circuit, seed=seed, epoch=epoch)
            apply_local_credit(
                base,
                raw_edge,
                raw_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )

    branches = {
        mode: {rule: copy.deepcopy(base) for rule in RULES} for mode, base in bases.items()
    }
    rows: list[dict[str, float | int | bool | str]] = []
    completed = 0

    for branch_step in BRANCH_STEPS:
        for offset in range(completed, branch_step):
            epoch = BRANCH_HORIZON + offset
            for mode in INIT_MODES:
                for rule in RULES:
                    _apply_rule_step(
                        branches[mode][rule],
                        circuit,
                        named_groups,
                        rule=rule,
                        seed=seed,
                        epoch=epoch,
                        learning_rate=learning_rate,
                        max_update=max_update,
                    )
        completed = branch_step

        for mode in INIT_MODES:
            diagnostics = {
                rule: _current_diag(
                    branches[mode][rule],
                    circuit,
                    named_groups,
                    rule=rule,
                    seed=seed,
                    epoch=BRANCH_HORIZON + branch_step,
                    learning_rate=learning_rate,
                    max_update=max_update,
                )
                for rule in RULES
            }
            power_norm_distance = float(
                torch.linalg.vector_norm(
                    branches[mode]["local_power"].weight - branches[mode]["delta_norm"].weight
                )
            )

            for eval_rep in range(EVAL_REPS):
                jitter_seed = EVAL_JITTER_BASE + eval_rep
                baseline_readout, classes = _heldout_readout(
                    bases[mode], circuit, jitter_seed=jitter_seed
                )
                outputs: dict[str, dict[str, float]] = {}
                for rule in RULES:
                    readout, paired_classes = _heldout_readout(
                        branches[mode][rule], circuit, jitter_seed=jitter_seed
                    )
                    if not torch.equal(classes, paired_classes):
                        raise RuntimeError("held-out class mismatch")
                    outputs[rule] = _metrics(readout, classes, local_readout=baseline_readout)

                metric_names = ("cross_entropy", "accuracy", "hard_margin", "soft_margin")
                values: dict[str, float] = {}
                for rule in RULES:
                    for metric in metric_names:
                        values[f"{rule}_{metric}"] = outputs[rule][metric]
                for metric in metric_names:
                    values[f"local_power_minus_threshold_{metric}"] = (
                        outputs["local_power"][metric] - outputs["threshold"][metric]
                    )
                    values[f"delta_norm_minus_threshold_{metric}"] = (
                        outputs["delta_norm"][metric] - outputs["threshold"][metric]
                    )
                    values[f"local_power_minus_delta_norm_{metric}"] = (
                        outputs["local_power"][metric] - outputs["delta_norm"][metric]
                    )
                values["local_power_delta_norm_weight_distance"] = power_norm_distance
                for rule, diag in diagnostics.items():
                    for key, value in diag.items():
                        values[f"{rule}_state_{key}"] = float(value)

                finite = (
                    all(
                        bool(torch.isfinite(branches[mode][rule].weight).all())
                        and bool(torch.isfinite(branches[mode][rule].bias).all())
                        for rule in RULES
                    )
                    and all(math.isfinite(float(value)) for value in values.values())
                )
                rows.append(
                    {
                        "seed": seed,
                        "init_mode": mode,
                        "branch_horizon": BRANCH_HORIZON,
                        "branch_step": branch_step,
                        "eval_rep": eval_rep,
                        **values,
                        "finite": finite,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Twenty-step paired trajectory of the locked local-power boundary gate against "
            "the rho>=0.5 threshold rule and an update-norm-only boundary control."
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
