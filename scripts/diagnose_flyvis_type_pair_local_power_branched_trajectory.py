from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_biological_logit_balance import _heldout_readout, _metrics
from diagnose_flyvis_type_pair_local_power_holdout import (
    LOCAL_POWER_THRESHOLD,
    _candidate_direction,
)
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
EVAL_JITTER_BASE = 6_500_000
INIT_MODES = ("random", "biological_strength")


def _selection_geometry(
    raw_edge: torch.Tensor,
    weight: torch.Tensor,
    named_groups,
) -> dict[str, float]:
    ratios: list[float] = []
    local_scores: list[float] = []
    threshold_suppressed = 0
    local_power_added = 0

    for _, _, indices in named_groups:
        values = raw_edge[indices]
        weights = weight[indices]
        mean = values.mean()
        residual = values - mean
        rho = float(
            torch.linalg.vector_norm(mean.expand_as(values))
            / torch.linalg.vector_norm(residual).clamp_min(1e-30)
        )
        local_score = float((weights * values).mean())
        boundary = 0.35 <= rho < 0.5
        threshold_suppressed += int(rho >= 0.5)
        local_power_added += int(boundary and local_score > LOCAL_POWER_THRESHOLD)
        ratios.append(rho)
        local_scores.append(local_score)

    count = max(len(ratios), 1)
    ratio_tensor = torch.tensor(ratios)
    score_tensor = torch.tensor(local_scores)
    return {
        "mean_rho": float(ratio_tensor.mean()),
        "median_rho": float(ratio_tensor.median()),
        "rho_ge_05_fraction": float((ratio_tensor >= 0.5).float().mean()),
        "local_power_added_group_fraction": local_power_added / count,
        "local_power_suppressed_group_fraction": (threshold_suppressed + local_power_added)
        / count,
        "mean_local_power_score": float(score_tensor.mean()),
        "positive_local_power_score_fraction": float((score_tensor > 0.0).float().mean()),
    }


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
) -> None:
    raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
    if rule == "threshold":
        direction, _ = _threshold_direction(raw_edge, named_groups)
    elif rule == "local_power":
        direction, _ = _candidate_direction(
            raw_edge,
            model.weight.detach(),
            named_groups,
            learning_rate=learning_rate,
            max_update=max_update,
            mode="local_power",
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
        mode: {"threshold": copy.deepcopy(base), "local_power": copy.deepcopy(base)}
        for mode, base in bases.items()
    }
    rows: list[dict[str, float | int | bool | str]] = []
    completed = 0

    for branch_step in BRANCH_STEPS:
        for offset in range(completed, branch_step):
            epoch = BRANCH_HORIZON + offset
            for mode in INIT_MODES:
                for rule in ("threshold", "local_power"):
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
            threshold_model = branches[mode]["threshold"]
            power_model = branches[mode]["local_power"]
            geometry: dict[str, dict[str, float]] = {}
            for rule, model in (("threshold", threshold_model), ("local_power", power_model)):
                raw_edge, _ = credit(
                    model,
                    circuit,
                    seed=seed,
                    epoch=BRANCH_HORIZON + branch_step,
                )
                geometry[rule] = _selection_geometry(
                    raw_edge,
                    model.weight.detach(),
                    named_groups,
                )

            weight_distance = float(
                torch.linalg.vector_norm(power_model.weight - threshold_model.weight)
            )
            bias_distance = float(
                torch.linalg.vector_norm(power_model.bias - threshold_model.bias)
            )

            for eval_rep in range(EVAL_REPS):
                jitter_seed = EVAL_JITTER_BASE + eval_rep
                baseline_readout, classes = _heldout_readout(
                    bases[mode], circuit, jitter_seed=jitter_seed
                )
                threshold_readout, threshold_classes = _heldout_readout(
                    threshold_model, circuit, jitter_seed=jitter_seed
                )
                power_readout, power_classes = _heldout_readout(
                    power_model, circuit, jitter_seed=jitter_seed
                )
                if not torch.equal(classes, threshold_classes) or not torch.equal(
                    classes, power_classes
                ):
                    raise RuntimeError("held-out class mismatch")

                threshold_metrics = _metrics(
                    threshold_readout, classes, local_readout=baseline_readout
                )
                power_metrics = _metrics(
                    power_readout, classes, local_readout=baseline_readout
                )
                metric_names = ("cross_entropy", "accuracy", "hard_margin", "soft_margin")
                values = {
                    **{f"threshold_{key}": threshold_metrics[key] for key in metric_names},
                    **{f"local_power_{key}": power_metrics[key] for key in metric_names},
                    **{
                        f"local_power_minus_threshold_{key}": power_metrics[key]
                        - threshold_metrics[key]
                        for key in metric_names
                    },
                    "branch_weight_distance": weight_distance,
                    "branch_bias_distance": bias_distance,
                    **{
                        f"threshold_state_{key}": value
                        for key, value in geometry["threshold"].items()
                    },
                    **{
                        f"local_power_state_{key}": value
                        for key, value in geometry["local_power"].items()
                    },
                }
                finite = (
                    bool(torch.isfinite(threshold_model.weight).all())
                    and bool(torch.isfinite(threshold_model.bias).all())
                    and bool(torch.isfinite(power_model.weight).all())
                    and bool(torch.isfinite(power_model.bias).all())
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
            "Branch the locked mean(weight*credit) boundary gate and rho>=0.5 threshold "
            "rule from the same epoch-80 local checkpoint and measure 20-step accumulation."
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
