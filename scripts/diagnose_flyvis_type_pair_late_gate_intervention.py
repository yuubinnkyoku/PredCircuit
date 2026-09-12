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

PRETRAIN_EPOCHS = 180
BRANCH_STEPS = (0, 1, 5, 10, 20)
EVAL_REPS = 4
EVAL_JITTER_BASE = 10_300_000
BRANCHES = ("standard", "rho05", "local_power", "local")
METRICS = ("cross_entropy", "accuracy", "hard_margin", "soft_margin")


def _direction(
    branch: str,
    raw_edge: torch.Tensor,
    weight: torch.Tensor,
    named_groups,
    *,
    learning_rate: float,
    max_update: float,
) -> torch.Tensor:
    if branch == "standard":
        return _standard_direction(raw_edge, named_groups)
    if branch == "rho05":
        return _threshold_direction(raw_edge, named_groups)[0]
    if branch == "local_power":
        return _candidate_direction(
            raw_edge,
            weight,
            named_groups,
            learning_rate=learning_rate,
            max_update=max_update,
            mode="local_power",
        )[0]
    if branch == "local":
        return raw_edge
    raise ValueError(branch)


def _evaluate(
    *,
    seed: int,
    branch_step: int,
    models: dict[str, PredictiveCodingGraph],
    circuit,
) -> list[dict[str, float | int | bool | str]]:
    rows: list[dict[str, float | int | bool | str]] = []
    for eval_rep in range(EVAL_REPS):
        jitter_seed = EVAL_JITTER_BASE + eval_rep
        reference_readout, classes = _heldout_readout(
            models["standard"], circuit, jitter_seed=jitter_seed
        )
        outputs: dict[str, dict[str, float]] = {}
        for branch in BRANCHES:
            readout, paired_classes = _heldout_readout(
                models[branch], circuit, jitter_seed=jitter_seed
            )
            if not torch.equal(classes, paired_classes):
                raise RuntimeError("held-out class mismatch")
            outputs[branch] = _metrics(
                readout,
                classes,
                local_readout=reference_readout,
            )

        values: dict[str, float] = {}
        for branch in BRANCHES:
            for metric in METRICS:
                values[f"{branch}_{metric}"] = outputs[branch][metric]
            values[f"{branch}_weight_norm"] = float(torch.linalg.vector_norm(models[branch].weight))

        for branch in BRANCHES:
            if branch == "standard":
                continue
            for metric in METRICS:
                values[f"{branch}_minus_standard_{metric}"] = (
                    outputs[branch][metric] - outputs["standard"][metric]
                )
            values[f"{branch}_minus_standard_weight_norm"] = (
                values[f"{branch}_weight_norm"] - values["standard_weight_norm"]
            )

        finite = all(
            bool(torch.isfinite(models[branch].weight).all())
            and bool(torch.isfinite(models[branch].bias).all())
            for branch in BRANCHES
        ) and all(math.isfinite(value) for value in values.values())
        rows.append(
            {
                "seed": seed,
                "branch_step": branch_step,
                "eval_rep": eval_rep,
                **values,
                "finite": finite,
            }
        )
    return rows


def run_seed(*, seed: int, learning_rate: float, max_update: float) -> pd.DataFrame:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    named_groups = _named_groups(circuit)
    base = PredictiveCodingGraph(
        circuit.graph,
        seed=seed,
        init_scale=0.08,
        use_biological_strength=True,
    )

    # Keep the trajectory identical and ungated through epoch 180.
    for step in range(1, PRETRAIN_EPOCHS + 1):
        epoch = step - 1
        raw_edge, raw_bias = credit(base, circuit, seed=seed, epoch=epoch)
        apply_local_credit(
            base,
            _standard_direction(raw_edge, named_groups),
            raw_bias,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )

    models = {branch: copy.deepcopy(base) for branch in BRANCHES}
    rows = _evaluate(seed=seed, branch_step=0, models=models, circuit=circuit)

    for branch_step in range(1, max(BRANCH_STEPS) + 1):
        epoch = PRETRAIN_EPOCHS + branch_step - 1
        for branch in BRANCHES:
            model = models[branch]
            raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
            direction = _direction(
                branch,
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

        if branch_step in BRANCH_STEPS:
            rows.extend(
                _evaluate(
                    seed=seed,
                    branch_step=branch_step,
                    models=models,
                    circuit=circuit,
                )
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Causal late-phase intervention: pretrain biological-strength models with "
            "ungated 4m+r to epoch 180, then branch for 20 epochs into standard, rho05, "
            "local_power, or purely local credit."
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
