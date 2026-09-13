from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_biological_logit_balance import _heldout_readout, _metrics
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

PRETRAIN_EPOCHS = 160
BRANCH_STEPS = (0, 5, 10, 20, 40)
EVAL_REPS = 4
EVAL_JITTER_BASE = 11_700_000
INIT_SCALES = (0.12, 0.14, 0.16)
SHARED_COEFFICIENT = 2.5
OUTWARD_GAINS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
BRANCHES = tuple(f"out{round(gain * 100):03d}" for gain in OUTWARD_GAINS) + ("local",)
METRICS = ("cross_entropy", "accuracy", "hard_margin", "soft_margin")


def _coefficient_direction(raw_edge: torch.Tensor, named_groups) -> torch.Tensor:
    result = raw_edge.clone()
    extra_gain = SHARED_COEFFICIENT - 1.0
    for _, _, indices in named_groups:
        values = raw_edge[indices]
        result[indices] = values + extra_gain * values.mean()
    return result


def _outward_gain_direction(
    raw_edge: torch.Tensor,
    weight: torch.Tensor,
    named_groups,
    *,
    outward_gain: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    result = raw_edge.clone()
    extra_gain = SHARED_COEFFICIENT - 1.0
    outward_groups = 0
    extra_norm = 0.0
    outward_norm = 0.0
    selected_norm = 0.0

    for _, _, indices in named_groups:
        values = raw_edge[indices]
        group_weight = weight[indices]
        extra = torch.ones_like(values) * (extra_gain * values.mean())
        extra_norm += float(torch.linalg.vector_norm(extra))
        weight_norm_sq = torch.dot(group_weight, group_weight)

        if float(weight_norm_sq) <= 1e-30:
            radial = torch.zeros_like(extra)
            tangent = extra
        else:
            coefficient = torch.dot(extra, group_weight) / weight_norm_sq
            radial = coefficient * group_weight
            tangent = extra - radial

        if float(torch.dot(radial, group_weight)) > 0.0:
            outward = radial
            inward = torch.zeros_like(radial)
            outward_groups += 1
        else:
            outward = torch.zeros_like(radial)
            inward = radial

        selected = tangent + inward + outward_gain * outward
        result[indices] = values + selected
        outward_norm += float(torch.linalg.vector_norm(outward))
        selected_norm += float(torch.linalg.vector_norm(selected))

    denominator = max(extra_norm, 1e-30)
    group_count = max(len(named_groups), 1)
    return result, {
        "outward_group_fraction": outward_groups / group_count,
        "outward_to_extra_norm": outward_norm / denominator,
        "selected_to_extra_norm": selected_norm / denominator,
    }


def _evaluate(
    *,
    seed: int,
    init_scale: float,
    branch_step: int,
    models: dict[str, PredictiveCodingGraph],
    circuit,
    diagnostic_sums: dict[str, dict[str, float]],
) -> list[dict[str, float | int | bool | str]]:
    rows: list[dict[str, float | int | bool | str]] = []
    for eval_rep in range(EVAL_REPS):
        jitter_seed = EVAL_JITTER_BASE + eval_rep
        reference_readout, classes = _heldout_readout(
            models["out100"], circuit, jitter_seed=jitter_seed
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
            for baseline in ("out000", "out100", "local"):
                if branch == baseline:
                    continue
                for metric in METRICS:
                    values[f"{branch}_minus_{baseline}_{metric}"] = (
                        outputs[branch][metric] - outputs[baseline][metric]
                    )
                values[f"{branch}_minus_{baseline}_weight_norm"] = (
                    values[f"{branch}_weight_norm"] - values[f"{baseline}_weight_norm"]
                )

        finite = all(
            bool(torch.isfinite(models[branch].weight).all())
            and bool(torch.isfinite(models[branch].bias).all())
            for branch in BRANCHES
        ) and all(math.isfinite(value) for value in values.values())
        row: dict[str, float | int | bool | str] = {
            "seed": seed,
            "init_scale": init_scale,
            "branch_step": branch_step,
            "eval_rep": eval_rep,
            **values,
            "finite": finite,
        }
        denominator = max(branch_step, 1)
        for branch, sums in diagnostic_sums.items():
            for key, total in sums.items():
                row[f"{branch}_{key}"] = total / denominator
        rows.append(row)
    return rows


def run_seed(*, seed: int, learning_rate: float, max_update: float) -> pd.DataFrame:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    named_groups = _named_groups(circuit)
    rows: list[dict[str, float | int | bool | str]] = []

    gain_by_branch = {f"out{round(gain * 100):03d}": gain for gain in OUTWARD_GAINS}

    for init_scale in INIT_SCALES:
        base = PredictiveCodingGraph(
            circuit.graph,
            seed=seed,
            init_scale=init_scale,
            use_biological_strength=True,
        )
        for step in range(1, PRETRAIN_EPOCHS + 1):
            epoch = step - 1
            raw_edge, raw_bias = credit(base, circuit, seed=seed, epoch=epoch)
            apply_local_credit(
                base,
                _coefficient_direction(raw_edge, named_groups),
                raw_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )

        models = {branch: copy.deepcopy(base) for branch in BRANCHES}
        diagnostic_sums = {
            branch: {
                "outward_group_fraction": 0.0,
                "outward_to_extra_norm": 0.0,
                "selected_to_extra_norm": 0.0,
            }
            for branch in gain_by_branch
        }
        rows.extend(
            _evaluate(
                seed=seed,
                init_scale=init_scale,
                branch_step=0,
                models=models,
                circuit=circuit,
                diagnostic_sums=diagnostic_sums,
            )
        )

        for branch_step in range(1, max(BRANCH_STEPS) + 1):
            epoch = PRETRAIN_EPOCHS + branch_step - 1
            for branch in BRANCHES:
                model = models[branch]
                raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
                if branch == "local":
                    direction = raw_edge
                    diagnostics: dict[str, float] = {}
                else:
                    direction, diagnostics = _outward_gain_direction(
                        raw_edge,
                        model.weight.detach(),
                        named_groups,
                        outward_gain=gain_by_branch[branch],
                    )
                apply_local_credit(
                    model,
                    direction,
                    raw_bias,
                    learning_rate=learning_rate,
                    weight_decay=0.0,
                    max_update=max_update,
                )
                if branch in diagnostic_sums:
                    for key, value in diagnostics.items():
                        diagnostic_sums[branch][key] += value

            if branch_step in BRANCH_STEPS:
                rows.extend(
                    _evaluate(
                        seed=seed,
                        init_scale=init_scale,
                        branch_step=branch_step,
                        models=models,
                        circuit=circuit,
                        diagnostic_sums=diagnostic_sums,
                    )
                )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "State-matched dose response for the outward radial component of extra "
            "type-pair shared credit."
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
