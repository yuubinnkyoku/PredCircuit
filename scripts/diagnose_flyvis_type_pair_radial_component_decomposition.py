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
EVAL_JITTER_BASE = 11_500_000
INIT_SCALES = (0.12, 0.14, 0.16)
SHARED_COEFFICIENT = 2.5
BRANCHES = (
    "fixed25",
    "no_outward25",
    "tangent25",
    "inward_only25",
    "outward_only25",
    "local",
)
METRICS = ("cross_entropy", "accuracy", "hard_margin", "soft_margin")


def _coefficient_direction(
    raw_edge: torch.Tensor,
    named_groups,
    *,
    coefficient: float,
) -> torch.Tensor:
    result = raw_edge.clone()
    extra_gain = coefficient - 1.0
    for _, _, indices in named_groups:
        values = raw_edge[indices]
        result[indices] = values + extra_gain * values.mean()
    return result


def _component_direction(
    branch: str,
    raw_edge: torch.Tensor,
    weight: torch.Tensor,
    named_groups,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Select radial/tangential pieces of the extra shared-credit component."""
    if branch == "local":
        return raw_edge, {}
    if branch == "fixed25":
        return (
            _coefficient_direction(
                raw_edge,
                named_groups,
                coefficient=SHARED_COEFFICIENT,
            ),
            {},
        )

    result = raw_edge.clone()
    extra_gain = SHARED_COEFFICIENT - 1.0
    outward_groups = 0
    inward_groups = 0
    extra_norm = 0.0
    outward_norm = 0.0
    inward_norm = 0.0
    tangent_norm = 0.0

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

        radial_dot = float(torch.dot(radial, group_weight))
        if radial_dot > 0.0:
            outward = radial
            inward = torch.zeros_like(radial)
            outward_groups += 1
        elif radial_dot < 0.0:
            outward = torch.zeros_like(radial)
            inward = radial
            inward_groups += 1
        else:
            outward = torch.zeros_like(radial)
            inward = torch.zeros_like(radial)

        outward_norm += float(torch.linalg.vector_norm(outward))
        inward_norm += float(torch.linalg.vector_norm(inward))
        tangent_norm += float(torch.linalg.vector_norm(tangent))

        if branch == "no_outward25":
            selected = tangent + inward
        elif branch == "tangent25":
            selected = tangent
        elif branch == "inward_only25":
            selected = inward
        elif branch == "outward_only25":
            selected = outward
        else:
            raise ValueError(branch)
        result[indices] = values + selected

    denominator = max(extra_norm, 1e-30)
    group_count = max(len(named_groups), 1)
    return result, {
        "component_outward_group_fraction": outward_groups / group_count,
        "component_inward_group_fraction": inward_groups / group_count,
        "component_outward_to_extra_norm": outward_norm / denominator,
        "component_inward_to_extra_norm": inward_norm / denominator,
        "component_tangent_to_extra_norm": tangent_norm / denominator,
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
            models["fixed25"], circuit, jitter_seed=jitter_seed
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
            for baseline in ("fixed25", "local"):
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
            direction = _coefficient_direction(
                raw_edge,
                named_groups,
                coefficient=SHARED_COEFFICIENT,
            )
            apply_local_credit(
                base,
                direction,
                raw_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )

        models = {branch: copy.deepcopy(base) for branch in BRANCHES}
        diagnostic_branches = BRANCHES[1:-1]
        diagnostic_sums = {
            branch: {
                "component_outward_group_fraction": 0.0,
                "component_inward_group_fraction": 0.0,
                "component_outward_to_extra_norm": 0.0,
                "component_inward_to_extra_norm": 0.0,
                "component_tangent_to_extra_norm": 0.0,
            }
            for branch in diagnostic_branches
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
                direction, diagnostics = _component_direction(
                    branch,
                    raw_edge,
                    model.weight.detach(),
                    named_groups,
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
            "State-matched decomposition of extra shared credit into outward radial, "
            "inward radial, and tangential components."
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
