from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_biological_logit_balance import _heldout_readout, _metrics
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from diagnose_flyvis_type_pair_signed_band_holdout import _signed_band_direction
from diagnose_flyvis_type_pair_signed_band_matched_control import _rho04_direction
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

BRANCH_HORIZON = 80
BRANCH_STEPS = (0, 1, 2, 5, 10, 20)
EVAL_REPS = 4
EVAL_JITTER_BASE = 5_700_000
INIT_MODES = ("random", "biological_strength")


def _selection_geometry(
    raw_edge: torch.Tensor,
    weight: torch.Tensor,
    named_groups,
) -> dict[str, float]:
    ratios: list[float] = []
    signed_only = 0
    rho04_only = 0
    common = 0
    signed_suppressed = 0
    rho04_suppressed = 0

    for _, _, indices in named_groups:
        values = raw_edge[indices]
        weights = weight[indices]
        mean = values.mean()
        mean_component = mean.expand_as(values)
        residual = values - mean
        ratio = float(
            torch.linalg.vector_norm(mean_component)
            / torch.linalg.vector_norm(residual).clamp_min(1e-30)
        )
        proxy = float(weights.mean() * mean)
        signed = ratio >= 0.5 or (ratio >= 0.35 and ratio < 0.5 and proxy > 0.0)
        rho04 = ratio >= 0.4
        ratios.append(ratio)
        signed_suppressed += int(signed)
        rho04_suppressed += int(rho04)
        signed_only += int(signed and not rho04)
        rho04_only += int(rho04 and not signed)
        common += int(signed and rho04)

    count = max(len(ratios), 1)
    ratio_tensor = torch.tensor(ratios)
    return {
        "mean_rho": float(ratio_tensor.mean()),
        "median_rho": float(ratio_tensor.median()),
        "rho_ge_05_fraction": float((ratio_tensor >= 0.5).float().mean()),
        "signed_suppressed_group_fraction": float(signed_suppressed / count),
        "rho04_suppressed_group_fraction": float(rho04_suppressed / count),
        "signed_only_group_fraction": float(signed_only / count),
        "rho04_only_group_fraction": float(rho04_only / count),
        "common_suppressed_group_fraction": float(common / count),
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
    if rule == "rho04":
        direction, _ = _rho04_direction(raw_edge, named_groups)
    elif rule == "signed_band":
        direction, _ = _signed_band_direction(raw_edge, model.weight.detach(), named_groups)
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
        mode: {"rho04": copy.deepcopy(base), "signed_band": copy.deepcopy(base)}
        for mode, base in bases.items()
    }
    rows: list[dict[str, float | int | bool | str]] = []
    completed = 0

    for branch_step in BRANCH_STEPS:
        for offset in range(completed, branch_step):
            epoch = BRANCH_HORIZON + offset
            for mode in INIT_MODES:
                for rule in ("rho04", "signed_band"):
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
            rho_model = branches[mode]["rho04"]
            signed_model = branches[mode]["signed_band"]
            geometry: dict[str, dict[str, float]] = {}
            for rule, model in (("rho04", rho_model), ("signed", signed_model)):
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
                torch.linalg.vector_norm(signed_model.weight - rho_model.weight)
            )
            bias_distance = float(torch.linalg.vector_norm(signed_model.bias - rho_model.bias))

            for eval_rep in range(EVAL_REPS):
                jitter_seed = EVAL_JITTER_BASE + eval_rep
                baseline_readout, classes = _heldout_readout(
                    bases[mode],
                    circuit,
                    jitter_seed=jitter_seed,
                )
                rho_readout, rho_classes = _heldout_readout(
                    rho_model,
                    circuit,
                    jitter_seed=jitter_seed,
                )
                signed_readout, signed_classes = _heldout_readout(
                    signed_model,
                    circuit,
                    jitter_seed=jitter_seed,
                )
                if not torch.equal(classes, rho_classes) or not torch.equal(classes, signed_classes):
                    raise RuntimeError("held-out class mismatch")
                rho_metrics = _metrics(rho_readout, classes, local_readout=baseline_readout)
                signed_metrics = _metrics(signed_readout, classes, local_readout=baseline_readout)
                metric_names = ("cross_entropy", "accuracy", "hard_margin", "soft_margin")
                values = {
                    **{f"rho04_{key}": rho_metrics[key] for key in metric_names},
                    **{f"signed_{key}": signed_metrics[key] for key in metric_names},
                    **{
                        f"signed_minus_rho04_{key}": signed_metrics[key] - rho_metrics[key]
                        for key in metric_names
                    },
                    "branch_weight_distance": weight_distance,
                    "branch_bias_distance": bias_distance,
                    **{f"rho04_state_{key}": value for key, value in geometry["rho04"].items()},
                    **{f"signed_state_{key}": value for key, value in geometry["signed"].items()},
                }
                finite = (
                    bool(torch.isfinite(rho_model.weight).all())
                    and bool(torch.isfinite(rho_model.bias).all())
                    and bool(torch.isfinite(signed_model.weight).all())
                    and bool(torch.isfinite(signed_model.bias).all())
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
            "Branch rho>=0.4 and signed-band rules from the same epoch-80 local checkpoint "
            "and measure whether their held-out hard-margin difference amplifies over 20 steps."
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
