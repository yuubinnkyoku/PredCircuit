from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_adaptive_mean_gain_holdout import _standard_direction
from diagnose_flyvis_type_pair_biological_logit_balance import _heldout_readout
from diagnose_flyvis_type_pair_l1_gain_geometry import _heldout_objectives
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from diagnose_flyvis_type_pair_peaknorm_expectation_match_holdout import (
    _fractional_candidate_update,
)
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

INIT_SCALES = (0.08, 0.10, 0.11, 0.115)
HORIZONS = (160, 180, 200)
EVAL_REPS = 4
EVAL_JITTER_BASE = 133_000_000
RULES = ("local", "standard", "soft020", "global020", "soft035", "global035")
PROBABILITY = {
    "soft020": 0.20,
    "global020": 0.20,
    "soft035": 0.35,
    "global035": 0.35,
}
DIAG_KEYS = (
    "mean_multiplier",
    "edge_weighted_multiplier",
    "candidate_group_fraction",
    "active_group_fraction",
    "attenuation_mass_fraction",
    "update_l2_ratio",
    "matched_l2_relative_error",
)


def _candidate_soft_update(
    direction: torch.Tensor,
    *,
    learning_rate: float,
    max_update: float,
    groups: list[tuple[str, str, torch.Tensor]],
    probability: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    clipped = (learning_rate * direction).clamp(-max_update, max_update)
    applied, diagnostics = _fractional_candidate_update(
        direction,
        learning_rate=learning_rate,
        max_update=max_update,
        groups=groups,
        probability=probability,
    )
    clipped_norm = torch.linalg.vector_norm(clipped)
    applied_norm = torch.linalg.vector_norm(applied)
    if float(clipped_norm) > 0.0:
        update_l2_ratio = float(applied_norm / clipped_norm)
    else:
        update_l2_ratio = 1.0
    return applied, {
        **diagnostics,
        "update_l2_ratio": update_l2_ratio,
        "matched_l2_relative_error": 0.0,
    }


def _global_l2_matched_update(
    direction: torch.Tensor,
    *,
    learning_rate: float,
    max_update: float,
    groups: list[tuple[str, str, torch.Tensor]],
    probability: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    clipped = (learning_rate * direction).clamp(-max_update, max_update)
    target, target_diagnostics = _candidate_soft_update(
        direction,
        learning_rate=learning_rate,
        max_update=max_update,
        groups=groups,
        probability=probability,
    )
    clipped_norm = torch.linalg.vector_norm(clipped)
    target_norm = torch.linalg.vector_norm(target)
    if float(clipped_norm) > 0.0:
        multiplier = min(1.0, float(target_norm / clipped_norm))
        applied = clipped * multiplier
    else:
        multiplier = 1.0
        applied = clipped
    applied_norm = torch.linalg.vector_norm(applied)
    denominator = max(float(target_norm), 1e-12)
    relative_error = abs(float(applied_norm) - float(target_norm)) / denominator
    return applied, {
        "mean_multiplier": multiplier,
        "edge_weighted_multiplier": multiplier,
        "candidate_group_fraction": target_diagnostics["candidate_group_fraction"],
        "active_group_fraction": float(multiplier < 1.0),
        "attenuation_mass_fraction": 1.0 - multiplier,
        "update_l2_ratio": multiplier,
        "matched_l2_relative_error": relative_error,
    }


@torch.no_grad()
def _apply_update(
    model: PredictiveCodingGraph,
    edge_direction: torch.Tensor,
    bias_direction: torch.Tensor,
    *,
    learning_rate: float,
    max_update: float,
    groups: list[tuple[str, str, torch.Tensor]],
    probability: float,
    global_match: bool,
) -> dict[str, float]:
    update_fn = _global_l2_matched_update if global_match else _candidate_soft_update
    edge_update, diagnostics = update_fn(
        edge_direction,
        learning_rate=learning_rate,
        max_update=max_update,
        groups=groups,
        probability=probability,
    )
    bias_update = (learning_rate * bias_direction).clamp(-max_update, max_update)
    model.weight.add_(edge_update)
    model.bias.add_(bias_update)
    return diagnostics


def run_seed(*, seed: int, learning_rate: float = 160.0, max_update: float = 0.05) -> pd.DataFrame:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    groups = _named_groups(circuit)
    rows: list[dict[str, float | int | bool | str]] = []

    for scale in INIT_SCALES:
        base = PredictiveCodingGraph(
            circuit.graph,
            seed=seed,
            init_scale=scale,
            use_biological_strength=True,
        )
        models = {rule: copy.deepcopy(base) for rule in RULES}
        diag_sums = {rule: {key: 0.0 for key in DIAG_KEYS} for rule in RULES[2:]}

        for step in range(1, max(HORIZONS) + 1):
            epoch = step - 1
            for rule, model in models.items():
                raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
                direction = raw_edge if rule == "local" else _standard_direction(raw_edge, groups)
                if rule in {"local", "standard"}:
                    apply_local_credit(
                        model,
                        direction,
                        raw_bias,
                        learning_rate=learning_rate,
                        weight_decay=0.0,
                        max_update=max_update,
                    )
                    continue

                diagnostics = _apply_update(
                    model,
                    direction,
                    raw_bias,
                    learning_rate=learning_rate,
                    max_update=max_update,
                    groups=groups,
                    probability=PROBABILITY[rule],
                    global_match=rule.startswith("global"),
                )
                for key in DIAG_KEYS:
                    diag_sums[rule][key] += diagnostics[key]

            if step not in HORIZONS:
                continue

            for eval_rep in range(EVAL_REPS):
                jitter_seed = EVAL_JITTER_BASE + eval_rep
                _, classes = _heldout_readout(models["local"], circuit, jitter_seed=jitter_seed)
                for rule, model in models.items():
                    readout, paired_classes = _heldout_readout(
                        model, circuit, jitter_seed=jitter_seed
                    )
                    if not torch.equal(classes, paired_classes):
                        raise RuntimeError("held-out class mismatch")
                    weight = model.weight.detach().clone().requires_grad_(True)
                    bias = model.bias.detach().clone()
                    ce, hard_margin, soft_margin = _heldout_objectives(circuit, weight, bias)
                    values = {
                        "cross_entropy": float(ce.detach()),
                        "accuracy": float((readout.argmax(dim=1) == classes).float().mean()),
                        "hard_margin": float(hard_margin.detach()),
                        "soft_margin": float(soft_margin.detach()),
                        "weight_norm": float(torch.linalg.vector_norm(model.weight.detach())),
                    }
                    diagnostics = {
                        key: (diag_sums[rule][key] / step if rule in diag_sums else math.nan)
                        for key in DIAG_KEYS
                    }
                    finite = (
                        bool(torch.isfinite(model.weight).all())
                        and bool(torch.isfinite(model.bias).all())
                        and all(math.isfinite(value) for value in values.values())
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "init_scale": scale,
                            "horizon": step,
                            "eval_rep": eval_rep,
                            "rule": rule,
                            "eval_jitter": jitter_seed,
                            **diagnostics,
                            **values,
                            "finite": finite,
                        }
                    )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fresh hold-out separating type-pair selective soft peak-norm attenuation "
            "from a global update with exactly matched edge-update L2 norm."
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
