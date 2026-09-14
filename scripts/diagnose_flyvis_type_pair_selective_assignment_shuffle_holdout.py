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
from diagnose_flyvis_type_pair_peaknorm_global_l2_match_holdout import (
    _apply_update,
    _candidate_soft_update,
)
from diagnose_flyvis_type_pair_selective_phase_factorial_holdout import _advance_standard
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

INIT_SCALE = 0.115
FORK_EPOCH = 140
SWITCH_EPOCH = 160
END_EPOCH = 200
EVAL_REPS = 4
EVAL_JITTER_BASE = 163_000_000
PROBABILITY = 0.35
POLICIES = ("standard", "global_global", "shuffle_global", "soft_global")
DIAG_KEYS = (
    "candidate_group_fraction",
    "attenuation_mass_fraction",
    "update_l2_ratio",
    "matched_l2_relative_error",
)


def _match_norm_without_amplifying_past_clip(
    shuffled: torch.Tensor,
    clipped: torch.Tensor,
    *,
    target_norm: float,
) -> torch.Tensor:
    shuffled_norm = float(torch.linalg.vector_norm(shuffled))
    if shuffled_norm > target_norm and shuffled_norm > 0.0:
        return shuffled * (target_norm / shuffled_norm)
    if shuffled_norm >= target_norm:
        return shuffled

    lo = 0.0
    hi = 1.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        trial = shuffled + mid * (clipped - shuffled)
        if float(torch.linalg.vector_norm(trial)) < target_norm:
            lo = mid
        else:
            hi = mid
    return shuffled + hi * (clipped - shuffled)


def _shuffled_candidate_update(
    direction: torch.Tensor,
    *,
    learning_rate: float,
    max_update: float,
    groups: list[tuple[str, str, torch.Tensor]],
    seed: int,
    epoch: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    clipped = (learning_rate * direction).clamp(-max_update, max_update)
    soft, soft_diag = _candidate_soft_update(
        direction,
        learning_rate=learning_rate,
        max_update=max_update,
        groups=groups,
        probability=PROBABILITY,
    )

    multipliers: list[float] = []
    for _, _, indices in groups:
        clipped_norm = float(torch.linalg.vector_norm(clipped[indices]))
        soft_norm = float(torch.linalg.vector_norm(soft[indices]))
        multiplier = soft_norm / clipped_norm if clipped_norm > 0.0 else 1.0
        multipliers.append(multiplier)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed * 1_000_003 + epoch * 97 + 17)
    permutation = torch.randperm(len(groups), generator=generator).tolist()
    shuffled = clipped.clone()
    for destination, (_, _, indices) in enumerate(groups):
        shuffled[indices] = clipped[indices] * multipliers[permutation[destination]]

    target_norm = float(torch.linalg.vector_norm(soft))
    applied = _match_norm_without_amplifying_past_clip(
        shuffled,
        clipped,
        target_norm=target_norm,
    )
    applied_norm = float(torch.linalg.vector_norm(applied))
    clipped_norm = float(torch.linalg.vector_norm(clipped))
    relative_error = abs(applied_norm - target_norm) / max(target_norm, 1e-12)

    return applied, {
        "candidate_group_fraction": soft_diag["candidate_group_fraction"],
        "attenuation_mass_fraction": soft_diag["attenuation_mass_fraction"],
        "update_l2_ratio": applied_norm / clipped_norm if clipped_norm > 0.0 else 1.0,
        "matched_l2_relative_error": relative_error,
    }


@torch.no_grad()
def _apply_shuffled_update(
    model: PredictiveCodingGraph,
    edge_direction: torch.Tensor,
    bias_direction: torch.Tensor,
    *,
    learning_rate: float,
    max_update: float,
    groups: list[tuple[str, str, torch.Tensor]],
    seed: int,
    epoch: int,
) -> dict[str, float]:
    edge_update, diagnostics = _shuffled_candidate_update(
        edge_direction,
        learning_rate=learning_rate,
        max_update=max_update,
        groups=groups,
        seed=seed,
        epoch=epoch,
    )
    bias_update = (learning_rate * bias_direction).clamp(-max_update, max_update)
    model.weight.add_(edge_update)
    model.bias.add_(bias_update)
    return diagnostics


def _advance_rule(
    model: PredictiveCodingGraph,
    circuit,
    groups: list[tuple[str, str, torch.Tensor]],
    *,
    rule: str,
    seed: int,
    start_epoch: int,
    end_epoch: int,
    learning_rate: float,
    max_update: float,
) -> dict[str, float]:
    sums = {key: 0.0 for key in DIAG_KEYS}
    steps = end_epoch - start_epoch
    for epoch in range(start_epoch, end_epoch):
        raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
        direction = _standard_direction(raw_edge, groups)
        if rule == "standard":
            apply_local_credit(
                model,
                direction,
                raw_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )
            continue
        if rule == "shuffle035":
            diag = _apply_shuffled_update(
                model,
                direction,
                raw_bias,
                learning_rate=learning_rate,
                max_update=max_update,
                groups=groups,
                seed=seed,
                epoch=epoch,
            )
        else:
            diag = _apply_update(
                model,
                direction,
                raw_bias,
                learning_rate=learning_rate,
                max_update=max_update,
                groups=groups,
                probability=PROBABILITY,
                global_match=rule == "global035",
            )
        for key in DIAG_KEYS:
            sums[key] += diag[key]
    if rule == "standard":
        return {key: math.nan for key in DIAG_KEYS}
    return {key: sums[key] / steps for key in DIAG_KEYS}


def run_seed(*, seed: int, learning_rate: float = 160.0, max_update: float = 0.05) -> pd.DataFrame:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    groups = _named_groups(circuit)
    base = PredictiveCodingGraph(
        circuit.graph,
        seed=seed,
        init_scale=INIT_SCALE,
        use_biological_strength=True,
    )
    _advance_standard(
        base,
        circuit,
        groups,
        seed=seed,
        start_epoch=0,
        end_epoch=FORK_EPOCH,
        learning_rate=learning_rate,
        max_update=max_update,
    )

    models = {policy: copy.deepcopy(base) for policy in POLICIES}
    diagnostics: dict[str, dict[str, float]] = {}
    early_rules = {
        "standard": "standard",
        "global_global": "global035",
        "shuffle_global": "shuffle035",
        "soft_global": "soft035",
    }
    for policy, model in models.items():
        early = _advance_rule(
            model,
            circuit,
            groups,
            rule=early_rules[policy],
            seed=seed,
            start_epoch=FORK_EPOCH,
            end_epoch=SWITCH_EPOCH,
            learning_rate=learning_rate,
            max_update=max_update,
        )
        late_rule = "standard" if policy == "standard" else "global035"
        late = _advance_rule(
            model,
            circuit,
            groups,
            rule=late_rule,
            seed=seed,
            start_epoch=SWITCH_EPOCH,
            end_epoch=END_EPOCH,
            learning_rate=learning_rate,
            max_update=max_update,
        )
        diagnostics[policy] = {
            **{f"early_{key}": value for key, value in early.items()},
            **{f"late_{key}": value for key, value in late.items()},
        }

    rows: list[dict[str, float | int | bool | str]] = []
    for eval_rep in range(EVAL_REPS):
        jitter_seed = EVAL_JITTER_BASE + eval_rep
        _, classes = _heldout_readout(models["standard"], circuit, jitter_seed=jitter_seed)
        for policy, model in models.items():
            readout, paired_classes = _heldout_readout(model, circuit, jitter_seed=jitter_seed)
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
            finite = (
                bool(torch.isfinite(model.weight).all())
                and bool(torch.isfinite(model.bias).all())
                and all(math.isfinite(value) for value in values.values())
            )
            rows.append(
                {
                    "seed": seed,
                    "init_scale": INIT_SCALE,
                    "fork_epoch": FORK_EPOCH,
                    "switch_epoch": SWITCH_EPOCH,
                    "end_epoch": END_EPOCH,
                    "eval_rep": eval_rep,
                    "eval_jitter": jitter_seed,
                    "policy": policy,
                    **diagnostics[policy],
                    **values,
                    "finite": finite,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fresh hold-out testing whether the epoch-140--160 benefit depends on assigning "
            "attenuation to the detected type-pairs rather than a shuffled type-pair assignment."
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
