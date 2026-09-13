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
from diagnose_flyvis_type_pair_peaknorm_persistence_holdout import _apply_persistent_peaknorm
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

INIT_SCALES = (0.08, 0.10, 0.11, 0.115)
HORIZONS = (160, 180, 200)
EVAL_REPS = 4
EVAL_JITTER_BASE = 121_000_000
RULES = ("local", "standard", "peak025x7", "random020", "random035")
PEAK_THRESHOLD = 0.25
RANDOM_PROBABILITY = {"random020": 0.20, "random035": 0.35}


def _deterministic_unit(seed: int, epoch: int, group_index: int) -> float:
    value = (
        seed * 1_000_003
        + epoch * 91_769
        + group_index * 6_113
        + seed * epoch * 37
    ) % 1_000_000
    return value / 1_000_000.0


def _random_candidate_update(
    direction: torch.Tensor,
    *,
    learning_rate: float,
    max_update: float,
    groups: list[tuple[str, str, torch.Tensor]],
    seed: int,
    epoch: int,
    probability: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    raw = learning_rate * direction
    clipped = raw.clamp(-max_update, max_update)
    applied = clipped.clone()
    multipliers: list[float] = []
    weights: list[int] = []
    candidate = 0
    active = 0

    for group_index, (_, _, indices) in enumerate(groups):
        raw_group = raw[indices]
        clipped_group = clipped[indices]
        raw_norm = torch.linalg.vector_norm(raw_group)
        clipped_norm = torch.linalg.vector_norm(clipped_group)
        peak = raw_group.abs().max()

        if float(raw_norm) == 0.0 or float(peak) <= max_update:
            peak_ratio = 1.0
        else:
            peak_ratio = float(max_update / peak)

        is_candidate = peak_ratio < PEAK_THRESHOLD
        if is_candidate:
            candidate += 1

        multiplier = 1.0
        selected = is_candidate and _deterministic_unit(seed, epoch, group_index) < probability
        if selected and float(clipped_norm) > 0.0:
            target_norm = raw_norm * peak_ratio
            multiplier = min(1.0, float(target_norm / clipped_norm))
            applied[indices] = clipped_group * multiplier
            active += 1

        multipliers.append(multiplier)
        weights.append(len(indices))

    total_weight = max(sum(weights), 1)
    weighted_multiplier = (
        sum(value * weight for value, weight in zip(multipliers, weights, strict=True))
        / total_weight
    )
    return applied, {
        "mean_multiplier": float(sum(multipliers) / max(len(multipliers), 1)),
        "edge_weighted_multiplier": float(weighted_multiplier),
        "candidate_group_fraction": float(candidate / max(len(groups), 1)),
        "active_group_fraction": float(active / max(len(groups), 1)),
    }


@torch.no_grad()
def _apply_random_candidate_peaknorm(
    model: PredictiveCodingGraph,
    edge_direction: torch.Tensor,
    bias_direction: torch.Tensor,
    *,
    learning_rate: float,
    max_update: float,
    groups: list[tuple[str, str, torch.Tensor]],
    seed: int,
    epoch: int,
    probability: float,
) -> dict[str, float]:
    edge_update, diagnostics = _random_candidate_update(
        edge_direction,
        learning_rate=learning_rate,
        max_update=max_update,
        groups=groups,
        seed=seed,
        epoch=epoch,
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
        streaks = [0 for _ in groups]
        diag_sums = {
            rule: {
                "mean_multiplier": 0.0,
                "edge_weighted_multiplier": 0.0,
                "candidate_group_fraction": 0.0,
                "active_group_fraction": 0.0,
            }
            for rule in ("peak025x7", "random020", "random035")
        }

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

                if rule == "peak025x7":
                    persistent = _apply_persistent_peaknorm(
                        model,
                        direction,
                        raw_bias,
                        learning_rate=learning_rate,
                        max_update=max_update,
                        groups=groups,
                        streaks=streaks,
                        persistence=7,
                    )
                    diagnostics = {
                        key: persistent[key]
                        for key in (
                            "mean_multiplier",
                            "edge_weighted_multiplier",
                            "candidate_group_fraction",
                            "active_group_fraction",
                        )
                    }
                else:
                    diagnostics = _apply_random_candidate_peaknorm(
                        model,
                        direction,
                        raw_bias,
                        learning_rate=learning_rate,
                        max_update=max_update,
                        groups=groups,
                        seed=seed,
                        epoch=epoch,
                        probability=RANDOM_PROBABILITY[rule],
                    )
                for key in diag_sums[rule]:
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
                        key: (
                            diag_sums[rule][key] / step
                            if rule in diag_sums
                            else math.nan
                        )
                        for key in (
                            "mean_multiplier",
                            "edge_weighted_multiplier",
                            "candidate_group_fraction",
                            "active_group_fraction",
                        )
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
            "Fresh hold-out testing whether seven-consecutive-burst peak-norm stabilization "
            "beats persistence-blind random intervention at matched candidate budgets."
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
