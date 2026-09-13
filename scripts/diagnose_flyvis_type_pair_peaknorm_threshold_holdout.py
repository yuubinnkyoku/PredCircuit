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
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

INIT_SCALES = (0.08, 0.10, 0.11, 0.115)
HORIZONS = (160, 180, 200)
EVAL_REPS = 4
EVAL_JITTER_BASE = 103_000_000
RULES = ("local", "standard", "peaknorm", "peak050", "peak033", "peak025")
THRESHOLDS = {"peaknorm": 1.0, "peak050": 0.50, "peak033": 0.33, "peak025": 0.25}


def _threshold_peaknorm_update(
    direction: torch.Tensor,
    *,
    learning_rate: float,
    max_update: float,
    groups: list[tuple[str, str, torch.Tensor]],
    threshold: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    raw = learning_rate * direction
    clipped = raw.clamp(-max_update, max_update)
    applied = clipped.clone()
    multipliers: list[float] = []
    peak_ratios: list[float] = []
    weights: list[int] = []
    active = 0

    for _, _, indices in groups:
        raw_group = raw[indices]
        clipped_group = clipped[indices]
        raw_norm = torch.linalg.vector_norm(raw_group)
        clipped_norm = torch.linalg.vector_norm(clipped_group)
        peak = raw_group.abs().max()

        if float(raw_norm) == 0.0 or float(peak) <= max_update:
            peak_ratio = 1.0
        else:
            peak_ratio = float(max_update / peak)

        multiplier = 1.0
        if peak_ratio < threshold and float(clipped_norm) > 0.0:
            target_norm = raw_norm * peak_ratio
            multiplier = min(1.0, float(target_norm / clipped_norm))
            applied[indices] = clipped_group * multiplier
            active += 1

        multipliers.append(multiplier)
        peak_ratios.append(peak_ratio)
        weights.append(len(indices))

    total_weight = max(sum(weights), 1)

    def weighted(values: list[float]) -> float:
        return float(
            sum(value * weight for value, weight in zip(values, weights, strict=True))
            / total_weight
        )

    return applied, {
        "threshold": threshold,
        "mean_multiplier": float(sum(multipliers) / max(len(multipliers), 1)),
        "edge_weighted_multiplier": weighted(multipliers),
        "active_group_fraction": float(active / max(len(groups), 1)),
        "mean_peak_ratio": float(sum(peak_ratios) / max(len(peak_ratios), 1)),
        "edge_weighted_peak_ratio": weighted(peak_ratios),
    }


@torch.no_grad()
def _apply_threshold_peaknorm(
    model: PredictiveCodingGraph,
    edge_direction: torch.Tensor,
    bias_direction: torch.Tensor,
    *,
    learning_rate: float,
    max_update: float,
    groups: list[tuple[str, str, torch.Tensor]],
    threshold: float,
) -> dict[str, float]:
    edge_update, diagnostics = _threshold_peaknorm_update(
        edge_direction,
        learning_rate=learning_rate,
        max_update=max_update,
        groups=groups,
        threshold=threshold,
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
        diag_sums = {
            rule: {
                "mean_multiplier": 0.0,
                "edge_weighted_multiplier": 0.0,
                "active_group_fraction": 0.0,
                "mean_peak_ratio": 0.0,
                "edge_weighted_peak_ratio": 0.0,
            }
            for rule in RULES
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
                else:
                    diagnostics = _apply_threshold_peaknorm(
                        model,
                        direction,
                        raw_bias,
                        learning_rate=learning_rate,
                        max_update=max_update,
                        groups=groups,
                        threshold=THRESHOLDS[rule],
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
                        key: (diag_sums[rule][key] / step if rule in THRESHOLDS else math.nan)
                        for key in diag_sums[rule]
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
            "Fresh hold-out of thresholded local peak-norm stabilization. Mildly saturated "
            "type-pairs keep ordinary elementwise-clipped 4m+r; only groups whose peak ratio "
            "max_update/max(|raw update|) falls below a local threshold are additionally damped."
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
