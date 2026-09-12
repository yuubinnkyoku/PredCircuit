from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_adaptive_mean_gain_holdout import _standard_direction
from diagnose_flyvis_type_pair_l1_gain_geometry import _heldout_objectives
from diagnose_flyvis_type_pair_local_power_holdout import _candidate_direction
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from diagnose_flyvis_type_pair_threshold_mean_gain_holdout import _threshold_direction
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

HORIZONS = (20, 40, 60, 80, 100, 120, 140, 160, 180, 200)
RULES = ("local", "standard", "threshold", "local_power")


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = (torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)).clamp_min(
        1e-30
    )
    return float(torch.dot(left, right) / denominator)


def _rule_direction(
    rule: str,
    raw_edge: torch.Tensor,
    weight: torch.Tensor,
    named_groups,
    *,
    learning_rate: float,
    max_update: float,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    """Return (direction, diag, shared_mean_component_mask_contribution)."""
    if rule == "local":
        return raw_edge, {}, torch.zeros_like(raw_edge)
    if rule == "standard":
        direction = _standard_direction(raw_edge, named_groups)
        shared = torch.zeros_like(raw_edge)
        for _, _, indices in named_groups:
            mean = raw_edge[indices].mean()
            shared[indices] = 3.0 * mean
        return direction, {"extra_gain_applied": 3.0}, shared
    if rule == "threshold":
        direction, diag = _threshold_direction(raw_edge, named_groups)
        # reconstruct shared component actually applied
        shared = direction - raw_edge
        diag = dict(diag)
        diag["extra_gain_applied"] = float(
            shared.abs().sum() / raw_edge.abs().sum().clamp_min(1e-30)
        )
        return direction, diag, shared
    if rule == "local_power":
        direction, diag = _candidate_direction(
            raw_edge,
            weight,
            named_groups,
            learning_rate=learning_rate,
            max_update=max_update,
            mode="local_power",
        )
        shared = direction - raw_edge
        diag = dict(diag)
        diag["extra_gain_applied"] = float(
            shared.abs().sum() / raw_edge.abs().sum().clamp_min(1e-30)
        )
        return direction, diag, shared
    raise ValueError(rule)


def run_seed(*, seed: int, learning_rate: float, max_update: float) -> pd.DataFrame:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    named_groups = _named_groups(circuit)
    model0 = PredictiveCodingGraph(
        circuit.graph,
        seed=seed,
        init_scale=0.08,
        use_biological_strength=True,
    )
    models = {rule: copy.deepcopy(model0) for rule in RULES}
    diag_sums: dict[str, dict[str, float]] = {rule: {} for rule in RULES}
    rows: list[dict[str, float | int | bool | str]] = []

    for step in range(1, max(HORIZONS) + 1):
        epoch = step - 1
        for rule in RULES:
            model = models[rule]
            raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
            direction, diagnostics, _shared = _rule_direction(
                rule,
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
            for key, value in diagnostics.items():
                diag_sums[rule][key] = diag_sums[rule].get(key, 0.0) + float(value)

        if step not in HORIZONS:
            continue

        # mechanism snapshot
        raw_edge, _ = credit(models["standard"], circuit, seed=seed, epoch=step)
        rho_ge_05 = 0
        rho_ge_035 = 0
        group_count = 0
        for _, _, indices in named_groups:
            values = raw_edge[indices]
            mean = values.mean()
            residual = values - mean
            rho = float(
                torch.linalg.vector_norm(mean.expand_as(values))
                / torch.linalg.vector_norm(residual).clamp_min(1e-30)
            )
            group_count += 1
            if rho >= 0.5:
                rho_ge_05 += 1
            if rho >= 0.35:
                rho_ge_035 += 1

        for rule in RULES:
            model = models[rule]
            weight = model.weight.detach().clone().requires_grad_(True)
            bias = model.bias.detach().clone()
            ce, hard_margin, soft_margin = _heldout_objectives(circuit, weight, bias)
            grad_hard = torch.autograd.grad(hard_margin, weight, retain_graph=True)[0]
            grad_ce = torch.autograd.grad(ce, weight)[0]

            weight_norm = float(torch.linalg.vector_norm(model.weight))
            bias_norm = float(torch.linalg.vector_norm(model.bias))
            # alignment of the current raw credit (from this rule's state) with hard-margin grad
            rule_raw, _ = credit(model, circuit, seed=seed, epoch=step)
            # shared-mean component of the standard direction at this state
            shared_std = torch.zeros_like(rule_raw)
            for _, _, indices in named_groups:
                shared_std[indices] = 3.0 * rule_raw[indices].mean()

            values = {
                "cross_entropy": float(ce.detach()),
                "hard_margin": float(hard_margin.detach()),
                "soft_margin": float(soft_margin.detach()),
                "weight_norm": weight_norm,
                "bias_norm": bias_norm,
                "credit_hard_cosine": _cosine(rule_raw, grad_hard.detach()),
                "credit_ce_cosine": _cosine(rule_raw, grad_ce.detach()),
                "shared_mean_hard_cosine": _cosine(shared_std, grad_hard.detach()),
                "shared_mean_norm": float(torch.linalg.vector_norm(shared_std)),
                "shared_mean_frac": float(
                    torch.linalg.vector_norm(shared_std)
                    / torch.linalg.vector_norm(rule_raw).clamp_min(1e-30)
                ),
                "rho_ge_05_group_fraction": rho_ge_05 / max(group_count, 1),
                "rho_ge_035_group_fraction": rho_ge_035 / max(group_count, 1),
            }
            for key, total in diag_sums[rule].items():
                values[f"cum_{key}"] = total / step

            finite = (
                bool(torch.isfinite(model.weight).all())
                and bool(torch.isfinite(model.bias).all())
                and all(math.isfinite(float(v)) for v in values.values())
            )
            rows.append(
                {
                    "seed": seed,
                    "horizon": step,
                    "rule": rule,
                    **values,
                    "finite": finite,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mechanism snapshot across the 200-epoch U-turn: rho distribution, "
            "weight-norm growth, and shared-mean / hard-margin gradient alignment "
            "for local, standard 4m+r, rho05, and local_power."
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
