from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_adaptive_mean_gain_holdout import _standard_direction
from diagnose_flyvis_type_pair_l1_gain_geometry import _heldout_objectives
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

INIT_SCALES = (0.100, 0.105, 0.110, 0.115)
HORIZONS = (160, 180, 190, 200)
RULES = ("local", "standard")


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = (torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)).clamp_min(
        1e-30
    )
    return float(torch.dot(left, right) / denominator)


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return parsed


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return parsed


def run_seed(
    *,
    seed: int,
    learning_rate: float,
    max_update: float,
    init_scales: tuple[float, ...] = INIT_SCALES,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    named_groups = _named_groups(circuit)
    models = {}
    for scale in init_scales:
        model0 = PredictiveCodingGraph(
            circuit.graph,
            seed=seed,
            init_scale=scale,
            use_biological_strength=True,
        )
        for rule in RULES:
            models[(scale, rule)] = copy.deepcopy(model0)

    rows: list[dict[str, float | int | bool | str]] = []
    for step in range(1, max(horizons) + 1):
        epoch = step - 1
        for scale in init_scales:
            for rule in RULES:
                model = models[(scale, rule)]
                raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
                direction = (
                    raw_edge if rule == "local" else _standard_direction(raw_edge, named_groups)
                )
                apply_local_credit(
                    model,
                    direction,
                    raw_bias,
                    learning_rate=learning_rate,
                    weight_decay=0.0,
                    max_update=max_update,
                )

        if step not in horizons:
            continue

        for scale in init_scales:
            for rule in RULES:
                model = models[(scale, rule)]
                weight = model.weight.detach().clone().requires_grad_(True)
                bias = model.bias.detach().clone()
                ce, hard_margin, soft_margin = _heldout_objectives(circuit, weight, bias)
                grad_hard = torch.autograd.grad(hard_margin, weight, retain_graph=True)[0]
                grad_ce = torch.autograd.grad(ce, weight)[0]

                raw_edge, _ = credit(model, circuit, seed=seed, epoch=step)
                shared = torch.zeros_like(raw_edge)
                rho_ge_05 = 0
                rho_ge_10 = 0
                outward_groups = 0
                for _, _, indices in named_groups:
                    values = raw_edge[indices]
                    mean = values.mean()
                    shared[indices] = 3.0 * mean
                    residual = values - mean
                    mean_component = mean.expand_as(values)
                    rho = float(
                        torch.linalg.vector_norm(mean_component)
                        / torch.linalg.vector_norm(residual).clamp_min(1e-30)
                    )
                    rho_ge_05 += int(rho >= 0.5)
                    rho_ge_10 += int(rho >= 1.0)
                    outward_groups += int(
                        torch.dot(model.weight.detach()[indices], shared[indices]) > 0.0
                    )

                standard_direction = raw_edge + shared
                unclipped_update = learning_rate * standard_direction
                clipped_update = (
                    unclipped_update.clamp(-max_update, max_update)
                    if max_update > 0.0
                    else unclipped_update
                )
                clip_fraction = (
                    float((unclipped_update.abs() >= max_update).float().mean())
                    if max_update > 0.0
                    else 0.0
                )
                group_count = max(len(named_groups), 1)
                model_weight = model.weight.detach()
                shared_norm = torch.linalg.vector_norm(shared)
                raw_norm = torch.linalg.vector_norm(raw_edge)
                values = {
                    "cross_entropy": float(ce.detach()),
                    "hard_margin": float(hard_margin.detach()),
                    "soft_margin": float(soft_margin.detach()),
                    "weight_norm": float(torch.linalg.vector_norm(model_weight)),
                    "raw_credit_norm": float(raw_norm),
                    "shared_mean_norm": float(shared_norm),
                    "shared_mean_frac": float(shared_norm / raw_norm.clamp_min(1e-30)),
                    "raw_hard_cosine": _cosine(raw_edge, grad_hard.detach()),
                    "shared_hard_cosine": _cosine(shared, grad_hard.detach()),
                    "standard_hard_cosine": _cosine(standard_direction, grad_hard.detach()),
                    "raw_ce_cosine": _cosine(raw_edge, grad_ce.detach()),
                    "shared_ce_cosine": _cosine(shared, grad_ce.detach()),
                    "shared_weight_cosine": _cosine(shared, model_weight),
                    "shared_weight_dot": float(torch.dot(shared, model_weight)),
                    "clipped_update_hard_cosine": _cosine(clipped_update, grad_hard.detach()),
                    "clipped_update_weight_cosine": _cosine(clipped_update, model_weight),
                    "clip_fraction": clip_fraction,
                    "rho_ge_05_group_fraction": rho_ge_05 / group_count,
                    "rho_ge_10_group_fraction": rho_ge_10 / group_count,
                    "outward_shared_group_fraction": outward_groups / group_count,
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
                        "rule": rule,
                        **values,
                        "finite": finite,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Track local and standard 4m+r credit geometry across the critical "
            "biological-init scale band where late recovery flips to collapse."
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--init-scales", type=_parse_float_tuple, default=INIT_SCALES)
    parser.add_argument("--horizons", type=_parse_int_tuple, default=HORIZONS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = run_seed(
        seed=args.seed,
        learning_rate=args.learning_rate,
        max_update=args.max_update,
        init_scales=args.init_scales,
        horizons=args.horizons,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
