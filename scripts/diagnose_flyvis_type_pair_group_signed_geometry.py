from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch

from diagnose_flyvis_type_pair_l1_gain_geometry import _heldout_objectives
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit
from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

HORIZONS = (60, 80, 100)
INIT_MODES = ("random", "biological_strength")
MEAN_DOMINANCE_THRESHOLD = 0.5


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = (torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)).clamp_min(
        1e-30
    )
    return float(torch.dot(left, right) / denominator)


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
    models = {mode: copy.deepcopy(base) for mode, base in bases.items()}
    rows: list[dict[str, float | int | bool | str]] = []
    trained = 0

    for horizon in HORIZONS:
        for epoch in range(trained, horizon):
            for mode in INIT_MODES:
                edge, bias = credit(models[mode], circuit, seed=seed, epoch=epoch)
                apply_local_credit(
                    models[mode],
                    edge,
                    bias,
                    learning_rate=learning_rate,
                    weight_decay=0.0,
                    max_update=max_update,
                )
        trained = horizon

        for mode in INIT_MODES:
            model = models[mode]
            raw_edge, _ = credit(model, circuit, seed=seed, epoch=horizon)
            weight = model.weight.detach().clone().requires_grad_(True)
            bias = model.bias.detach().clone()
            ce, hard_margin, soft_margin = _heldout_objectives(circuit, weight, bias)
            grad_ce = torch.autograd.grad(ce, weight, retain_graph=True)[0]
            grad_hard = torch.autograd.grad(hard_margin, weight, retain_graph=True)[0]
            grad_soft = torch.autograd.grad(soft_margin, weight)[0]

            for source_type, target_type, indices in named_groups:
                values = raw_edge[indices]
                mean = values.mean()
                mean_component = torch.zeros_like(raw_edge)
                mean_component[indices] = mean
                residual = values - mean
                mean_norm = torch.linalg.vector_norm(mean_component[indices])
                residual_norm = torch.linalg.vector_norm(residual)
                ratio = float(mean_norm / residual_norm.clamp_min(1e-30))
                group_weight = model.weight.detach()[indices]
                weight_mean = group_weight.mean()
                weight_std = group_weight.std(unbiased=False)
                local_weight_credit = float(weight_mean * mean)

                ce_dot = float(torch.dot(grad_ce, mean_component))
                hard_dot = float(torch.dot(grad_hard, mean_component))
                soft_dot = float(torch.dot(grad_soft, mean_component))
                values_out = {
                    "mean_credit": float(mean),
                    "mean_credit_abs": float(mean.abs()),
                    "mean_component_norm": float(mean_norm),
                    "residual_norm": float(residual_norm),
                    "mean_to_residual_norm": ratio,
                    "high_rho": ratio >= MEAN_DOMINANCE_THRESHOLD,
                    "weight_mean": float(weight_mean),
                    "weight_std": float(weight_std),
                    "weight_mean_times_credit_mean": local_weight_credit,
                    "ce_grad_dot_mean": ce_dot,
                    "hard_margin_grad_dot_mean": hard_dot,
                    "soft_margin_grad_dot_mean": soft_dot,
                    "ce_grad_cos_mean": _cosine(grad_ce, mean_component),
                    "hard_margin_grad_cos_mean": _cosine(grad_hard, mean_component),
                    "soft_margin_grad_cos_mean": _cosine(grad_soft, mean_component),
                }
                finite = (
                    bool(torch.isfinite(model.weight).all())
                    and bool(torch.isfinite(model.bias).all())
                    and all(
                        math.isfinite(value)
                        for value in values_out.values()
                        if not isinstance(value, bool)
                    )
                )
                rows.append(
                    {
                        "seed": seed,
                        "init_mode": mode,
                        "horizon": horizon,
                        "source_type": source_type,
                        "target_type": target_type,
                        "edge_count": len(indices),
                        **values_out,
                        "finite": finite,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose type-pair shared-credit groups by signed held-out CE/margin geometry "
            "while recording topology-agnostic local candidate gate features."
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
