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
from diagnose_flyvis_type_pair_ontrajectory_geometry import (
    END_EPOCH,
    FORK_EPOCH,
    INIT_SCALE,
    _apply_precomputed,
    _objective_values,
    _state_diagnostic,
)
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denom) == 0.0:
        return 0.0
    return float(torch.dot(left, right) / denom)


def _gradient_geometry(circuit, model: PredictiveCodingGraph) -> tuple[torch.Tensor, ...]:
    weight = model.weight.detach().clone().requires_grad_(True)
    bias = model.bias.detach().clone()
    ce, hard_margin, soft_margin = _heldout_objectives(circuit, weight, bias)
    grad_ce = torch.autograd.grad(ce, weight, retain_graph=True)[0]
    grad_hard = torch.autograd.grad(hard_margin, weight, retain_graph=True)[0]
    grad_soft = torch.autograd.grad(soft_margin, weight)[0]
    return grad_ce, grad_hard, grad_soft


def run_seed(*, seed: int, learning_rate: float = 160.0, max_update: float = 0.05) -> pd.DataFrame:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    groups = _named_groups(circuit)
    base = PredictiveCodingGraph(
        circuit.graph,
        seed=seed,
        init_scale=INIT_SCALE,
        use_biological_strength=True,
    )
    for epoch in range(FORK_EPOCH):
        raw_edge, raw_bias = credit(base, circuit, seed=seed, epoch=epoch)
        direction = _standard_direction(raw_edge, groups)
        apply_local_credit(
            base,
            direction,
            raw_bias,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )

    soft_model = copy.deepcopy(base)
    global_model = copy.deepcopy(base)
    rows: list[dict[str, float | int | bool]] = []
    for epoch in range(FORK_EPOCH, END_EPOCH):
        soft_diag, soft_update, _, soft_bias = _state_diagnostic(
            soft_model,
            circuit,
            groups,
            seed=seed,
            epoch=epoch,
            learning_rate=learning_rate,
            max_update=max_update,
        )
        global_diag, _, global_update, global_bias = _state_diagnostic(
            global_model,
            circuit,
            groups,
            seed=seed,
            epoch=epoch,
            learning_rate=learning_rate,
            max_update=max_update,
        )
        _apply_precomputed(soft_model, soft_update, soft_bias)
        _apply_precomputed(global_model, global_update, global_bias)

        displacement = (soft_model.weight - global_model.weight).detach()
        displacement_l2 = float(torch.linalg.vector_norm(displacement))
        bias_displacement_l2 = float(
            torch.linalg.vector_norm(soft_model.bias - global_model.bias)
        )
        soft_values = _objective_values(circuit, soft_model.weight, soft_model.bias)
        global_values = _objective_values(circuit, global_model.weight, global_model.bias)
        soft_grads = _gradient_geometry(circuit, soft_model)
        global_grads = _gradient_geometry(circuit, global_model)

        values: dict[str, float] = {
            "weight_displacement_l2": displacement_l2,
            "bias_displacement_l2": bias_displacement_l2,
            "observed_ce_difference": soft_values[0] - global_values[0],
            "observed_hard_margin_difference": soft_values[1] - global_values[1],
            "observed_soft_margin_difference": soft_values[2] - global_values[2],
            "soft_l2_match_relative_error": soft_diag["l2_match_relative_error"],
            "global_l2_match_relative_error": global_diag["l2_match_relative_error"],
        }
        for name, soft_grad, global_grad in zip(
            ("ce", "hard_margin", "soft_margin"), soft_grads, global_grads, strict=True
        ):
            global_dot = float(torch.dot(global_grad, displacement))
            soft_dot = float(torch.dot(soft_grad, displacement))
            values[f"global_{name}_grad_dot_displacement"] = global_dot
            values[f"soft_{name}_grad_dot_displacement"] = soft_dot
            values[f"global_{name}_grad_cos_displacement"] = _cosine(
                global_grad, displacement
            )
            values[f"soft_{name}_grad_cos_displacement"] = _cosine(
                soft_grad, displacement
            )
            observed = values[
                "observed_ce_difference"
                if name == "ce"
                else f"observed_{name}_difference"
            ]
            values[f"global_{name}_linearization_residual"] = observed - global_dot
            values[f"soft_{name}_linearization_residual"] = observed - soft_dot

        finite = (
            bool(torch.isfinite(soft_model.weight).all())
            and bool(torch.isfinite(global_model.weight).all())
            and all(math.isfinite(value) for value in values.values())
        )
        rows.append({"seed": seed, "epoch": epoch, **values, "finite": finite})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether the accumulated soft-vs-global parameter displacement points "
            "along held-out CE and margin gradients after the epoch-140 fork."
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
    if len(frame) != END_EPOCH - FORK_EPOCH or not frame["finite"].all():
        raise RuntimeError("incomplete or non-finite displacement geometry diagnostic")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
