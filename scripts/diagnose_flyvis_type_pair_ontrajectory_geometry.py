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
from diagnose_flyvis_type_pair_peaknorm_global_l2_match_holdout import (
    _candidate_soft_update,
    _global_l2_matched_update,
)
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

INIT_SCALE = 0.115
FORK_EPOCH = 140
END_EPOCH = 160
PROBABILITY = 0.35


def _objective_values(
    circuit, weight: torch.Tensor, bias: torch.Tensor
) -> tuple[float, float, float]:
    ce, hard_margin, soft_margin = _heldout_objectives(circuit, weight, bias)
    return float(ce.detach()), float(hard_margin.detach()), float(soft_margin.detach())


def _state_diagnostic(
    model: PredictiveCodingGraph,
    circuit,
    groups: list[tuple[str, str, torch.Tensor]],
    *,
    seed: int,
    epoch: int,
    learning_rate: float,
    max_update: float,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
    direction = _standard_direction(raw_edge, groups)
    soft_update, soft_diag = _candidate_soft_update(
        direction,
        learning_rate=learning_rate,
        max_update=max_update,
        groups=groups,
        probability=PROBABILITY,
    )
    global_update, global_diag = _global_l2_matched_update(
        direction,
        learning_rate=learning_rate,
        max_update=max_update,
        groups=groups,
        probability=PROBABILITY,
    )
    bias_update = (learning_rate * raw_bias).clamp(-max_update, max_update)

    weight = model.weight.detach().clone().requires_grad_(True)
    bias = model.bias.detach().clone()
    ce, hard_margin, soft_margin = _heldout_objectives(circuit, weight, bias)
    grad_ce = torch.autograd.grad(ce, weight, retain_graph=True)[0]
    grad_hard = torch.autograd.grad(hard_margin, weight, retain_graph=True)[0]
    grad_soft = torch.autograd.grad(soft_margin, weight)[0]
    redistribution = soft_update - global_update

    base = (float(ce.detach()), float(hard_margin.detach()), float(soft_margin.detach()))
    soft_next = _objective_values(circuit, weight.detach() + soft_update, bias + bias_update)
    global_next = _objective_values(circuit, weight.detach() + global_update, bias + bias_update)
    values = {
        "base_ce": base[0],
        "base_hard_margin": base[1],
        "base_soft_margin": base[2],
        "candidate_group_fraction": soft_diag["candidate_group_fraction"],
        "redistribution_l2": float(torch.linalg.vector_norm(redistribution)),
        "l2_match_relative_error": global_diag["matched_l2_relative_error"],
        "ce_grad_dot_redistribution": float(torch.dot(grad_ce, redistribution)),
        "hard_grad_dot_redistribution": float(torch.dot(grad_hard, redistribution)),
        "soft_grad_dot_redistribution": float(torch.dot(grad_soft, redistribution)),
        "soft_minus_global_next_ce": soft_next[0] - global_next[0],
        "soft_minus_global_next_hard": soft_next[1] - global_next[1],
        "soft_minus_global_next_soft": soft_next[2] - global_next[2],
    }
    return values, soft_update, global_update, bias_update


@torch.no_grad()
def _apply_precomputed(
    model: PredictiveCodingGraph, edge_update: torch.Tensor, bias_update: torch.Tensor
) -> None:
    model.weight.add_(edge_update)
    model.bias.add_(bias_update)


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
        soft_diag, soft_on_soft, _, soft_bias = _state_diagnostic(
            soft_model,
            circuit,
            groups,
            seed=seed,
            epoch=epoch,
            learning_rate=learning_rate,
            max_update=max_update,
        )
        global_diag, _, global_on_global, global_bias = _state_diagnostic(
            global_model,
            circuit,
            groups,
            seed=seed,
            epoch=epoch,
            learning_rate=learning_rate,
            max_update=max_update,
        )
        _apply_precomputed(soft_model, soft_on_soft, soft_bias)
        _apply_precomputed(global_model, global_on_global, global_bias)

        soft_post = _objective_values(circuit, soft_model.weight.detach(), soft_model.bias.detach())
        global_post = _objective_values(
            circuit, global_model.weight.detach(), global_model.bias.detach()
        )
        values = {
            **{f"soft_path_{key}": value for key, value in soft_diag.items()},
            **{f"global_path_{key}": value for key, value in global_diag.items()},
            "post_weight_l2_distance": float(
                torch.linalg.vector_norm(soft_model.weight - global_model.weight)
            ),
            "post_ce_difference": soft_post[0] - global_post[0],
            "post_hard_margin_difference": soft_post[1] - global_post[1],
            "post_soft_margin_difference": soft_post[2] - global_post[2],
        }
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
            "Track how soft035 and L2-matched global035 create different credit geometry "
            "after branching from the same epoch-140 state."
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
        raise RuntimeError("incomplete or non-finite on-trajectory geometry diagnostic")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
