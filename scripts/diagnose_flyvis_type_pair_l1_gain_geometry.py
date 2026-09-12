from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from diagnose_flyvis_type_pair_l1_gain_sweep import _direction
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from run_flyvis_exact_pc_gradient import differentiable_sequence
from run_flyvis_retinotopic_contrastive import (
    DIRECTIONS,
    output_nodes,
    render_motion_batch,
    targets_for,
)
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

HORIZONS = (0, 20, 60, 100)
EVAL_JITTER = 4_500_000
EVAL_REPEATS = 4
INIT_MODES = ("random", "biological_strength")


def _l1_components(raw: torch.Tensor, named_groups):
    mean_component = torch.zeros_like(raw)
    l1_raw = torch.zeros_like(raw)
    l1_mask = torch.zeros_like(raw, dtype=torch.bool)
    for _, target_type, indices in named_groups:
        if target_type != "L1":
            continue
        values = raw[indices]
        mean_component[indices] = values.mean()
        l1_raw[indices] = values
        l1_mask[indices] = True
    residual = l1_raw - mean_component
    return mean_component, residual, l1_mask


def _heldout_objectives(circuit, weight: torch.Tensor, bias: torch.Tensor):
    directions = list(DIRECTIONS) * EVAL_REPEATS
    stimulus = render_motion_batch(
        circuit,
        directions,
        frames=7,
        width=0.5,
        jitter_seed=EVAL_JITTER,
    )
    _, classes = targets_for(directions)
    state = differentiable_sequence(
        circuit,
        stimulus,
        weight,
        bias,
        frame_steps=2,
        step_size=0.015,
    )
    readout = state[:, output_nodes(circuit)]
    batch = torch.arange(len(classes))
    correct = readout[batch, classes]
    wrong = readout.clone()
    wrong[batch, classes] = -torch.inf
    max_wrong = wrong.max(dim=1).values
    wrong_lse = torch.logsumexp(wrong, dim=1)
    ce = F.cross_entropy(readout, classes)
    hard_margin = (correct - max_wrong).mean()
    soft_margin = (correct - wrong_lse).mean()
    return ce, hard_margin, soft_margin


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = (torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)).clamp_min(
        1e-30
    )
    return float(torch.dot(left, right) / denominator)


def _clipped_update(direction: torch.Tensor, *, learning_rate: float, max_update: float):
    update = learning_rate * direction
    if max_update > 0.0:
        update = update.clamp(-max_update, max_update)
    return update


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
            mean_component, residual, l1_mask = _l1_components(raw_edge, named_groups)
            l1_raw = torch.where(l1_mask, raw_edge, torch.zeros_like(raw_edge))
            direction_gain1 = _direction(raw_edge, named_groups, l1_gain=1.0)
            direction_gain3 = _direction(raw_edge, named_groups, l1_gain=3.0)
            update_gain1 = _clipped_update(
                direction_gain1,
                learning_rate=learning_rate,
                max_update=max_update,
            )
            update_gain3 = _clipped_update(
                direction_gain3,
                learning_rate=learning_rate,
                max_update=max_update,
            )
            extra_update = update_gain3 - update_gain1

            weight = model.weight.detach().clone().requires_grad_(True)
            bias = model.bias.detach().clone()
            ce, hard_margin, soft_margin = _heldout_objectives(circuit, weight, bias)
            grad_ce = torch.autograd.grad(ce, weight, retain_graph=True)[0]
            grad_hard = torch.autograd.grad(hard_margin, weight, retain_graph=True)[0]
            grad_soft = torch.autograd.grad(soft_margin, weight)[0]

            raw_norm = torch.linalg.vector_norm(l1_raw)
            mean_norm = torch.linalg.vector_norm(mean_component)
            residual_norm = torch.linalg.vector_norm(residual)
            extra_norm = torch.linalg.vector_norm(extra_update)
            l1_count = int(l1_mask.sum())
            gain1_clip = (
                (learning_rate * direction_gain1[l1_mask]).abs() >= max_update
                if max_update > 0.0
                else torch.zeros(l1_count, dtype=torch.bool)
            )
            gain3_clip = (
                (learning_rate * direction_gain3[l1_mask]).abs() >= max_update
                if max_update > 0.0
                else torch.zeros(l1_count, dtype=torch.bool)
            )

            values = {
                "l1_raw_norm": float(raw_norm),
                "l1_mean_norm": float(mean_norm),
                "l1_residual_norm": float(residual_norm),
                "l1_mean_energy_fraction": float(
                    mean_norm.square() / raw_norm.square().clamp_min(1e-30)
                ),
                "l1_mean_to_residual_norm": float(mean_norm / residual_norm.clamp_min(1e-30)),
                "gain1_gain3_direction_cosine": _cosine(direction_gain1, direction_gain3),
                "gain3_extra_update_norm": float(extra_norm),
                "gain3_extra_update_relative_norm": float(
                    extra_norm / torch.linalg.vector_norm(update_gain1).clamp_min(1e-30)
                ),
                "gain1_l1_clip_fraction": float(gain1_clip.float().mean()),
                "gain3_l1_clip_fraction": float(gain3_clip.float().mean()),
                "heldout_ce": float(ce.detach()),
                "heldout_hard_margin": float(hard_margin.detach()),
                "heldout_soft_margin": float(soft_margin.detach()),
                "ce_grad_dot_gain3_extra_update": float(torch.dot(grad_ce, extra_update)),
                "hard_margin_grad_dot_gain3_extra_update": float(
                    torch.dot(grad_hard, extra_update)
                ),
                "soft_margin_grad_dot_gain3_extra_update": float(
                    torch.dot(grad_soft, extra_update)
                ),
                "ce_grad_cos_l1_mean": _cosine(grad_ce, mean_component),
                "hard_margin_grad_cos_l1_mean": _cosine(grad_hard, mean_component),
                "soft_margin_grad_cos_l1_mean": _cosine(grad_soft, mean_component),
            }
            finite = (
                bool(torch.isfinite(model.weight).all())
                and bool(torch.isfinite(model.bias).all())
                and all(math.isfinite(value) for value in values.values())
            )
            rows.append(
                {
                    "seed": seed,
                    "init_mode": mode,
                    "horizon": horizon,
                    "l1_edge_count": l1_count,
                    **values,
                    "finite": finite,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare random and biological-strength initialization for the local geometry "
            "that makes L1 shared-credit gain 3 differ from gain 1"
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
