from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from diagnose_flyvis_type_pair_horizon_causal import _match_local_state, _mixed_direction
from diagnose_flyvis_type_pair_state_geometry import _group_constant_projection
from run_flyvis_exact_pc_gradient import differentiable_sequence
from run_flyvis_retinotopic_contrastive import (
    DIRECTIONS,
    output_nodes,
    render_motion_batch,
    targets_for,
)
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit
from run_flyvis_type_pair_consensus import type_pair_indices
from run_flyvis_type_pair_shuffle_control import shuffled_groups_like

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

HORIZONS = (60, 80, 100)
EVAL_REPS = 2
EVAL_JITTER_BASE = 2_500_000
EVAL_REPEATS = 4
GROUPINGS = ("bio", "shuffle")


def _heldout_ce(
    circuit,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    jitter_seed: int,
) -> torch.Tensor:
    directions = list(DIRECTIONS) * EVAL_REPEATS
    stimulus = render_motion_batch(
        circuit,
        directions,
        frames=7,
        width=0.5,
        jitter_seed=jitter_seed,
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
    return F.cross_entropy(state[:, output_nodes(circuit)], classes)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = (torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)).clamp_min(
        1e-30
    )
    return float(torch.dot(left, right) / denominator)


def run_seed(
    *,
    seed: int,
    shuffle_seed: int,
    learning_rate: float,
    max_update: float,
) -> pd.DataFrame:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    base = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    local = copy.deepcopy(base)
    type_ = copy.deepcopy(base)
    biological_groups = type_pair_indices(circuit)
    shuffled_groups = shuffled_groups_like(
        biological_groups,
        edge_count=base.weight.numel(),
        shuffle_seed=shuffle_seed,
    )
    rows: list[dict[str, float | int | bool | str]] = []

    for step in range(1, max(HORIZONS) + 1):
        epoch = step - 1
        local_edge, local_bias = credit(local, circuit, seed=seed, epoch=epoch)
        apply_local_credit(
            local,
            local_edge,
            local_bias,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )

        raw_edge, raw_bias = credit(type_, circuit, seed=seed, epoch=epoch)
        direction = _mixed_direction(raw_edge, biological_groups)
        apply_local_credit(
            type_,
            direction,
            raw_bias,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )
        _match_local_state(type_, local)

        if step not in HORIZONS:
            continue

        delta = type_.weight.detach() - local.weight.detach()
        for eval_rep in range(EVAL_REPS):
            jitter_seed = EVAL_JITTER_BASE + eval_rep
            weight = local.weight.detach().clone().requires_grad_(True)
            bias = local.bias.detach().clone()
            loss = _heldout_ce(circuit, weight, bias, jitter_seed=jitter_seed)
            (gradient,) = torch.autograd.grad(loss, weight, create_graph=True)

            for grouping, groups in (
                ("bio", biological_groups),
                ("shuffle", shuffled_groups),
            ):
                projection_delta = _group_constant_projection(delta, groups).detach()
                residual_delta = (delta - projection_delta).detach()

                grad_projection = torch.dot(gradient, projection_delta)
                grad_residual = torch.dot(gradient, residual_delta)

                residual_hvp = torch.autograd.grad(
                    grad_residual,
                    weight,
                    retain_graph=True,
                )[0]
                projection_hvp = torch.autograd.grad(
                    grad_projection,
                    weight,
                    retain_graph=True,
                )[0]
                cross_curvature = torch.dot(projection_delta, residual_hvp)
                projection_curvature = torch.dot(projection_delta, projection_hvp)
                residual_curvature = torch.dot(residual_delta, residual_hvp)

                values = {
                    "baseline_ce": float(loss.detach()),
                    "projection_norm": float(torch.linalg.vector_norm(projection_delta)),
                    "residual_norm": float(torch.linalg.vector_norm(residual_delta)),
                    "projection_residual_cosine": _cosine(projection_delta, residual_delta),
                    "grad_dot_projection": float(grad_projection.detach()),
                    "grad_dot_residual": float(grad_residual.detach()),
                    "projection_curvature": float(projection_curvature.detach()),
                    "residual_curvature": float(residual_curvature.detach()),
                    "cross_curvature": float(cross_curvature.detach()),
                    "predicted_ce_synergy": float(-cross_curvature.detach()),
                }
                finite = (
                    bool(torch.isfinite(local.weight).all())
                    and bool(torch.isfinite(type_.weight).all())
                    and all(math.isfinite(value) for value in values.values())
                )
                rows.append(
                    {
                        "seed": seed,
                        "horizon": step,
                        "eval_rep": eval_rep,
                        "grouping": grouping,
                        **values,
                        "finite": finite,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure held-out CE Hessian cross-curvature between the learned type-pair "
            "group-constant state displacement and its within-group residual displacement"
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shuffle-seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frame = run_seed(
        seed=args.seed,
        shuffle_seed=args.shuffle_seed,
        learning_rate=args.learning_rate,
        max_update=args.max_update,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
