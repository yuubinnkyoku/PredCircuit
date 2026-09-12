from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from diagnose_flyvis_type_pair_adaptive_mean_gain_holdout import _standard_direction
from diagnose_flyvis_type_pair_horizon_causal import _match_local_state
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

HORIZONS = (60, 100)
EVAL_JITTER_BASE = 4_800_000
EVAL_REPS = 2
EVAL_REPEATS = 4
INIT_MODES = ("random", "biological_strength")
MEAN_DOMINANCE_THRESHOLD = 0.5
LOW_EXTRA_GAIN = 1.0
HIGH_EXTRA_GAIN = 3.0


def _heldout_objectives(
    circuit,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    jitter_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def _group_extra_update(
    values: torch.Tensor,
    *,
    learning_rate: float,
    max_update: float,
) -> torch.Tensor:
    mean = values.mean()
    direction_high = values + HIGH_EXTRA_GAIN * mean
    direction_low = values + LOW_EXTRA_GAIN * mean
    update_high = learning_rate * direction_high
    update_low = learning_rate * direction_low
    if max_update > 0.0:
        update_high = update_high.clamp(-max_update, max_update)
        update_low = update_low.clamp(-max_update, max_update)
    return update_high - update_low


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
    models = {
        mode: {
            "local": copy.deepcopy(base),
            "standard": copy.deepcopy(base),
        }
        for mode, base in bases.items()
    }
    rows: list[dict[str, float | int | bool | str]] = []

    for step in range(1, max(HORIZONS) + 1):
        epoch = step - 1
        for mode in INIT_MODES:
            local = models[mode]["local"]
            local_edge, local_bias = credit(local, circuit, seed=seed, epoch=epoch)
            apply_local_credit(
                local,
                local_edge,
                local_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )

            standard = models[mode]["standard"]
            raw_edge, raw_bias = credit(standard, circuit, seed=seed, epoch=epoch)
            apply_local_credit(
                standard,
                _standard_direction(raw_edge, named_groups),
                raw_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )
            _match_local_state(standard, local)

        if step not in HORIZONS:
            continue

        for mode in INIT_MODES:
            model = models[mode]["standard"]
            raw_edge, _ = credit(model, circuit, seed=seed, epoch=step)

            for eval_rep in range(EVAL_REPS):
                weight = model.weight.detach().clone().requires_grad_(True)
                bias = model.bias.detach().clone()
                ce, hard_margin, soft_margin = _heldout_objectives(
                    circuit,
                    weight,
                    bias,
                    jitter_seed=EVAL_JITTER_BASE + eval_rep,
                )
                grad_ce = torch.autograd.grad(ce, weight, retain_graph=True)[0]
                grad_hard = torch.autograd.grad(hard_margin, weight, retain_graph=True)[0]
                grad_soft = torch.autograd.grad(soft_margin, weight)[0]

                for group_index, (source_type, target_type, indices) in enumerate(named_groups):
                    values = raw_edge[indices]
                    mean = values.mean()
                    mean_component = mean.expand_as(values)
                    residual = values - mean
                    raw_norm = torch.linalg.vector_norm(values)
                    mean_norm = torch.linalg.vector_norm(mean_component)
                    residual_norm = torch.linalg.vector_norm(residual)
                    q = float(mean_norm / residual_norm.clamp_min(1e-30))
                    extra_update_group = _group_extra_update(
                        values,
                        learning_rate=learning_rate,
                        max_update=max_update,
                    )
                    extra_update = torch.zeros_like(raw_edge)
                    extra_update[indices] = extra_update_group
                    extra_norm = torch.linalg.vector_norm(extra_update_group)
                    values_out = {
                        "raw_norm": float(raw_norm),
                        "mean_norm": float(mean_norm),
                        "residual_norm": float(residual_norm),
                        "mean_energy_fraction": float(
                            mean_norm.square() / raw_norm.square().clamp_min(1e-30)
                        ),
                        "mean_to_residual_norm": q,
                        "flagged_mean_dominant": q >= MEAN_DOMINANCE_THRESHOLD,
                        "gain3_minus_gain1_update_norm": float(extra_norm),
                        "ce_grad_dot_gain3_extra": float(torch.dot(grad_ce, extra_update)),
                        "hard_margin_grad_dot_gain3_extra": float(
                            torch.dot(grad_hard, extra_update)
                        ),
                        "soft_margin_grad_dot_gain3_extra": float(
                            torch.dot(grad_soft, extra_update)
                        ),
                        "ce_grad_cos_gain3_extra": _cosine(grad_ce, extra_update),
                        "hard_margin_grad_cos_gain3_extra": _cosine(grad_hard, extra_update),
                        "soft_margin_grad_cos_gain3_extra": _cosine(grad_soft, extra_update),
                    }
                    finite = (
                        bool(torch.isfinite(model.weight).all())
                        and bool(torch.isfinite(model.bias).all())
                        and all(
                            math.isfinite(float(value))
                            for key, value in values_out.items()
                            if key != "flagged_mean_dominant"
                        )
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "init_mode": mode,
                            "horizon": step,
                            "eval_rep": eval_rep,
                            "group_index": group_index,
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
            "Measure whether type-pair mean dominance predicts the held-out gradient cost of "
            "keeping the fixed extra shared-credit gain at 3 instead of reducing it to 1."
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
