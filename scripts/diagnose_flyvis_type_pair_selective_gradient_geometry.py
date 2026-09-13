from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_adaptive_mean_gain_holdout import _standard_direction
from diagnose_flyvis_type_pair_l1_gain_geometry import _cosine, _heldout_objectives
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

INIT_SCALES = (0.10, 0.115)
HORIZONS = (160, 180, 200)
PROBABILITY = 0.35
PEAK_THRESHOLD = 0.25


def _candidate_mask(
    direction: torch.Tensor,
    *,
    learning_rate: float,
    max_update: float,
    groups: list[tuple[str, str, torch.Tensor]],
) -> torch.Tensor:
    raw = learning_rate * direction
    mask = torch.zeros_like(raw, dtype=torch.bool)
    for _, _, indices in groups:
        raw_group = raw[indices]
        raw_norm = torch.linalg.vector_norm(raw_group)
        peak = raw_group.abs().max()
        if float(raw_norm) == 0.0 or float(peak) <= max_update:
            peak_ratio = 1.0
        else:
            peak_ratio = float(max_update / peak)
        if peak_ratio < PEAK_THRESHOLD:
            mask[indices] = True
    return mask


def _masked_cosine(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    return _cosine(left[mask], right[mask])


def _objective_values(circuit, weight: torch.Tensor, bias: torch.Tensor) -> tuple[float, float, float]:
    ce, hard_margin, soft_margin = _heldout_objectives(circuit, weight, bias)
    return float(ce.detach()), float(hard_margin.detach()), float(soft_margin.detach())


def run_seed(*, seed: int, learning_rate: float = 160.0, max_update: float = 0.05) -> pd.DataFrame:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    groups = _named_groups(circuit)
    rows: list[dict[str, float | int | bool]] = []

    for scale in INIT_SCALES:
        model = PredictiveCodingGraph(
            circuit.graph,
            seed=seed,
            init_scale=scale,
            use_biological_strength=True,
        )
        trained = 0
        for horizon in HORIZONS:
            for epoch in range(trained, horizon):
                raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
                direction = _standard_direction(raw_edge, groups)
                apply_local_credit(
                    model,
                    direction,
                    raw_bias,
                    learning_rate=learning_rate,
                    weight_decay=0.0,
                    max_update=max_update,
                )
            trained = horizon

            raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=horizon)
            direction = _standard_direction(raw_edge, groups)
            standard_update = (learning_rate * direction).clamp(-max_update, max_update)
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
            candidate_mask = _candidate_mask(
                direction,
                learning_rate=learning_rate,
                max_update=max_update,
                groups=groups,
            )
            noncandidate_mask = ~candidate_mask

            weight = model.weight.detach().clone().requires_grad_(True)
            bias = model.bias.detach().clone()
            ce, hard_margin, soft_margin = _heldout_objectives(circuit, weight, bias)
            grad_ce = torch.autograd.grad(ce, weight, retain_graph=True)[0]
            grad_hard = torch.autograd.grad(hard_margin, weight, retain_graph=True)[0]
            grad_soft = torch.autograd.grad(soft_margin, weight)[0]

            redistribution = soft_update - global_update
            soft_removed = standard_update - soft_update
            global_removed = standard_update - global_update
            base_ce = float(ce.detach())
            base_hard = float(hard_margin.detach())
            base_soft = float(soft_margin.detach())
            soft_next = _objective_values(
                circuit,
                model.weight.detach() + soft_update,
                model.bias.detach() + bias_update,
            )
            global_next = _objective_values(
                circuit,
                model.weight.detach() + global_update,
                model.bias.detach() + bias_update,
            )

            values = {
                "base_ce": base_ce,
                "base_hard_margin": base_hard,
                "base_soft_margin": base_soft,
                "candidate_edge_fraction": float(candidate_mask.float().mean()),
                "candidate_group_fraction": soft_diag["candidate_group_fraction"],
                "soft_update_l2": float(torch.linalg.vector_norm(soft_update)),
                "global_update_l2": float(torch.linalg.vector_norm(global_update)),
                "l2_match_relative_error": global_diag["matched_l2_relative_error"],
                "redistribution_l2": float(torch.linalg.vector_norm(redistribution)),
                "ce_grad_dot_standard": float(torch.dot(grad_ce, standard_update)),
                "ce_grad_dot_soft": float(torch.dot(grad_ce, soft_update)),
                "ce_grad_dot_global": float(torch.dot(grad_ce, global_update)),
                "hard_grad_dot_standard": float(torch.dot(grad_hard, standard_update)),
                "hard_grad_dot_soft": float(torch.dot(grad_hard, soft_update)),
                "hard_grad_dot_global": float(torch.dot(grad_hard, global_update)),
                "soft_grad_dot_standard": float(torch.dot(grad_soft, standard_update)),
                "soft_grad_dot_soft": float(torch.dot(grad_soft, soft_update)),
                "soft_grad_dot_global": float(torch.dot(grad_soft, global_update)),
                "ce_grad_dot_redistribution": float(torch.dot(grad_ce, redistribution)),
                "hard_grad_dot_redistribution": float(torch.dot(grad_hard, redistribution)),
                "soft_grad_dot_redistribution": float(torch.dot(grad_soft, redistribution)),
                "ce_grad_dot_soft_removed": float(torch.dot(grad_ce, soft_removed)),
                "hard_grad_dot_soft_removed": float(torch.dot(grad_hard, soft_removed)),
                "soft_grad_dot_soft_removed": float(torch.dot(grad_soft, soft_removed)),
                "ce_grad_dot_global_removed": float(torch.dot(grad_ce, global_removed)),
                "hard_grad_dot_global_removed": float(torch.dot(grad_hard, global_removed)),
                "soft_grad_dot_global_removed": float(torch.dot(grad_soft, global_removed)),
                "candidate_ce_grad_cos_standard": _masked_cosine(
                    grad_ce, standard_update, candidate_mask
                ),
                "candidate_hard_grad_cos_standard": _masked_cosine(
                    grad_hard, standard_update, candidate_mask
                ),
                "candidate_soft_grad_cos_standard": _masked_cosine(
                    grad_soft, standard_update, candidate_mask
                ),
                "noncandidate_ce_grad_cos_standard": _masked_cosine(
                    grad_ce, standard_update, noncandidate_mask
                ),
                "noncandidate_hard_grad_cos_standard": _masked_cosine(
                    grad_hard, standard_update, noncandidate_mask
                ),
                "noncandidate_soft_grad_cos_standard": _masked_cosine(
                    grad_soft, standard_update, noncandidate_mask
                ),
                "soft_next_ce_delta": soft_next[0] - base_ce,
                "soft_next_hard_delta": soft_next[1] - base_hard,
                "soft_next_soft_delta": soft_next[2] - base_soft,
                "global_next_ce_delta": global_next[0] - base_ce,
                "global_next_hard_delta": global_next[1] - base_hard,
                "global_next_soft_delta": global_next[2] - base_soft,
                "soft_minus_global_next_ce": soft_next[0] - global_next[0],
                "soft_minus_global_next_hard": soft_next[1] - global_next[1],
                "soft_minus_global_next_soft": soft_next[2] - global_next[2],
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
                    "horizon": horizon,
                    **values,
                    "finite": finite,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether type-pair selective attenuation redistributes an exactly "
            "L2-matched update toward better held-out CE and margin gradient geometry."
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
