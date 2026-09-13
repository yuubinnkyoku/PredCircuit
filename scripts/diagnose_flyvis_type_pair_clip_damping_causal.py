from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_adaptive_mean_gain_holdout import _standard_direction
from diagnose_flyvis_type_pair_biological_logit_balance import _heldout_readout
from diagnose_flyvis_type_pair_l1_gain_geometry import EVAL_JITTER, _heldout_objectives
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

INIT_SCALES = (0.100, 0.110, 0.115)
BRANCH_EPOCH = 180
FINAL_EPOCH = 200
RULES = (
    "local",
    "standard_clip",
    "standard_group_rescale",
    "standard_clip_groupnorm",
)


def _group_rescaled(
    raw_edge: torch.Tensor,
    *,
    max_update: float,
    groups: list[tuple[str, str, torch.Tensor]],
) -> torch.Tensor:
    applied = raw_edge.clone()
    for _, _, indices in groups:
        raw_group = raw_edge[indices]
        peak = raw_group.abs().max()
        if float(peak) > max_update:
            applied[indices] = raw_group * (max_update / peak)
    return applied


def _clip_groupnorm_matched(
    raw_edge: torch.Tensor,
    *,
    max_update: float,
    groups: list[tuple[str, str, torch.Tensor]],
) -> torch.Tensor:
    clipped = raw_edge.clamp(-max_update, max_update)
    target = _group_rescaled(raw_edge, max_update=max_update, groups=groups)
    applied = clipped.clone()
    for _, _, indices in groups:
        clipped_group = clipped[indices]
        target_group = target[indices]
        clipped_norm = torch.linalg.vector_norm(clipped_group)
        if float(clipped_norm) == 0.0:
            applied[indices] = torch.zeros_like(clipped_group)
        else:
            target_norm = torch.linalg.vector_norm(target_group)
            applied[indices] = clipped_group * (target_norm / clipped_norm)
    return applied


@torch.no_grad()
def _apply_edge_rule(
    model: PredictiveCodingGraph,
    edge_direction: torch.Tensor,
    bias_direction: torch.Tensor,
    *,
    learning_rate: float,
    max_update: float,
    groups: list[tuple[str, str, torch.Tensor]],
    mode: str,
) -> None:
    raw_edge = learning_rate * edge_direction
    if mode == "group_rescale":
        edge_update = _group_rescaled(
            raw_edge,
            max_update=max_update,
            groups=groups,
        )
    elif mode == "clip_groupnorm":
        edge_update = _clip_groupnorm_matched(
            raw_edge,
            max_update=max_update,
            groups=groups,
        )
    else:
        raise ValueError(f"unknown edge-update mode: {mode}")

    # Bias handling is held fixed so the intervention isolates edge-update
    # magnitude versus direction within each type-pair group.
    bias_update = (learning_rate * bias_direction).clamp(-max_update, max_update)
    model.weight.add_(edge_update)
    model.bias.add_(bias_update)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = (
        torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    ).clamp_min(1e-30)
    return float(torch.dot(left, right) / denominator)


def _geometry(
    direction: torch.Tensor,
    *,
    learning_rate: float,
    max_update: float,
    groups: list[tuple[str, str, torch.Tensor]],
) -> dict[str, float]:
    raw = learning_rate * direction
    clipped = raw.clamp(-max_update, max_update)
    group_rescale = _group_rescaled(raw, max_update=max_update, groups=groups)
    clip_groupnorm = _clip_groupnorm_matched(
        raw,
        max_update=max_update,
        groups=groups,
    )
    raw_norm = torch.linalg.vector_norm(raw).clamp_min(1e-30)
    clip_norm = torch.linalg.vector_norm(clipped).clamp_min(1e-30)
    group_norm = torch.linalg.vector_norm(group_rescale).clamp_min(1e-30)
    matched_norm = torch.linalg.vector_norm(clip_groupnorm).clamp_min(1e-30)
    return {
        "branch_clip_raw_cosine": _cosine(raw, clipped),
        "branch_group_raw_cosine": _cosine(raw, group_rescale),
        "branch_matched_raw_cosine": _cosine(raw, clip_groupnorm),
        "branch_group_vs_matched_cosine": _cosine(group_rescale, clip_groupnorm),
        "branch_clip_norm_ratio": float(clip_norm / raw_norm),
        "branch_group_norm_ratio": float(group_norm / raw_norm),
        "branch_matched_norm_ratio": float(matched_norm / raw_norm),
        "branch_group_to_matched_l2": float(group_norm / matched_norm),
        "branch_group_to_clip_l2": float(group_norm / clip_norm),
    }


def run_seed(
    *,
    seed: int,
    learning_rate: float = 160.0,
    max_update: float = 0.05,
) -> pd.DataFrame:
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
        for step in range(1, BRANCH_EPOCH + 1):
            raw_edge, raw_bias = credit(base, circuit, seed=seed, epoch=step - 1)
            apply_local_credit(
                base,
                _standard_direction(raw_edge, groups),
                raw_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )

        models = {rule: copy.deepcopy(base) for rule in RULES}
        branch_raw, _ = credit(base, circuit, seed=seed, epoch=BRANCH_EPOCH)
        geometry = _geometry(
            _standard_direction(branch_raw, groups),
            learning_rate=learning_rate,
            max_update=max_update,
            groups=groups,
        )

        for step in range(BRANCH_EPOCH + 1, FINAL_EPOCH + 1):
            epoch = step - 1
            for rule, model in models.items():
                raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
                direction = raw_edge if rule == "local" else _standard_direction(raw_edge, groups)
                if rule in {"local", "standard_clip"}:
                    apply_local_credit(
                        model,
                        direction,
                        raw_bias,
                        learning_rate=learning_rate,
                        weight_decay=0.0,
                        max_update=max_update,
                    )
                elif rule == "standard_group_rescale":
                    _apply_edge_rule(
                        model,
                        direction,
                        raw_bias,
                        learning_rate=learning_rate,
                        max_update=max_update,
                        groups=groups,
                        mode="group_rescale",
                    )
                else:
                    _apply_edge_rule(
                        model,
                        direction,
                        raw_bias,
                        learning_rate=learning_rate,
                        max_update=max_update,
                        groups=groups,
                        mode="clip_groupnorm",
                    )

        for rule, model in models.items():
            weight = model.weight.detach().clone().requires_grad_(True)
            bias = model.bias.detach().clone()
            ce, hard_margin, soft_margin = _heldout_objectives(circuit, weight, bias)
            readout, classes = _heldout_readout(model, circuit, jitter_seed=EVAL_JITTER)
            values = {
                "cross_entropy": float(ce.detach()),
                "accuracy": float((readout.argmax(dim=1) == classes).float().mean()),
                "hard_margin": float(hard_margin.detach()),
                "soft_margin": float(soft_margin.detach()),
                "weight_norm": float(torch.linalg.vector_norm(model.weight.detach())),
            }
            rows.append(
                {
                    "seed": seed,
                    "init_scale": scale,
                    "rule": rule,
                    "branch_epoch": BRANCH_EPOCH,
                    "final_epoch": FINAL_EPOCH,
                    **geometry,
                    **values,
                    "finite": bool(torch.isfinite(model.weight).all())
                    and bool(torch.isfinite(model.bias).all())
                    and all(math.isfinite(value) for value in values.values()),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Separate update-magnitude damping from clipping direction: compare "
            "type-pair radial rescaling with an elementwise-clipped update matched "
            "to the exact same per-group L2 magnitude."
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
