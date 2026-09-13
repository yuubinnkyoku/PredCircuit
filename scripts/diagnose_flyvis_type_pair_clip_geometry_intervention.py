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

INIT_SCALES = (0.100, 0.110, 0.115)
BRANCH_EPOCH = 180
FINAL_EPOCH = 200
RULES = ("local", "standard_clip", "standard_group_rescale", "standard_global_rescale")


@torch.no_grad()
def _apply_direction_preserving(
    model: PredictiveCodingGraph,
    edge_direction: torch.Tensor,
    bias_direction: torch.Tensor,
    *,
    learning_rate: float,
    max_update: float,
    groups: list[tuple[str, str, torch.Tensor]],
    mode: str,
) -> None:
    edge_update = learning_rate * edge_direction
    bias_update = learning_rate * bias_direction

    if max_update > 0.0:
        if mode == "global":
            peak = edge_update.abs().max()
            if float(peak) > max_update:
                edge_update.mul_(max_update / peak)
        elif mode == "group":
            for _, _, indices in groups:
                peak = edge_update[indices].abs().max()
                if float(peak) > max_update:
                    edge_update[indices].mul_(max_update / peak)
        else:
            raise ValueError(f"unknown rescale mode: {mode}")
        bias_update.clamp_(-max_update, max_update)

    model.weight.add_(edge_update)
    model.bias.add_(bias_update)


def _update_geometry(
    direction: torch.Tensor,
    *,
    learning_rate: float,
    max_update: float,
    groups: list[tuple[str, str, torch.Tensor]],
    mode: str,
) -> tuple[float, float]:
    raw_update = learning_rate * direction
    if mode == "clip":
        applied = raw_update.clamp(-max_update, max_update)
    elif mode == "global":
        applied = raw_update.clone()
        peak = applied.abs().max()
        if float(peak) > max_update:
            applied.mul_(max_update / peak)
    elif mode == "group":
        applied = raw_update.clone()
        for _, _, indices in groups:
            peak = applied[indices].abs().max()
            if float(peak) > max_update:
                applied[indices].mul_(max_update / peak)
    else:
        raise ValueError(mode)

    denominator = (
        torch.linalg.vector_norm(raw_update) * torch.linalg.vector_norm(applied)
    ).clamp_min(1e-30)
    cosine = float(torch.dot(raw_update, applied) / denominator)
    norm_ratio = float(
        torch.linalg.vector_norm(applied)
        / torch.linalg.vector_norm(raw_update).clamp_min(1e-30)
    )
    return cosine, norm_ratio


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
        branch_standard = _standard_direction(branch_raw, groups)
        clip_cos, clip_ratio = _update_geometry(
            branch_standard,
            learning_rate=learning_rate,
            max_update=max_update,
            groups=groups,
            mode="clip",
        )
        group_cos, group_ratio = _update_geometry(
            branch_standard,
            learning_rate=learning_rate,
            max_update=max_update,
            groups=groups,
            mode="group",
        )
        global_cos, global_ratio = _update_geometry(
            branch_standard,
            learning_rate=learning_rate,
            max_update=max_update,
            groups=groups,
            mode="global",
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
                    _apply_direction_preserving(
                        model,
                        direction,
                        raw_bias,
                        learning_rate=learning_rate,
                        max_update=max_update,
                        groups=groups,
                        mode="group",
                    )
                else:
                    _apply_direction_preserving(
                        model,
                        direction,
                        raw_bias,
                        learning_rate=learning_rate,
                        max_update=max_update,
                        groups=groups,
                        mode="global",
                    )

        for rule, model in models.items():
            weight = model.weight.detach().clone().requires_grad_(True)
            bias = model.bias.detach().clone()
            ce, hard_margin, soft_margin = _heldout_objectives(circuit, weight, bias)
            values = {
                "cross_entropy": float(ce.detach()),
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
                    "branch_clip_direction_cosine": clip_cos,
                    "branch_group_direction_cosine": group_cos,
                    "branch_global_direction_cosine": global_cos,
                    "branch_clip_norm_ratio": clip_ratio,
                    "branch_group_norm_ratio": group_ratio,
                    "branch_global_norm_ratio": global_ratio,
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
            "Causally test whether elementwise update clipping drives the late "
            "4m+r collapse by comparing it with direction-preserving rescaling."
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
