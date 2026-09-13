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
RULES = ("local", "standard_clip", "standard_group_l2", "standard_global_l2")


def _l2_match(raw: torch.Tensor, clipped: torch.Tensor) -> torch.Tensor:
    raw_norm = torch.linalg.vector_norm(raw)
    if float(raw_norm) == 0.0:
        return torch.zeros_like(raw)
    return raw * (torch.linalg.vector_norm(clipped) / raw_norm)


@torch.no_grad()
def _apply_l2_matched(
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
    clipped_edge = raw_edge.clamp(-max_update, max_update)

    if mode == "global":
        edge_update = _l2_match(raw_edge, clipped_edge)
    elif mode == "group":
        edge_update = raw_edge.clone()
        for _, _, indices in groups:
            raw_group = raw_edge[indices]
            clipped_group = clipped_edge[indices]
            edge_update[indices] = _l2_match(raw_group, clipped_group)
    else:
        raise ValueError(f"unknown L2-match mode: {mode}")

    # Keep bias treatment identical to the ordinary elementwise-clipped rule so
    # the intervention isolates edge-update geometry.
    bias_update = (learning_rate * bias_direction).clamp(-max_update, max_update)
    model.weight.add_(edge_update)
    model.bias.add_(bias_update)


def _geometry(
    direction: torch.Tensor,
    *,
    learning_rate: float,
    max_update: float,
    groups: list[tuple[str, str, torch.Tensor]],
    mode: str,
) -> tuple[float, float, float]:
    raw = learning_rate * direction
    clipped = raw.clamp(-max_update, max_update)
    if mode == "clip":
        applied = clipped
    elif mode == "global":
        applied = _l2_match(raw, clipped)
    elif mode == "group":
        applied = raw.clone()
        for _, _, indices in groups:
            applied[indices] = _l2_match(raw[indices], clipped[indices])
    else:
        raise ValueError(mode)

    raw_norm = torch.linalg.vector_norm(raw).clamp_min(1e-30)
    applied_norm = torch.linalg.vector_norm(applied).clamp_min(1e-30)
    cosine = float(torch.dot(raw, applied) / (raw_norm * applied_norm))
    norm_ratio = float(applied_norm / raw_norm)
    clip_norm = torch.linalg.vector_norm(clipped).clamp_min(1e-30)
    clip_l2_ratio = float(applied_norm / clip_norm)
    return cosine, norm_ratio, clip_l2_ratio


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
        branch_direction = _standard_direction(branch_raw, groups)
        geometry: dict[str, tuple[float, float, float]] = {}
        for mode in ("clip", "group", "global"):
            geometry[mode] = _geometry(
                branch_direction,
                learning_rate=learning_rate,
                max_update=max_update,
                groups=groups,
                mode=mode,
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
                elif rule == "standard_group_l2":
                    _apply_l2_matched(
                        model,
                        direction,
                        raw_bias,
                        learning_rate=learning_rate,
                        max_update=max_update,
                        groups=groups,
                        mode="group",
                    )
                else:
                    _apply_l2_matched(
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
                    "branch_clip_direction_cosine": geometry["clip"][0],
                    "branch_group_direction_cosine": geometry["group"][0],
                    "branch_global_direction_cosine": geometry["global"][0],
                    "branch_clip_norm_ratio": geometry["clip"][1],
                    "branch_group_norm_ratio": geometry["group"][1],
                    "branch_global_norm_ratio": geometry["global"][1],
                    "branch_group_to_clip_l2": geometry["group"][2],
                    "branch_global_to_clip_l2": geometry["global"][2],
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
            "Test whether late 4m+r collapse comes from clipping-induced direction "
            "distortion while matching the elementwise-clipped L2 update magnitude."
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
