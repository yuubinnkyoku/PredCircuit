from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_biological_logit_balance import _heldout_readout, _metrics
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from diagnose_flyvis_type_pair_signed_band_holdout import (
    HIGH_EXTRA_GAIN,
    HIGH_RHO_THRESHOLD,
    LOW_EXTRA_GAIN,
    SIGNED_BAND_THRESHOLD,
    _signed_band_direction,
)
from diagnose_flyvis_type_pair_signed_band_matched_control import _rho04_direction
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

HORIZONS = (80, 100)
INIT_MODES = ("random", "biological_strength")
EVAL_REPS = 4
EVAL_JITTER_BASE = 6_100_000
RANDOM_CANDIDATES = 256
RHO04_THRESHOLD = 0.4


def _clipped_update(
    direction: torch.Tensor, learning_rate: float, max_update: float
) -> torch.Tensor:
    update = learning_rate * direction
    return update.clamp(-max_update, max_update) if max_update > 0.0 else update


def _boundary_geometry(raw_edge: torch.Tensor, weight: torch.Tensor, named_groups):
    boundary = []
    signed_selected = []
    for group_index, (_, _, indices) in enumerate(named_groups):
        values = raw_edge[indices]
        mean = values.mean()
        residual = values - mean
        rho = float(
            torch.linalg.vector_norm(mean.expand_as(values))
            / torch.linalg.vector_norm(residual).clamp_min(1e-30)
        )
        proxy = float(weight[indices].mean() * mean)
        if SIGNED_BAND_THRESHOLD <= rho < HIGH_RHO_THRESHOLD:
            boundary.append(group_index)
            if proxy > 0.0:
                signed_selected.append(group_index)
    return boundary, signed_selected


def _masked_direction(raw_edge: torch.Tensor, named_groups, selected_boundary: set[int]):
    result = raw_edge.clone()
    for group_index, (_, _, indices) in enumerate(named_groups):
        values = raw_edge[indices]
        mean = values.mean()
        residual = values - mean
        rho = float(
            torch.linalg.vector_norm(mean.expand_as(values))
            / torch.linalg.vector_norm(residual).clamp_min(1e-30)
        )
        suppress = rho >= HIGH_RHO_THRESHOLD or group_index in selected_boundary
        extra_gain = LOW_EXTRA_GAIN if suppress else HIGH_EXTRA_GAIN
        result[indices] = values + extra_gain * mean
    return result


def _matched_random_direction(
    raw_edge: torch.Tensor,
    weight: torch.Tensor,
    named_groups,
    rho04_update: torch.Tensor,
    signed_update: torch.Tensor,
    *,
    seed: int,
    horizon: int,
    mode_index: int,
    learning_rate: float,
    max_update: float,
):
    boundary, signed_selected = _boundary_geometry(raw_edge, weight, named_groups)
    selected_count = len(signed_selected)
    target_norm = float(torch.linalg.vector_norm(signed_update - rho04_update))
    if selected_count == 0 or selected_count == len(boundary):
        direction = _masked_direction(raw_edge, named_groups, set(signed_selected))
        return direction, {
            "target_difference_norm": target_norm,
            "matched_difference_norm": target_norm,
            "relative_norm_error": 0.0,
            "boundary_group_count": len(boundary),
            "selected_group_count": selected_count,
            "selection_overlap_fraction": 1.0,
            "degenerate_match": True,
        }

    generator = torch.Generator().manual_seed(seed * 100_000 + horizon * 10 + mode_index)
    signed_set = set(signed_selected)
    best = None
    for _ in range(RANDOM_CANDIDATES):
        order = torch.randperm(len(boundary), generator=generator).tolist()
        chosen = {boundary[index] for index in order[:selected_count]}
        if chosen == signed_set:
            continue
        direction = _masked_direction(raw_edge, named_groups, chosen)
        update = _clipped_update(direction, learning_rate, max_update)
        difference_norm = float(torch.linalg.vector_norm(update - rho04_update))
        norm_error = abs(difference_norm - target_norm)
        overlap = len(chosen & signed_set) / max(selected_count, 1)
        score = (norm_error, overlap)
        if best is None or score < best[0]:
            best = (score, direction, difference_norm, overlap)

    if best is None:
        raise RuntimeError("failed to construct non-identical matched random mask")
    _, direction, difference_norm, overlap = best
    relative_error = abs(difference_norm - target_norm) / max(target_norm, 1e-30)
    return direction, {
        "target_difference_norm": target_norm,
        "matched_difference_norm": difference_norm,
        "relative_norm_error": relative_error,
        "boundary_group_count": len(boundary),
        "selected_group_count": selected_count,
        "selection_overlap_fraction": overlap,
        "degenerate_match": False,
    }


def _apply(model, direction, bias, learning_rate: float, max_update: float):
    apply_local_credit(
        model,
        direction,
        bias,
        learning_rate=learning_rate,
        weight_decay=0.0,
        max_update=max_update,
    )


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
    rows = []
    trained = 0
    for horizon in HORIZONS:
        for epoch in range(trained, horizon):
            for base in bases.values():
                raw_edge, raw_bias = credit(base, circuit, seed=seed, epoch=epoch)
                _apply(base, raw_edge, raw_bias, learning_rate, max_update)
        trained = horizon

        for mode_index, mode in enumerate(INIT_MODES):
            base = bases[mode]
            raw_edge, raw_bias = credit(base, circuit, seed=seed, epoch=horizon)
            rho_direction, _ = _rho04_direction(raw_edge, named_groups)
            signed_direction, _ = _signed_band_direction(
                raw_edge,
                base.weight.detach(),
                named_groups,
            )
            rho_update = _clipped_update(rho_direction, learning_rate, max_update)
            signed_update = _clipped_update(signed_direction, learning_rate, max_update)
            matched_direction, diagnostics = _matched_random_direction(
                raw_edge,
                base.weight.detach(),
                named_groups,
                rho_update,
                signed_update,
                seed=seed,
                horizon=horizon,
                mode_index=mode_index,
                learning_rate=learning_rate,
                max_update=max_update,
            )

            rho_model = copy.deepcopy(base)
            signed_model = copy.deepcopy(base)
            matched_model = copy.deepcopy(base)
            _apply(rho_model, rho_direction, raw_bias, learning_rate, max_update)
            _apply(signed_model, signed_direction, raw_bias, learning_rate, max_update)
            _apply(matched_model, matched_direction, raw_bias, learning_rate, max_update)

            for eval_rep in range(EVAL_REPS):
                jitter_seed = EVAL_JITTER_BASE + eval_rep
                base_readout, classes = _heldout_readout(base, circuit, jitter_seed=jitter_seed)
                outputs = {}
                for name, model in (
                    ("rho04", rho_model),
                    ("signed", signed_model),
                    ("matched_random", matched_model),
                ):
                    readout, paired_classes = _heldout_readout(
                        model,
                        circuit,
                        jitter_seed=jitter_seed,
                    )
                    if not torch.equal(classes, paired_classes):
                        raise RuntimeError("held-out class mismatch")
                    outputs[name] = _metrics(readout, classes, local_readout=base_readout)

                values = {}
                for metric in ("cross_entropy", "accuracy", "hard_margin", "soft_margin"):
                    values[f"signed_minus_matched_{metric}"] = (
                        outputs["signed"][metric] - outputs["matched_random"][metric]
                    )
                    values[f"signed_minus_rho04_{metric}"] = (
                        outputs["signed"][metric] - outputs["rho04"][metric]
                    )
                finite = (
                    all(
                        torch.isfinite(model.weight).all()
                        for model in (rho_model, signed_model, matched_model)
                    )
                    and all(
                        torch.isfinite(model.bias).all()
                        for model in (rho_model, signed_model, matched_model)
                    )
                    and all(math.isfinite(float(value)) for value in values.values())
                    and all(math.isfinite(float(value)) for value in diagnostics.values())
                )
                rows.append(
                    {
                        "seed": seed,
                        "init_mode": mode,
                        "horizon": horizon,
                        "eval_rep": eval_rep,
                        **values,
                        **diagnostics,
                        "finite": bool(finite),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "One-step signed-band causal control with a random boundary-group mask matched "
            "to the signed rule in selected-group count and clipped update-difference norm."
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = run_seed(seed=args.seed, learning_rate=args.learning_rate, max_update=args.max_update)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
