from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_biological_logit_balance import _heldout_readout, _metrics
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from diagnose_flyvis_type_pair_signed_band_holdout import _signed_band_direction
from diagnose_flyvis_type_pair_signed_band_matched_control import _rho04_direction
from diagnose_flyvis_type_pair_signed_band_norm_matched_random_control import (
    _apply,
    _boundary_geometry,
    _clipped_update,
    _masked_direction,
)
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

HORIZONS = (80, 100)
EVAL_REPS = 2
EVAL_JITTER_BASE = 6_200_000
RANDOM_POOL = 1024
ENSEMBLE_SIZE = 12


def _matched_ensemble(
    raw_edge: torch.Tensor,
    weight: torch.Tensor,
    named_groups,
    rho04_update: torch.Tensor,
    signed_update: torch.Tensor,
    *,
    seed: int,
    horizon: int,
    learning_rate: float,
    max_update: float,
):
    boundary, signed_selected = _boundary_geometry(raw_edge, weight, named_groups)
    selected_count = len(signed_selected)
    signed_set = set(signed_selected)
    target_norm = float(torch.linalg.vector_norm(signed_update - rho04_update))
    if selected_count == 0 or selected_count == len(boundary):
        raise RuntimeError("degenerate boundary selection cannot support an ensemble control")

    generator = torch.Generator().manual_seed(seed * 100_000 + horizon)
    candidates = {}
    for _ in range(RANDOM_POOL):
        order = torch.randperm(len(boundary), generator=generator).tolist()
        chosen = frozenset(boundary[index] for index in order[:selected_count])
        if chosen == signed_set or chosen in candidates:
            continue
        direction = _masked_direction(raw_edge, named_groups, set(chosen))
        update = _clipped_update(direction, learning_rate, max_update)
        norm = float(torch.linalg.vector_norm(update - rho04_update))
        relative_error = abs(norm - target_norm) / max(target_norm, 1e-30)
        overlap = len(chosen & signed_set) / max(selected_count, 1)
        candidates[chosen] = (relative_error, overlap, norm, direction)

    ranked = sorted(candidates.values(), key=lambda item: (item[0], item[1]))
    if len(ranked) < ENSEMBLE_SIZE:
        raise RuntimeError(f"only {len(ranked)} distinct random masks for ensemble")
    return ranked[:ENSEMBLE_SIZE], {
        "target_difference_norm": target_norm,
        "boundary_group_count": len(boundary),
        "selected_group_count": selected_count,
    }


def run_seed(*, seed: int, learning_rate: float, max_update: float) -> pd.DataFrame:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    named_groups = _named_groups(circuit)
    base = PredictiveCodingGraph(
        circuit.graph,
        seed=seed,
        init_scale=0.08,
        use_biological_strength=True,
    )
    rows = []
    trained = 0
    for horizon in HORIZONS:
        for epoch in range(trained, horizon):
            raw_edge, raw_bias = credit(base, circuit, seed=seed, epoch=epoch)
            _apply(base, raw_edge, raw_bias, learning_rate, max_update)
        trained = horizon

        raw_edge, raw_bias = credit(base, circuit, seed=seed, epoch=horizon)
        rho_direction, _ = _rho04_direction(raw_edge, named_groups)
        signed_direction, _ = _signed_band_direction(
            raw_edge,
            base.weight.detach(),
            named_groups,
        )
        rho_update = _clipped_update(rho_direction, learning_rate, max_update)
        signed_update = _clipped_update(signed_direction, learning_rate, max_update)
        controls, geometry = _matched_ensemble(
            raw_edge,
            base.weight.detach(),
            named_groups,
            rho_update,
            signed_update,
            seed=seed,
            horizon=horizon,
            learning_rate=learning_rate,
            max_update=max_update,
        )

        signed_model = copy.deepcopy(base)
        _apply(signed_model, signed_direction, raw_bias, learning_rate, max_update)
        control_models = []
        for relative_error, overlap, norm, direction in controls:
            model = copy.deepcopy(base)
            _apply(model, direction, raw_bias, learning_rate, max_update)
            control_models.append((model, relative_error, overlap, norm))

        for eval_rep in range(EVAL_REPS):
            jitter_seed = EVAL_JITTER_BASE + eval_rep
            base_readout, classes = _heldout_readout(base, circuit, jitter_seed=jitter_seed)
            signed_readout, signed_classes = _heldout_readout(
                signed_model,
                circuit,
                jitter_seed=jitter_seed,
            )
            if not torch.equal(classes, signed_classes):
                raise RuntimeError("held-out class mismatch")
            signed_metrics = _metrics(signed_readout, classes, local_readout=base_readout)

            for control_id, (model, relative_error, overlap, norm) in enumerate(control_models):
                readout, paired_classes = _heldout_readout(
                    model,
                    circuit,
                    jitter_seed=jitter_seed,
                )
                if not torch.equal(classes, paired_classes):
                    raise RuntimeError("held-out class mismatch")
                control_metrics = _metrics(readout, classes, local_readout=base_readout)
                values = {
                    f"signed_minus_control_{metric}": signed_metrics[metric]
                    - control_metrics[metric]
                    for metric in ("cross_entropy", "accuracy", "hard_margin", "soft_margin")
                }
                finite = (
                    bool(torch.isfinite(signed_model.weight).all())
                    and bool(torch.isfinite(signed_model.bias).all())
                    and bool(torch.isfinite(model.weight).all())
                    and bool(torch.isfinite(model.bias).all())
                    and all(math.isfinite(float(value)) for value in values.values())
                    and math.isfinite(relative_error)
                    and math.isfinite(overlap)
                    and math.isfinite(norm)
                )
                rows.append(
                    {
                        "seed": seed,
                        "horizon": horizon,
                        "eval_rep": eval_rep,
                        "control_id": control_id,
                        **values,
                        **geometry,
                        "control_difference_norm": norm,
                        "relative_norm_error": relative_error,
                        "selection_overlap_fraction": overlap,
                        "finite": finite,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the signed boundary-band rule with an ensemble of random boundary masks "
            "matched in selected-group count and clipped update-difference norm."
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
