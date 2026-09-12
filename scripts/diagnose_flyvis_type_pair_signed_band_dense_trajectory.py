from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_biological_logit_balance import _heldout_readout, _metrics
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from diagnose_flyvis_type_pair_signed_band_branched_trajectory import _selection_geometry
from diagnose_flyvis_type_pair_signed_band_holdout import _signed_band_direction
from diagnose_flyvis_type_pair_signed_band_matched_control import _rho04_direction
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

BRANCH_HORIZON = 80
BRANCH_STEPS = tuple(range(21))
EVAL_REPS = 2
EVAL_JITTER_BASE = 5_900_000


def _apply_direction(
    model: PredictiveCodingGraph,
    direction: torch.Tensor,
    bias_direction: torch.Tensor,
    *,
    learning_rate: float,
    max_update: float,
) -> None:
    apply_local_credit(
        model,
        direction,
        bias_direction,
        learning_rate=learning_rate,
        weight_decay=0.0,
        max_update=max_update,
    )


def run_seed(*, seed: int, learning_rate: float, max_update: float) -> pd.DataFrame:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    named_groups = _named_groups(circuit)
    base = PredictiveCodingGraph(
        circuit.graph,
        seed=seed,
        init_scale=0.08,
        use_biological_strength=True,
    )

    for epoch in range(BRANCH_HORIZON):
        raw_edge, raw_bias = credit(base, circuit, seed=seed, epoch=epoch)
        _apply_direction(
            base,
            raw_edge,
            raw_bias,
            learning_rate=learning_rate,
            max_update=max_update,
        )

    rho_model = copy.deepcopy(base)
    signed_model = copy.deepcopy(base)
    rows: list[dict[str, float | int | bool]] = []

    for branch_step in BRANCH_STEPS:
        if branch_step > 0:
            update_epoch = BRANCH_HORIZON + branch_step - 1
            rho_raw, rho_bias = credit(rho_model, circuit, seed=seed, epoch=update_epoch)
            signed_raw, signed_bias = credit(
                signed_model,
                circuit,
                seed=seed,
                epoch=update_epoch,
            )
            rho_geometry = _selection_geometry(
                rho_raw,
                rho_model.weight.detach(),
                named_groups,
            )
            signed_geometry = _selection_geometry(
                signed_raw,
                signed_model.weight.detach(),
                named_groups,
            )
            rho_direction, _ = _rho04_direction(rho_raw, named_groups)
            signed_direction, _ = _signed_band_direction(
                signed_raw,
                signed_model.weight.detach(),
                named_groups,
            )
            rho_update = (learning_rate * rho_direction).clamp(-max_update, max_update)
            signed_update = (learning_rate * signed_direction).clamp(-max_update, max_update)
            update_difference_norm = float(torch.linalg.vector_norm(signed_update - rho_update))
            _apply_direction(
                rho_model,
                rho_direction,
                rho_bias,
                learning_rate=learning_rate,
                max_update=max_update,
            )
            _apply_direction(
                signed_model,
                signed_direction,
                signed_bias,
                learning_rate=learning_rate,
                max_update=max_update,
            )
        else:
            rho_raw, _ = credit(rho_model, circuit, seed=seed, epoch=BRANCH_HORIZON)
            signed_raw, _ = credit(signed_model, circuit, seed=seed, epoch=BRANCH_HORIZON)
            rho_geometry = _selection_geometry(
                rho_raw,
                rho_model.weight.detach(),
                named_groups,
            )
            signed_geometry = _selection_geometry(
                signed_raw,
                signed_model.weight.detach(),
                named_groups,
            )
            update_difference_norm = 0.0

        weight_distance = float(torch.linalg.vector_norm(signed_model.weight - rho_model.weight))
        bias_distance = float(torch.linalg.vector_norm(signed_model.bias - rho_model.bias))

        for eval_rep in range(EVAL_REPS):
            jitter_seed = EVAL_JITTER_BASE + eval_rep
            baseline_readout, classes = _heldout_readout(base, circuit, jitter_seed=jitter_seed)
            rho_readout, rho_classes = _heldout_readout(
                rho_model,
                circuit,
                jitter_seed=jitter_seed,
            )
            signed_readout, signed_classes = _heldout_readout(
                signed_model,
                circuit,
                jitter_seed=jitter_seed,
            )
            if not torch.equal(classes, rho_classes) or not torch.equal(
                classes, signed_classes
            ):
                raise RuntimeError("held-out class mismatch")
            rho_metrics = _metrics(rho_readout, classes, local_readout=baseline_readout)
            signed_metrics = _metrics(signed_readout, classes, local_readout=baseline_readout)
            metric_names = ("cross_entropy", "accuracy", "hard_margin", "soft_margin")
            values = {
                **{
                    f"signed_minus_rho04_{key}": signed_metrics[key] - rho_metrics[key]
                    for key in metric_names
                },
                "branch_weight_distance": weight_distance,
                "branch_bias_distance": bias_distance,
                "current_update_difference_norm": update_difference_norm,
                **{f"rho04_pre_{key}": value for key, value in rho_geometry.items()},
                **{f"signed_pre_{key}": value for key, value in signed_geometry.items()},
            }
            finite = (
                bool(torch.isfinite(rho_model.weight).all())
                and bool(torch.isfinite(rho_model.bias).all())
                and bool(torch.isfinite(signed_model.weight).all())
                and bool(torch.isfinite(signed_model.bias).all())
                and all(math.isfinite(float(value)) for value in values.values())
            )
            rows.append(
                {
                    "seed": seed,
                    "branch_horizon": BRANCH_HORIZON,
                    "branch_step": branch_step,
                    "eval_rep": eval_rep,
                    **values,
                    "finite": finite,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Dense biological-strength trajectory after branching rho>=0.4 and signed-band "
            "rules from the same epoch-80 local checkpoint."
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
