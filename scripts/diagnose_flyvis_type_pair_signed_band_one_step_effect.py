from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_group_signed_geometry import _heldout_objectives
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from diagnose_flyvis_type_pair_signed_band_holdout import _signed_band_direction
from diagnose_flyvis_type_pair_signed_band_matched_control import _rho04_direction
from diagnose_flyvis_type_pair_signed_band_swap_update_geometry import _clipped_update
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

HORIZONS = (60, 80, 100)
INIT_MODES = ("random", "biological_strength")


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
            raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=horizon)
            rho04_direction, rho04_diag = _rho04_direction(raw_edge, named_groups)
            signed_direction, signed_diag = _signed_band_direction(
                raw_edge,
                model.weight.detach(),
                named_groups,
            )
            rho04_update = _clipped_update(
                rho04_direction,
                learning_rate=learning_rate,
                max_update=max_update,
            )
            signed_update = _clipped_update(
                signed_direction,
                learning_rate=learning_rate,
                max_update=max_update,
            )
            bias_update = _clipped_update(
                raw_bias,
                learning_rate=learning_rate,
                max_update=max_update,
            )

            weight0 = model.weight.detach().clone()
            bias0 = model.bias.detach().clone()
            ce0, hard0, soft0 = _heldout_objectives(circuit, weight0, bias0)
            ce_rho04, hard_rho04, soft_rho04 = _heldout_objectives(
                circuit,
                weight0 + rho04_update,
                bias0 + bias_update,
            )
            ce_signed, hard_signed, soft_signed = _heldout_objectives(
                circuit,
                weight0 + signed_update,
                bias0 + bias_update,
            )

            values_out = {
                "heldout_ce_before": float(ce0),
                "heldout_hard_margin_before": float(hard0),
                "heldout_soft_margin_before": float(soft0),
                "rho04_one_step_ce_change": float(ce_rho04 - ce0),
                "rho04_one_step_hard_margin_change": float(hard_rho04 - hard0),
                "rho04_one_step_soft_margin_change": float(soft_rho04 - soft0),
                "signed_one_step_ce_change": float(ce_signed - ce0),
                "signed_one_step_hard_margin_change": float(hard_signed - hard0),
                "signed_one_step_soft_margin_change": float(soft_signed - soft0),
                "signed_minus_rho04_one_step_ce": float(ce_signed - ce_rho04),
                "signed_minus_rho04_one_step_hard_margin": float(hard_signed - hard_rho04),
                "signed_minus_rho04_one_step_soft_margin": float(soft_signed - soft_rho04),
                "signed_minus_rho04_update_norm": float(
                    torch.linalg.vector_norm(signed_update - rho04_update)
                ),
                **rho04_diag,
                **signed_diag,
            }
            finite = (
                bool(torch.isfinite(model.weight).all())
                and bool(torch.isfinite(model.bias).all())
                and all(math.isfinite(float(value)) for value in values_out.values())
            )
            rows.append(
                {
                    "seed": seed,
                    "init_mode": mode,
                    "horizon": horizon,
                    **values_out,
                    "finite": finite,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Direct held-out finite-difference comparison after one exact clipped update "
            "from the signed boundary-band rule versus the matched rho>=0.4 rule."
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
