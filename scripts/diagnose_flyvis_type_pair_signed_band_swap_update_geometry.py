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
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

HORIZONS = (60, 80, 100)
INIT_MODES = ("random", "biological_strength")


def _clipped_update(
    direction: torch.Tensor, *, learning_rate: float, max_update: float
) -> torch.Tensor:
    update = learning_rate * direction
    if max_update > 0.0:
        update = update.clamp(-max_update, max_update)
    return update


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
            raw_edge, _ = credit(model, circuit, seed=seed, epoch=horizon)
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
            delta_update = signed_update - rho04_update

            weight = model.weight.detach().clone().requires_grad_(True)
            bias = model.bias.detach().clone()
            ce, hard_margin, soft_margin = _heldout_objectives(circuit, weight, bias)
            grad_ce = torch.autograd.grad(ce, weight, retain_graph=True)[0]
            grad_hard = torch.autograd.grad(hard_margin, weight, retain_graph=True)[0]
            grad_soft = torch.autograd.grad(soft_margin, weight)[0]

            signed_only_groups = 0
            rho04_only_groups = 0
            signed_only_edges = 0
            rho04_only_edges = 0
            for _, _, indices in named_groups:
                values = raw_edge[indices]
                mean = values.mean()
                residual = values - mean
                rho = float(
                    torch.linalg.vector_norm(mean.expand_as(values))
                    / torch.linalg.vector_norm(residual).clamp_min(1e-30)
                )
                local_sign = float(model.weight.detach()[indices].mean() * mean)
                signed_suppress = rho >= 0.5 or (0.35 <= rho < 0.5 and local_sign > 0.0)
                rho04_suppress = rho >= 0.4
                if signed_suppress and not rho04_suppress:
                    signed_only_groups += 1
                    signed_only_edges += len(indices)
                elif rho04_suppress and not signed_suppress:
                    rho04_only_groups += 1
                    rho04_only_edges += len(indices)

            values_out = {
                "heldout_ce": float(ce.detach()),
                "heldout_hard_margin": float(hard_margin.detach()),
                "heldout_soft_margin": float(soft_margin.detach()),
                "signed_minus_rho04_update_norm": float(torch.linalg.vector_norm(delta_update)),
                "ce_grad_dot_signed_minus_rho04_update": float(torch.dot(grad_ce, delta_update)),
                "hard_margin_grad_dot_signed_minus_rho04_update": float(
                    torch.dot(grad_hard, delta_update)
                ),
                "soft_margin_grad_dot_signed_minus_rho04_update": float(
                    torch.dot(grad_soft, delta_update)
                ),
                "signed_only_group_count": signed_only_groups,
                "rho04_only_group_count": rho04_only_groups,
                "signed_only_edge_count": signed_only_edges,
                "rho04_only_edge_count": rho04_only_edges,
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
            "Fresh local-state geometry test of the exact clipped update swapped by the "
            "signed boundary-band gate versus the suppression-rate-matched rho>=0.4 gate."
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
