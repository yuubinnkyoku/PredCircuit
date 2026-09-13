from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_adaptive_mean_gain_holdout import _standard_direction
from diagnose_flyvis_type_pair_biological_logit_balance import _heldout_readout
from diagnose_flyvis_type_pair_l1_gain_geometry import _heldout_objectives
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from diagnose_flyvis_type_pair_peaknorm_global_l2_match_holdout import (
    _apply_update,
)
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

INIT_SCALES = (0.10, 0.115)
FORK_EPOCHS = (160, 180)
END_EPOCH = 200
EVAL_REPS = 4
EVAL_JITTER_BASE = 139_000_000
RULES = ("standard", "soft035", "global035")
PROBABILITY = 0.35
DIAG_KEYS = (
    "candidate_group_fraction",
    "attenuation_mass_fraction",
    "update_l2_ratio",
    "matched_l2_relative_error",
)


def _advance_standard(
    model: PredictiveCodingGraph,
    circuit,
    groups: list[tuple[str, str, torch.Tensor]],
    *,
    seed: int,
    start_epoch: int,
    end_epoch: int,
    learning_rate: float,
    max_update: float,
) -> None:
    for epoch in range(start_epoch, end_epoch):
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


def _branch_to_end(
    model: PredictiveCodingGraph,
    circuit,
    groups: list[tuple[str, str, torch.Tensor]],
    *,
    rule: str,
    seed: int,
    fork_epoch: int,
    learning_rate: float,
    max_update: float,
) -> dict[str, float]:
    diag_sums = {key: 0.0 for key in DIAG_KEYS}
    steps = END_EPOCH - fork_epoch
    for epoch in range(fork_epoch, END_EPOCH):
        raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
        direction = _standard_direction(raw_edge, groups)
        if rule == "standard":
            apply_local_credit(
                model,
                direction,
                raw_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )
            continue
        diagnostics = _apply_update(
            model,
            direction,
            raw_bias,
            learning_rate=learning_rate,
            max_update=max_update,
            groups=groups,
            probability=PROBABILITY,
            global_match=rule == "global035",
        )
        for key in DIAG_KEYS:
            diag_sums[key] += diagnostics[key]
    if rule == "standard":
        return {key: math.nan for key in DIAG_KEYS}
    return {key: diag_sums[key] / steps for key in DIAG_KEYS}


def run_seed(*, seed: int, learning_rate: float = 160.0, max_update: float = 0.05) -> pd.DataFrame:
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
        snapshots: dict[int, PredictiveCodingGraph] = {}
        trained = 0
        for fork_epoch in FORK_EPOCHS:
            _advance_standard(
                base,
                circuit,
                groups,
                seed=seed,
                start_epoch=trained,
                end_epoch=fork_epoch,
                learning_rate=learning_rate,
                max_update=max_update,
            )
            trained = fork_epoch
            snapshots[fork_epoch] = copy.deepcopy(base)

        for fork_epoch, snapshot in snapshots.items():
            models = {rule: copy.deepcopy(snapshot) for rule in RULES}
            diagnostics = {
                rule: _branch_to_end(
                    model,
                    circuit,
                    groups,
                    rule=rule,
                    seed=seed,
                    fork_epoch=fork_epoch,
                    learning_rate=learning_rate,
                    max_update=max_update,
                )
                for rule, model in models.items()
            }
            for eval_rep in range(EVAL_REPS):
                jitter_seed = EVAL_JITTER_BASE + eval_rep
                _, classes = _heldout_readout(
                    models["standard"], circuit, jitter_seed=jitter_seed
                )
                for rule, model in models.items():
                    readout, paired_classes = _heldout_readout(
                        model, circuit, jitter_seed=jitter_seed
                    )
                    if not torch.equal(classes, paired_classes):
                        raise RuntimeError("held-out class mismatch")
                    weight = model.weight.detach().clone().requires_grad_(True)
                    bias = model.bias.detach().clone()
                    ce, hard_margin, soft_margin = _heldout_objectives(circuit, weight, bias)
                    values = {
                        "cross_entropy": float(ce.detach()),
                        "accuracy": float((readout.argmax(dim=1) == classes).float().mean()),
                        "hard_margin": float(hard_margin.detach()),
                        "soft_margin": float(soft_margin.detach()),
                        "weight_norm": float(torch.linalg.vector_norm(model.weight.detach())),
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
                            "fork_epoch": fork_epoch,
                            "end_epoch": END_EPOCH,
                            "eval_rep": eval_rep,
                            "eval_jitter": jitter_seed,
                            "rule": rule,
                            **diagnostics[rule],
                            **values,
                            "finite": finite,
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fresh common-state fork hold-out testing whether type-pair selective "
            "attenuation must begin before the late-collapse geometry is degraded."
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
