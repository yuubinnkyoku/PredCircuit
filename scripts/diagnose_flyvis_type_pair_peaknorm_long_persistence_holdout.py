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
from diagnose_flyvis_type_pair_peaknorm_persistence_holdout import _apply_persistent_peaknorm
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

INIT_SCALES = (0.08, 0.10, 0.11, 0.115)
HORIZONS = (160, 180, 200)
EVAL_REPS = 4
EVAL_JITTER_BASE = 115_000_000
RULES = ("local", "standard", "peak025x3", "peak025x5", "peak025x7")
PERSISTENCE = {"peak025x3": 3, "peak025x5": 5, "peak025x7": 7}


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
        models = {rule: copy.deepcopy(base) for rule in RULES}
        streaks = {rule: [0 for _ in groups] for rule in PERSISTENCE}
        diag_sums = {
            rule: {
                "mean_multiplier": 0.0,
                "edge_weighted_multiplier": 0.0,
                "candidate_group_fraction": 0.0,
                "active_group_fraction": 0.0,
                "mean_streak": 0.0,
                "max_streak": 0.0,
            }
            for rule in PERSISTENCE
        }

        for step in range(1, max(HORIZONS) + 1):
            epoch = step - 1
            for rule, model in models.items():
                raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
                direction = raw_edge if rule == "local" else _standard_direction(raw_edge, groups)
                if rule in {"local", "standard"}:
                    apply_local_credit(
                        model,
                        direction,
                        raw_bias,
                        learning_rate=learning_rate,
                        weight_decay=0.0,
                        max_update=max_update,
                    )
                else:
                    diagnostics = _apply_persistent_peaknorm(
                        model,
                        direction,
                        raw_bias,
                        learning_rate=learning_rate,
                        max_update=max_update,
                        groups=groups,
                        streaks=streaks[rule],
                        persistence=PERSISTENCE[rule],
                    )
                    for key in diag_sums[rule]:
                        diag_sums[rule][key] += diagnostics[key]

            if step not in HORIZONS:
                continue

            for eval_rep in range(EVAL_REPS):
                jitter_seed = EVAL_JITTER_BASE + eval_rep
                _, classes = _heldout_readout(models["local"], circuit, jitter_seed=jitter_seed)
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
                    diagnostics = {
                        key: (diag_sums[rule][key] / step if rule in PERSISTENCE else math.nan)
                        for key in (
                            "mean_multiplier",
                            "edge_weighted_multiplier",
                            "candidate_group_fraction",
                            "active_group_fraction",
                            "mean_streak",
                            "max_streak",
                        )
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
                            "horizon": step,
                            "eval_rep": eval_rep,
                            "rule": rule,
                            "eval_jitter": jitter_seed,
                            **diagnostics,
                            **values,
                            "finite": finite,
                        }
                    )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fresh hold-out mapping whether 3, 5, or 7 consecutive type-pair peak-update "
            "bursts best separate safe 4m+r recovery from late collapse."
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
