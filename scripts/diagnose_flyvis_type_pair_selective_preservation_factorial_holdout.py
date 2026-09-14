from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_biological_logit_balance import _heldout_readout
from diagnose_flyvis_type_pair_l1_gain_geometry import _heldout_objectives
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from diagnose_flyvis_type_pair_selective_phase_factorial_holdout import (
    _advance_phase,
    _advance_standard,
)

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

INIT_SCALE = 0.115
FORK_EPOCH = 140
SWITCH_EPOCH = 160
END_EPOCH = 200
EVAL_REPS = 4
EVAL_JITTER_BASE = 163_000_000
POLICIES = {
    "standard": ("standard", "standard"),
    "global_standard": ("global035", "standard"),
    "soft_standard": ("soft035", "standard"),
    "global_global": ("global035", "global035"),
    "soft_global": ("soft035", "global035"),
}


def run_seed(*, seed: int, learning_rate: float = 160.0, max_update: float = 0.05) -> pd.DataFrame:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    groups = _named_groups(circuit)
    base = PredictiveCodingGraph(
        circuit.graph,
        seed=seed,
        init_scale=INIT_SCALE,
        use_biological_strength=True,
    )
    _advance_standard(
        base,
        circuit,
        groups,
        seed=seed,
        start_epoch=0,
        end_epoch=FORK_EPOCH,
        learning_rate=learning_rate,
        max_update=max_update,
    )

    models = {policy: copy.deepcopy(base) for policy in POLICIES}
    diagnostics: dict[str, dict[str, float]] = {}
    for policy, (early_rule, late_rule) in POLICIES.items():
        early = _advance_phase(
            models[policy],
            circuit,
            groups,
            rule=early_rule,
            seed=seed,
            start_epoch=FORK_EPOCH,
            end_epoch=SWITCH_EPOCH,
            learning_rate=learning_rate,
            max_update=max_update,
        )
        late = _advance_phase(
            models[policy],
            circuit,
            groups,
            rule=late_rule,
            seed=seed,
            start_epoch=SWITCH_EPOCH,
            end_epoch=END_EPOCH,
            learning_rate=learning_rate,
            max_update=max_update,
        )
        diagnostics[policy] = {
            **{f"early_{key}": value for key, value in early.items()},
            **{f"late_{key}": value for key, value in late.items()},
        }

    rows: list[dict[str, float | int | bool | str]] = []
    for eval_rep in range(EVAL_REPS):
        jitter_seed = EVAL_JITTER_BASE + eval_rep
        _, classes = _heldout_readout(models["standard"], circuit, jitter_seed=jitter_seed)
        for policy, model in models.items():
            readout, paired_classes = _heldout_readout(model, circuit, jitter_seed=jitter_seed)
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
            early_rule, late_rule = POLICIES[policy]
            rows.append(
                {
                    "seed": seed,
                    "init_scale": INIT_SCALE,
                    "fork_epoch": FORK_EPOCH,
                    "switch_epoch": SWITCH_EPOCH,
                    "end_epoch": END_EPOCH,
                    "eval_rep": eval_rep,
                    "eval_jitter": jitter_seed,
                    "policy": policy,
                    "early_rule": early_rule,
                    "late_rule": late_rule,
                    **diagnostics[policy],
                    **values,
                    "finite": finite,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fresh 2x2 hold-out testing whether late global damping preserves the benefit "
            "of selective type-pair attenuation during epochs 140-160."
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
