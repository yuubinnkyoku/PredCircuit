from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_adaptive_mean_gain_holdout import _standard_direction
from diagnose_flyvis_type_pair_biological_logit_balance import _heldout_readout, _metrics
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

HORIZONS = (100, 160, 200)
EVAL_REPS = 4
EVAL_JITTER_BASES = (9_100_000, 20_000_000, 40_000_000, 80_000_000)
RULES = ("local", "standard")
METRICS = ("cross_entropy", "accuracy", "hard_margin", "soft_margin")


def run_seed(*, seed: int, learning_rate: float, max_update: float) -> pd.DataFrame:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    named_groups = _named_groups(circuit)
    base = PredictiveCodingGraph(
        circuit.graph,
        seed=seed,
        init_scale=0.08,
        use_biological_strength=True,
    )
    models = {rule: copy.deepcopy(base) for rule in RULES}
    rows: list[dict[str, float | int | bool | str]] = []

    for step in range(1, max(HORIZONS) + 1):
        epoch = step - 1
        for rule in RULES:
            model = models[rule]
            raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
            direction = raw_edge if rule == "local" else _standard_direction(raw_edge, named_groups)
            apply_local_credit(
                model,
                direction,
                raw_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )

        if step not in HORIZONS:
            continue

        for jitter_base in EVAL_JITTER_BASES:
            for eval_rep in range(EVAL_REPS):
                jitter_seed = jitter_base + eval_rep
                local_readout, classes = _heldout_readout(
                    models["local"],
                    circuit,
                    jitter_seed=jitter_seed,
                )
                standard_readout, paired_classes = _heldout_readout(
                    models["standard"],
                    circuit,
                    jitter_seed=jitter_seed,
                )
                if not torch.equal(classes, paired_classes):
                    raise RuntimeError("held-out class mismatch")

                local_metrics = _metrics(
                    local_readout,
                    classes,
                    local_readout=local_readout,
                )
                standard_metrics = _metrics(
                    standard_readout,
                    classes,
                    local_readout=local_readout,
                )
                values: dict[str, float] = {}
                for metric in METRICS:
                    values[f"local_{metric}"] = local_metrics[metric]
                    values[f"standard_{metric}"] = standard_metrics[metric]
                    values[f"standard_minus_local_{metric}"] = (
                        standard_metrics[metric] - local_metrics[metric]
                    )

                finite = all(
                    bool(torch.isfinite(models[rule].weight).all())
                    and bool(torch.isfinite(models[rule].bias).all())
                    for rule in RULES
                ) and all(math.isfinite(float(value)) for value in values.values())
                rows.append(
                    {
                        "seed": seed,
                        "horizon": step,
                        "jitter_base": jitter_base,
                        "eval_rep": eval_rep,
                        **values,
                        "finite": finite,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train local and standard 4m+r once per seed, then evaluate the identical "
            "learned weights on several widely separated held-out jitter bases."
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
