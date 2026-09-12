from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_biological_logit_balance import _heldout_readout, _metrics
from diagnose_flyvis_type_pair_horizon_causal import _match_local_state
from diagnose_flyvis_type_pair_l1_gain_sweep import _direction
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

HORIZONS = (60, 80, 100)
EVAL_REPS = 4
EVAL_JITTER_BASE = 4_200_000
RULE_GAINS = {"gain1": 1.0, "standard": 3.0}


def run_seed(*, seed: int, learning_rate: float, max_update: float) -> pd.DataFrame:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    named_groups = _named_groups(circuit)
    base = PredictiveCodingGraph(
        circuit.graph,
        seed=seed,
        init_scale=0.08,
        use_biological_strength=True,
    )
    models = {"local": copy.deepcopy(base)}
    models.update({rule: copy.deepcopy(base) for rule in RULE_GAINS})
    weight_error_sum = {rule: 0.0 for rule in RULE_GAINS}
    bias_error_sum = {rule: 0.0 for rule in RULE_GAINS}
    rows: list[dict[str, float | int | bool | str]] = []

    for step in range(1, max(HORIZONS) + 1):
        epoch = step - 1
        local_edge, local_bias = credit(models["local"], circuit, seed=seed, epoch=epoch)
        apply_local_credit(
            models["local"],
            local_edge,
            local_bias,
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_update=max_update,
        )

        for rule, l1_gain in RULE_GAINS.items():
            model = models[rule]
            raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
            direction = _direction(raw_edge, named_groups, l1_gain=l1_gain)
            apply_local_credit(
                model,
                direction,
                raw_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )
            weight_error, bias_error = _match_local_state(model, models["local"])
            weight_error_sum[rule] += weight_error
            bias_error_sum[rule] += bias_error

        if step not in HORIZONS:
            continue

        for eval_rep in range(EVAL_REPS):
            jitter_seed = EVAL_JITTER_BASE + eval_rep
            local_readout, classes = _heldout_readout(
                models["local"], circuit, jitter_seed=jitter_seed
            )
            for rule in ("local", "gain1", "standard"):
                if rule == "local":
                    readout = local_readout
                    mean_weight_error = 0.0
                    mean_bias_error = 0.0
                else:
                    readout, paired_classes = _heldout_readout(
                        models[rule], circuit, jitter_seed=jitter_seed
                    )
                    if not torch.equal(classes, paired_classes):
                        raise RuntimeError("held-out class mismatch")
                    mean_weight_error = weight_error_sum[rule] / step
                    mean_bias_error = bias_error_sum[rule] / step
                metrics = _metrics(readout, classes, local_readout=local_readout)
                finite = (
                    bool(torch.isfinite(models[rule].weight).all())
                    and bool(torch.isfinite(models[rule].bias).all())
                    and all(math.isfinite(value) for value in metrics.values())
                    and math.isfinite(mean_weight_error)
                    and math.isfinite(mean_bias_error)
                )
                rows.append(
                    {
                        "seed": seed,
                        "horizon": step,
                        "eval_rep": eval_rep,
                        "rule": rule,
                        **metrics,
                        "mean_weight_norm_error": mean_weight_error,
                        "mean_bias_error": mean_bias_error,
                        "finite": finite,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Independent hold-out of the selected asymmetric type-pair rule: L1 mean "
            "gain 1 versus standard gain 3 and local credit under biological initialization"
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
