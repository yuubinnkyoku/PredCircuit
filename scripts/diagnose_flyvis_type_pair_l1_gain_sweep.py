from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_biological_logit_balance import _heldout_readout, _metrics
from diagnose_flyvis_type_pair_horizon_causal import _match_local_state
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit
from run_flyvis_type_pair_norm_matched_control import match_norm

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

HORIZONS = (60, 80, 100)
EVAL_REPS = 4
EVAL_JITTER_BASE = 4_000_000
L1_GAINS = (0.0, 1.0, 2.0, 3.0)


def _direction(
    raw_edge: torch.Tensor,
    named_groups: list[tuple[str, str, torch.Tensor]],
    *,
    l1_gain: float,
) -> torch.Tensor:
    result = raw_edge.clone()
    for _, target_type, indices in named_groups:
        gain = l1_gain if target_type == "L1" else 3.0
        result[indices] = raw_edge[indices] + gain * raw_edge[indices].mean()
    return match_norm(result, raw_edge)


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
    for gain in L1_GAINS:
        models[f"l1_gain_{gain:g}"] = copy.deepcopy(base)
    error_sum = {rule: 0.0 for rule in models if rule != "local"}
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

        for gain in L1_GAINS:
            rule = f"l1_gain_{gain:g}"
            model = models[rule]
            raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
            direction = _direction(raw_edge, named_groups, l1_gain=gain)
            apply_local_credit(
                model,
                direction,
                raw_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )
            weight_error, _ = _match_local_state(model, models["local"])
            error_sum[rule] += weight_error

        if step not in HORIZONS:
            continue

        for eval_rep in range(EVAL_REPS):
            jitter_seed = EVAL_JITTER_BASE + eval_rep
            local_readout, classes = _heldout_readout(
                models["local"], circuit, jitter_seed=jitter_seed
            )
            local_metrics = _metrics(local_readout, classes, local_readout=local_readout)
            rows.append(
                {
                    "seed": seed,
                    "horizon": step,
                    "eval_rep": eval_rep,
                    "rule": "local",
                    "l1_gain": -1.0,
                    **local_metrics,
                    "mean_weight_norm_error": 0.0,
                    "finite": True,
                }
            )
            for gain in L1_GAINS:
                rule = f"l1_gain_{gain:g}"
                readout, paired_classes = _heldout_readout(
                    models[rule], circuit, jitter_seed=jitter_seed
                )
                if not torch.equal(classes, paired_classes):
                    raise RuntimeError("held-out class mismatch")
                metrics = _metrics(readout, classes, local_readout=local_readout)
                mean_error = error_sum[rule] / step
                finite = (
                    bool(torch.isfinite(models[rule].weight).all())
                    and bool(torch.isfinite(models[rule].bias).all())
                    and all(math.isfinite(value) for value in metrics.values())
                    and math.isfinite(mean_error)
                )
                rows.append(
                    {
                        "seed": seed,
                        "horizon": step,
                        "eval_rep": eval_rep,
                        "rule": rule,
                        "l1_gain": gain,
                        **metrics,
                        "mean_weight_norm_error": mean_error,
                        "finite": finite,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep L1-target shared-credit gain while keeping non-L1 gain fixed at 3, "
            "to test the CE-versus-hard-decision tradeoff under biological initialization"
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
