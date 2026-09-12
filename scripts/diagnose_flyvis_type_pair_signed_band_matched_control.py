from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_adaptive_mean_gain_holdout import _standard_direction
from diagnose_flyvis_type_pair_biological_logit_balance import _heldout_readout, _metrics
from diagnose_flyvis_type_pair_horizon_causal import _match_local_state
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from diagnose_flyvis_type_pair_signed_band_holdout import _signed_band_direction
from diagnose_flyvis_type_pair_threshold_mean_gain_holdout import _threshold_direction
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

HORIZONS = (60, 80, 100)
EVAL_REPS = 4
EVAL_JITTER_BASE = 5_100_000
INIT_MODES = ("random", "biological_strength")
RULES = ("local", "standard", "threshold", "rho04", "signed_band")
MATCHED_RHO_THRESHOLD = 0.4
LOW_EXTRA_GAIN = 1.0
HIGH_EXTRA_GAIN = 3.0


def _rho04_direction(raw_edge: torch.Tensor, named_groups) -> tuple[torch.Tensor, dict[str, float]]:
    result = raw_edge.clone()
    suppressed_groups = 0
    suppressed_edges = 0
    total_edges = 0
    for _, _, indices in named_groups:
        values = raw_edge[indices]
        mean = values.mean()
        mean_component = mean.expand_as(values)
        residual = values - mean
        ratio = float(
            torch.linalg.vector_norm(mean_component)
            / torch.linalg.vector_norm(residual).clamp_min(1e-30)
        )
        suppress = ratio >= MATCHED_RHO_THRESHOLD
        result[indices] = values + (LOW_EXTRA_GAIN if suppress else HIGH_EXTRA_GAIN) * mean
        total_edges += len(indices)
        if suppress:
            suppressed_groups += 1
            suppressed_edges += len(indices)
    return result, {
        "rho04_suppressed_group_fraction": float(suppressed_groups / max(len(named_groups), 1)),
        "rho04_suppressed_edge_fraction": float(suppressed_edges / max(total_edges, 1)),
    }


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
    models = {mode: {rule: copy.deepcopy(base) for rule in RULES} for mode, base in bases.items()}
    nonlocal_rules = ("standard", "threshold", "rho04", "signed_band")
    weight_error_sum = {mode: {rule: 0.0 for rule in nonlocal_rules} for mode in INIT_MODES}
    bias_error_sum = {mode: {rule: 0.0 for rule in nonlocal_rules} for mode in INIT_MODES}
    diag_sum = {
        mode: {
            "rho04_suppressed_group_fraction": 0.0,
            "rho04_suppressed_edge_fraction": 0.0,
            "signed_suppressed_group_fraction": 0.0,
            "signed_band_added_group_fraction": 0.0,
            "signed_suppressed_edge_fraction": 0.0,
        }
        for mode in INIT_MODES
    }
    rows: list[dict[str, float | int | bool | str]] = []

    for step in range(1, max(HORIZONS) + 1):
        epoch = step - 1
        for mode in INIT_MODES:
            local = models[mode]["local"]
            local_edge, local_bias = credit(local, circuit, seed=seed, epoch=epoch)
            apply_local_credit(
                local,
                local_edge,
                local_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )

            for rule in nonlocal_rules:
                model = models[mode][rule]
                raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
                if rule == "standard":
                    direction = _standard_direction(raw_edge, named_groups)
                    diagnostics: dict[str, float] = {}
                elif rule == "threshold":
                    direction, _ = _threshold_direction(raw_edge, named_groups)
                    diagnostics = {}
                elif rule == "rho04":
                    direction, diagnostics = _rho04_direction(raw_edge, named_groups)
                else:
                    direction, diagnostics = _signed_band_direction(
                        raw_edge,
                        model.weight.detach(),
                        named_groups,
                    )
                apply_local_credit(
                    model,
                    direction,
                    raw_bias,
                    learning_rate=learning_rate,
                    weight_decay=0.0,
                    max_update=max_update,
                )
                weight_error, bias_error = _match_local_state(model, local)
                weight_error_sum[mode][rule] += weight_error
                bias_error_sum[mode][rule] += bias_error
                for key, value in diagnostics.items():
                    diag_sum[mode][key] += value

        if step not in HORIZONS:
            continue

        for mode in INIT_MODES:
            local = models[mode]["local"]
            for eval_rep in range(EVAL_REPS):
                jitter_seed = EVAL_JITTER_BASE + eval_rep
                local_readout, classes = _heldout_readout(local, circuit, jitter_seed=jitter_seed)
                for rule in RULES:
                    model = models[mode][rule]
                    if rule == "local":
                        readout = local_readout
                        mean_weight_error = 0.0
                        mean_bias_error = 0.0
                    else:
                        readout, paired_classes = _heldout_readout(
                            model,
                            circuit,
                            jitter_seed=jitter_seed,
                        )
                        if not torch.equal(classes, paired_classes):
                            raise RuntimeError("held-out class mismatch")
                        mean_weight_error = weight_error_sum[mode][rule] / step
                        mean_bias_error = bias_error_sum[mode][rule] / step
                    metrics = _metrics(readout, classes, local_readout=local_readout)
                    diagnostics = {key: diag_sum[mode][key] / step for key in diag_sum[mode]}
                    finite = (
                        bool(torch.isfinite(model.weight).all())
                        and bool(torch.isfinite(model.bias).all())
                        and all(math.isfinite(value) for value in metrics.values())
                        and all(math.isfinite(value) for value in diagnostics.values())
                        and math.isfinite(mean_weight_error)
                        and math.isfinite(mean_bias_error)
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "init_mode": mode,
                            "horizon": step,
                            "eval_rep": eval_rep,
                            "rule": rule,
                            "matched_rho_threshold": MATCHED_RHO_THRESHOLD,
                            **metrics,
                            **diagnostics,
                            "mean_weight_norm_error": mean_weight_error,
                            "mean_bias_error": mean_bias_error,
                            "finite": finite,
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fresh hold-out comparing the signed boundary-band gate with an unconditional "
            "rho>=0.4 control chosen to match its suppression rate."
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
