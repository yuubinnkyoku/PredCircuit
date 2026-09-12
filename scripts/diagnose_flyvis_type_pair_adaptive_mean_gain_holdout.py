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

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

HORIZONS = (60, 80, 100)
EVAL_REPS = 4
EVAL_JITTER_BASE = 4_600_000
INIT_MODES = ("random", "biological_strength")
RULES = ("local", "standard", "adaptive")
ADAPTIVE_TAU = 0.75
MAX_EXTRA_GAIN = 3.0


def _standard_direction(raw_edge: torch.Tensor, named_groups) -> torch.Tensor:
    result = raw_edge.clone()
    for _, _, indices in named_groups:
        result[indices] = raw_edge[indices] + MAX_EXTRA_GAIN * raw_edge[indices].mean()
    return result


def _adaptive_direction(
    raw_edge: torch.Tensor,
    named_groups,
    *,
    tau: float = ADAPTIVE_TAU,
) -> tuple[torch.Tensor, dict[str, float]]:
    result = raw_edge.clone()
    gains: list[float] = []
    weights: list[int] = []
    ratios: list[float] = []
    capped = 0

    for _, _, indices in named_groups:
        values = raw_edge[indices]
        mean = values.mean()
        mean_component = mean.expand_as(values)
        residual = values - mean
        mean_norm = torch.linalg.vector_norm(mean_component)
        residual_norm = torch.linalg.vector_norm(residual)
        ratio = float(mean_norm / residual_norm.clamp_min(1e-30))
        if ratio <= 1e-30:
            extra_gain = MAX_EXTRA_GAIN
        else:
            extra_gain = min(MAX_EXTRA_GAIN, tau / ratio)
        if extra_gain >= MAX_EXTRA_GAIN - 1e-12:
            capped += 1
        result[indices] = values + extra_gain * mean
        gains.append(extra_gain)
        weights.append(len(indices))
        ratios.append(ratio)

    weight_total = max(sum(weights), 1)
    edge_weighted_gain = (
        sum(gain * weight for gain, weight in zip(gains, weights, strict=True)) / weight_total
    )
    return result, {
        "adaptive_mean_extra_gain": float(sum(gains) / max(len(gains), 1)),
        "adaptive_edge_weighted_extra_gain": float(edge_weighted_gain),
        "adaptive_min_extra_gain": float(min(gains, default=MAX_EXTRA_GAIN)),
        "adaptive_capped_group_fraction": float(capped / max(len(gains), 1)),
        "adaptive_mean_mean_to_residual_norm": float(sum(ratios) / max(len(ratios), 1)),
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
    weight_error_sum = {mode: {"standard": 0.0, "adaptive": 0.0} for mode in INIT_MODES}
    bias_error_sum = {mode: {"standard": 0.0, "adaptive": 0.0} for mode in INIT_MODES}
    adaptive_diag_sum = {
        mode: {
            "adaptive_mean_extra_gain": 0.0,
            "adaptive_edge_weighted_extra_gain": 0.0,
            "adaptive_min_extra_gain": 0.0,
            "adaptive_capped_group_fraction": 0.0,
            "adaptive_mean_mean_to_residual_norm": 0.0,
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

            standard = models[mode]["standard"]
            raw_edge, raw_bias = credit(standard, circuit, seed=seed, epoch=epoch)
            standard_direction = _standard_direction(raw_edge, named_groups)
            apply_local_credit(
                standard,
                standard_direction,
                raw_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )
            weight_error, bias_error = _match_local_state(standard, local)
            weight_error_sum[mode]["standard"] += weight_error
            bias_error_sum[mode]["standard"] += bias_error

            adaptive = models[mode]["adaptive"]
            raw_edge, raw_bias = credit(adaptive, circuit, seed=seed, epoch=epoch)
            adaptive_direction, diagnostics = _adaptive_direction(raw_edge, named_groups)
            apply_local_credit(
                adaptive,
                adaptive_direction,
                raw_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )
            weight_error, bias_error = _match_local_state(adaptive, local)
            weight_error_sum[mode]["adaptive"] += weight_error
            bias_error_sum[mode]["adaptive"] += bias_error
            for key, value in diagnostics.items():
                adaptive_diag_sum[mode][key] += value

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
                            model, circuit, jitter_seed=jitter_seed
                        )
                        if not torch.equal(classes, paired_classes):
                            raise RuntimeError("held-out class mismatch")
                        mean_weight_error = weight_error_sum[mode][rule] / step
                        mean_bias_error = bias_error_sum[mode][rule] / step
                    metrics = _metrics(readout, classes, local_readout=local_readout)
                    adaptive_diagnostics = {
                        key: adaptive_diag_sum[mode][key] / step for key in adaptive_diag_sum[mode]
                    }
                    finite = (
                        bool(torch.isfinite(model.weight).all())
                        and bool(torch.isfinite(model.bias).all())
                        and all(math.isfinite(value) for value in metrics.values())
                        and math.isfinite(mean_weight_error)
                        and math.isfinite(mean_bias_error)
                        and all(math.isfinite(value) for value in adaptive_diagnostics.values())
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "init_mode": mode,
                            "horizon": step,
                            "eval_rep": eval_rep,
                            "rule": rule,
                            "adaptive_tau": ADAPTIVE_TAU,
                            **metrics,
                            **adaptive_diagnostics,
                            "mean_weight_norm_error": mean_weight_error,
                            "mean_bias_error": mean_bias_error,
                            "finite": finite,
                        }
                    )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Independent hold-out of a topology-agnostic adaptive type-pair shared-credit rule. "
            "The extra mean gain is min(3, 0.75 / (||m||/||r||))."
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
