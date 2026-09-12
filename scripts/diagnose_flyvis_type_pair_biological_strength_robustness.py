from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_horizon_causal import _match_local_state, _mixed_direction
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from diagnose_flyvis_type_pair_multi_eval_holdout import heldout_metrics
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit
from run_flyvis_type_pair_norm_matched_control import match_norm

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

HORIZONS = (60, 80, 100)
EVAL_REPS = 4
EVAL_JITTER_BASE = 3_300_000
INIT_MODES = ("random", "biological_strength")
RULES = ("local", "type", "mi9_only", "r7r8_mi9")


def _selective_direction(
    raw_edge: torch.Tensor,
    named_groups: list[tuple[str, str, torch.Tensor]],
    *,
    sources: set[str] | None,
) -> torch.Tensor:
    result = raw_edge.clone()
    for source_type, target_type, indices in named_groups:
        selected = target_type == "Mi9" and (sources is None or source_type in sources)
        if selected:
            result[indices] = raw_edge[indices] + 3.0 * raw_edge[indices].mean()
    return match_norm(result, raw_edge)


def run_seed(
    *,
    seed: int,
    learning_rate: float,
    max_update: float,
) -> pd.DataFrame:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    named_groups = _named_groups(circuit)
    groups = [indices for _, _, indices in named_groups]
    bases = {
        "random": PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08),
        "biological_strength": PredictiveCodingGraph(
            circuit.graph,
            seed=seed,
            init_scale=0.08,
            use_biological_strength=True,
        ),
    }
    models = {
        init_mode: {rule: copy.deepcopy(base) for rule in RULES}
        for init_mode, base in bases.items()
    }
    direction_change_sum = {
        init_mode: {rule: 0.0 for rule in RULES if rule != "local"} for init_mode in INIT_MODES
    }
    weight_error_sum = copy.deepcopy(direction_change_sum)
    bias_error_sum = copy.deepcopy(direction_change_sum)
    rows: list[dict[str, float | int | bool | str]] = []

    for step in range(1, max(HORIZONS) + 1):
        epoch = step - 1
        for init_mode in INIT_MODES:
            init_models = models[init_mode]
            local_edge, local_bias = credit(init_models["local"], circuit, seed=seed, epoch=epoch)
            apply_local_credit(
                init_models["local"],
                local_edge,
                local_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )

            for rule in ("type", "mi9_only", "r7r8_mi9"):
                model = init_models[rule]
                raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
                if rule == "type":
                    direction = _mixed_direction(raw_edge, groups)
                elif rule == "mi9_only":
                    direction = _selective_direction(raw_edge, named_groups, sources=None)
                else:
                    direction = _selective_direction(
                        raw_edge,
                        named_groups,
                        sources={"R7", "R8"},
                    )
                direction_change_sum[init_mode][rule] += float(
                    torch.linalg.vector_norm(direction - raw_edge)
                    / torch.linalg.vector_norm(raw_edge).clamp_min(1e-30)
                )
                apply_local_credit(
                    model,
                    direction,
                    raw_bias,
                    learning_rate=learning_rate,
                    weight_decay=0.0,
                    max_update=max_update,
                )
                weight_error, bias_error = _match_local_state(model, init_models["local"])
                weight_error_sum[init_mode][rule] += weight_error
                bias_error_sum[init_mode][rule] += bias_error

        if step not in HORIZONS:
            continue

        for eval_rep in range(EVAL_REPS):
            jitter_seed = EVAL_JITTER_BASE + eval_rep
            for init_mode in INIT_MODES:
                for rule in RULES:
                    model = models[init_mode][rule]
                    metrics = heldout_metrics(model, circuit, jitter_seed=jitter_seed)
                    if rule == "local":
                        mean_direction_change = 0.0
                        mean_weight_error = 0.0
                        mean_bias_error = 0.0
                    else:
                        mean_direction_change = direction_change_sum[init_mode][rule] / step
                        mean_weight_error = weight_error_sum[init_mode][rule] / step
                        mean_bias_error = bias_error_sum[init_mode][rule] / step
                    finite = (
                        bool(torch.isfinite(model.weight).all())
                        and bool(torch.isfinite(model.bias).all())
                        and all(math.isfinite(value) for value in metrics.values())
                        and math.isfinite(mean_direction_change)
                        and math.isfinite(mean_weight_error)
                        and math.isfinite(mean_bias_error)
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "horizon": step,
                            "eval_rep": eval_rep,
                            "init_mode": init_mode,
                            "rule": rule,
                            "mean_relative_direction_change": mean_direction_change,
                            "mean_weight_norm_error": mean_weight_error,
                            "mean_bias_error": mean_bias_error,
                            **metrics,
                            "finite": finite,
                        }
                    )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test full, Mi9-only, and R7/R8->Mi9 shared credit under paired random and "
            "FlyVis synapse-strength/sign initialization"
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
