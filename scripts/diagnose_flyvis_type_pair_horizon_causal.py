from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_multi_eval_holdout import heldout_metrics
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit
from run_flyvis_type_pair_consensus import type_pair_indices
from run_flyvis_type_pair_linear_mix import linear_mix_direction
from run_flyvis_type_pair_norm_matched_control import match_norm
from run_flyvis_type_pair_shuffle_control import shuffled_groups_like

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

CHECKPOINTS = (0, 25, 50, 75)
HORIZONS = (1, 5, 10, 25)
EVAL_REPS = 4
EVAL_JITTER_BASE = 1_800_000
RULES = ("local", "type", "shuffle")


def _mixed_direction(raw_edge: torch.Tensor, groups: list[torch.Tensor]) -> torch.Tensor:
    mixed = linear_mix_direction(
        raw_edge,
        groups,
        coarse_gain=4.0,
        residual_gain=1.0,
    )
    return match_norm(mixed, raw_edge)


def _match_local_state(
    treatment: PredictiveCodingGraph,
    local: PredictiveCodingGraph,
) -> tuple[float, float]:
    target_norm = torch.linalg.vector_norm(local.weight).clamp_min(1e-30)
    treatment_norm = torch.linalg.vector_norm(treatment.weight).clamp_min(1e-30)
    treatment.weight.mul_(target_norm / treatment_norm)
    treatment.bias.copy_(local.bias)
    weight_error = float(
        (torch.linalg.vector_norm(treatment.weight) - target_norm).abs() / target_norm
    )
    bias_error = float(torch.linalg.vector_norm(treatment.bias - local.bias))
    return weight_error, bias_error


def run_seed(
    *,
    seed: int,
    shuffle_seed: int,
    learning_rate: float,
    max_update: float,
) -> pd.DataFrame:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    base = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    biological_groups = type_pair_indices(circuit)
    shuffled_groups = shuffled_groups_like(
        biological_groups,
        edge_count=base.weight.numel(),
        shuffle_seed=shuffle_seed,
    )

    rows: list[dict[str, float | int | bool | str]] = []
    trained_epochs = 0
    for checkpoint in CHECKPOINTS:
        for epoch in range(trained_epochs, checkpoint):
            edge, bias = credit(base, circuit, seed=seed, epoch=epoch)
            apply_local_credit(
                base,
                edge,
                bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )
        trained_epochs = checkpoint

        baseline = [
            heldout_metrics(
                base,
                circuit,
                jitter_seed=EVAL_JITTER_BASE + eval_rep,
            )
            for eval_rep in range(EVAL_REPS)
        ]
        models = {rule: copy.deepcopy(base) for rule in RULES}
        weight_error_sum = {"type": 0.0, "shuffle": 0.0}
        bias_error_sum = {"type": 0.0, "shuffle": 0.0}
        direction_change_sum = {"type": 0.0, "shuffle": 0.0}
        clip_sum = {rule: 0.0 for rule in RULES}

        for step in range(1, max(HORIZONS) + 1):
            epoch = checkpoint + step - 1
            local_edge, local_bias = credit(models["local"], circuit, seed=seed, epoch=epoch)
            clip_sum["local"] += float(
                (learning_rate * local_edge).abs().gt(max_update).float().mean()
            )
            apply_local_credit(
                models["local"],
                local_edge,
                local_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )

            for rule, groups in (
                ("type", biological_groups),
                ("shuffle", shuffled_groups),
            ):
                raw_edge, raw_bias = credit(models[rule], circuit, seed=seed, epoch=epoch)
                direction = _mixed_direction(raw_edge, groups)
                raw_norm = torch.linalg.vector_norm(raw_edge).clamp_min(1e-30)
                direction_change_sum[rule] += float(
                    torch.linalg.vector_norm(direction - raw_edge) / raw_norm
                )
                clip_sum[rule] += float(
                    (learning_rate * direction).abs().gt(max_update).float().mean()
                )
                apply_local_credit(
                    models[rule],
                    direction,
                    raw_bias,
                    learning_rate=learning_rate,
                    weight_decay=0.0,
                    max_update=max_update,
                )
                weight_error, bias_error = _match_local_state(
                    models[rule], models["local"]
                )
                weight_error_sum[rule] += weight_error
                bias_error_sum[rule] += bias_error

            if step not in HORIZONS:
                continue

            for eval_rep in range(EVAL_REPS):
                before = baseline[eval_rep]
                for rule in RULES:
                    after = heldout_metrics(
                        models[rule],
                        circuit,
                        jitter_seed=EVAL_JITTER_BASE + eval_rep,
                    )
                    finite = (
                        bool(torch.isfinite(models[rule].weight).all())
                        and bool(torch.isfinite(models[rule].bias).all())
                        and all(math.isfinite(value) for value in before.values())
                        and all(math.isfinite(value) for value in after.values())
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "checkpoint": checkpoint,
                            "horizon": step,
                            "eval_rep": eval_rep,
                            "rule": rule,
                            "mean_relative_direction_change": (
                                0.0
                                if rule == "local"
                                else direction_change_sum[rule] / step
                            ),
                            "mean_clip_fraction": clip_sum[rule] / step,
                            "mean_weight_norm_error": (
                                0.0 if rule == "local" else weight_error_sum[rule] / step
                            ),
                            "mean_bias_error": (
                                0.0 if rule == "local" else bias_error_sum[rule] / step
                            ),
                            "before_cross_entropy": before["cross_entropy"],
                            "before_accuracy": before["accuracy"],
                            "before_hard_margin": before["hard_margin"],
                            "after_cross_entropy": after["cross_entropy"],
                            "after_accuracy": after["accuracy"],
                            "after_hard_margin": after["hard_margin"],
                            "after_margin_identity_error": after["margin_identity_error"],
                            "finite": finite,
                        }
                    )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Trace when norm- and bias-matched 4m+r begins to help across 1/5/10/25 "
            "steps from identical local-training checkpoints"
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shuffle-seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frame = run_seed(
        seed=args.seed,
        shuffle_seed=args.shuffle_seed,
        learning_rate=args.learning_rate,
        max_update=args.max_update,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
