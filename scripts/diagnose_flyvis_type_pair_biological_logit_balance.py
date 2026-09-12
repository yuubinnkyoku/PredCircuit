from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from diagnose_flyvis_type_pair_horizon_causal import _match_local_state, _mixed_direction
from diagnose_flyvis_type_pair_l1_complement_interaction import _subset_direction
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from run_flyvis_retinotopic_contrastive import (
    DIRECTIONS,
    infer_sequence,
    output_nodes,
    render_motion_batch,
    targets_for,
)
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

HORIZONS = (60, 80, 100)
EVAL_REPS = 4
EVAL_JITTER_BASE = 3_800_000
RULES = ("local", "l1_only", "non_l1_only", "type")


@torch.no_grad()
def _heldout_readout(model: PredictiveCodingGraph, circuit, *, jitter_seed: int):
    directions = list(DIRECTIONS) * 12
    stimulus = render_motion_batch(
        circuit,
        directions,
        frames=7,
        width=0.5,
        jitter_seed=jitter_seed,
    )
    _, classes = targets_for(directions)
    state = infer_sequence(
        model,
        circuit,
        stimulus,
        frame_steps=2,
        step_size=0.015,
    )
    return state[:, output_nodes(circuit)], classes


def _safe_mean(values: torch.Tensor) -> float:
    return float(values.mean()) if values.numel() else 0.0


def _metrics(
    readout: torch.Tensor,
    classes: torch.Tensor,
    *,
    local_readout: torch.Tensor,
) -> dict[str, float]:
    batch = torch.arange(len(classes))
    correct = readout[batch, classes]
    wrong = readout.clone()
    wrong[batch, classes] = -torch.inf
    max_wrong = wrong.max(dim=1).values
    wrong_lse = torch.logsumexp(wrong, dim=1)
    hard_margin = correct - max_wrong
    soft_margin = correct - wrong_lse
    tail_pressure = wrong_lse - max_wrong
    nll = F.cross_entropy(readout, classes, reduction="none")
    probabilities = readout.softmax(dim=1)[batch, classes]
    predicted = readout.argmax(dim=1)
    is_correct = predicted == classes

    local_predicted = local_readout.argmax(dim=1)
    local_correct = local_predicted == classes
    local_wrong = ~local_correct
    wrong_to_correct = local_wrong & is_correct
    correct_to_wrong = local_correct & ~is_correct

    class_ce = []
    class_accuracy = []
    for class_index in range(readout.shape[1]):
        mask = classes == class_index
        class_ce.append(_safe_mean(nll[mask]))
        class_accuracy.append(_safe_mean(is_correct[mask].float()))
    class_ce_tensor = torch.tensor(class_ce)
    class_accuracy_tensor = torch.tensor(class_accuracy)

    return {
        "cross_entropy": float(nll.mean()),
        "accuracy": float(is_correct.float().mean()),
        "hard_margin": float(hard_margin.mean()),
        "soft_margin": float(soft_margin.mean()),
        "correct_logit": float(correct.mean()),
        "max_wrong_logit": float(max_wrong.mean()),
        "wrong_logsumexp": float(wrong_lse.mean()),
        "wrong_tail_pressure": float(tail_pressure.mean()),
        "correct_probability": float(probabilities.mean()),
        "ce_on_local_correct": _safe_mean(nll[local_correct]),
        "ce_on_local_wrong": _safe_mean(nll[local_wrong]),
        "prob_on_local_correct": _safe_mean(probabilities[local_correct]),
        "prob_on_local_wrong": _safe_mean(probabilities[local_wrong]),
        "margin_on_local_correct": _safe_mean(hard_margin[local_correct]),
        "margin_on_local_wrong": _safe_mean(hard_margin[local_wrong]),
        "wrong_to_correct_fraction": float(wrong_to_correct.float().mean()),
        "correct_to_wrong_fraction": float(correct_to_wrong.float().mean()),
        "local_correct_fraction": float(local_correct.float().mean()),
        "class_ce_std": float(class_ce_tensor.std(correction=0)),
        "worst_class_ce": float(class_ce_tensor.max()),
        "class_accuracy_std": float(class_accuracy_tensor.std(correction=0)),
        "worst_class_accuracy": float(class_accuracy_tensor.min()),
    }


def run_seed(*, seed: int, learning_rate: float, max_update: float) -> pd.DataFrame:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    named_groups = _named_groups(circuit)
    groups = [indices for _, _, indices in named_groups]
    base = PredictiveCodingGraph(
        circuit.graph,
        seed=seed,
        init_scale=0.08,
        use_biological_strength=True,
    )
    models = {rule: copy.deepcopy(base) for rule in RULES}
    weight_error_sum = {rule: 0.0 for rule in RULES if rule != "local"}
    bias_error_sum = copy.deepcopy(weight_error_sum)
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

        for rule in RULES[1:]:
            model = models[rule]
            raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
            if rule == "type":
                direction = _mixed_direction(raw_edge, groups)
            elif rule == "l1_only":
                direction = _subset_direction(raw_edge, named_groups, select_l1=True)
            else:
                direction = _subset_direction(raw_edge, named_groups, select_l1=False)
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
            for rule in RULES:
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
            "Decompose the biological-strength CE/accuracy tradeoff for local, L1-only, "
            "non-L1-only, and full type-pair shared credit"
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
