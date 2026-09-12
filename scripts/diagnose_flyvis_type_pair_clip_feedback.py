from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_biological_logit_balance import _heldout_readout
from diagnose_flyvis_type_pair_gain_scale_product import _direction
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

INIT_SCALES = (0.12, 0.14, 0.16)
RULES = ("local", "fixed25", "clip03_gate", "clip03_smooth")
HORIZONS = (80, 120, 160, 180, 200)
EVAL_REPS = 4
EVAL_JITTER_BASE = 10_960_000
BASE_COEFFICIENT = 2.5
CLIP_THRESHOLD = 0.03


def _prospective_clip_fraction(
    raw_edge: torch.Tensor,
    named_groups,
    *,
    coefficient: float,
    learning_rate: float,
    max_update: float,
) -> float:
    if max_update <= 0.0:
        return 0.0
    direction = _direction(
        raw_edge,
        named_groups,
        shared_coefficient=coefficient,
    )
    preclip_update = learning_rate * direction
    return float((preclip_update.abs() >= max_update).float().mean())


def _coefficient_for_rule(
    rule: str,
    prospective_clip_fraction: float,
    *,
    base_coefficient: float,
    clip_threshold: float,
) -> float:
    if rule == "local":
        return 1.0
    if rule == "fixed25":
        return base_coefficient
    if rule == "clip03_gate":
        return base_coefficient if prospective_clip_fraction <= clip_threshold else 1.0
    if rule == "clip03_smooth":
        if prospective_clip_fraction <= clip_threshold:
            return base_coefficient
        ratio = clip_threshold / max(prospective_clip_fraction, 1e-12)
        return 1.0 + (base_coefficient - 1.0) * ratio
    raise ValueError(f"unknown rule: {rule}")


def run_seed(
    *,
    seed: int,
    learning_rate: float,
    max_update: float,
    init_scales: tuple[float, ...] = INIT_SCALES,
    horizons: tuple[int, ...] = HORIZONS,
    eval_jitter_base: int = EVAL_JITTER_BASE,
    base_coefficient: float = BASE_COEFFICIENT,
    clip_threshold: float = CLIP_THRESHOLD,
) -> pd.DataFrame:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    named_groups = _named_groups(circuit)
    models = {
        (scale, rule): PredictiveCodingGraph(
            circuit.graph,
            seed=seed,
            init_scale=scale,
            use_biological_strength=True,
        )
        for scale in init_scales
        for rule in RULES
    }
    coefficient_sum = {(scale, rule): 0.0 for scale in init_scales for rule in RULES}
    gated_steps = {(scale, rule): 0 for scale in init_scales for rule in RULES}
    last_prospective_clip = {(scale, rule): 0.0 for scale in init_scales for rule in RULES}
    last_actual_clip = {(scale, rule): 0.0 for scale in init_scales for rule in RULES}
    last_coefficient = {(scale, rule): 1.0 for scale in init_scales for rule in RULES}
    rows: list[dict[str, float | int | bool | str]] = []

    for step in range(1, max(horizons) + 1):
        epoch = step - 1
        for scale in init_scales:
            for rule in RULES:
                model = models[(scale, rule)]
                raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
                prospective_clip = _prospective_clip_fraction(
                    raw_edge,
                    named_groups,
                    coefficient=base_coefficient,
                    learning_rate=learning_rate,
                    max_update=max_update,
                )
                coefficient = _coefficient_for_rule(
                    rule,
                    prospective_clip,
                    base_coefficient=base_coefficient,
                    clip_threshold=clip_threshold,
                )
                direction = _direction(
                    raw_edge,
                    named_groups,
                    shared_coefficient=coefficient,
                )
                preclip_update = learning_rate * direction
                if max_update > 0.0:
                    actual_clip = float((preclip_update.abs() >= max_update).float().mean())
                else:
                    actual_clip = 0.0
                apply_local_credit(
                    model,
                    direction,
                    raw_bias,
                    learning_rate=learning_rate,
                    weight_decay=0.0,
                    max_update=max_update,
                )
                key = (scale, rule)
                coefficient_sum[key] += coefficient
                if coefficient < base_coefficient - 1e-12:
                    gated_steps[key] += 1
                last_prospective_clip[key] = prospective_clip
                last_actual_clip[key] = actual_clip
                last_coefficient[key] = coefficient

        if step not in horizons:
            continue

        for scale in init_scales:
            for rule in RULES:
                key = (scale, rule)
                model = models[key]
                weight_norm = float(torch.linalg.vector_norm(model.weight))
                mean_coefficient = coefficient_sum[key] / step
                gated_fraction = gated_steps[key] / step
                for eval_rep in range(EVAL_REPS):
                    jitter_seed = eval_jitter_base + eval_rep
                    readout, classes = _heldout_readout(
                        model,
                        circuit,
                        jitter_seed=jitter_seed,
                    )
                    batch = torch.arange(len(classes))
                    correct = readout[batch, classes]
                    wrong = readout.clone()
                    wrong[batch, classes] = -torch.inf
                    max_wrong = wrong.max(dim=1).values
                    wrong_lse = torch.logsumexp(wrong, dim=1)
                    hard_margin = float((correct - max_wrong).mean())
                    soft_margin = float((correct - wrong_lse).mean())
                    ce = float(torch.nn.functional.cross_entropy(readout, classes))
                    acc = float((readout.argmax(dim=1) == classes).float().mean())
                    diagnostic_values = (
                        weight_norm,
                        mean_coefficient,
                        gated_fraction,
                        last_prospective_clip[key],
                        last_actual_clip[key],
                        last_coefficient[key],
                    )
                    finite = (
                        bool(torch.isfinite(model.weight).all())
                        and bool(torch.isfinite(model.bias).all())
                        and all(
                            math.isfinite(value)
                            for value in (
                                ce,
                                acc,
                                hard_margin,
                                soft_margin,
                                *diagnostic_values,
                            )
                        )
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "horizon": step,
                            "init_scale": scale,
                            "rule": rule,
                            "eval_rep": eval_rep,
                            "cross_entropy": ce,
                            "accuracy": acc,
                            "hard_margin": hard_margin,
                            "soft_margin": soft_margin,
                            "weight_norm": weight_norm,
                            "mean_shared_coefficient": mean_coefficient,
                            "gated_step_fraction": gated_fraction,
                            "prospective_clip_fraction": last_prospective_clip[key],
                            "actual_clip_fraction": last_actual_clip[key],
                            "last_shared_coefficient": last_coefficient[key],
                            "finite": finite,
                        }
                    )

    return pd.DataFrame(rows)


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return parsed


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether a purely local clip-fraction feedback gate can preserve "
            "useful shared credit while preventing high-scale late collapse."
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--init-scales", type=_parse_float_tuple, default=INIT_SCALES)
    parser.add_argument("--horizons", type=_parse_int_tuple, default=HORIZONS)
    parser.add_argument("--eval-jitter-base", type=int, default=EVAL_JITTER_BASE)
    parser.add_argument("--base-coefficient", type=float, default=BASE_COEFFICIENT)
    parser.add_argument("--clip-threshold", type=float, default=CLIP_THRESHOLD)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = run_seed(
        seed=args.seed,
        learning_rate=args.learning_rate,
        max_update=args.max_update,
        init_scales=args.init_scales,
        horizons=args.horizons,
        eval_jitter_base=args.eval_jitter_base,
        base_coefficient=args.base_coefficient,
        clip_threshold=args.clip_threshold,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
