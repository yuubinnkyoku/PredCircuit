from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_biological_logit_balance import _heldout_readout
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

INIT_SCALES = (0.10, 0.12, 0.14, 0.16)
HORIZONS = (160, 200)
RULES = ("local", "standard", "iso32", "iso40")
EVAL_REPS = 4
EVAL_JITTER_BASE = 10_900_000


def _shared_coefficient(rule: str, init_scale: float) -> float:
    if rule == "local":
        return 1.0
    if rule == "standard":
        return 4.0
    if rule == "iso32":
        return 0.32 / init_scale
    if rule == "iso40":
        return 0.40 / init_scale
    raise ValueError(f"unknown rule: {rule}")


def _direction(
    raw_edge: torch.Tensor,
    named_groups,
    *,
    shared_coefficient: float,
) -> torch.Tensor:
    result = raw_edge.clone()
    extra_gain = shared_coefficient - 1.0
    for _, _, indices in named_groups:
        values = raw_edge[indices]
        result[indices] = values + extra_gain * values.mean()
    return result


def run_seed(
    *,
    seed: int,
    learning_rate: float,
    max_update: float,
    init_scales: tuple[float, ...] = INIT_SCALES,
    horizons: tuple[int, ...] = HORIZONS,
    eval_jitter_base: int = EVAL_JITTER_BASE,
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
    rows: list[dict[str, float | int | bool | str]] = []

    for step in range(1, max(horizons) + 1):
        epoch = step - 1
        for scale in init_scales:
            for rule in RULES:
                model = models[(scale, rule)]
                raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
                coefficient = _shared_coefficient(rule, scale)
                direction = _direction(
                    raw_edge,
                    named_groups,
                    shared_coefficient=coefficient,
                )
                apply_local_credit(
                    model,
                    direction,
                    raw_bias,
                    learning_rate=learning_rate,
                    weight_decay=0.0,
                    max_update=max_update,
                )

        if step not in horizons:
            continue

        for scale in init_scales:
            for rule in RULES:
                model = models[(scale, rule)]
                coefficient = _shared_coefficient(rule, scale)
                weight_norm = float(torch.linalg.vector_norm(model.weight))
                for eval_rep in range(EVAL_REPS):
                    jitter_seed = eval_jitter_base + eval_rep
                    readout, classes = _heldout_readout(model, circuit, jitter_seed=jitter_seed)
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
                    finite = (
                        bool(torch.isfinite(model.weight).all())
                        and bool(torch.isfinite(model.bias).all())
                        and all(
                            math.isfinite(v)
                            for v in (ce, acc, hard_margin, soft_margin, weight_norm)
                        )
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "horizon": step,
                            "init_scale": scale,
                            "rule": rule,
                            "shared_coefficient": coefficient,
                            "coefficient_times_scale": coefficient * scale,
                            "eval_rep": eval_rep,
                            "cross_entropy": ce,
                            "accuracy": acc,
                            "hard_margin": hard_margin,
                            "soft_margin": soft_margin,
                            "weight_norm": weight_norm,
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
            "Test whether late shared-credit stability is controlled by the product "
            "of biological initialization scale and the shared-mean coefficient."
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--init-scales", type=_parse_float_tuple, default=INIT_SCALES)
    parser.add_argument("--horizons", type=_parse_int_tuple, default=HORIZONS)
    parser.add_argument("--eval-jitter-base", type=int, default=EVAL_JITTER_BASE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = run_seed(
        seed=args.seed,
        learning_rate=args.learning_rate,
        max_update=args.max_update,
        init_scales=args.init_scales,
        horizons=args.horizons,
        eval_jitter_base=args.eval_jitter_base,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
