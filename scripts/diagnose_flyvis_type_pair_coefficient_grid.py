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
COEFFICIENTS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
HORIZONS = (160, 200)
EVAL_REPS = 4
EVAL_JITTER_BASE = 10_920_000


def run_seed(
    *,
    seed: int,
    learning_rate: float,
    max_update: float,
    init_scales: tuple[float, ...] = INIT_SCALES,
    coefficients: tuple[float, ...] = COEFFICIENTS,
    horizons: tuple[int, ...] = HORIZONS,
    eval_jitter_base: int = EVAL_JITTER_BASE,
) -> pd.DataFrame:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    named_groups = _named_groups(circuit)
    models = {
        (scale, coefficient): PredictiveCodingGraph(
            circuit.graph,
            seed=seed,
            init_scale=scale,
            use_biological_strength=True,
        )
        for scale in init_scales
        for coefficient in coefficients
    }
    rows: list[dict[str, float | int | bool]] = []

    for step in range(1, max(horizons) + 1):
        epoch = step - 1
        for scale in init_scales:
            for coefficient in coefficients:
                model = models[(scale, coefficient)]
                raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
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
            for coefficient in coefficients:
                model = models[(scale, coefficient)]
                weight_norm = float(torch.linalg.vector_norm(model.weight))
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
                    finite = (
                        bool(torch.isfinite(model.weight).all())
                        and bool(torch.isfinite(model.bias).all())
                        and all(
                            math.isfinite(value)
                            for value in (ce, acc, hard_margin, soft_margin, weight_norm)
                        )
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "horizon": step,
                            "init_scale": scale,
                            "shared_coefficient": coefficient,
                            "coefficient_times_scale": coefficient * scale,
                            "coefficient_times_scale_sq": coefficient * scale * scale,
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
            "Map the biological shared-credit stability boundary directly over a "
            "fixed coefficient grid instead of assuming a scale exponent."
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--init-scales", type=_parse_float_tuple, default=INIT_SCALES)
    parser.add_argument("--coefficients", type=_parse_float_tuple, default=COEFFICIENTS)
    parser.add_argument("--horizons", type=_parse_int_tuple, default=HORIZONS)
    parser.add_argument("--eval-jitter-base", type=int, default=EVAL_JITTER_BASE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = run_seed(
        seed=args.seed,
        learning_rate=args.learning_rate,
        max_update=args.max_update,
        init_scales=args.init_scales,
        coefficients=args.coefficients,
        horizons=args.horizons,
        eval_jitter_base=args.eval_jitter_base,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
