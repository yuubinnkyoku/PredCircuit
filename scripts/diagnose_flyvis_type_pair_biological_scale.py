from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_adaptive_mean_gain_holdout import _standard_direction
from diagnose_flyvis_type_pair_biological_logit_balance import _heldout_readout
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

HORIZONS = (100, 160, 200)
INIT_SCALES = (0.04, 0.08, 0.16)
RULES = ("local", "standard")
EVAL_REPS = 4
EVAL_JITTER_BASE = 10_700_000
METRICS = ("cross_entropy", "accuracy", "hard_margin", "soft_margin")


def run_seed(*, seed: int, learning_rate: float, max_update: float) -> pd.DataFrame:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    named_groups = _named_groups(circuit)
    models = {
        (scale, rule): PredictiveCodingGraph(
            circuit.graph,
            seed=seed,
            init_scale=scale,
            use_biological_strength=True,
        )
        for scale in INIT_SCALES
        for rule in RULES
    }
    rows: list[dict[str, float | int | bool | str]] = []

    for step in range(1, max(HORIZONS) + 1):
        epoch = step - 1
        for scale in INIT_SCALES:
            for rule in RULES:
                model = models[(scale, rule)]
                raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
                direction = (
                    raw_edge if rule == "local" else _standard_direction(raw_edge, named_groups)
                )
                apply_local_credit(
                    model,
                    direction,
                    raw_bias,
                    learning_rate=learning_rate,
                    weight_decay=0.0,
                    max_update=max_update,
                )

        if step not in HORIZONS:
            continue

        for scale in INIT_SCALES:
            for rule in RULES:
                model = models[(scale, rule)]
                weight_norm = float(torch.linalg.vector_norm(model.weight))
                for eval_rep in range(EVAL_REPS):
                    jitter_seed = EVAL_JITTER_BASE + eval_rep
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Perturb the biological-strength init scale to test whether the late "
            "4m+r recovery is specific to one initialization magnitude."
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
