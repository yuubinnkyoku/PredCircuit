from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from diagnose_flyvis_type_pair_adaptive_mean_gain_holdout import _standard_direction
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_axis_identity_control import credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

START_EPOCH = 120
END_EPOCH = 160
FOCUS_START = 140
PEAK_THRESHOLD = 0.25


def _candidate_groups(
    direction: torch.Tensor,
    *,
    learning_rate: float,
    max_update: float,
    groups: list[tuple[str, str, torch.Tensor]],
) -> set[int]:
    raw = learning_rate * direction
    candidate: set[int] = set()
    for group_idx, (_, _, indices) in enumerate(groups):
        raw_group = raw[indices]
        raw_norm = torch.linalg.vector_norm(raw_group)
        peak = raw_group.abs().max()
        if float(raw_norm) == 0.0 or float(peak) <= max_update:
            peak_ratio = 1.0
        else:
            peak_ratio = float(max_update / peak)
        if peak_ratio < PEAK_THRESHOLD:
            candidate.add(group_idx)
    return candidate


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def run_seed(
    *,
    seed: int,
    init_scale: float = 0.115,
    learning_rate: float = 160.0,
    max_update: float = 0.05,
) -> pd.DataFrame:
    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=2)
    groups = _named_groups(circuit)
    model = PredictiveCodingGraph(
        circuit.graph,
        seed=seed,
        init_scale=init_scale,
        use_biological_strength=True,
    )

    by_epoch: dict[int, set[int]] = {}
    for epoch in range(END_EPOCH + 1):
        raw_edge, raw_bias = credit(model, circuit, seed=seed, epoch=epoch)
        direction = _standard_direction(raw_edge, groups)
        if epoch >= START_EPOCH:
            by_epoch[epoch] = _candidate_groups(
                direction,
                learning_rate=learning_rate,
                max_update=max_update,
                groups=groups,
            )
        if epoch < END_EPOCH:
            apply_local_credit(
                model,
                direction,
                raw_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )

    focus_epochs = list(range(FOCUS_START, END_EPOCH + 1))
    focus_sets = [by_epoch[epoch] for epoch in focus_epochs]
    all_focus = set().union(*focus_sets)
    frequencies = {
        group_idx: sum(group_idx in candidate for candidate in focus_sets) / len(focus_sets)
        for group_idx in all_focus
    }
    consecutive_jaccard = [
        _jaccard(by_epoch[epoch], by_epoch[epoch + 1]) for epoch in range(FOCUS_START, END_EPOCH)
    ]
    reference_jaccard = [
        _jaccard(by_epoch[FOCUS_START], by_epoch[epoch])
        for epoch in range(FOCUS_START + 1, END_EPOCH + 1)
    ]
    counts = [len(candidate) for candidate in focus_sets]
    total_groups = len(groups)

    persistent_half = sum(frequency >= 0.5 for frequency in frequencies.values())
    persistent_three_quarters = sum(frequency >= 0.75 for frequency in frequencies.values())
    mean_frequency = sum(frequencies.values()) / len(frequencies) if frequencies else 0.0
    max_frequency = max(frequencies.values()) if frequencies else 0.0

    values = {
        "candidate_group_fraction_epoch120": len(by_epoch[120]) / total_groups,
        "candidate_group_fraction_epoch140": len(by_epoch[140]) / total_groups,
        "candidate_group_fraction_epoch160": len(by_epoch[160]) / total_groups,
        "candidate_group_fraction_focus_mean": sum(counts) / (len(counts) * total_groups),
        "candidate_group_fraction_focus_sd": float(
            torch.tensor(counts, dtype=torch.float64).std(unbiased=True)
        )
        / total_groups,
        "ever_candidate_group_fraction_focus": len(all_focus) / total_groups,
        "persistent_half_group_fraction_focus": persistent_half / total_groups,
        "persistent_three_quarters_group_fraction_focus": persistent_three_quarters / total_groups,
        "mean_candidate_frequency_among_ever": mean_frequency,
        "max_candidate_frequency": max_frequency,
        "consecutive_jaccard_mean": sum(consecutive_jaccard) / len(consecutive_jaccard),
        "reference140_jaccard_mean": sum(reference_jaccard) / len(reference_jaccard),
        "jaccard_140_160": _jaccard(by_epoch[140], by_epoch[160]),
    }
    finite = (
        bool(torch.isfinite(model.weight).all())
        and bool(torch.isfinite(model.bias).all())
        and all(math.isfinite(value) for value in values.values())
    )
    return pd.DataFrame(
        [
            {
                "seed": seed,
                "init_scale": init_scale,
                "n_groups": total_groups,
                **values,
                "finite": finite,
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether type-pair candidates persist or churn during the "
            "epoch-140..160 interval where selective attenuation can alter the "
            "long-run trajectory."
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--init-scale", type=float, default=0.115)
    parser.add_argument("--learning-rate", type=float, default=160.0)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = run_seed(
        seed=args.seed,
        init_scale=args.init_scale,
        learning_rate=args.learning_rate,
        max_update=args.max_update,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
