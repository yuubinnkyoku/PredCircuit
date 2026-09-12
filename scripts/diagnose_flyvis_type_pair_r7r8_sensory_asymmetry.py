from __future__ import annotations

import argparse
import copy
import math
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from diagnose_flyvis_branch_nudge_alignment import branched_local_direction
from diagnose_flyvis_type_pair_horizon_causal import _match_local_state
from diagnose_flyvis_type_pair_mi9_causal import _named_groups
from run_flyvis_retinotopic_contrastive import (
    DIRECTIONS,
    infer_sequence,
    output_nodes,
    render_motion_batch,
    targets_for,
)
from run_flyvis_temporal_coherence_gate import apply_local_credit
from run_flyvis_type_pair_norm_matched_control import match_norm

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import RetinotopicFlyVisCircuit, graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph

HORIZONS = (60, 80, 100)
EVAL_REPS = 4
EVAL_JITTER_BASE = 3_100_000
PROFILES = {
    "r8_high": {"R7": 0.75, "R8": 1.25},
    "r7_high": {"R7": 1.25, "R8": 0.75},
}
RULES = ("local", "r7_only", "r8_only", "r7r8")


def asymmetric_render_motion_batch(
    circuit: RetinotopicFlyVisCircuit,
    directions: list[float],
    *,
    frames: int,
    width: float,
    jitter_seed: int,
    gains: dict[str, float],
) -> torch.Tensor:
    stimulus = render_motion_batch(
        circuit,
        directions,
        frames=frames,
        width=width,
        jitter_seed=jitter_seed,
    )
    input_gains = torch.tensor(
        [gains.get(circuit.node_types[index], 1.0) for index in circuit.input_nodes],
        dtype=stimulus.dtype,
        device=stimulus.device,
    )
    return stimulus * input_gains


def asymmetric_credit(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    *,
    seed: int,
    epoch: int,
    gains: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_sum = torch.zeros_like(model.weight)
    bias_sum = torch.zeros_like(model.bias)
    directions = list(DIRECTIONS)
    random.Random(700_000 * seed + epoch).shuffle(directions)
    for sample_index, direction in enumerate(directions):
        stimulus = asymmetric_render_motion_batch(
            circuit,
            [direction],
            frames=7,
            width=0.5,
            jitter_seed=100_000 * seed + 100 * epoch + sample_index,
            gains=gains,
        )
        _, classes = targets_for([direction])
        edge, bias = branched_local_direction(
            model,
            circuit,
            stimulus,
            classes,
            beta=0.03,
            frame_steps=2,
            nudge_steps=2,
            step_size=0.015,
            terminal_only=False,
        )
        edge_sum += edge
        bias_sum += bias
    return edge_sum, bias_sum


@torch.no_grad()
def asymmetric_heldout_metrics(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    *,
    jitter_seed: int,
    gains: dict[str, float],
) -> dict[str, float]:
    directions = list(DIRECTIONS) * 12
    stimulus = asymmetric_render_motion_batch(
        circuit,
        directions,
        frames=7,
        width=0.5,
        jitter_seed=jitter_seed,
        gains=gains,
    )
    _, classes = targets_for(directions)
    state = infer_sequence(
        model,
        circuit,
        stimulus,
        frame_steps=2,
        step_size=0.015,
    )
    readout = state[:, output_nodes(circuit)]
    correct = readout.gather(1, classes[:, None]).squeeze(1)
    wrong = readout.clone()
    wrong[torch.arange(len(classes)), classes] = -torch.inf
    hard_margin = correct - wrong.max(dim=1).values
    return {
        "cross_entropy": float(F.cross_entropy(readout, classes)),
        "accuracy": float((readout.argmax(dim=1) == classes).float().mean()),
        "hard_margin": float(hard_margin.mean()),
    }


def _selected_direction(
    raw_edge: torch.Tensor,
    named_groups: list[tuple[str, str, torch.Tensor]],
    rule: str,
) -> torch.Tensor:
    result = raw_edge.clone()
    for source_type, target_type, indices in named_groups:
        selected = target_type == "Mi9" and (
            (rule == "r7_only" and source_type == "R7")
            or (rule == "r8_only" and source_type == "R8")
            or (rule == "r7r8" and source_type in {"R7", "R8"})
        )
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
    base = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    named_groups = _named_groups(circuit)
    pair_counts = {
        source: sum(
            source_type == source and target_type == "Mi9"
            for source_type, target_type, _ in named_groups
        )
        for source in ("R7", "R8")
    }
    if pair_counts != {"R7": 1, "R8": 1}:
        raise RuntimeError(f"unexpected R7/R8->Mi9 group counts: {pair_counts}")

    models = {
        profile: {rule: copy.deepcopy(base) for rule in RULES}
        for profile in PROFILES
    }
    direction_change_sum = {
        profile: {rule: 0.0 for rule in RULES if rule != "local"}
        for profile in PROFILES
    }
    weight_error_sum = copy.deepcopy(direction_change_sum)
    bias_error_sum = copy.deepcopy(direction_change_sum)
    rows: list[dict[str, float | int | bool | str]] = []

    for step in range(1, max(HORIZONS) + 1):
        epoch = step - 1
        for profile, gains in PROFILES.items():
            profile_models = models[profile]
            local_edge, local_bias = asymmetric_credit(
                profile_models["local"],
                circuit,
                seed=seed,
                epoch=epoch,
                gains=gains,
            )
            apply_local_credit(
                profile_models["local"],
                local_edge,
                local_bias,
                learning_rate=learning_rate,
                weight_decay=0.0,
                max_update=max_update,
            )

            for rule in ("r7_only", "r8_only", "r7r8"):
                model = profile_models[rule]
                raw_edge, raw_bias = asymmetric_credit(
                    model,
                    circuit,
                    seed=seed,
                    epoch=epoch,
                    gains=gains,
                )
                direction = _selected_direction(raw_edge, named_groups, rule)
                direction_change_sum[profile][rule] += float(
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
                weight_error, bias_error = _match_local_state(model, profile_models["local"])
                weight_error_sum[profile][rule] += weight_error
                bias_error_sum[profile][rule] += bias_error

        if step not in HORIZONS:
            continue

        for eval_rep in range(EVAL_REPS):
            jitter_seed = EVAL_JITTER_BASE + eval_rep
            for profile, gains in PROFILES.items():
                for rule in RULES:
                    model = models[profile][rule]
                    metrics = asymmetric_heldout_metrics(
                        model,
                        circuit,
                        jitter_seed=jitter_seed,
                        gains=gains,
                    )
                    if rule == "local":
                        mean_direction_change = 0.0
                        mean_weight_error = 0.0
                        mean_bias_error = 0.0
                    else:
                        mean_direction_change = direction_change_sum[profile][rule] / step
                        mean_weight_error = weight_error_sum[profile][rule] / step
                        mean_bias_error = bias_error_sum[profile][rule] / step
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
                            "profile": profile,
                            "r7_gain": gains["R7"],
                            "r8_gain": gains["R8"],
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
            "Break the synthetic R7/R8 sensory symmetry with mirrored gain profiles and test "
            "whether sparse R7/R8->Mi9 shared credit remains causal"
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
