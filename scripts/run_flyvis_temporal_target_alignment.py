from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import pandas as pd
import torch
from run_flyvis_retinotopic_contrastive import (
    DIRECTIONS,
    TARGET_CORRECT,
    TARGET_OTHER,
    infer_sequence,
    output_nodes,
    render_motion_batch,
)
from run_flyvis_retinotopic_temporal_contrastive import temporal_contrastive_step

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import (
    RetinotopicFlyVisCircuit,
    graph_from_flyvis_retinotopy,
    type_pair_preserving_rewire,
)
from predcircuit.model import PredictiveCodingGraph

CANONICAL_CLASSES = (1, 2, 0, 3)


def shifted_classes(shift: int) -> tuple[int, ...]:
    return tuple((class_index + shift) % 4 for class_index in CANONICAL_CLASSES)


def targets_for_mapping(
    directions: list[float], class_map: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.full((len(directions), 4), TARGET_OTHER, dtype=torch.float32)
    classes = torch.empty(len(directions), dtype=torch.long)
    for sample, direction in enumerate(directions):
        direction_index = DIRECTIONS.index(direction)
        class_index = class_map[direction_index]
        targets[sample, class_index] = TARGET_CORRECT
        classes[sample] = class_index
    return targets, classes


@torch.no_grad()
def evaluate_mapping(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    *,
    class_map: tuple[int, ...],
    repeats: int,
    frames: int,
    width: float,
    frame_steps: int,
    step_size: float,
    jitter_seed: int,
) -> tuple[float, float, float]:
    directions = list(DIRECTIONS) * repeats
    stimulus = render_motion_batch(
        circuit,
        directions,
        frames=frames,
        width=width,
        jitter_seed=jitter_seed,
    )
    targets, classes = targets_for_mapping(directions, class_map)
    state = infer_sequence(
        model,
        circuit,
        stimulus,
        frame_steps=frame_steps,
        step_size=step_size,
    )
    readout = state[:, output_nodes(circuit)]
    mse = float((readout - targets).square().mean())
    accuracy = float((readout.argmax(dim=1) == classes).float().mean())
    correct = readout.gather(1, classes[:, None]).squeeze(1)
    masked = readout.clone()
    masked[torch.arange(len(classes)), classes] = -torch.inf
    margin = float((correct - masked.max(dim=1).values).mean())
    return mse, accuracy, margin


@torch.no_grad()
def run_one(
    circuit: RetinotopicFlyVisCircuit,
    *,
    topology: str,
    shift: int,
    seed: int,
    epochs: int,
    frames: int,
    width: float,
    frame_steps: int,
    step_size: float,
    beta: float,
    weight_lr: float,
    test_repeats: int,
) -> dict[str, float | int | str | bool]:
    class_map = shifted_classes(shift)
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    mse_before, accuracy_before, margin_before = evaluate_mapping(
        model,
        circuit,
        class_map=class_map,
        repeats=test_repeats,
        frames=frames,
        width=width,
        frame_steps=frame_steps,
        step_size=step_size,
        jitter_seed=900_000 + seed,
    )

    update_sum = 0.0
    max_update = 0.0
    updates = 0
    for epoch in range(epochs):
        directions = list(DIRECTIONS)
        random.Random(700_000 * seed + epoch).shuffle(directions)
        for sample_index, direction in enumerate(directions):
            stimulus = render_motion_batch(
                circuit,
                [direction],
                frames=frames,
                width=width,
                jitter_seed=100_000 * seed + 100 * epoch + sample_index,
            )
            targets, _ = targets_for_mapping([direction], class_map)
            mean_update, step_max, _ = temporal_contrastive_step(
                model,
                circuit,
                stimulus,
                targets,
                beta=beta,
                frame_steps=frame_steps,
                step_size=step_size,
                weight_lr=weight_lr,
            )
            update_sum += mean_update
            max_update = max(max_update, step_max)
            updates += 1

    mse_after, accuracy_after, margin_after = evaluate_mapping(
        model,
        circuit,
        class_map=class_map,
        repeats=test_repeats,
        frames=frames,
        width=width,
        frame_steps=frame_steps,
        step_size=step_size,
        jitter_seed=900_000 + seed,
    )
    return {
        "topology": topology,
        "target_shift": shift,
        "target_map": ",".join(str(value) for value in class_map),
        "seed": seed,
        "mse_before": mse_before,
        "mse_after": mse_after,
        "mse_improvement": mse_before - mse_after,
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "margin_before": margin_before,
        "margin_after": margin_after,
        "mean_abs_update": update_sum / max(updates, 1),
        "max_abs_update": max_update,
        "finite": bool(torch.isfinite(model.weight).all()) and math.isfinite(mse_after),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test whether local-PC topology benefit depends on biological T4 target alignment"
    )
    parser.add_argument("--target-shift", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--extent", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--weight-lr", type=float, default=10.0)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_temporal_target_alignment.csv"),
    )
    args = parser.parse_args()

    biological = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=args.extent)
    rows: list[dict[str, float | int | str | bool]] = []
    for seed in range(args.seeds):
        rewired = type_pair_preserving_rewire(
            biological,
            swaps=max(biological.graph.num_edges * 2, 1),
            seed=10_000 + seed,
        )
        for topology, circuit in (
            ("biological", biological),
            ("type_pair_rewire", rewired),
        ):
            rows.append(
                run_one(
                    circuit,
                    topology=topology,
                    shift=args.target_shift,
                    seed=seed,
                    epochs=args.epochs,
                    frames=args.frames,
                    width=args.bar_width,
                    frame_steps=args.frame_steps,
                    step_size=args.step_size,
                    beta=args.beta,
                    weight_lr=args.weight_lr,
                    test_repeats=args.test_repeats,
                )
            )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = (
        frame.groupby("topology")[
            ["mse_after", "mse_improvement", "accuracy_after", "margin_after"]
        ]
        .agg(["mean", "median", "std"])
        .sort_index()
    )
    print(
        f"Temporal target alignment: shift={args.target_shift}, "
        f"map={shifted_classes(args.target_shift)}, lr={args.weight_lr:g}"
    )
    print(summary.to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
