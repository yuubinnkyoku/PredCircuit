from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import pandas as pd
import torch
from run_flyvis_retinotopic_contrastive import (
    DIRECTIONS,
    evaluate,
    render_motion_batch,
    targets_for,
)
from run_flyvis_retinotopic_temporal_contrastive import temporal_contrastive_step

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import (
    RetinotopicFlyVisCircuit,
    graph_from_flyvis_retinotopy,
    type_pair_preserving_rewire,
)
from predcircuit.model import PredictiveCodingGraph


@torch.no_grad()
def run_one(
    circuit: RetinotopicFlyVisCircuit,
    *,
    topology: str,
    init_mode: str,
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
    use_biological_strength = init_mode == "connectome_strength"
    model = PredictiveCodingGraph(
        circuit.graph,
        seed=seed,
        init_scale=0.08,
        use_biological_strength=use_biological_strength,
    )
    mse_before, accuracy_before, margin_before = evaluate(
        model,
        circuit,
        repeats=test_repeats,
        frames=frames,
        width=width,
        frame_steps=frame_steps,
        step_size=step_size,
        jitter_seed=900_000 + seed,
    )

    mean_update_sum = 0.0
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
            targets, _ = targets_for([direction])
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
            mean_update_sum += mean_update
            max_update = max(max_update, step_max)
            updates += 1

    mse_after, accuracy_after, margin_after = evaluate(
        model,
        circuit,
        repeats=test_repeats,
        frames=frames,
        width=width,
        frame_steps=frame_steps,
        step_size=step_size,
        jitter_seed=900_000 + seed,
    )
    return {
        "topology": topology,
        "init_mode": init_mode,
        "seed": seed,
        "epochs": epochs,
        "beta": beta,
        "weight_lr": weight_lr,
        "mse_before": mse_before,
        "mse_after": mse_after,
        "mse_improvement": mse_before - mse_after,
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "margin_before": margin_before,
        "margin_after": margin_after,
        "mean_abs_update": mean_update_sum / max(updates, 1),
        "max_abs_update": max_update,
        "mean_abs_weight": float(model.weight.abs().mean()),
        "finite": bool(torch.isfinite(model.weight).all()) and math.isfinite(mse_after),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare measured connectome-strength and random initialization under local PC"
    )
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
        default=Path("results/generated/flyvis_temporal_strength_init.csv"),
    )
    args = parser.parse_args()

    spec = load_flyvis_spec()
    biological = graph_from_flyvis_retinotopy(spec, extent=args.extent)
    rows: list[dict[str, float | int | str | bool]] = []
    for seed in range(args.seeds):
        pair_rotation = graph_from_flyvis_retinotopy(
            spec,
            extent=args.extent,
            pair_rotation_seed=20_000 + seed,
        )
        type_pair = type_pair_preserving_rewire(
            biological,
            swaps=max(biological.graph.num_edges * 2, 1),
            seed=10_000 + seed,
        )
        for topology, circuit in (
            ("biological", biological),
            ("pair_rotation", pair_rotation),
            ("type_pair_rewire", type_pair),
        ):
            for init_mode in ("random", "connectome_strength"):
                rows.append(
                    run_one(
                        circuit,
                        topology=topology,
                        init_mode=init_mode,
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
        frame.groupby(["topology", "init_mode"])[
            [
                "mse_before",
                "mse_after",
                "accuracy_before",
                "accuracy_after",
                "margin_after",
            ]
        ]
        .agg(["mean", "median", "std"])
        .sort_index()
    )
    print(
        f"Temporal strength-init comparison: lr={args.weight_lr:g}, "
        f"seeds={args.seeds}, extent={args.extent}"
    )
    print(summary.to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
