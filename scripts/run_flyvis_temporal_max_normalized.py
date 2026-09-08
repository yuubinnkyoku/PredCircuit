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
from run_flyvis_retinotopic_temporal_adam import temporal_local_direction

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import (
    RetinotopicFlyVisCircuit,
    graph_from_flyvis_retinotopy,
    type_pair_preserving_rewire,
)
from predcircuit.model import PredictiveCodingGraph


@torch.no_grad()
def max_normalized_step(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    stimulus: torch.Tensor,
    targets: torch.Tensor,
    *,
    beta: float,
    frame_steps: int,
    step_size: float,
    target_max_edge_update: float,
    epsilon: float,
) -> tuple[float, float, float, float, float]:
    """Scale the whole local direction so its largest edge step has a fixed magnitude.

    A single scalar is shared by all edge and bias coordinates, so the update vector is
    not rotated. Controlling the maximum rather than the mean avoids the large outlier
    steps observed with mean-absolute normalization on the heavy-tailed local signal.
    """
    edge_direction, bias_direction, output_shift = temporal_local_direction(
        model,
        circuit,
        stimulus,
        targets,
        beta=beta,
        frame_steps=frame_steps,
        step_size=step_size,
    )
    raw_max = float(edge_direction.abs().max())
    raw_mean = float(edge_direction.abs().mean())
    scale = target_max_edge_update / max(raw_max, epsilon)
    edge_delta = scale * edge_direction
    bias_delta = scale * bias_direction
    model.weight.add_(edge_delta)
    model.bias.add_(bias_delta)
    return (
        raw_mean,
        raw_max,
        float(edge_delta.abs().mean()),
        float(edge_delta.abs().max()),
        output_shift,
    )


def run_one(
    circuit: RetinotopicFlyVisCircuit,
    *,
    topology: str,
    seed: int,
    epochs: int,
    frames: int,
    width: float,
    frame_steps: int,
    step_size: float,
    beta: float,
    target_max_edge_update: float,
    epsilon: float,
    test_repeats: int,
) -> dict[str, float | int | str | bool]:
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
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

    raw_mean_sum = 0.0
    raw_max_sum = 0.0
    applied_mean_sum = 0.0
    applied_max_sum = 0.0
    nudge_sum = 0.0
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
            raw_mean, raw_max, applied_mean, applied_max, output_shift = max_normalized_step(
                model,
                circuit,
                stimulus,
                targets,
                beta=beta,
                frame_steps=frame_steps,
                step_size=step_size,
                target_max_edge_update=target_max_edge_update,
                epsilon=epsilon,
            )
            raw_mean_sum += raw_mean
            raw_max_sum += raw_max
            applied_mean_sum += applied_mean
            applied_max_sum += applied_max
            nudge_sum += output_shift
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
        "learning_rule": "temporal_contrastive_max_normalized",
        "topology": topology,
        "seed": seed,
        "epochs": epochs,
        "beta": beta,
        "target_max_edge_update": target_max_edge_update,
        "frame_steps": frame_steps,
        "mse_before": mse_before,
        "mse_after": mse_after,
        "mse_improvement": mse_before - mse_after,
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "margin_before": margin_before,
        "margin_after": margin_after,
        "mean_abs_raw_direction": raw_mean_sum / max(updates, 1),
        "mean_max_abs_raw_direction": raw_max_sum / max(updates, 1),
        "mean_abs_applied_update": applied_mean_sum / max(updates, 1),
        "mean_max_abs_applied_update": applied_max_sum / max(updates, 1),
        "mean_output_nudge": nudge_sum / max(updates, 1),
        "mean_abs_weight": float(model.weight.abs().mean()),
        "nodes": circuit.graph.num_nodes,
        "edges": circuit.graph.num_edges,
        "finite": bool(torch.isfinite(model.weight).all())
        and bool(torch.isfinite(model.bias).all())
        and math.isfinite(mse_after),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direction-preserving max-controlled normalization for online local PC"
    )
    parser.add_argument("--extent", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--target-max-edge-update", type=float, required=True)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_temporal_max_normalized.csv"),
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
                    seed=seed,
                    epochs=args.epochs,
                    frames=args.frames,
                    width=args.bar_width,
                    frame_steps=args.frame_steps,
                    step_size=args.step_size,
                    beta=args.beta,
                    target_max_edge_update=args.target_max_edge_update,
                    epsilon=args.epsilon,
                    test_repeats=args.test_repeats,
                )
            )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = (
        frame.groupby("topology")[["mse_after", "accuracy_after", "margin_after"]]
        .agg(["mean", "median", "std"])
        .sort_index()
    )
    print(
        f"Max-normalized local PC: extent={args.extent}, "
        f"target_max_edge_update={args.target_max_edge_update:g}"
    )
    print(summary.to_string())
    print("\nPer-seed results:")
    print(
        frame[
            [
                "topology",
                "seed",
                "mse_before",
                "mse_after",
                "accuracy_before",
                "accuracy_after",
                "margin_after",
                "mean_abs_raw_direction",
                "mean_max_abs_raw_direction",
                "mean_abs_applied_update",
                "mean_max_abs_applied_update",
                "finite",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
