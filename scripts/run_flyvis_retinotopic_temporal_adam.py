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
    output_nodes,
    render_motion_batch,
    targets_for,
)
from run_flyvis_retinotopic_temporal_contrastive import trajectory_statistics

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import (
    RetinotopicFlyVisCircuit,
    graph_from_flyvis_retinotopy,
    type_pair_preserving_rewire,
)
from predcircuit.model import PredictiveCodingGraph


@torch.no_grad()
def local_adam_step(
    parameter: torch.Tensor,
    direction: torch.Tensor,
    first_moment: torch.Tensor,
    second_moment: torch.Tensor,
    *,
    step: int,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
) -> torch.Tensor:
    """Adam-style adaptation for a local descent direction; returns the added delta."""
    first_moment.mul_(beta1).add_(direction, alpha=1.0 - beta1)
    second_moment.mul_(beta2).addcmul_(direction, direction, value=1.0 - beta2)
    first_hat = first_moment / (1.0 - beta1**step)
    second_hat = second_moment / (1.0 - beta2**step)
    delta = learning_rate * first_hat / (second_hat.sqrt() + epsilon)
    parameter.add_(delta)
    return delta


@torch.no_grad()
def temporal_local_direction(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    stimulus: torch.Tensor,
    targets: torch.Tensor,
    *,
    beta: float,
    frame_steps: int,
    step_size: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    free_state, free_edge, free_bias = trajectory_statistics(
        model,
        circuit,
        stimulus,
        targets=None,
        beta=beta,
        frame_steps=frame_steps,
        step_size=step_size,
    )
    nudged_state, nudged_edge, nudged_bias = trajectory_statistics(
        model,
        circuit,
        stimulus,
        targets=targets,
        beta=beta,
        frame_steps=frame_steps,
        step_size=step_size,
    )
    edge_direction = (nudged_edge - free_edge) / beta
    bias_direction = (nudged_bias - free_bias) / beta
    output_shift = float(
        (nudged_state[:, output_nodes(circuit)] - free_state[:, output_nodes(circuit)]).abs().mean()
    )
    return edge_direction, bias_direction, output_shift


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
    learning_rate: float,
    beta1: float,
    beta2: float,
    adam_epsilon: float,
    test_repeats: int,
) -> dict[str, float | int | str | bool]:
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    edge_m = torch.zeros_like(model.weight)
    edge_v = torch.zeros_like(model.weight)
    bias_m = torch.zeros_like(model.bias)
    bias_v = torch.zeros_like(model.bias)

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
    raw_direction_sum = 0.0
    applied_update_sum = 0.0
    output_shift_sum = 0.0
    max_applied_update = 0.0
    update_count = 0
    for epoch in range(epochs):
        directions = list(DIRECTIONS)
        random.Random(900_000 * seed + epoch).shuffle(directions)
        for sample_index, direction in enumerate(directions):
            stimulus = render_motion_batch(
                circuit,
                [direction],
                frames=frames,
                width=width,
                jitter_seed=100_000 * seed + 100 * epoch + sample_index,
            )
            targets, _ = targets_for([direction])
            edge_direction, bias_direction, output_shift = temporal_local_direction(
                model,
                circuit,
                stimulus,
                targets,
                beta=beta,
                frame_steps=frame_steps,
                step_size=step_size,
            )
            update_count += 1
            edge_delta = local_adam_step(
                model.weight,
                edge_direction,
                edge_m,
                edge_v,
                step=update_count,
                learning_rate=learning_rate,
                beta1=beta1,
                beta2=beta2,
                epsilon=adam_epsilon,
            )
            local_adam_step(
                model.bias,
                bias_direction,
                bias_m,
                bias_v,
                step=update_count,
                learning_rate=learning_rate,
                beta1=beta1,
                beta2=beta2,
                epsilon=adam_epsilon,
            )
            raw_direction_sum += float(edge_direction.abs().mean())
            applied_update_sum += float(edge_delta.abs().mean())
            max_applied_update = max(max_applied_update, float(edge_delta.abs().max()))
            output_shift_sum += output_shift

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
        "learning_rule": "temporal_contrastive_local_adam",
        "topology": topology,
        "seed": seed,
        "epochs": epochs,
        "beta": beta,
        "learning_rate": learning_rate,
        "beta1": beta1,
        "beta2": beta2,
        "adam_epsilon": adam_epsilon,
        "frame_steps": frame_steps,
        "mse_before": mse_before,
        "mse_after": mse_after,
        "mse_improvement": mse_before - mse_after,
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "margin_before": margin_before,
        "margin_after": margin_after,
        "mean_abs_raw_direction": raw_direction_sum / max(update_count, 1),
        "mean_abs_applied_update": applied_update_sum / max(update_count, 1),
        "max_abs_applied_update": max_applied_update,
        "mean_output_nudge": output_shift_sum / max(update_count, 1),
        "mean_abs_weight": float(model.weight.abs().mean()),
        "nodes": circuit.graph.num_nodes,
        "edges": circuit.graph.num_edges,
        "finite": bool(torch.isfinite(model.weight).all()) and math.isfinite(mse_after),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temporal contrastive local PC with per-synapse Adam adaptation"
    )
    parser.add_argument("--extent", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out", type=Path, default=Path("results/generated/flyvis_temporal_local_adam.csv")
    )
    args = parser.parse_args()

    base = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=args.extent)
    rows: list[dict[str, float | int | str | bool]] = []
    for seed in range(args.seeds):
        rewired = type_pair_preserving_rewire(
            base,
            swaps=max(base.graph.num_edges * 2, 1),
            seed=10_000 + seed,
        )
        for topology, circuit in (("biological", base), ("type_pair_rewire", rewired)):
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
                    learning_rate=args.learning_rate,
                    beta1=args.beta1,
                    beta2=args.beta2,
                    adam_epsilon=args.adam_epsilon,
                    test_repeats=args.test_repeats,
                )
            )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = (
        frame.groupby("topology")[
            ["mse_after", "accuracy_after", "margin_after", "mse_improvement"]
        ]
        .agg(["mean", "median", "std"])
        .sort_index()
    )
    print(
        f"Temporal local Adam: extent={args.extent}, {base.graph.num_nodes} nodes, "
        f"{base.graph.num_edges} edges, beta={args.beta:g}, lr={args.learning_rate:g}"
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
                "mean_abs_applied_update",
                "finite",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
