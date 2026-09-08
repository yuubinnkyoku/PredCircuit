from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import pandas as pd
import torch
from run_flyvis_credit_residual_scaling import evaluate_metrics
from run_flyvis_retinotopic_contrastive import (
    DIRECTIONS,
    output_nodes,
    render_motion_batch,
    targets_for,
)
from run_flyvis_temporal_coherence_gate import apply_local_credit

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import RetinotopicFlyVisCircuit, graph_from_flyvis_retinotopy
from predcircuit.model import PredictiveCodingGraph


@torch.no_grad()
def infer_classification_nudged(
    model: PredictiveCodingGraph,
    initial_state: torch.Tensor,
    *,
    clamp_mask: torch.Tensor,
    clamp_values: torch.Tensor,
    nudged_nodes: list[int],
    classes: torch.Tensor,
    beta: float,
    steps: int,
    step_size: float,
) -> torch.Tensor:
    """Infer under a cross-entropy force applied only at the output population.

    The supervised force is beta * (softmax(z) - one_hot(y)), i.e. the exact gradient of
    cross entropy with respect to the four output activities. No task gradient is supplied to
    hidden states or synapses; it can reach them only through predictive-coding state dynamics.
    """
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if classes.shape != (initial_state.shape[0],):
        raise ValueError("classes must have shape [batch]")

    state = initial_state.clone().float()
    clamp_mask = clamp_mask.to(device=state.device, dtype=torch.bool)
    clamp_values = clamp_values.to(state.device, dtype=state.dtype)
    classes = classes.to(state.device)
    state[:, clamp_mask] = clamp_values[:, clamp_mask]

    for _ in range(steps):
        grad = model._internal_gradient(state)
        logits = state[:, nudged_nodes]
        probabilities = torch.softmax(logits, dim=1)
        output_force = probabilities
        output_force[torch.arange(state.shape[0], device=state.device), classes] -= 1.0
        grad[:, nudged_nodes] += beta * output_force
        grad[:, clamp_mask] = 0.0
        state -= step_size * grad
        state[:, clamp_mask] = clamp_values[:, clamp_mask]
    return state


@torch.no_grad()
def classification_trajectory_statistics(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    stimulus: torch.Tensor,
    *,
    classes: torch.Tensor | None,
    beta: float,
    frame_steps: int,
    step_size: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = torch.zeros(stimulus.shape[0], circuit.graph.num_nodes, dtype=torch.float32)
    clamp_mask = torch.zeros(circuit.graph.num_nodes, dtype=torch.bool)
    clamp_mask[list(circuit.input_nodes)] = True
    edge_stats = torch.zeros(circuit.graph.num_edges, dtype=torch.float32)
    bias_stats = torch.zeros(circuit.graph.num_nodes, dtype=torch.float32)
    outs = output_nodes(circuit)

    for frame in stimulus.unbind(dim=1):
        clamp_values = torch.zeros_like(state)
        clamp_values[:, list(circuit.input_nodes)] = frame
        if classes is None:
            state, _ = model.infer(
                state,
                clamp_mask=clamp_mask,
                clamp_values=clamp_values,
                steps=frame_steps,
                step_size=step_size,
            )
        else:
            state = infer_classification_nudged(
                model,
                state,
                clamp_mask=clamp_mask,
                clamp_values=clamp_values,
                nudged_nodes=outs,
                classes=classes,
                beta=beta,
                steps=frame_steps,
                step_size=step_size,
            )
        edge_stats += model.local_edge_statistics(state)
        bias_stats += model.errors(state).mean(dim=0)

    scale = 1.0 / stimulus.shape[1]
    return state, edge_stats * scale, bias_stats * scale


@torch.no_grad()
def classification_local_direction(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    stimulus: torch.Tensor,
    classes: torch.Tensor,
    *,
    beta: float,
    frame_steps: int,
    step_size: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    free_state, free_edge, free_bias = classification_trajectory_statistics(
        model,
        circuit,
        stimulus,
        classes=None,
        beta=beta,
        frame_steps=frame_steps,
        step_size=step_size,
    )
    nudged_state, nudged_edge, nudged_bias = classification_trajectory_statistics(
        model,
        circuit,
        stimulus,
        classes=classes,
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


def cycle_classification_credit(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    *,
    seed: int,
    epoch: int,
    frames: int,
    width: float,
    beta: float,
    frame_steps: int,
    step_size: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    edge_sum = torch.zeros_like(model.weight)
    bias_sum = torch.zeros_like(model.bias)
    output_shift_sum = 0.0
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
        _, classes = targets_for([direction])
        edge, bias, output_shift = classification_local_direction(
            model,
            circuit,
            stimulus,
            classes,
            beta=beta,
            frame_steps=frame_steps,
            step_size=step_size,
        )
        edge_sum += edge
        bias_sum += bias
        output_shift_sum += output_shift
    return edge_sum, bias_sum, output_shift_sum / len(directions)


def run_one(
    circuit: RetinotopicFlyVisCircuit,
    *,
    seed: int,
    epochs: int,
    frames: int,
    width: float,
    frame_steps: int,
    step_size: float,
    beta: float,
    learning_rate: float,
    weight_decay: float,
    max_update: float,
    test_repeats: int,
) -> dict[str, float | int | bool]:
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    before = evaluate_metrics(
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
    output_shift_sum = 0.0
    max_abs_update = 0.0
    for epoch in range(epochs):
        edge_direction, bias_direction, output_shift = cycle_classification_credit(
            model,
            circuit,
            seed=seed,
            epoch=epoch,
            frames=frames,
            width=width,
            beta=beta,
            frame_steps=frame_steps,
            step_size=step_size,
        )
        mean_update, step_max = apply_local_credit(
            model,
            edge_direction,
            bias_direction,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_update=max_update,
        )
        mean_update_sum += mean_update
        output_shift_sum += output_shift
        max_abs_update = max(max_abs_update, step_max)

    after = evaluate_metrics(
        model,
        circuit,
        repeats=test_repeats,
        frames=frames,
        width=width,
        frame_steps=frame_steps,
        step_size=step_size,
        jitter_seed=900_000 + seed,
    )
    count = max(epochs, 1)
    return {
        "seed": seed,
        "epochs": epochs,
        "beta": beta,
        "learning_rate": learning_rate,
        "mse_before": before["mse"],
        "mse_after": after["mse"],
        "accuracy_before": before["accuracy"],
        "accuracy_after": after["accuracy"],
        "margin_before": before["margin"],
        "margin_after": after["margin"],
        "cross_entropy_before": before["cross_entropy"],
        "cross_entropy_after": after["cross_entropy"],
        "cross_entropy_improvement": before["cross_entropy"] - after["cross_entropy"],
        "mean_abs_update": mean_update_sum / count,
        "max_abs_update": max_abs_update,
        "mean_output_nudge": output_shift_sum / count,
        "nodes": circuit.graph.num_nodes,
        "edges": circuit.graph.num_edges,
        "finite": bool(torch.isfinite(model.weight).all()) and math.isfinite(after["mse"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train FlyVis with a cross-entropy force restricted to the four output neurons; "
            "hidden credit remains predictive-coding-local"
        )
    )
    parser.add_argument("--extent", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--learning-rate", type=float, default=20.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-update", type=float, default=0.05)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/flyvis_classification_nudge.csv"),
    )
    args = parser.parse_args()

    circuit = graph_from_flyvis_retinotopy(load_flyvis_spec(), extent=args.extent)
    rows = [
        run_one(
            circuit,
            seed=seed,
            epochs=args.epochs,
            frames=args.frames,
            width=args.bar_width,
            frame_steps=args.frame_steps,
            step_size=args.step_size,
            beta=args.beta,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_update=args.max_update,
            test_repeats=args.test_repeats,
        )
        for seed in range(args.seeds)
    ]
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = frame[
        [
            "mse_after",
            "accuracy_after",
            "margin_after",
            "cross_entropy_after",
            "cross_entropy_improvement",
            "mean_abs_update",
            "mean_output_nudge",
        ]
    ].agg(["mean", "median", "std"])
    print(
        f"Classification nudge: extent={args.extent}, beta={args.beta:g}, "
        f"lr={args.learning_rate:g}, {circuit.graph.num_nodes} nodes, "
        f"{circuit.graph.num_edges} edges"
    )
    print(summary.to_string())
    print("\nPer-seed results:")
    print(
        frame[
            [
                "seed",
                "accuracy_after",
                "mse_after",
                "cross_entropy_after",
                "margin_after",
                "mean_abs_update",
                "mean_output_nudge",
                "finite",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
