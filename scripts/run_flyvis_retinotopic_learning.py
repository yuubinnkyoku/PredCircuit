from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import pandas as pd
import torch

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import (
    RetinotopicFlyVisCircuit,
    axial_hex_radius,
    graph_from_flyvis_retinotopy,
    type_pair_preserving_rewire,
)
from predcircuit.model import PredictiveCodingGraph

DIRECTIONS = (0.0, 90.0, 180.0, 270.0)
T4_TYPES = ("T4a", "T4b", "T4c", "T4d")
DIRECTION_TO_T4 = {180.0: "T4a", 0.0: "T4b", 90.0: "T4c", 270.0: "T4d"}
TARGET_CORRECT = 0.5
TARGET_OTHER = -TARGET_CORRECT / (len(T4_TYPES) - 1)


def _input_xy(circuit: RetinotopicFlyVisCircuit) -> torch.Tensor:
    indices = torch.tensor(circuit.input_nodes, dtype=torch.long)
    u = circuit.node_u[indices].float()
    v = circuit.node_v[indices].float()
    return torch.stack([u + 0.5 * v, (math.sqrt(3.0) / 2.0) * v], dim=1)


def render_motion_batch(
    circuit: RetinotopicFlyVisCircuit,
    directions: list[float],
    *,
    frames: int,
    width: float,
    jitter_seed: int,
) -> torch.Tensor:
    """Render bright moving bars with small held-out phase/width/amplitude variation."""
    xy = _input_xy(circuit)
    batch: list[torch.Tensor] = []
    for sample, direction_deg in enumerate(directions):
        rng = random.Random(jitter_seed + 1009 * sample)
        theta = math.radians(direction_deg)
        direction = torch.tensor([math.cos(theta), math.sin(theta)], dtype=torch.float32)
        projection = xy @ direction
        span = float(projection.abs().max()) + 0.8
        phase = rng.uniform(-0.3, 0.3)
        sample_width = width * rng.uniform(0.85, 1.15)
        amplitude = rng.uniform(0.85, 1.0)
        centers = torch.linspace(-span, span, frames) + phase
        rendered = [
            amplitude * torch.exp(-0.5 * ((projection - center) / sample_width).square())
            for center in centers
        ]
        batch.append(torch.stack(rendered))
    return torch.stack(batch)


def output_nodes(circuit: RetinotopicFlyVisCircuit) -> list[int]:
    return [circuit.central_node(cell_type) for cell_type in T4_TYPES]


def target_values(directions: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-mean simplex-like targets so a constant output cannot improve MSE."""
    targets = torch.full((len(directions), len(T4_TYPES)), TARGET_OTHER, dtype=torch.float32)
    classes = torch.empty(len(directions), dtype=torch.long)
    for sample, direction in enumerate(directions):
        target_type = DIRECTION_TO_T4[direction]
        class_index = T4_TYPES.index(target_type)
        targets[sample, class_index] = TARGET_CORRECT
        classes[sample] = class_index
    return targets, classes


@torch.no_grad()
def infer_sequence(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    stimulus: torch.Tensor,
    *,
    frame_steps: int,
    step_size: float,
) -> torch.Tensor:
    batch = stimulus.shape[0]
    state = torch.zeros(batch, circuit.graph.num_nodes, dtype=torch.float32)
    clamp_mask = torch.zeros(circuit.graph.num_nodes, dtype=torch.bool)
    clamp_mask[list(circuit.input_nodes)] = True
    clamp_values = torch.zeros_like(state)
    state, _ = model.infer(
        state,
        clamp_mask=clamp_mask,
        clamp_values=clamp_values,
        steps=max(frame_steps * 2, 1),
        step_size=step_size,
    )
    for frame in stimulus.unbind(dim=1):
        clamp_values = torch.zeros_like(state)
        clamp_values[:, list(circuit.input_nodes)] = frame
        state, _ = model.infer(
            state,
            clamp_mask=clamp_mask,
            clamp_values=clamp_values,
            steps=frame_steps,
            step_size=step_size,
        )
    return state


@torch.no_grad()
def supervised_local_step(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    stimulus: torch.Tensor,
    targets: torch.Tensor,
    *,
    frame_steps: int,
    target_steps: int,
    step_size: float,
    weight_lr: float,
) -> tuple[float, float, float]:
    state = infer_sequence(
        model,
        circuit,
        stimulus,
        frame_steps=frame_steps,
        step_size=step_size,
    )
    out_nodes = output_nodes(circuit)
    clamp_mask = torch.zeros(circuit.graph.num_nodes, dtype=torch.bool)
    clamp_mask[list(circuit.input_nodes)] = True
    clamp_mask[out_nodes] = True
    clamp_values = torch.zeros_like(state)
    clamp_values[:, list(circuit.input_nodes)] = stimulus[:, -1]
    clamp_values[:, out_nodes] = targets
    state, _ = model.infer(
        state,
        clamp_mask=clamp_mask,
        clamp_values=clamp_values,
        steps=target_steps,
        step_size=step_size,
    )
    delta = model.local_weight_step(
        state,
        learning_rate=weight_lr,
        weight_decay=1e-5,
        clip=0.05,
    )
    return float(model.energy(state)), float(delta.abs().mean()), float(delta.abs().max())


@torch.no_grad()
def evaluate(
    model: PredictiveCodingGraph,
    circuit: RetinotopicFlyVisCircuit,
    *,
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
    targets, classes = target_values(directions)
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


def mean_spatial_edge_length(circuit: RetinotopicFlyVisCircuit) -> float:
    if not circuit.graph.num_edges:
        return 0.0
    src, dst = circuit.graph.edge_index
    lengths = [
        axial_hex_radius(
            int(circuit.node_u[int(target)] - circuit.node_u[int(source)]),
            int(circuit.node_v[int(target)] - circuit.node_v[int(source)]),
        )
        for source, target in zip(src.tolist(), dst.tolist(), strict=True)
    ]
    return sum(lengths) / len(lengths)


def run_one(
    circuit: RetinotopicFlyVisCircuit,
    *,
    topology: str,
    seed: int,
    epochs: int,
    frames: int,
    width: float,
    frame_steps: int,
    target_steps: int,
    step_size: float,
    weight_lr: float,
    train_batch_size: int,
    test_repeats: int,
) -> dict[str, float | int | str | bool]:
    if train_batch_size <= 0 or train_batch_size > len(DIRECTIONS):
        raise ValueError("train_batch_size must be between 1 and 4")
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

    final_energy = float("nan")
    update_sum = 0.0
    max_abs_update = 0.0
    update_count = 0
    for epoch in range(epochs):
        directions = list(DIRECTIONS)
        random.Random(500_000 * seed + epoch).shuffle(directions)
        for batch_index, start in enumerate(range(0, len(directions), train_batch_size)):
            batch_directions = directions[start : start + train_batch_size]
            stimulus = render_motion_batch(
                circuit,
                batch_directions,
                frames=frames,
                width=width,
                jitter_seed=100_000 * seed + 100 * epoch + batch_index,
            )
            targets, _ = target_values(batch_directions)
            final_energy, mean_update, step_max_update = supervised_local_step(
                model,
                circuit,
                stimulus,
                targets,
                frame_steps=frame_steps,
                target_steps=target_steps,
                step_size=step_size,
                weight_lr=weight_lr,
            )
            update_sum += mean_update
            max_abs_update = max(max_abs_update, step_max_update)
            update_count += 1

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
        "learning_rule": "predictive_coding_local",
        "topology": topology,
        "seed": seed,
        "epochs": epochs,
        "weight_lr": weight_lr,
        "train_batch_size": train_batch_size,
        "target_steps": target_steps,
        "mse_before": mse_before,
        "mse_after": mse_after,
        "mse_improvement": mse_before - mse_after,
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "accuracy_improvement": accuracy_after - accuracy_before,
        "margin_before": margin_before,
        "margin_after": margin_after,
        "final_clamped_energy": final_energy,
        "mean_abs_update": update_sum / max(update_count, 1),
        "max_abs_update": max_abs_update,
        "mean_abs_weight": float(model.weight.abs().mean()),
        "nodes": circuit.graph.num_nodes,
        "edges": circuit.graph.num_edges,
        "mean_spatial_edge_length": mean_spatial_edge_length(circuit),
        "finite": bool(torch.isfinite(model.weight).all()) and math.isfinite(mse_after),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Local-PC learning on a retinotopic FlyVis crop")
    parser.add_argument("--extent", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--target-steps", type=int, default=8)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--weight-lr", type=float, default=1e-3)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--test-repeats", type=int, default=4)
    parser.add_argument(
        "--out", type=Path, default=Path("results/generated/flyvis_retinotopic_learning.csv")
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
        before_in, before_out = base.graph.degrees()
        after_in, after_out = rewired.graph.degrees()
        if not torch.equal(before_in, after_in) or not torch.equal(before_out, after_out):
            raise RuntimeError("type-pair rewire failed degree-preservation check")
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
                    target_steps=args.target_steps,
                    step_size=args.step_size,
                    weight_lr=args.weight_lr,
                    train_batch_size=args.train_batch_size,
                    test_repeats=args.test_repeats,
                )
            )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = (
        frame.groupby("topology")[
            [
                "mse_after",
                "accuracy_after",
                "margin_after",
                "mse_improvement",
                "mean_abs_update",
            ]
        ]
        .agg(["mean", "median", "std"])
        .sort_index()
    )
    chance_mse = (TARGET_CORRECT**2 + (len(T4_TYPES) - 1) * TARGET_OTHER**2) / len(T4_TYPES)
    print(
        f"Learning crop: extent={args.extent}, {base.graph.num_nodes} nodes, "
        f"{base.graph.num_edges} edges, weight_lr={args.weight_lr:g}, "
        f"batch={args.train_batch_size}, target_steps={args.target_steps}, "
        f"zero-output MSE={chance_mse:.6f}"
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
                "mean_abs_update",
                "max_abs_update",
                "finite",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
