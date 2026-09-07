from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import pandas as pd
import torch

from predcircuit.baselines import BPTTGraphNetwork
from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import (
    RetinotopicFlyVisCircuit,
    graph_from_flyvis_retinotopy,
    type_pair_preserving_rewire,
)

DIRECTIONS = (0.0, 90.0, 180.0, 270.0)
T4_TYPES = ("T4a", "T4b", "T4c", "T4d")
DIRECTION_TO_T4 = {180.0: "T4a", 0.0: "T4b", 90.0: "T4c", 270.0: "T4d"}
TARGET_CORRECT = 0.5
TARGET_OTHER = -TARGET_CORRECT / (len(T4_TYPES) - 1)


def input_xy(circuit: RetinotopicFlyVisCircuit) -> torch.Tensor:
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
    xy = input_xy(circuit)
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
        batch.append(
            torch.stack(
                [
                    amplitude * torch.exp(-0.5 * ((projection - center) / sample_width).square())
                    for center in centers
                ]
            )
        )
    return torch.stack(batch)


def targets_for(directions: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.full((len(directions), len(T4_TYPES)), TARGET_OTHER, dtype=torch.float32)
    classes = torch.empty(len(directions), dtype=torch.long)
    for sample, direction in enumerate(directions):
        class_index = T4_TYPES.index(DIRECTION_TO_T4[direction])
        targets[sample, class_index] = TARGET_CORRECT
        classes[sample] = class_index
    return targets, classes


def output_nodes(circuit: RetinotopicFlyVisCircuit) -> list[int]:
    return [circuit.central_node(cell_type) for cell_type in T4_TYPES]


@torch.no_grad()
def evaluate(
    model: BPTTGraphNetwork,
    circuit: RetinotopicFlyVisCircuit,
    *,
    repeats: int,
    frames: int,
    width: float,
    steps_per_frame: int,
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
    targets, classes = targets_for(directions)
    readout = model.forward_sequence(
        stimulus,
        input_nodes=list(circuit.input_nodes),
        output_nodes=output_nodes(circuit),
        steps_per_frame=steps_per_frame,
    )
    mse = float((readout - targets).square().mean())
    accuracy = float((readout.argmax(dim=1) == classes).float().mean())
    correct = readout.gather(1, classes[:, None]).squeeze(1)
    masked = readout.clone()
    masked[torch.arange(len(classes)), classes] = -torch.inf
    margin = float((correct - masked.max(dim=1).values).mean())
    return mse, accuracy, margin


def run_one(
    circuit: RetinotopicFlyVisCircuit,
    *,
    topology: str,
    seed: int,
    epochs: int,
    frames: int,
    width: float,
    steps_per_frame: int,
    learning_rate: float,
    train_repeats: int,
    test_repeats: int,
) -> dict[str, float | int | str | bool]:
    model = BPTTGraphNetwork(circuit.graph, seed=seed, init_scale=0.08)
    mse_before, accuracy_before, margin_before = evaluate(
        model,
        circuit,
        repeats=test_repeats,
        frames=frames,
        width=width,
        steps_per_frame=steps_per_frame,
        jitter_seed=900_000 + seed,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    final_train_loss = float("nan")
    max_grad_norm = 0.0
    for epoch in range(epochs):
        directions = list(DIRECTIONS) * train_repeats
        stimulus = render_motion_batch(
            circuit,
            directions,
            frames=frames,
            width=width,
            jitter_seed=100_000 * seed + epoch,
        )
        targets, _ = targets_for(directions)
        optimizer.zero_grad(set_to_none=True)
        readout = model.forward_sequence(
            stimulus,
            input_nodes=list(circuit.input_nodes),
            output_nodes=output_nodes(circuit),
            steps_per_frame=steps_per_frame,
        )
        loss = (readout - targets).square().mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        max_grad_norm = max(max_grad_norm, float(grad_norm))
        optimizer.step()
        final_train_loss = float(loss.detach())

    mse_after, accuracy_after, margin_after = evaluate(
        model,
        circuit,
        repeats=test_repeats,
        frames=frames,
        width=width,
        steps_per_frame=steps_per_frame,
        jitter_seed=900_000 + seed,
    )
    return {
        "learning_rule": "bptt",
        "topology": topology,
        "seed": seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "mse_before": mse_before,
        "mse_after": mse_after,
        "mse_improvement": mse_before - mse_after,
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "accuracy_improvement": accuracy_after - accuracy_before,
        "margin_before": margin_before,
        "margin_after": margin_after,
        "final_train_loss": final_train_loss,
        "max_grad_norm": max_grad_norm,
        "mean_abs_weight": float(model.weight.detach().abs().mean()),
        "nodes": circuit.graph.num_nodes,
        "edges": circuit.graph.num_edges,
        "finite": bool(torch.isfinite(model.weight).all()) and math.isfinite(mse_after),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="BPTT control on retinotopic FlyVis motion")
    parser.add_argument("--extent", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--steps-per-frame", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-repeats", type=int, default=4)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out", type=Path, default=Path("results/generated/flyvis_retinotopic_bptt.csv")
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
                    steps_per_frame=args.steps_per_frame,
                    learning_rate=args.learning_rate,
                    train_repeats=args.train_repeats,
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
        f"BPTT crop: extent={args.extent}, {base.graph.num_nodes} nodes, "
        f"{base.graph.num_edges} edges, lr={args.learning_rate:g}"
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
                "max_grad_norm",
                "finite",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
