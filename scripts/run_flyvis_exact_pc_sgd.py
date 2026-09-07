from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from run_flyvis_exact_pc_gradient import differentiable_sequence, sync_model
from run_flyvis_retinotopic_contrastive import (
    DIRECTIONS,
    evaluate,
    output_nodes,
    render_motion_batch,
    targets_for,
)

from predcircuit.flyvis import load_flyvis_spec
from predcircuit.flyvis_retinotopy import (
    RetinotopicFlyVisCircuit,
    graph_from_flyvis_retinotopy,
    type_pair_preserving_rewire,
)
from predcircuit.model import PredictiveCodingGraph


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
    learning_rate: float,
    momentum: float,
    train_repeats: int,
    test_repeats: int,
) -> dict[str, float | int | str | bool]:
    model = PredictiveCodingGraph(circuit.graph, seed=seed, init_scale=0.08)
    weight = torch.nn.Parameter(model.weight.detach().clone())
    bias = torch.nn.Parameter(model.bias.detach().clone())
    optimizer = torch.optim.SGD([weight, bias], lr=learning_rate, momentum=momentum)

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
    max_grad_norm = 0.0
    final_train_loss = float("nan")
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
        state = differentiable_sequence(
            circuit,
            stimulus,
            weight,
            bias,
            frame_steps=frame_steps,
            step_size=step_size,
        )
        readout = state[:, output_nodes(circuit)]
        loss = (readout - targets).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([weight, bias], 10.0)
        max_grad_norm = max(max_grad_norm, float(grad_norm))
        optimizer.step()
        final_train_loss = float(loss.detach())

    sync_model(model, weight, bias)
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
        "learning_rule": "exact_unrolled_pc_gradient",
        "optimizer": "sgd",
        "topology": topology,
        "seed": seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "mse_before": mse_before,
        "mse_after": mse_after,
        "mse_improvement": mse_before - mse_after,
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "margin_before": margin_before,
        "margin_after": margin_after,
        "final_train_loss": final_train_loss,
        "max_grad_norm": max_grad_norm,
        "mean_abs_weight": float(weight.detach().abs().mean()),
        "nodes": circuit.graph.num_nodes,
        "edges": circuit.graph.num_edges,
        "finite": bool(torch.isfinite(weight).all()) and math.isfinite(mse_after),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact same-PC gradient trained with SGD")
    parser.add_argument("--extent", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--bar-width", type=float, default=0.5)
    parser.add_argument("--frame-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.015)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--train-repeats", type=int, default=4)
    parser.add_argument("--test-repeats", type=int, default=8)
    parser.add_argument(
        "--out", type=Path, default=Path("results/generated/flyvis_exact_pc_sgd.csv")
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
                    learning_rate=args.learning_rate,
                    momentum=args.momentum,
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
        f"Exact PC SGD: extent={args.extent}, {base.graph.num_nodes} nodes, "
        f"{base.graph.num_edges} edges, lr={args.learning_rate:g}, momentum={args.momentum:g}"
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
