#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from predcircuit.damage import edge_lesion, node_lesion, weight_noise
from predcircuit.model import PredictiveCodingGraph
from predcircuit.tasks import linear_mapping_task
from predcircuit.topology import layered_graph


def train(seed: int, epochs: int) -> tuple[PredictiveCodingGraph, torch.Tensor, torch.Tensor]:
    graph = layered_graph([2, 8, 1], recurrent_probability=0.2, feedback_probability=0.08, seed=seed)
    model = PredictiveCodingGraph(graph, seed=seed)
    x, y = linear_mapping_task()
    for _ in range(epochs):
        model.train_batch(
            x,
            y,
            input_nodes=[0, 1],
            output_nodes=[graph.num_nodes - 1],
            inference_steps=25,
            inference_lr=0.05,
            weight_lr=1e-2,
        )
    return model, x, y


def mse(model: PredictiveCodingGraph, x: torch.Tensor, y: torch.Tensor) -> float:
    pred = model.predict(
        x,
        input_nodes=[0, 1],
        output_nodes=[model.graph.num_nodes - 1],
        inference_steps=100,
        inference_lr=0.05,
    )
    return float(torch.mean((pred - y).square()))


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a trained PC graph under structural damage")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--out", type=Path, default=Path("results/generated/robustness.csv"))
    args = p.parse_args()

    fractions = [0.0, 0.1, 0.2, 0.3, 0.5]
    rows: list[dict[str, float | int | str]] = []
    for seed in range(args.seeds):
        model, x, y = train(seed, args.epochs)
        for frac in fractions:
            for kind, damaged in (
                ("edge_lesion", edge_lesion(model, frac, seed=1000 + seed)),
                (
                    "node_lesion",
                    node_lesion(
                        model,
                        frac,
                        protected_nodes=[0, 1, model.graph.num_nodes - 1],
                        seed=2000 + seed,
                    ),
                ),
                ("weight_noise", weight_noise(model, frac, seed=3000 + seed)),
            ):
                rows.append(
                    {"seed": seed, "damage": kind, "fraction": frac, "mse": mse(damaged, x, y)}
                )

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.groupby(["damage", "fraction"])["mse"].agg(["mean", "std"]).to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
