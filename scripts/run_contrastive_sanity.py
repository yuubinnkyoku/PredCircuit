from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from predcircuit.model import PredictiveCodingGraph
from predcircuit.tasks import linear_mapping_task
from predcircuit.topology import layered_graph


@torch.no_grad()
def evaluate(model: PredictiveCodingGraph) -> float:
    x, y = linear_mapping_task()
    pred = model.predict(
        x,
        input_nodes=[0, 1],
        output_nodes=[model.graph.num_nodes - 1],
        inference_steps=120,
        inference_lr=0.05,
    )
    return float((pred - y).square().mean())


@torch.no_grad()
def run_one(*, seed: int, epochs: int, beta: float, weight_lr: float) -> dict[str, float | int | bool]:
    graph = layered_graph(
        [2, 8, 1], recurrent_probability=0.2, feedback_probability=0.08, seed=seed
    )
    model = PredictiveCodingGraph(graph, seed=seed)
    x, y = linear_mapping_task()
    input_nodes = [0, 1]
    output_nodes = [graph.num_nodes - 1]
    clamp_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
    clamp_mask[input_nodes] = True
    clamp_values = torch.zeros(x.shape[0], graph.num_nodes, dtype=torch.float32)
    clamp_values[:, input_nodes] = x

    mse_before = evaluate(model)
    mean_update_sum = 0.0
    max_update = 0.0
    for _ in range(epochs):
        initial = torch.zeros_like(clamp_values)
        free_state, _ = model.infer(
            initial,
            clamp_mask=clamp_mask,
            clamp_values=clamp_values,
            steps=120,
            step_size=0.05,
        )
        nudged_state = model.infer_nudged(
            free_state,
            clamp_mask=clamp_mask,
            clamp_values=clamp_values,
            nudged_nodes=output_nodes,
            nudged_values=y,
            beta=beta,
            steps=120,
            step_size=0.05,
        )
        delta = model.contrastive_weight_step(
            free_state,
            nudged_state,
            beta=beta,
            learning_rate=weight_lr,
            weight_decay=1e-5,
            clip=0.05,
        )
        mean_update_sum += float(delta.abs().mean())
        max_update = max(max_update, float(delta.abs().max()))

    mse_after = evaluate(model)
    return {
        "seed": seed,
        "epochs": epochs,
        "beta": beta,
        "weight_lr": weight_lr,
        "mse_before": mse_before,
        "mse_after": mse_after,
        "improvement": mse_before - mse_after,
        "mean_abs_update": mean_update_sum / epochs,
        "max_abs_update": max_update,
        "finite": bool(torch.isfinite(model.weight).all()) and math.isfinite(mse_after),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Equilibrium contrastive-PC sanity check")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--weight-lr", type=float, default=3e-3)
    parser.add_argument(
        "--out", type=Path, default=Path("results/generated/contrastive_sanity.csv")
    )
    args = parser.parse_args()

    rows = [
        run_one(seed=seed, epochs=args.epochs, beta=args.beta, weight_lr=args.weight_lr)
        for seed in range(args.seeds)
    ]
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(
        frame[["mse_before", "mse_after", "improvement", "mean_abs_update"]]
        .agg(["mean", "median", "std"])
        .to_string()
    )
    print("\nPer-seed results:")
    print(frame.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
