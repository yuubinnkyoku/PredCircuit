from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
import torch

from .metrics import graph_metrics
from .model import PredictiveCodingGraph
from .tasks import linear_mapping_task
from .topology import CircuitGraph


@dataclass
class RunResult:
    topology: str
    seed: int
    epochs: int
    mse_before: float
    mse_after: float
    improvement: float
    final_energy: float
    mean_abs_weight: float
    nodes: int
    edges: int
    density: float
    reciprocal_fraction: float


def train_linear_mapping(
    graph: CircuitGraph,
    *,
    topology_name: str,
    seed: int,
    epochs: int = 800,
    inference_steps: int = 25,
    inference_lr: float = 0.05,
    weight_lr: float = 1e-2,
) -> RunResult:
    """Run a tiny regression sanity check.

    The task is intentionally simple. It verifies that inference and local weight updates can
    lower a supervised prediction error; it is not a biological benchmark.
    """
    x, y = linear_mapping_task()
    model = PredictiveCodingGraph(graph, seed=seed)
    input_nodes = [0, 1]
    output_nodes = [graph.num_nodes - 1]
    pred0 = model.predict(
        x,
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        inference_steps=80,
        inference_lr=inference_lr,
    )
    mse_before = float(torch.mean((pred0 - y).square()))

    final = {"energy": float("nan")}
    for _ in range(epochs):
        final = model.train_batch(
            x,
            y,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            inference_steps=inference_steps,
            inference_lr=inference_lr,
            weight_lr=weight_lr,
        )
    pred = model.predict(
        x,
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        inference_steps=100,
        inference_lr=inference_lr,
    )
    mse_after = float(torch.mean((pred - y).square()))
    gm = graph_metrics(graph)
    return RunResult(
        topology=topology_name,
        seed=seed,
        epochs=epochs,
        mse_before=mse_before,
        mse_after=mse_after,
        improvement=mse_before - mse_after,
        final_energy=float(final["energy"]),
        mean_abs_weight=float(model.weight.abs().mean()),
        nodes=gm.nodes,
        edges=gm.edges,
        density=gm.density,
        reciprocal_fraction=gm.reciprocal_fraction,
    )


def results_frame(results: list[RunResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in results])
