from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
import torch

from .baselines import BPTTGraphNetwork
from .metrics import graph_metrics
from .model import PredictiveCodingGraph
from .tasks import linear_mapping_task
from .topology import CircuitGraph


@dataclass
class RunResult:
    learning_rule: str
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
        learning_rule="predictive_coding",
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


def train_bptt_linear_mapping(
    graph: CircuitGraph,
    *,
    topology_name: str,
    seed: int,
    epochs: int = 800,
    learning_rate: float = 1e-2,
    unroll_steps: int = 20,
) -> RunResult:
    """Train the same graph topology with autograd/BPTT as a control."""
    x, y = linear_mapping_task()
    model = BPTTGraphNetwork(graph, seed=seed)
    input_nodes = [0, 1]
    output_nodes = [graph.num_nodes - 1]
    with torch.no_grad():
        pred0 = model(x, input_nodes=input_nodes, output_nodes=output_nodes, steps=unroll_steps)
        mse_before = float(torch.mean((pred0 - y).square()))
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    final_loss = float("nan")
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        pred = model(x, input_nodes=input_nodes, output_nodes=output_nodes, steps=unroll_steps)
        loss = torch.mean((pred - y).square())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        final_loss = float(loss.detach())
    with torch.no_grad():
        pred = model(x, input_nodes=input_nodes, output_nodes=output_nodes, steps=unroll_steps)
        mse_after = float(torch.mean((pred - y).square()))
    gm = graph_metrics(graph)
    return RunResult(
        learning_rule="bptt",
        topology=topology_name,
        seed=seed,
        epochs=epochs,
        mse_before=mse_before,
        mse_after=mse_after,
        improvement=mse_before - mse_after,
        final_energy=final_loss,
        mean_abs_weight=float(model.weight.detach().abs().mean()),
        nodes=gm.nodes,
        edges=gm.edges,
        density=gm.density,
        reciprocal_fraction=gm.reciprocal_fraction,
    )


def results_frame(results: list[RunResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in results])
