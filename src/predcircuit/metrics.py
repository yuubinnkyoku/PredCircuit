from __future__ import annotations

from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np
import torch

from .topology import CircuitGraph


@dataclass
class GraphMetrics:
    nodes: int
    edges: int
    density: float
    reciprocal_fraction: float
    strongly_connected_components: int
    largest_scc_fraction: float
    mean_in_degree: float
    mean_out_degree: float


def graph_metrics(graph: CircuitGraph) -> GraphMetrics:
    g = graph.to_networkx()
    in_deg, out_deg = graph.degrees()
    reciprocal_fraction = float(graph.reciprocal_mask().float().mean()) if graph.num_edges else 0.0
    scc = list(nx.strongly_connected_components(g))
    largest = max((len(c) for c in scc), default=0)
    return GraphMetrics(
        nodes=graph.num_nodes,
        edges=graph.num_edges,
        density=graph.density,
        reciprocal_fraction=reciprocal_fraction,
        strongly_connected_components=len(scc),
        largest_scc_fraction=largest / max(graph.num_nodes, 1),
        mean_in_degree=float(in_deg.float().mean()),
        mean_out_degree=float(out_deg.float().mean()),
    )


def graph_metrics_dict(graph: CircuitGraph) -> dict[str, float | int]:
    return asdict(graph_metrics(graph))


def binary_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(((pred >= 0) == (target >= 0)).float().mean())


def perturb_edges(graph: CircuitGraph, drop_probability: float, seed: int = 0) -> CircuitGraph:
    rng = np.random.default_rng(seed)
    keep = torch.tensor(rng.random(graph.num_edges) >= drop_probability, dtype=torch.bool)
    return graph._subset_edges(keep)
