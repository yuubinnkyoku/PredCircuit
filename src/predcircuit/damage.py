from __future__ import annotations

import copy

import numpy as np
import torch

from .model import PredictiveCodingGraph


@torch.no_grad()
def edge_lesion(
    model: PredictiveCodingGraph, fraction: float, seed: int = 0
) -> PredictiveCodingGraph:
    """Return a copy with a random fraction of synaptic weights set to zero."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    damaged = copy.deepcopy(model)
    rng = np.random.default_rng(seed)
    count = int(round(fraction * damaged.graph.num_edges))
    if count:
        idx = torch.as_tensor(rng.choice(damaged.graph.num_edges, size=count, replace=False))
        damaged.weight[idx] = 0.0
    return damaged


@torch.no_grad()
def node_lesion(
    model: PredictiveCodingGraph,
    fraction: float,
    *,
    protected_nodes: list[int] | None = None,
    seed: int = 0,
) -> PredictiveCodingGraph:
    """Return a copy with randomly selected nodes functionally silenced.

    Incoming and outgoing weights plus the node bias are zeroed. Protected nodes are never
    selected, which is useful for keeping task input/output nodes intact.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    protected = set(protected_nodes or [])
    candidates = [i for i in range(model.graph.num_nodes) if i not in protected]
    count = min(len(candidates), int(round(fraction * len(candidates))))
    damaged = copy.deepcopy(model)
    if not count:
        return damaged
    rng = np.random.default_rng(seed)
    nodes = torch.as_tensor(rng.choice(candidates, size=count, replace=False), dtype=torch.long)
    src, dst = damaged.graph.edge_index
    incident = torch.isin(src, nodes) | torch.isin(dst, nodes)
    damaged.weight[incident] = 0.0
    damaged.bias[nodes] = 0.0
    return damaged


@torch.no_grad()
def weight_noise(
    model: PredictiveCodingGraph, relative_std: float, seed: int = 0
) -> PredictiveCodingGraph:
    """Return a copy with additive Gaussian weight noise scaled by weight RMS."""
    if relative_std < 0:
        raise ValueError("relative_std must be non-negative")
    damaged = copy.deepcopy(model)
    gen = torch.Generator(device=damaged.weight.device).manual_seed(seed)
    scale = damaged.weight.square().mean().sqrt().clamp_min(1e-12)
    noise = torch.randn(
        damaged.weight.shape,
        generator=gen,
        device=damaged.weight.device,
        dtype=damaged.weight.dtype,
    )
    damaged.weight.add_(noise * (relative_std * scale))
    return damaged
