from __future__ import annotations

import torch

from .topology import CircuitGraph


class BPTTGraphNetwork(torch.nn.Module):
    """A recurrent graph baseline trained by backpropagation through time.

    It uses the same directed edge set as a PredCircuit model, allowing topology-controlled
    comparisons between local predictive-coding learning and autograd/BPTT.
    """

    def __init__(self, graph: CircuitGraph, *, seed: int = 0, init_scale: float = 0.15) -> None:
        super().__init__()
        self.graph = graph
        gen = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(torch.randn(graph.num_edges, generator=gen) * init_scale)

    def forward(
        self,
        input_values: torch.Tensor,
        *,
        input_nodes: list[int],
        output_nodes: list[int],
        steps: int = 20,
        leak: float = 0.35,
    ) -> torch.Tensor:
        state = torch.zeros(
            input_values.shape[0],
            self.graph.num_nodes,
            device=input_values.device,
            dtype=input_values.dtype,
        )
        src, dst = self.graph.edge_index.to(input_values.device)
        for _ in range(steps):
            state = state.clone()
            state[:, input_nodes] = input_values
            drive = torch.zeros_like(state)
            drive.index_add_(1, dst, torch.tanh(state[:, src]) * self.weight)
            updated = torch.tanh(drive)
            state = leak * state + (1.0 - leak) * updated
            state = state.clone()
            state[:, input_nodes] = input_values
        return state[:, output_nodes]
