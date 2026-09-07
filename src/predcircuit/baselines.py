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

    def _step(
        self,
        state: torch.Tensor,
        input_values: torch.Tensor,
        *,
        input_nodes: list[int],
        leak: float,
    ) -> torch.Tensor:
        state = state.clone()
        state[:, input_nodes] = input_values
        src, dst = self.graph.edge_index.to(input_values.device)
        drive = torch.zeros_like(state)
        drive.index_add_(1, dst, torch.tanh(state[:, src]) * self.weight)
        updated = torch.tanh(drive)
        state = leak * state + (1.0 - leak) * updated
        state = state.clone()
        state[:, input_nodes] = input_values
        return state

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
        for _ in range(steps):
            state = self._step(state, input_values, input_nodes=input_nodes, leak=leak)
        return state[:, output_nodes]

    def forward_sequence(
        self,
        input_sequence: torch.Tensor,
        *,
        input_nodes: list[int],
        output_nodes: list[int],
        steps_per_frame: int = 2,
        leak: float = 0.35,
    ) -> torch.Tensor:
        """Unroll a time-varying input sequence while retaining graph state between frames.

        ``input_sequence`` has shape ``[batch, frames, input_nodes]``. Gradients flow through
        every frame and every recurrent step, making this the global-credit-assignment control
        for the retinotopic motion task.
        """
        if input_sequence.ndim != 3:
            raise ValueError("input_sequence must have shape [batch, frames, input_nodes]")
        if input_sequence.shape[2] != len(input_nodes):
            raise ValueError("last input_sequence dimension must match input_nodes")
        if steps_per_frame <= 0:
            raise ValueError("steps_per_frame must be positive")

        state = torch.zeros(
            input_sequence.shape[0],
            self.graph.num_nodes,
            device=input_sequence.device,
            dtype=input_sequence.dtype,
        )
        for frame in input_sequence.unbind(dim=1):
            for _ in range(steps_per_frame):
                state = self._step(state, frame, input_nodes=input_nodes, leak=leak)
        return state[:, output_nodes]
