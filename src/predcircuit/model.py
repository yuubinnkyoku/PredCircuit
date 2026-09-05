from __future__ import annotations

from dataclasses import dataclass

import torch

from .topology import CircuitGraph


@dataclass
class InferenceTrace:
    energy: list[float]


class PredictiveCodingGraph(torch.nn.Module):
    """Predictive-coding network on an arbitrary directed graph.

    Each node holds a scalar activity x_i. An edge j->i predicts node i with
    w_ji * tanh(x_j). Inference minimizes the local prediction-error energy

        E = 1/2 sum_i (x_i - sum_j w_ji tanh(x_j))^2.

    Synaptic learning is local:

        Delta w_ji propto epsilon_i * tanh(x_j)

    where epsilon_i is the postsynaptic prediction error. No autograd graph is used for
    the predictive-coding weight update.
    """

    def __init__(
        self,
        graph: CircuitGraph,
        *,
        init_scale: float = 0.15,
        use_biological_strength: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.graph = graph
        gen = torch.Generator().manual_seed(seed)
        if use_biological_strength and graph.edge_weight is not None:
            base = graph.edge_weight.float()
            base = base / base.abs().mean().clamp_min(1e-8)
            init = init_scale * base
        else:
            init = torch.randn(graph.num_edges, generator=gen) * init_scale
        self.weight = torch.nn.Parameter(init, requires_grad=False)
        self.bias = torch.nn.Parameter(torch.zeros(graph.num_nodes), requires_grad=False)

    @staticmethod
    def activation(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    @staticmethod
    def activation_prime(x: torch.Tensor) -> torch.Tensor:
        y = torch.tanh(x)
        return 1.0 - y.square()

    def predict_nodes(self, state: torch.Tensor) -> torch.Tensor:
        src, dst = self.graph.edge_index.to(state.device)
        contribution = self.activation(state[:, src]) * self.weight.to(state.device)
        pred = torch.zeros_like(state)
        pred.index_add_(1, dst, contribution)
        pred += self.bias.to(state.device)
        return pred

    def errors(self, state: torch.Tensor) -> torch.Tensor:
        return state - self.predict_nodes(state)

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.errors(state).square().sum(dim=1).mean()

    @torch.no_grad()
    def infer(
        self,
        initial_state: torch.Tensor,
        *,
        clamp_mask: torch.Tensor,
        clamp_values: torch.Tensor,
        steps: int = 40,
        step_size: float = 0.08,
        record_trace: bool = False,
    ) -> tuple[torch.Tensor, InferenceTrace | None]:
        state = initial_state.clone().float()
        clamp_mask = clamp_mask.to(device=state.device, dtype=torch.bool)
        clamp_values = clamp_values.to(state.device, dtype=state.dtype)
        src, dst = self.graph.edge_index.to(state.device)
        weight = self.weight.to(state.device)
        trace: list[float] = []

        state[:, clamp_mask] = clamp_values[:, clamp_mask]
        for _ in range(steps):
            eps = self.errors(state)
            downstream = torch.zeros_like(state)
            # dE/dx_i = eps_i - tanh'(x_i) * sum_{i->k} w_ik eps_k
            downstream.index_add_(1, src, eps[:, dst] * weight)
            grad = eps - self.activation_prime(state) * downstream
            grad[:, clamp_mask] = 0.0
            state -= step_size * grad
            state[:, clamp_mask] = clamp_values[:, clamp_mask]
            if record_trace:
                trace.append(float(self.energy(state)))
        return state, InferenceTrace(trace) if record_trace else None

    @torch.no_grad()
    def local_weight_step(
        self,
        state: torch.Tensor,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        clip: float | None = 5.0,
    ) -> torch.Tensor:
        """Apply one local Hebbian/prediction-error update and return delta weights."""
        src, dst = self.graph.edge_index.to(state.device)
        eps = self.errors(state)
        pre = self.activation(state[:, src])
        delta = learning_rate * (eps[:, dst] * pre).mean(dim=0)
        if weight_decay:
            delta -= learning_rate * weight_decay * self.weight
        if clip is not None:
            delta = delta.clamp(-clip, clip)
        self.weight.add_(delta.to(self.weight.device))
        self.bias.add_(learning_rate * eps.mean(dim=0).to(self.bias.device))
        return delta.detach().cpu()

    @torch.no_grad()
    def train_batch(
        self,
        input_values: torch.Tensor,
        target_values: torch.Tensor,
        *,
        input_nodes: list[int],
        output_nodes: list[int],
        inference_steps: int = 40,
        inference_lr: float = 0.08,
        weight_lr: float = 3e-3,
    ) -> dict[str, float]:
        batch = input_values.shape[0]
        state = torch.zeros(batch, self.graph.num_nodes, dtype=torch.float32)
        clamp_mask = torch.zeros(self.graph.num_nodes, dtype=torch.bool)
        clamp_mask[input_nodes] = True
        clamp_mask[output_nodes] = True
        clamp_values = torch.zeros_like(state)
        clamp_values[:, input_nodes] = input_values
        clamp_values[:, output_nodes] = target_values
        state, trace = self.infer(
            state,
            clamp_mask=clamp_mask,
            clamp_values=clamp_values,
            steps=inference_steps,
            step_size=inference_lr,
            record_trace=True,
        )
        delta = self.local_weight_step(state, learning_rate=weight_lr)
        return {
            "energy": float(self.energy(state)),
            "mean_abs_update": float(delta.abs().mean()),
            "final_inference_energy": trace.energy[-1] if trace and trace.energy else float("nan"),
        }

    @torch.no_grad()
    def predict(
        self,
        input_values: torch.Tensor,
        *,
        input_nodes: list[int],
        output_nodes: list[int],
        inference_steps: int = 80,
        inference_lr: float = 0.08,
    ) -> torch.Tensor:
        batch = input_values.shape[0]
        state = torch.zeros(batch, self.graph.num_nodes, dtype=torch.float32)
        clamp_mask = torch.zeros(self.graph.num_nodes, dtype=torch.bool)
        clamp_mask[input_nodes] = True
        clamp_values = torch.zeros_like(state)
        clamp_values[:, input_nodes] = input_values
        state, _ = self.infer(
            state,
            clamp_mask=clamp_mask,
            clamp_values=clamp_values,
            steps=inference_steps,
            step_size=inference_lr,
        )
        return state[:, output_nodes]
