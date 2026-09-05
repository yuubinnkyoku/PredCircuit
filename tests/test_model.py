import torch

from predcircuit.model import PredictiveCodingGraph
from predcircuit.topology import CircuitGraph


def test_local_weight_update_matches_formula() -> None:
    graph = CircuitGraph(num_nodes=2, edge_index=torch.tensor([[0], [1]]))
    model = PredictiveCodingGraph(graph, seed=0)
    model.weight.zero_()
    state = torch.tensor([[0.5, 0.25]])
    eps = model.errors(state)
    expected = 0.1 * eps[:, 1] * torch.tanh(state[:, 0])
    delta = model.local_weight_step(state, learning_rate=0.1)
    assert torch.allclose(delta, expected)


def test_inference_respects_clamps() -> None:
    graph = CircuitGraph(num_nodes=2, edge_index=torch.tensor([[0], [1]]))
    model = PredictiveCodingGraph(graph, seed=1)
    init = torch.zeros(3, 2)
    mask = torch.tensor([True, False])
    values = torch.tensor([[0.7, 0.0]]).repeat(3, 1)
    state, _ = model.infer(init, clamp_mask=mask, clamp_values=values, steps=10)
    assert torch.all(state[:, 0] == 0.7)
