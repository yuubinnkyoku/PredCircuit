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


def test_soft_nudge_moves_output_toward_target_without_unclamping_input() -> None:
    graph = CircuitGraph(num_nodes=2, edge_index=torch.tensor([[0], [1]]))
    model = PredictiveCodingGraph(graph, seed=2)
    model.weight.fill_(0.5)
    init = torch.zeros(1, 2)
    mask = torch.tensor([True, False])
    values = torch.tensor([[0.8, 0.0]])
    target = torch.tensor([[0.6]])
    free, _ = model.infer(
        init,
        clamp_mask=mask,
        clamp_values=values,
        steps=20,
        step_size=0.05,
    )
    nudged = model.infer_nudged(
        free,
        clamp_mask=mask,
        clamp_values=values,
        nudged_nodes=[1],
        nudged_values=target,
        beta=0.5,
        steps=20,
        step_size=0.05,
    )
    assert nudged[0, 1] > free[0, 1]
    assert nudged[0, 0] == values[0, 0]


def test_contrastive_weight_update_matches_local_statistic_difference() -> None:
    graph = CircuitGraph(num_nodes=2, edge_index=torch.tensor([[0], [1]]))
    model = PredictiveCodingGraph(graph, seed=3)
    model.weight.zero_()
    model.bias.zero_()
    free = torch.tensor([[0.4, 0.1], [0.2, -0.1]])
    nudged = torch.tensor([[0.5, 0.3], [0.1, -0.2]])
    beta = 0.25
    learning_rate = 0.2
    expected = (learning_rate / beta) * (
        model.local_edge_statistics(nudged) - model.local_edge_statistics(free)
    )
    delta = model.contrastive_weight_step(
        free,
        nudged,
        beta=beta,
        learning_rate=learning_rate,
        clip=None,
    )
    assert torch.allclose(delta, expected)
