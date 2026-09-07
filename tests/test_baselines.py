import torch

from predcircuit.baselines import BPTTGraphNetwork
from predcircuit.topology import CircuitGraph


def test_bptt_sequence_has_expected_shape_and_gradients() -> None:
    graph = CircuitGraph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 1]], dtype=torch.long),
    )
    model = BPTTGraphNetwork(graph, seed=0)
    sequence = torch.tensor(
        [
            [[0.0], [1.0], [0.5]],
            [[1.0], [0.0], [0.25]],
        ],
        dtype=torch.float32,
    )
    output = model.forward_sequence(
        sequence,
        input_nodes=[0],
        output_nodes=[2],
        steps_per_frame=2,
    )
    assert output.shape == (2, 1)
    output.square().mean().backward()
    assert model.weight.grad is not None
    assert torch.isfinite(model.weight.grad).all()
