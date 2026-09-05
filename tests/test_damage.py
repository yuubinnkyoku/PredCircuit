import torch

from predcircuit.damage import edge_lesion, node_lesion, weight_noise
from predcircuit.model import PredictiveCodingGraph
from predcircuit.topology import layered_graph


def test_damage_helpers_do_not_modify_original() -> None:
    graph = layered_graph([2, 4, 1], seed=0)
    model = PredictiveCodingGraph(graph, seed=0)
    original = model.weight.clone()
    lesioned = edge_lesion(model, 0.5, seed=1)
    assert torch.equal(model.weight, original)
    assert int((lesioned.weight == 0).sum()) > 0

    node_damaged = node_lesion(model, 0.5, protected_nodes=[0, 1, 6], seed=2)
    assert torch.equal(model.weight, original)
    assert int((node_damaged.weight == 0).sum()) > 0

    noisy = weight_noise(model, 0.2, seed=3)
    assert torch.equal(model.weight, original)
    assert not torch.equal(noisy.weight, original)
