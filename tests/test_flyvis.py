import torch

from predcircuit.flyvis import graph_from_flyvis_spec, with_sensory_distance_rank


def tiny_spec() -> dict[str, object]:
    return {
        "nodes": [{"name": "R"}, {"name": "M"}, {"name": "T"}],
        "input_units": ["R"],
        "output_units": ["T"],
        "edges": [
            {"src": "R", "tar": "M", "alpha": 1, "offsets": [[[0, 0], 2.0], [[1, 0], 3.0]]},
            {"src": "M", "tar": "T", "alpha": -1, "offsets": [[[0, 0], 4.0]]},
        ],
    }


def test_flyvis_type_graph_aggregates_offsets_and_sign() -> None:
    circuit = graph_from_flyvis_spec(tiny_spec())
    assert circuit.graph.node_ids == ("R", "M", "T")
    assert circuit.input_nodes == (0,)
    assert circuit.output_nodes == (2,)
    assert torch.equal(circuit.graph.edge_index, torch.tensor([[0, 1], [1, 2]]))
    assert circuit.graph.edge_weight is not None
    assert torch.allclose(circuit.graph.edge_weight, torch.tensor([5.0, -4.0]))


def test_sensory_distance_rank() -> None:
    circuit = with_sensory_distance_rank(graph_from_flyvis_spec(tiny_spec()))
    assert circuit.graph.node_rank is not None
    assert torch.equal(circuit.graph.node_rank, torch.tensor([0.0, 1.0, 2.0]))
