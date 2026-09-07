import torch

from predcircuit.flyvis_retinotopy import (
    axial_hex_radius,
    graph_from_flyvis_retinotopy,
    rotate_axial,
)


def tiny_retinotopic_spec() -> dict[str, object]:
    return {
        "nodes": [
            {"name": "R", "pattern": ["stride", [1, 1]]},
            {"name": "M", "pattern": ["stride", [1, 1]]},
            {"name": "T", "pattern": ["single", None]},
        ],
        "input_units": ["R"],
        "output_units": ["T"],
        "edges": [
            {"src": "R", "tar": "M", "alpha": 1, "offsets": [[[0, 0], 2.0]]},
            {"src": "M", "tar": "T", "alpha": -1, "offsets": [[[0, 0], 3.0]]},
        ],
    }


def test_hex_rotation_preserves_radius() -> None:
    points = [(1, 0), (1, -2), (-2, 1), (0, 0)]
    for point in points:
        radii = {axial_hex_radius(*rotate_axial(point[0], point[1], turns)) for turns in range(6)}
        assert radii == {axial_hex_radius(*point)}
        assert rotate_axial(point[0], point[1], 6) == point


def test_retinotopic_compiler_expands_hex_columns() -> None:
    circuit = graph_from_flyvis_retinotopy(tiny_retinotopic_spec(), extent=1)
    assert circuit.graph.num_nodes == 15
    assert circuit.graph.num_edges == 8
    assert len(circuit.input_nodes) == 7
    assert len(circuit.output_nodes) == 1
    assert circuit.central_node("R") in circuit.input_nodes
    assert circuit.central_node("T") in circuit.output_nodes
    assert circuit.graph.edge_weight is not None
    assert torch.count_nonzero(circuit.graph.edge_weight == 2.0) == 7
    assert torch.count_nonzero(circuit.graph.edge_weight == -3.0) == 1


def test_pair_rotation_scramble_keeps_hex_radius_and_edge_count_on_symmetric_crop() -> None:
    spec = tiny_retinotopic_spec()
    edges = spec["edges"]
    assert isinstance(edges, list)
    first = edges[0]
    assert isinstance(first, dict)
    first["offsets"] = [[[1, 0], 2.0]]

    biological = graph_from_flyvis_retinotopy(spec, extent=2)
    scrambled = graph_from_flyvis_retinotopy(spec, extent=2, pair_rotation_seed=7)
    assert biological.graph.num_nodes == scrambled.graph.num_nodes
    assert biological.graph.num_edges == scrambled.graph.num_edges
    assert biological.graph.edge_weight is not None
    assert scrambled.graph.edge_weight is not None
    assert torch.allclose(
        biological.graph.edge_weight.abs().sort().values,
        scrambled.graph.edge_weight.abs().sort().values,
    )
