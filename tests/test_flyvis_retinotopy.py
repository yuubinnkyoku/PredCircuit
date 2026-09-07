from collections import Counter

import torch

from predcircuit.flyvis_retinotopy import (
    axial_hex_radius,
    graph_from_flyvis_retinotopy,
    rotate_axial,
    type_pair_preserving_rewire,
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


def test_type_pair_rewire_preserves_degrees_edge_count_and_type_pairs() -> None:
    circuit = graph_from_flyvis_retinotopy(tiny_retinotopic_spec(), extent=2)
    rewired = type_pair_preserving_rewire(circuit, swaps=25, seed=3)
    before_in, before_out = circuit.graph.degrees()
    after_in, after_out = rewired.graph.degrees()
    assert torch.equal(before_in, after_in)
    assert torch.equal(before_out, after_out)
    assert circuit.graph.num_edges == rewired.graph.num_edges
    assert not torch.equal(circuit.graph.edge_index, rewired.graph.edge_index)
    assert circuit.graph.edge_weight is not None
    assert rewired.graph.edge_weight is not None
    assert torch.equal(circuit.graph.edge_weight, rewired.graph.edge_weight)

    def type_pairs(edge_index: torch.Tensor) -> Counter[tuple[str, str]]:
        return Counter(
            (circuit.node_types[int(source)], circuit.node_types[int(target)])
            for source, target in edge_index.t().tolist()
        )

    assert type_pairs(circuit.graph.edge_index) == type_pairs(rewired.graph.edge_index)
