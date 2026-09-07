from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch

from .topology import CircuitGraph


@dataclass(frozen=True)
class RetinotopicFlyVisCircuit:
    """Finite hexagonal crop of the FlyVis connectome with spatial offsets intact."""

    graph: CircuitGraph
    input_nodes: tuple[int, ...]
    output_nodes: tuple[int, ...]
    node_types: tuple[str, ...]
    node_u: torch.Tensor
    node_v: torch.Tensor

    def nodes_of_type(self, cell_type: str) -> tuple[int, ...]:
        return tuple(i for i, typ in enumerate(self.node_types) if typ == cell_type)

    def central_node(self, cell_type: str) -> int:
        matches = [
            i
            for i, typ in enumerate(self.node_types)
            if typ == cell_type and int(self.node_u[i]) == 0 and int(self.node_v[i]) == 0
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one central {cell_type!r} node, found {len(matches)}")
        return matches[0]


def axial_hex_radius(u: int, v: int) -> int:
    """Return hex distance from the origin for axial/oblique coordinates."""
    return max(abs(u), abs(v), abs(u + v))


def rotate_axial(u: int, v: int, turns: int) -> tuple[int, int]:
    """Rotate an axial offset counter-clockwise by ``turns * 60`` degrees."""
    turns %= 6
    for _ in range(turns):
        u, v = -v, u + v
    return u, v


def _node_coordinates(node: dict[str, Any], extent: int) -> list[tuple[int, int]]:
    pattern_raw = node.get("pattern", ["stride", [1, 1]])
    if not isinstance(pattern_raw, list) or len(pattern_raw) != 2:
        raise TypeError("FlyVis node pattern must be [kind, args]")
    pattern, args = pattern_raw
    if pattern == "single":
        return [(0, 0)]
    if pattern == "tile":
        stride = int(args)
        u_stride = v_stride = stride
    elif pattern == "stride":
        if not isinstance(args, list) or len(args) != 2:
            raise TypeError("stride pattern args must be [u_stride, v_stride]")
        u_stride, v_stride = int(args[0]), int(args[1])
    else:
        raise ValueError(f"unsupported FlyVis node pattern: {pattern!r}")
    if u_stride <= 0 or v_stride <= 0:
        raise ValueError("node strides must be positive")

    coordinates: list[tuple[int, int]] = []
    for u in range(-extent, extent + 1):
        for v in range(max(-extent, -extent - u), min(extent, extent - u) + 1):
            if u % u_stride == 0 and v % v_stride == 0:
                coordinates.append((u, v))
    return coordinates


def graph_from_flyvis_retinotopy(
    spec: dict[str, Any],
    *,
    extent: int = 2,
    signed: bool = True,
    pair_rotation_seed: int | None = None,
) -> RetinotopicFlyVisCircuit:
    """Compile the published FlyVis convolutional scaffold into explicit neurons/edges.

    ``extent`` is the radius of the finite hexagonal crop. The compiler follows FlyVis'
    stride/tile/single node patterns and reported source-to-target offsets, but deliberately
    does not fill missing offsets inside convex hulls. This keeps the pilot tied to reported
    offsets and avoids importing the full FlyVis runtime.

    If ``pair_rotation_seed`` is set, every cell-type edge specification receives an
    independent random multiple-of-60-degree rotation of all its spatial offsets. This null
    preserves the internal receptive-field geometry and synapse values of each type pair while
    scrambling the orientation alignment between different type pairs. On finite crops it may
    change a small number of boundary edges, so it is a secondary rather than degree-matched
    null.
    """
    if extent < 0:
        raise ValueError("extent must be non-negative")
    nodes_raw = spec.get("nodes")
    edges_raw = spec.get("edges")
    if not isinstance(nodes_raw, list) or not isinstance(edges_raw, list):
        raise TypeError("FlyVis spec must contain list-valued 'nodes' and 'edges'")

    node_types: list[str] = []
    node_u: list[int] = []
    node_v: list[int] = []
    lookup: dict[tuple[str, int, int], int] = {}
    nodes_by_type: dict[str, list[tuple[int, int, int]]] = {}
    for node in nodes_raw:
        cell_type = str(node["name"])
        for u, v in _node_coordinates(node, extent):
            key = (cell_type, u, v)
            if key in lookup:
                raise ValueError(f"duplicate FlyVis node: {key!r}")
            index = len(node_types)
            lookup[key] = index
            nodes_by_type.setdefault(cell_type, []).append((index, u, v))
            node_types.append(cell_type)
            node_u.append(u)
            node_v.append(v)

    rng = random.Random(pair_rotation_seed)
    aggregated: dict[tuple[int, int], float] = {}
    for edge in edges_raw:
        source_type = str(edge["src"])
        target_type = str(edge["tar"])
        offsets = edge.get("offsets", [])
        if not isinstance(offsets, list):
            raise TypeError("FlyVis edge offsets must be a list")
        turns = rng.randrange(6) if pair_rotation_seed is not None else 0
        sign = float(edge.get("alpha", 1.0)) if signed else 1.0
        source_nodes = nodes_by_type.get(source_type, [])
        for offset in offsets:
            if not isinstance(offset, list) or len(offset) != 2:
                raise TypeError("FlyVis offset entries must be [[du, dv], n_syn]")
            delta, n_syn = offset
            if not isinstance(delta, list) or len(delta) != 2:
                raise TypeError("FlyVis spatial offset must be [du, dv]")
            du, dv = rotate_axial(int(delta[0]), int(delta[1]), turns)
            weight = sign * float(n_syn)
            if weight == 0.0:
                continue
            for source_idx, u, v in source_nodes:
                target_idx = lookup.get((target_type, u + du, v + dv))
                if target_idx is None:
                    continue
                key = (source_idx, target_idx)
                aggregated[key] = aggregated.get(key, 0.0) + weight

    pairs = sorted((edge, weight) for edge, weight in aggregated.items() if weight != 0.0)
    if pairs:
        edge_index = torch.tensor([edge for edge, _ in pairs], dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor([weight for _, weight in pairs], dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty(0, dtype=torch.float32)

    input_types = {str(name) for name in spec.get("input_units", [])}
    output_types = {str(name) for name in spec.get("output_units", [])}
    input_nodes = tuple(i for i, typ in enumerate(node_types) if typ in input_types)
    output_nodes = tuple(i for i, typ in enumerate(node_types) if typ in output_types)
    node_ids = tuple(
        f"{typ}[{u},{v}]" for typ, u, v in zip(node_types, node_u, node_v, strict=True)
    )

    graph = CircuitGraph(
        num_nodes=len(node_types),
        edge_index=edge_index,
        edge_weight=edge_weight,
        node_ids=node_ids,
    )
    return RetinotopicFlyVisCircuit(
        graph=graph,
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        node_types=tuple(node_types),
        node_u=torch.tensor(node_u, dtype=torch.long),
        node_v=torch.tensor(node_v, dtype=torch.long),
    )


def type_pair_preserving_rewire(
    circuit: RetinotopicFlyVisCircuit,
    *,
    swaps: int,
    seed: int = 0,
) -> RetinotopicFlyVisCircuit:
    """Rewire explicit edges while preserving type pairs and every node's degrees.

    Directed double-edge swaps are restricted to edges with the same source and target cell
    types. Therefore the null keeps the complete cell-type graph, exact edge count, each
    neuron's in/out degree, and the measured weight attached to each edge slot, while destroying
    fine retinotopic partner identity. It is the primary fine-wiring null for learned tasks.
    """
    if swaps < 0:
        raise ValueError("swaps must be non-negative")
    graph = circuit.graph
    if graph.num_edges < 2 or swaps == 0:
        return circuit

    edges = [tuple(map(int, edge)) for edge in graph.edge_index.t().tolist()]
    blocks: dict[tuple[str, str], list[int]] = {}
    for index, (source, target) in enumerate(edges):
        key = (circuit.node_types[source], circuit.node_types[target])
        blocks.setdefault(key, []).append(index)
    eligible = [indices for indices in blocks.values() if len(indices) >= 2]
    if not eligible:
        return circuit

    rng = random.Random(seed)
    edge_set = set(edges)
    successful = 0
    attempts = 0
    max_attempts = max(100, swaps * 100)
    while successful < swaps and attempts < max_attempts:
        attempts += 1
        indices = rng.choice(eligible)
        i, j = rng.sample(indices, 2)
        source_a, target_a = edges[i]
        source_b, target_b = edges[j]
        if source_a == source_b or target_a == target_b:
            continue
        edge_a = (source_a, target_b)
        edge_b = (source_b, target_a)
        if source_a == target_b or source_b == target_a:
            continue
        if edge_a in edge_set or edge_b in edge_set:
            continue
        edge_set.remove(edges[i])
        edge_set.remove(edges[j])
        edge_set.add(edge_a)
        edge_set.add(edge_b)
        edges[i] = edge_a
        edges[j] = edge_b
        successful += 1

    rewired = torch.tensor(edges, dtype=torch.long).t().contiguous()
    new_graph = CircuitGraph(
        num_nodes=graph.num_nodes,
        edge_index=rewired,
        edge_weight=None if graph.edge_weight is None else graph.edge_weight.clone(),
        node_ids=graph.node_ids,
        node_rank=graph.node_rank,
    )
    return RetinotopicFlyVisCircuit(
        graph=new_graph,
        input_nodes=circuit.input_nodes,
        output_nodes=circuit.output_nodes,
        node_types=circuit.node_types,
        node_u=circuit.node_u.clone(),
        node_v=circuit.node_v.clone(),
    )
