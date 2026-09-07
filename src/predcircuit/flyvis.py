from __future__ import annotations

import json
import urllib.request
from collections import deque
from dataclasses import dataclass
from typing import Any

import torch

from .topology import CircuitGraph

FLYVIS_CONNECTOME_COMMIT = "92b3845cc426dd309a1a0e1b3890156c42e14021"
FLYVIS_CONNECTOME_URL = (
    "https://raw.githubusercontent.com/TuragaLab/flyvis/"
    f"{FLYVIS_CONNECTOME_COMMIT}/flyvis/connectome/fib25-fib19_v2.2.json"
)


@dataclass(frozen=True)
class FlyVisTypeCircuit:
    """Cell-type-level projection of the published FlyVis connectome scaffold."""

    graph: CircuitGraph
    input_nodes: tuple[int, ...]
    output_nodes: tuple[int, ...]


def load_flyvis_spec(url: str = FLYVIS_CONNECTOME_URL) -> dict[str, Any]:
    """Download the pinned FlyVis connectome specification."""
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())


def graph_from_flyvis_spec(spec: dict[str, Any], *, signed: bool = True) -> FlyVisTypeCircuit:
    """Collapse FlyVis' retinotopic filters to a cell-type-level directed graph.

    Each type-to-type edge weight is the sum of the reported synapse counts over spatial
    offsets. When ``signed`` is true, FlyVis' ``alpha`` field (+1/-1) is applied.

    This deliberately discards retinotopic offsets. It is therefore suitable for topology
    and inference-conditioning pilots, not for claims about visual motion computation.
    """
    nodes_raw = spec.get("nodes")
    edges_raw = spec.get("edges")
    if not isinstance(nodes_raw, list) or not isinstance(edges_raw, list):
        raise TypeError("FlyVis spec must contain list-valued 'nodes' and 'edges'")

    names = [str(node["name"]) for node in nodes_raw]
    index = {name: i for i, name in enumerate(names)}
    if len(index) != len(names):
        raise ValueError("FlyVis node names must be unique")

    aggregated: dict[tuple[int, int], float] = {}
    for edge in edges_raw:
        src_name = str(edge["src"])
        dst_name = str(edge["tar"])
        if src_name not in index or dst_name not in index:
            raise ValueError(f"unknown FlyVis edge endpoint: {src_name!r}->{dst_name!r}")
        offsets = edge.get("offsets", [])
        if not isinstance(offsets, list):
            raise TypeError("FlyVis edge offsets must be a list")
        magnitude = sum(float(offset[1]) for offset in offsets)
        sign = float(edge.get("alpha", 1.0)) if signed else 1.0
        key = (index[src_name], index[dst_name])
        aggregated[key] = aggregated.get(key, 0.0) + sign * magnitude

    pairs = sorted((pair, weight) for pair, weight in aggregated.items() if weight != 0.0)
    if not pairs:
        raise ValueError("FlyVis spec produced no non-zero type-level edges")
    edge_index = torch.tensor([pair for pair, _ in pairs], dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor([weight for _, weight in pairs], dtype=torch.float32)

    input_names = [str(name) for name in spec.get("input_units", [])]
    output_names = [str(name) for name in spec.get("output_units", [])]
    missing = [name for name in input_names + output_names if name not in index]
    if missing:
        raise ValueError(f"unknown FlyVis input/output cell types: {missing[:5]!r}")

    graph = CircuitGraph(
        num_nodes=len(names),
        edge_index=edge_index,
        edge_weight=edge_weight,
        node_ids=tuple(names),
    )
    return FlyVisTypeCircuit(
        graph=graph,
        input_nodes=tuple(index[name] for name in input_names),
        output_nodes=tuple(index[name] for name in output_names),
    )


def with_sensory_distance_rank(circuit: FlyVisTypeCircuit) -> FlyVisTypeCircuit:
    """Attach shortest directed distance from any FlyVis input type as a coarse rank.

    Unreachable nodes are placed one rank beyond the farthest reachable node. The resulting
    rank is a graph-derived control variable, not an anatomical layer annotation.
    """
    graph = circuit.graph
    adjacency: list[list[int]] = [[] for _ in range(graph.num_nodes)]
    for src, dst in graph.edge_index.t().tolist():
        adjacency[int(src)].append(int(dst))

    distance = [-1] * graph.num_nodes
    queue: deque[int] = deque()
    for node in circuit.input_nodes:
        distance[node] = 0
        queue.append(node)
    while queue:
        src = queue.popleft()
        for dst in adjacency[src]:
            if distance[dst] >= 0:
                continue
            distance[dst] = distance[src] + 1
            queue.append(dst)

    farthest = max((value for value in distance if value >= 0), default=0)
    unreachable_rank = farthest + 1
    rank = torch.tensor(
        [value if value >= 0 else unreachable_rank for value in distance], dtype=torch.float32
    )
    ranked = CircuitGraph(
        num_nodes=graph.num_nodes,
        edge_index=graph.edge_index.clone(),
        edge_weight=None if graph.edge_weight is None else graph.edge_weight.clone(),
        node_ids=graph.node_ids,
        node_rank=rank,
    )
    return FlyVisTypeCircuit(ranked, circuit.input_nodes, circuit.output_nodes)
