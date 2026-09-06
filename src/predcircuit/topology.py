from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch


@dataclass(frozen=True)
class CircuitGraph:
    """A directed neural-circuit graph with one scalar state per node.

    `edge_index[0]` contains source nodes and `edge_index[1]` destination nodes.
    `edge_weight` is an optional biological strength used only for initialization/analysis;
    trainable model parameters are kept by the model itself.
    """

    num_nodes: int
    edge_index: torch.Tensor
    edge_weight: torch.Tensor | None = None
    node_ids: tuple[str, ...] | None = None
    node_rank: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if self.edge_index.dtype != torch.long:
            object.__setattr__(self, "edge_index", self.edge_index.long())
        if self.edge_weight is not None and len(self.edge_weight) != self.num_edges:
            raise ValueError("edge_weight must have one value per edge")
        if self.node_ids is not None and len(self.node_ids) != self.num_nodes:
            raise ValueError("node_ids must have one value per node")
        if self.num_edges and (
            int(self.edge_index.min()) < 0 or int(self.edge_index.max()) >= self.num_nodes
        ):
            raise ValueError("edge_index contains an out-of-range node")

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def density(self) -> float:
        possible = self.num_nodes * max(self.num_nodes - 1, 1)
        return self.num_edges / possible

    def degrees(self) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = self.edge_index
        out_deg = torch.bincount(src, minlength=self.num_nodes)
        in_deg = torch.bincount(dst, minlength=self.num_nodes)
        return in_deg, out_deg

    def reciprocal_mask(self) -> torch.Tensor:
        edges = set(map(tuple, self.edge_index.t().tolist()))
        return torch.tensor(
            [(int(v), int(u)) in edges for u, v in edges_from(self)], dtype=torch.bool
        )

    def remove_reciprocal_edges(self) -> CircuitGraph:
        """Remove edges that participate in a reciprocal two-node motif."""
        keep = ~self.reciprocal_mask()
        return self._subset_edges(keep)

    def remove_feedback_edges(self, rank: torch.Tensor | None = None) -> CircuitGraph:
        """Remove edges that go from an equal/higher rank to a lower rank.

        Rank can encode anatomical depth, sensory-to-motor ordering, or hand-defined layers.
        Equal-rank edges are treated as recurrent/lateral and removed as feedback here.
        """
        rank = rank if rank is not None else self.node_rank
        if rank is None:
            raise ValueError("feedback ablation requires node_rank")
        src, dst = self.edge_index
        keep = rank[src] < rank[dst]
        return self._subset_edges(keep)

    def degree_preserving_rewire(self, swaps: int, seed: int = 0) -> CircuitGraph:
        """Directed double-edge swaps preserving every node's in/out degree.

        Edge (a->b, c->d) becomes (a->d, c->b) when that introduces neither self loops
        nor duplicate edges. Biological edge strengths are shuffled across rewired edges.
        """
        rng = np.random.default_rng(seed)
        edges = [tuple(map(int, e)) for e in self.edge_index.t().tolist()]
        edge_set = set(edges)
        if len(edges) < 2:
            return self

        successful = 0
        attempts = 0
        max_attempts = max(100, swaps * 50)
        while successful < swaps and attempts < max_attempts:
            attempts += 1
            i, j = rng.choice(len(edges), size=2, replace=False)
            a, b = edges[i]
            c, d = edges[j]
            if a == c or b == d:
                continue
            e1 = (a, d)
            e2 = (c, b)
            if a == d or c == b:
                continue
            old1, old2 = edges[i], edges[j]
            if e1 in edge_set or e2 in edge_set:
                continue
            edge_set.remove(old1)
            edge_set.remove(old2)
            edge_set.add(e1)
            edge_set.add(e2)
            edges[i] = e1
            edges[j] = e2
            successful += 1

        rewired = torch.tensor(edges, dtype=torch.long).t().contiguous()
        weight = self.edge_weight
        if weight is not None:
            perm = torch.as_tensor(rng.permutation(self.num_edges), dtype=torch.long)
            weight = weight[perm]
        return CircuitGraph(
            num_nodes=self.num_nodes,
            edge_index=rewired,
            edge_weight=weight,
            node_ids=self.node_ids,
            node_rank=self.node_rank,
        )

    def shuffle_edge_weights(self, seed: int = 0) -> CircuitGraph:
        """Keep adjacency fixed and randomly permute measured edge strengths."""
        if self.edge_weight is None:
            raise ValueError("shuffle_edge_weights requires edge_weight")
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(self.num_edges, generator=gen)
        return CircuitGraph(
            num_nodes=self.num_nodes,
            edge_index=self.edge_index.clone(),
            edge_weight=self.edge_weight[perm].clone(),
            node_ids=self.node_ids,
            node_rank=self.node_rank,
        )

    def binary_edge_weights(self) -> CircuitGraph:
        """Keep adjacency fixed but replace measured strengths with unit weights."""
        return CircuitGraph(
            num_nodes=self.num_nodes,
            edge_index=self.edge_index.clone(),
            edge_weight=torch.ones(self.num_edges, dtype=torch.float32),
            node_ids=self.node_ids,
            node_rank=self.node_rank,
        )

    def _subset_edges(self, keep: torch.Tensor) -> CircuitGraph:
        return CircuitGraph(
            num_nodes=self.num_nodes,
            edge_index=self.edge_index[:, keep],
            edge_weight=None if self.edge_weight is None else self.edge_weight[keep],
            node_ids=self.node_ids,
            node_rank=self.node_rank,
        )

    def to_networkx(self) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from(range(self.num_nodes))
        weights = (
            self.edge_weight.tolist() if self.edge_weight is not None else [1.0] * self.num_edges
        )
        for (u, v), w in zip(edges_from(self), weights, strict=True):
            g.add_edge(u, v, weight=float(w))
        return g


def edges_from(graph: CircuitGraph) -> Iterable[tuple[int, int]]:
    for u, v in graph.edge_index.t().tolist():
        yield int(u), int(v)


def layered_graph(
    layer_sizes: list[int],
    *,
    recurrent_probability: float = 0.15,
    feedback_probability: float = 0.05,
    seed: int = 0,
) -> CircuitGraph:
    """Create a sparse layered graph with optional lateral/recurrent and feedback edges."""
    if len(layer_sizes) < 2 or any(s <= 0 for s in layer_sizes):
        raise ValueError("layer_sizes must contain at least two positive layers")
    rng = np.random.default_rng(seed)
    offsets = np.cumsum([0] + layer_sizes)
    ranks = np.concatenate([np.full(s, i) for i, s in enumerate(layer_sizes)])
    edges: set[tuple[int, int]] = set()

    for li in range(len(layer_sizes) - 1):
        for u in range(offsets[li], offsets[li + 1]):
            for v in range(offsets[li + 1], offsets[li + 2]):
                edges.add((u, v))

    for li in range(1, len(layer_sizes) - 1):
        nodes = range(offsets[li], offsets[li + 1])
        for u in nodes:
            for v in nodes:
                if u != v and rng.random() < recurrent_probability:
                    edges.add((u, v))

    for li in range(1, len(layer_sizes)):
        for u in range(offsets[li], offsets[li + 1]):
            for v in range(offsets[li - 1], offsets[li]):
                if rng.random() < feedback_probability:
                    edges.add((u, v))

    edge_index = torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()
    return CircuitGraph(
        num_nodes=sum(layer_sizes),
        edge_index=edge_index,
        node_rank=torch.tensor(ranks, dtype=torch.long),
    )


def erdos_renyi_matched(graph: CircuitGraph, seed: int = 0) -> CircuitGraph:
    """Directed simple random graph with exactly the same node and edge counts.

    This is intentionally a weak null model: unlike degree-preserving rewiring, it does
    not preserve the in/out-degree sequence. It is useful only alongside stronger controls.
    """
    n = graph.num_nodes
    max_edges = n * max(n - 1, 0)
    if graph.num_edges > max_edges:
        raise ValueError("graph has more edges than a directed simple graph can contain")
    rng = np.random.default_rng(seed)
    all_edges = np.array([(u, v) for u in range(n) for v in range(n) if u != v], dtype=np.int64)
    if graph.num_edges:
        chosen = rng.choice(len(all_edges), size=graph.num_edges, replace=False)
        edges = torch.from_numpy(all_edges[chosen]).long().t().contiguous()
    else:
        edges = torch.empty((2, 0), dtype=torch.long)
    weight = None
    if graph.edge_weight is not None:
        perm = torch.as_tensor(rng.permutation(graph.num_edges), dtype=torch.long)
        weight = graph.edge_weight[perm].clone()
    return CircuitGraph(
        num_nodes=n,
        edge_index=edges,
        edge_weight=weight,
        node_ids=graph.node_ids,
        node_rank=graph.node_rank,
    )
