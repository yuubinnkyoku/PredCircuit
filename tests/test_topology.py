import collections

import torch

from predcircuit.topology import erdos_renyi_matched, layered_graph


def test_degree_preserving_rewire_keeps_degree_sequence() -> None:
    graph = layered_graph([3, 7, 2], recurrent_probability=0.4, feedback_probability=0.2, seed=3)
    rewired = graph.degree_preserving_rewire(100, seed=4)
    before = graph.degrees()
    after = rewired.degrees()
    assert torch.equal(before[0], after[0])
    assert torch.equal(before[1], after[1])
    assert graph.num_edges == rewired.num_edges


def test_rank_pair_preserving_rewire_keeps_degrees_and_rank_pairs() -> None:
    graph = layered_graph([3, 8, 3], recurrent_probability=0.45, feedback_probability=0.35, seed=7)
    rewired = graph.rank_pair_preserving_rewire(200, seed=8)
    before = graph.degrees()
    after = rewired.degrees()
    assert torch.equal(before[0], after[0])
    assert torch.equal(before[1], after[1])
    assert graph.num_edges == rewired.num_edges
    assert rewired.node_rank is not None

    def rank_pairs(edge_index: torch.Tensor) -> collections.Counter[tuple[int, int]]:
        assert graph.node_rank is not None
        src, dst = edge_index
        return collections.Counter(
            (int(graph.node_rank[u]), int(graph.node_rank[v]))
            for u, v in zip(src.tolist(), dst.tolist(), strict=True)
        )

    assert rank_pairs(graph.edge_index) == rank_pairs(rewired.edge_index)
    assert torch.all(rewired.edge_index[0] != rewired.edge_index[1])
    assert len(set(map(tuple, rewired.edge_index.t().tolist()))) == rewired.num_edges


def test_remove_feedback_obeys_rank() -> None:
    graph = layered_graph([2, 4, 1], recurrent_probability=0.5, feedback_probability=0.5, seed=2)
    ff = graph.remove_feedback_edges()
    src, dst = ff.edge_index
    assert ff.node_rank is not None
    assert torch.all(ff.node_rank[src] < ff.node_rank[dst])


def test_er_matched_preserves_n_and_e_only() -> None:
    graph = layered_graph([3, 5, 2], recurrent_probability=0.3, feedback_probability=0.1, seed=5)
    null = erdos_renyi_matched(graph, seed=6)
    assert null.num_nodes == graph.num_nodes
    assert null.num_edges == graph.num_edges
    assert torch.all(null.edge_index[0] != null.edge_index[1])
