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


def test_remove_feedback_obeys_rank() -> None:
    graph = layered_graph([2, 4, 1], recurrent_probability=0.5, feedback_probability=0.5, seed=2)
    ff = graph.remove_feedback_edges()
    src, dst = ff.edge_index
    assert torch.all(ff.node_rank[src] < ff.node_rank[dst])


def test_er_matched_preserves_n_and_e_only() -> None:
    graph = layered_graph([3, 5, 2], recurrent_probability=0.3, feedback_probability=0.1, seed=5)
    null = erdos_renyi_matched(graph, seed=6)
    assert null.num_nodes == graph.num_nodes
    assert null.num_edges == graph.num_edges
    assert torch.all(null.edge_index[0] != null.edge_index[1])
