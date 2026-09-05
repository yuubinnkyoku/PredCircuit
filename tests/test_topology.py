import torch

from predcircuit.topology import layered_graph


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
