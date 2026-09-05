import pandas as pd

from predcircuit.connectome import attach_node_rank, graph_from_connectivity, normalize_connectivity_table


def test_normalize_connectivity_and_extract() -> None:
    raw = pd.DataFrame(
        {
            "body_pre": [10, 10, 20, 30],
            "body_post": [20, 30, 30, 10],
            "weight": [5, 2, 7, 1],
        }
    )
    table = normalize_connectivity_table(raw)
    graph = graph_from_connectivity(table, min_weight=2, max_nodes=3)
    assert graph.num_nodes == 3
    assert graph.num_edges == 3


def test_attach_node_rank() -> None:
    raw = pd.DataFrame({"pre": [10, 20], "post": [20, 30], "weight": [3, 4]})
    graph = graph_from_connectivity(raw, max_nodes=3)
    ranks = pd.DataFrame({"node": [10, 20, 30], "layer_median": [0.0, 1.0, 2.0]})
    ranked = attach_node_rank(graph, ranks)
    assert ranked.node_rank is not None
    assert len(ranked.node_rank) == 3
