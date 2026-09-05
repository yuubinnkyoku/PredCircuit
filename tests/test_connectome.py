import pandas as pd

from predcircuit.connectome import graph_from_connectivity, normalize_connectivity_table


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
