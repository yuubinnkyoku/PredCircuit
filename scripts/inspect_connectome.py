from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from predcircuit.connectome import graph_from_connectivity, load_connectivity
from predcircuit.metrics import graph_metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Threshold/extract a connectome and report graph statistics")
    p.add_argument("path", type=Path)
    p.add_argument("--min-weight", type=float, default=5.0)
    p.add_argument("--max-nodes", type=int, default=1000)
    p.add_argument("--seed-node", action="append", default=[])
    args = p.parse_args()
    table = load_connectivity(args.path)
    graph = graph_from_connectivity(
        table,
        min_weight=args.min_weight,
        max_nodes=args.max_nodes,
        seed_nodes=args.seed_node or None,
    )
    print(json.dumps(asdict(graph_metrics(graph)), indent=2))


if __name__ == "__main__":
    main()
