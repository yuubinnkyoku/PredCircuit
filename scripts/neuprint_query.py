"""Fetch a small MaleCNS subgraph via neuPrint without embedding credentials in the repo."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--type", required=True, help="MaleCNS neuron type, e.g. DNge104")
    p.add_argument("--out", type=Path, default=Path("data/raw/neuprint_subgraph.csv"))
    args = p.parse_args()

    token = os.environ.get("NEUPRINT_TOKEN")
    if not token:
        raise SystemExit("Set NEUPRINT_TOKEN in the environment; never commit the token.")
    try:
        from neuprint import Client, fetch_adjacencies
    except ImportError as exc:
        raise SystemExit("Install with: uv sync --extra malecns") from exc

    Client("https://neuprint.janelia.org", dataset="male-cns:v1.0", token=token)
    outgoing, _ = fetch_adjacencies(args.type)
    incoming, _ = fetch_adjacencies(None, args.type)
    edges = pd.concat([outgoing, incoming], ignore_index=True).drop_duplicates()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(args.out, index=False)
    print(f"wrote {len(edges)} edges to {args.out}")


if __name__ == "__main__":
    main()
