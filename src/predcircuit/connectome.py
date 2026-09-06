from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .topology import CircuitGraph

MALCNS_ANNOTATIONS_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
    "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)
MALCNS_CONNECTIVITY_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
)
MALCNS_NEUROTRANSMITTER_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
    "body-neurotransmitters-male-cns-v1.0.feather"
)


@dataclass
class ConnectomeTables:
    annotations: pd.DataFrame | None
    connectivity: pd.DataFrame


def _pick(columns: Iterable[str], candidates: tuple[str, ...]) -> str:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise ValueError(f"none of {candidates!r} found in columns: {list(columns)!r}")


def normalize_connectivity_table(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common neuPrint/MaleCNS edge-table schemas to pre/post/weight."""
    pre = _pick(df.columns, ("body_pre", "bodyId_pre", "pre", "source", "bodyId"))
    post = _pick(df.columns, ("body_post", "bodyId_post", "post", "target"))
    try:
        weight = _pick(df.columns, ("weight", "count", "syn_count", "synapse_count", "w"))
    except ValueError:
        out = df[[pre, post]].copy()
        out["weight"] = 1.0
    else:
        out = df[[pre, post, weight]].copy()
        out = out.rename(columns={weight: "weight"})
    out = out.rename(columns={pre: "pre", post: "post"})
    out = out[["pre", "post", "weight"]]
    out = out.dropna()
    out["weight"] = pd.to_numeric(out["weight"], errors="raise")
    return out


def load_connectivity(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".feather":
        df = pd.read_feather(path)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".tsv"}:
        df = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    else:
        raise ValueError(f"unsupported connectivity format: {path.suffix}")
    return normalize_connectivity_table(df)


def graph_from_connectivity(
    table: pd.DataFrame,
    *,
    min_weight: float = 1.0,
    max_nodes: int | None = None,
    seed_nodes: Iterable[int | str] | None = None,
) -> CircuitGraph:
    """Build a weighted CircuitGraph, optionally extracting a strong local subgraph.

    If max_nodes is set, nodes are ranked by total incident synapse weight. If seed_nodes are
    supplied, they are kept first and the remaining slots are filled with strongest 1-hop
    neighbors, then strongest global nodes.
    """
    df = (
        normalize_connectivity_table(table)
        if set(table.columns) != {"pre", "post", "weight"}
        else table.copy()
    )
    df = df[df["weight"] >= min_weight].copy()
    if df.empty:
        raise ValueError("no edges remain after thresholding")

    incident = (
        pd.concat(
            [
                df[["pre", "weight"]].rename(columns={"pre": "node"}),
                df[["post", "weight"]].rename(columns={"post": "node"}),
            ],
            ignore_index=True,
        )
        .groupby("node", sort=False)["weight"]
        .sum()
        .sort_values(ascending=False)
    )

    selected: list[object]
    if max_nodes is None:
        selected = incident.index.tolist()
    else:
        selected = []
        if seed_nodes is not None:
            seeds = list(seed_nodes)
            selected.extend([s for s in seeds if s in incident.index])
            local = df[df["pre"].isin(seeds) | df["post"].isin(seeds)]
            neigh = pd.concat([local["pre"], local["post"]]).drop_duplicates().tolist()
            neigh = [n for n in neigh if n not in selected]
            neigh.sort(key=lambda n: float(incident.get(n, 0.0)), reverse=True)
            selected.extend(neigh[: max(0, max_nodes - len(selected))])
        if len(selected) < max_nodes:
            selected.extend(
                [n for n in incident.index if n not in selected][: max_nodes - len(selected)]
            )
        selected = selected[:max_nodes]

    selected_set = set(selected)
    sub = df[df["pre"].isin(selected_set) & df["post"].isin(selected_set)].copy()
    ids = list(dict.fromkeys(selected))
    index = {node: i for i, node in enumerate(ids)}
    src = torch.tensor([index[x] for x in sub["pre"]], dtype=torch.long)
    dst = torch.tensor([index[x] for x in sub["post"]], dtype=torch.long)
    weight = torch.tensor(sub["weight"].to_numpy(dtype=np.float32), dtype=torch.float32)
    return CircuitGraph(
        num_nodes=len(ids),
        edge_index=torch.stack([src, dst]),
        edge_weight=weight,
        node_ids=tuple(map(str, ids)),
    )


def attach_node_rank(
    graph: CircuitGraph,
    table: pd.DataFrame,
    *,
    node_col: str = "node",
    rank_col: str = "layer_median",
) -> CircuitGraph:
    """Attach an anatomical/graph-traversal rank to a graph from a metadata table.

    `graph.node_ids` are compared as strings so integer body IDs and string IDs interoperate.
    Missing ranks are rejected instead of silently inventing an ordering.
    """
    if graph.node_ids is None:
        raise ValueError("attach_node_rank requires graph.node_ids")
    if node_col not in table or rank_col not in table:
        raise ValueError(f"table must contain {node_col!r} and {rank_col!r}")
    mapping = dict(zip(table[node_col].astype(str), table[rank_col], strict=True))
    missing = [node for node in graph.node_ids if node not in mapping or pd.isna(mapping[node])]
    if missing:
        sample = ", ".join(missing[:5])
        raise ValueError(f"missing rank for {len(missing)} nodes (e.g. {sample})")
    rank = torch.tensor([float(mapping[node]) for node in graph.node_ids], dtype=torch.float32)
    return CircuitGraph(
        num_nodes=graph.num_nodes,
        edge_index=graph.edge_index.clone(),
        edge_weight=None if graph.edge_weight is None else graph.edge_weight.clone(),
        node_ids=graph.node_ids,
        node_rank=rank,
    )
