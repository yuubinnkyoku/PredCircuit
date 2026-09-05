from __future__ import annotations

from dataclasses import asdict, dataclass

from .topology import CircuitGraph


@dataclass(frozen=True)
class LocalCostEstimate:
    nodes: int
    edges: int
    inference_steps: int
    pc_edge_messages_per_example: int
    pc_min_state_scalars: int
    bptt_min_saved_state_scalars: int


def local_cost_estimate(graph: CircuitGraph, inference_steps: int) -> LocalCostEstimate:
    """Simple architecture-independent lower-bound bookkeeping estimates.

    Predictive-coding inference evaluates each edge in the prediction and error-feedback
    directions, then once more for the local weight update. BPTT must at least retain one node
    state per unrolled step. These are counting estimates, not wall-clock or energy claims.
    """
    if inference_steps < 0:
        raise ValueError("inference_steps must be non-negative")
    return LocalCostEstimate(
        nodes=graph.num_nodes,
        edges=graph.num_edges,
        inference_steps=inference_steps,
        pc_edge_messages_per_example=(2 * inference_steps + 1) * graph.num_edges,
        pc_min_state_scalars=2 * graph.num_nodes + graph.num_edges,
        bptt_min_saved_state_scalars=(inference_steps + 1) * graph.num_nodes + graph.num_edges,
    )


def local_cost_dict(graph: CircuitGraph, inference_steps: int) -> dict[str, int]:
    return asdict(local_cost_estimate(graph, inference_steps))
