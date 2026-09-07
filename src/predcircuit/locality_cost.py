from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TrainingStateCost:
    """Analytical storage lower bounds for sparse recurrent training.

    Model parameters and the externally supplied input sequence are excluded so the
    comparison isolates temporary training state. The BPTT estimate is deliberately
    a lower bound: real autodiff systems usually retain additional intermediates.
    """

    nodes: int
    edges: int
    batch_size: int
    recurrent_steps: int
    bytes_per_value: int
    local_elements: int
    bptt_state_lower_bound_elements: int

    @property
    def local_bytes(self) -> int:
        return self.local_elements * self.bytes_per_value

    @property
    def bptt_state_lower_bound_bytes(self) -> int:
        return self.bptt_state_lower_bound_elements * self.bytes_per_value

    @property
    def bptt_over_local_ratio(self) -> float:
        return self.bptt_state_lower_bound_elements / self.local_elements

    @property
    def crossover_recurrent_steps(self) -> int:
        """First recurrent depth where the BPTT state-history lower bound exceeds local storage."""

        state_width = self.batch_size * self.nodes
        return max(math.floor(self.local_elements / state_width), 0)


def estimate_two_phase_pc_vs_bptt(
    *,
    nodes: int,
    edges: int,
    batch_size: int,
    recurrent_steps: int,
    bytes_per_value: int = 4,
) -> TrainingStateCost:
    """Compare a streaming two-phase PC update with unrolled BPTT state history.

    The local-PC estimate stores one current node-state vector, the free-phase edge
    statistic, and the free-phase node/bias statistic. During the nudged phase the
    corresponding statistics can be accumulated and differenced in place.

    The BPTT estimate stores only recurrent node states for the initial state plus
    every unrolled recurrent step. It excludes prediction/error tensors, activation
    derivatives, framework bookkeeping, and optimizer state, so it should not be
    interpreted as a full implementation memory estimate.
    """

    if nodes <= 0:
        raise ValueError("nodes must be positive")
    if edges < 0:
        raise ValueError("edges must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if recurrent_steps <= 0:
        raise ValueError("recurrent_steps must be positive")
    if bytes_per_value <= 0:
        raise ValueError("bytes_per_value must be positive")

    state_width = batch_size * nodes
    local_elements = state_width + edges + nodes
    bptt_elements = state_width * (recurrent_steps + 1)
    return TrainingStateCost(
        nodes=nodes,
        edges=edges,
        batch_size=batch_size,
        recurrent_steps=recurrent_steps,
        bytes_per_value=bytes_per_value,
        local_elements=local_elements,
        bptt_state_lower_bound_elements=bptt_elements,
    )
