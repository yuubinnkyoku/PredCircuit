from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareCost:
    weight_scalars: int
    free_state_scalars: int
    dual_state_scalars: int
    macs_per_relaxation_step: int
    dual_scalar_updates_per_step: int
    persistent_state_bits: int
    weight_bits: int

    def cycles_per_step_lower_bound(
        self,
        *,
        mac_lanes: int,
        dual_lanes: int = 1,
        overlap_dual_update: bool = True,
    ) -> int:
        if mac_lanes < 1 or dual_lanes < 1:
            raise ValueError("lane counts must be positive")
        mac_cycles = math.ceil(self.macs_per_relaxation_step / mac_lanes)
        dual_cycles = math.ceil(self.dual_scalar_updates_per_step / dual_lanes)
        if self.dual_scalar_updates_per_step == 0:
            return mac_cycles
        if overlap_dual_update:
            return max(mac_cycles, dual_cycles)
        return mac_cycles + dual_cycles


def residual_mlp_weight_scalars(
    *,
    depth: int,
    width: int,
    input_dim: int,
    output_dim: int,
) -> int:
    if depth < 2:
        raise ValueError("depth must be at least 2")
    if min(width, input_dim, output_dim) < 1:
        raise ValueError("dimensions must be positive")
    return width * input_dim + (depth - 2) * width * width + output_dim * width


def estimate_local_learning_cost(
    *,
    family: str,
    depth: int,
    width: int,
    input_dim: int,
    output_dim: int,
    batch_size: int,
    state_bits: int,
    weight_bits: int,
    dual_bits: int | None = None,
) -> HardwareCost:
    if family not in {"spc", "pcalm"}:
        raise ValueError("family must be 'spc' or 'pcalm'")
    if batch_size < 1 or state_bits < 1 or weight_bits < 1:
        raise ValueError("batch size and bit widths must be positive")

    weights = residual_mlp_weight_scalars(
        depth=depth,
        width=width,
        input_dim=input_dim,
        output_dim=output_dim,
    )
    free = (depth - 1) * batch_size * width
    dual = free if family == "pcalm" else 0
    if family == "pcalm":
        if dual_bits is None or dual_bits < 1:
            raise ValueError("PC-ALM requires a positive dual_bits value")
    else:
        dual_bits = 0

    # One forward Wz path and one transpose W^T credit path per layer.
    # This is a major-MAC count, not a claim that every implementation must
    # materialize both paths separately.
    macs = 2 * batch_size * weights
    state_storage = free * state_bits + dual * dual_bits

    return HardwareCost(
        weight_scalars=weights,
        free_state_scalars=free,
        dual_state_scalars=dual,
        macs_per_relaxation_step=macs,
        dual_scalar_updates_per_step=dual,
        persistent_state_bits=state_storage,
        weight_bits=weights * weight_bits,
    )
