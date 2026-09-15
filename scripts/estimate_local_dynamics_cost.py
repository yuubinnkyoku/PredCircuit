"""Symbolic first-order cost model for PC-ALM versus a BP/ePC reverse sweep.

This is intentionally architecture-parametric.  It avoids pretending that a
MAC count is a cycle count: the key distinction is total arithmetic work versus
critical-path layer waves when layers have dedicated parallel datapaths.
"""

from __future__ import annotations

import argparse
import json


def costs(depth: int, width: int, pcalm_steps: int) -> dict[str, int | float]:
    if depth < 2 or width < 1 or pcalm_steps < 1:
        raise ValueError("depth>=2, width>=1 and pcalm_steps>=1 are required")

    hidden_layers = depth - 1
    # Dense equal-width approximation.  One PC-ALM state update needs a forward
    # prediction W_l z_{l-1} and a transpose-neighbour term W_{l+1}^T q_{l+1}.
    # Boundary corrections are O(N^2) and omitted deliberately.
    pcalm_matvec_mac_per_step = 2 * hidden_layers * width * width
    pcalm_dual_scalar_ops_per_step = 2 * hidden_layers * width  # alpha*r + lambda

    # A reverse-mode credit sweep needs one transpose matvec per hidden-layer
    # boundary.  This counts credit propagation only, not parameter-gradient
    # outer products, which both learning methods eventually need.
    reverse_matvec_mac = hidden_layers * width * width

    # With one dedicated layer engine per layer, all PC-ALM local updates form
    # one layer-wave per relaxation step. BP/ePC reverse credit remains ordered
    # across hidden boundaries. These are dependency waves, not clock cycles.
    pcalm_parallel_waves = pcalm_steps
    reverse_sequential_waves = hidden_layers

    return {
        "depth": depth,
        "width": width,
        "pcalm_steps": pcalm_steps,
        "pcalm_matvec_mac_per_step": pcalm_matvec_mac_per_step,
        "pcalm_total_matvec_mac": pcalm_steps * pcalm_matvec_mac_per_step,
        "pcalm_total_dual_scalar_ops": pcalm_steps * pcalm_dual_scalar_ops_per_step,
        "reverse_credit_matvec_mac": reverse_matvec_mac,
        "mac_ratio_pcalm_over_reverse_credit": (
            pcalm_steps * pcalm_matvec_mac_per_step / reverse_matvec_mac
        ),
        "pcalm_parallel_dependency_waves": pcalm_parallel_waves,
        "reverse_sequential_dependency_waves": reverse_sequential_waves,
        "wave_ratio_pcalm_over_reverse": pcalm_parallel_waves / reverse_sequential_waves,
        "pcalm_state_words": 2 * hidden_layers * width,  # z and lambda
        "spc_state_words": hidden_layers * width,
        "lambda_state_overhead_words": hidden_layers * width,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--pcalm-steps", type=int, default=96)
    args = parser.parse_args()
    print(json.dumps(costs(args.depth, args.width, args.pcalm_steps), indent=2))


if __name__ == "__main__":
    main()
