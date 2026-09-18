from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

import diagnose_pcalm_width64_update_lattice_alignment as alignment
from diagnose_pcalm_width64_leaky_mixed_precision import quantize as base_quantize
from predcircuit.pcalm import (
    ResidualMLP,
    Schedule,
    gradient_cosine,
    gradient_relative_error,
    method_grad,
)

STATE_PRECISION = "fixed15_i3"
UPDATE_PRECISION = "fixed14_i1"
DUAL_PRECISION = "fixed12_i1"
PHASES = (0.0, 0.25, 0.5)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test whether sub-LSB state-grid phase changes leaky PC-ALM lattice lock."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--state-lr", type=float, default=0.234285)
    parser.add_argument("--dual-leak", type=float, default=0.02)
    args = parser.parse_args()

    depth, width = 32, 64
    model_seed = args.seed + 1000 * width + depth
    data_seed = args.seed + 10_000 + 1000 * width + depth
    generator = torch.Generator().manual_seed(data_seed)
    x = torch.randn(4, 8, generator=generator)
    y = torch.randn(4, 4, generator=generator)

    bp_model = ResidualMLP(
        depth=depth, width=width, input_dim=8, output_dim=4, activation="relu", seed=model_seed
    )
    bp = method_grad(bp_model, x, y, Schedule("bp", budget=0), state_lr=args.state_lr, rho=1.0)
    bp_first = bp[0]
    bp_norm = float(bp_first.norm())
    state_lsb = alignment.fixed_lsb(STATE_PRECISION)
    assert state_lsb is not None

    rows: list[dict[str, object]] = []
    original_quantize = alignment.quantize
    try:
        for phase_lsb in PHASES:
            offset = phase_lsb * state_lsb

            def phase_quantize(
                value: torch.Tensor, precision: str, *, _offset: float = offset
            ) -> tuple[torch.Tensor, int, int]:
                if precision != STATE_PRECISION or _offset == 0.0:
                    return base_quantize(value, precision)
                shifted, saturated, total = base_quantize(value - _offset, precision)
                return shifted + _offset, saturated, total

            alignment.quantize = phase_quantize
            model = ResidualMLP(
                depth=depth,
                width=width,
                input_dim=8,
                output_dim=4,
                activation="relu",
                seed=model_seed,
            )
            grads, stats = alignment.run_alignment(
                model,
                x,
                y,
                update_precision=UPDATE_PRECISION,
                state_precision=STATE_PRECISION,
                dual_precision=DUAL_PRECISION,
                budget=args.budget,
                state_lr=args.state_lr,
                dual_leak=args.dual_leak,
            )
            first = grads[0]
            norm_ratio = float(first.norm()) / bp_norm if bp_norm else math.nan
            cosine = gradient_cosine([first], [bp_first])
            relative_error = gradient_relative_error([first], [bp_first])
            finite = bool(stats["finite"])
            rows.append(
                {
                    "seed": args.seed,
                    "phase_lsb": phase_lsb,
                    "phase_offset": offset,
                    "state_lsb": state_lsb,
                    "state_lr": args.state_lr,
                    "lattice_ratio_r": args.state_lr * x.shape[0],
                    "state_precision": STATE_PRECISION,
                    "update_precision": UPDATE_PRECISION,
                    "dual_precision": DUAL_PRECISION,
                    **stats,
                    "first_layer_cosine_to_bp": cosine,
                    "first_layer_grad_norm_ratio_to_bp": norm_ratio,
                    "first_layer_relative_error_to_bp": relative_error,
                    "useful_first_layer_credit": finite
                    and cosine >= 0.9
                    and 0.5 <= norm_ratio <= 2.0
                    and relative_error <= 0.6,
                }
            )
    finally:
        alignment.quantize = original_quantize

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
