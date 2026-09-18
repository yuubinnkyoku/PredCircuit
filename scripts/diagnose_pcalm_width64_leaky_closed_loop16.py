from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from diagnose_pcalm_width64_leaky_mixed_precision import run
from predcircuit.pcalm import (
    ResidualMLP,
    Schedule,
    gradient_cosine,
    gradient_relative_error,
    method_grad,
)

STATE_LRS = (0.234285, 0.25, 0.26)
UPDATE_PRECISION = "fixed16_i2"
STATE_PRECISION = "fixed16_i3"
DUAL_PRECISION = "fixed12_i1"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test the 16/16/12 leaky PC-ALM closed loop on both sides of the "
            "predicted state/update lattice boundary."
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--dual-leak", type=float, default=0.02)
    args = parser.parse_args()

    depth, width = 32, 64
    model_seed = args.seed + 1000 * width + depth
    data_seed = args.seed + 10_000 + 1000 * width + depth
    generator = torch.Generator().manual_seed(data_seed)
    x = torch.randn(4, 8, generator=generator)
    y = torch.randn(4, 4, generator=generator)

    bp_model = ResidualMLP(
        depth=depth,
        width=width,
        input_dim=8,
        output_dim=4,
        activation="relu",
        seed=model_seed,
    )
    bp = method_grad(bp_model, x, y, Schedule("bp", budget=0), state_lr=0.25, rho=1.0)
    bp_first = bp[0]
    bp_norm = float(bp_first.norm())

    # fixed16_i3 state has 12 fractional bits, so half an LSB is 2^-13.
    # fixed16_i2 update has 13 fractional bits, so one update LSB is also 2^-13.
    # With batch=4, R = eta_eff * delta_u / (delta_h / 2) = 4 * state_lr.
    state_half_lsb = 2.0**-13
    update_lsb = 2.0**-13

    rows: list[dict[str, object]] = []
    for state_lr in STATE_LRS:
        model = ResidualMLP(
            depth=depth,
            width=width,
            input_dim=8,
            output_dim=4,
            activation="relu",
            seed=model_seed,
        )
        grads, stats = run(
            model,
            x,
            y,
            update_precision=UPDATE_PRECISION,
            state_precision=STATE_PRECISION,
            dual_precision=DUAL_PRECISION,
            budget=args.budget,
            state_lr=state_lr,
            dual_leak=args.dual_leak,
        )
        first = grads[0]
        ratio = float(first.norm()) / bp_norm if bp_norm else math.nan
        cosine = gradient_cosine([first], [bp_first])
        relative_error = gradient_relative_error([first], [bp_first])
        effective_lr = state_lr * x.shape[0]
        lattice_ratio = effective_lr * update_lsb / state_half_lsb
        finite = bool(stats["finite"])
        rows.append(
            {
                "seed": args.seed,
                "budget": args.budget,
                "state_lr": state_lr,
                "effective_lr": effective_lr,
                "dual_leak": args.dual_leak,
                "update_precision": UPDATE_PRECISION,
                "state_precision": STATE_PRECISION,
                "dual_precision": DUAL_PRECISION,
                "update_lsb": update_lsb,
                "state_half_lsb": state_half_lsb,
                "lattice_ratio_r": lattice_ratio,
                "predicted_unlocked": lattice_ratio >= 1.0,
                **stats,
                "useful_first_layer_credit": finite
                and cosine >= 0.9
                and 0.5 <= ratio <= 2.0
                and relative_error <= 0.6,
                "first_layer_cosine_to_bp": cosine,
                "first_layer_grad_norm_ratio_to_bp": ratio,
                "first_layer_relative_error_to_bp": relative_error,
            }
        )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
