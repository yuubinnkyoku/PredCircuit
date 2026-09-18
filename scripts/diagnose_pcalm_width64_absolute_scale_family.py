from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from diagnose_pcalm_width64_update_lattice_alignment import fixed_lsb, run_alignment
from predcircuit.pcalm import (
    ResidualMLP,
    Schedule,
    gradient_cosine,
    gradient_relative_error,
    method_grad,
)

# Same state/update LSB ratio at three absolute grid scales.  The middle pair is
# the already-tested fixed15_i3/fixed14_i1 regime; this experiment adds the
# matched coarse and fine scales while recording state-motion diagnostics.
SCALE_CONFIGS = {
    "coarse": ("fixed14_i3", "fixed13_i1"),
    "middle": ("fixed15_i3", "fixed14_i1"),
    "fine": ("fixed16_i3", "fixed15_i1"),
}
STATE_LRS = (0.234285, 0.25)
DUAL_PRECISION = "fixed12_i1"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matched-R absolute lattice-scale test for leaky width-64 PC-ALM."
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
        depth=depth, width=width, input_dim=8, output_dim=4, activation="relu", seed=model_seed
    )
    bp = method_grad(bp_model, x, y, Schedule("bp", budget=0), state_lr=0.25, rho=1.0)
    bp_first = bp[0]
    bp_norm = float(bp_first.norm())

    rows: list[dict[str, object]] = []
    for state_lr in STATE_LRS:
        effective_lr = state_lr * x.shape[0]
        for scale, (state_precision, update_precision) in SCALE_CONFIGS.items():
            state_lsb = fixed_lsb(state_precision)
            update_lsb = fixed_lsb(update_precision)
            assert state_lsb is not None and update_lsb is not None
            lattice_ratio = effective_lr * update_lsb / (0.5 * state_lsb)

            model = ResidualMLP(
                depth=depth,
                width=width,
                input_dim=8,
                output_dim=4,
                activation="relu",
                seed=model_seed,
            )
            grads, stats = run_alignment(
                model,
                x,
                y,
                update_precision=update_precision,
                state_precision=state_precision,
                dual_precision=DUAL_PRECISION,
                budget=args.budget,
                state_lr=state_lr,
                dual_leak=args.dual_leak,
            )
            first = grads[0]
            ratio = float(first.norm()) / bp_norm if bp_norm else math.nan
            cosine = gradient_cosine([first], [bp_first])
            relative_error = gradient_relative_error([first], [bp_first])
            finite = bool(stats["finite"])
            rows.append(
                {
                    "seed": args.seed,
                    "scale": scale,
                    "budget": args.budget,
                    "state_lr": state_lr,
                    "effective_lr": effective_lr,
                    "dual_leak": args.dual_leak,
                    "state_precision": state_precision,
                    "update_precision": update_precision,
                    "dual_precision": DUAL_PRECISION,
                    "state_lsb": state_lsb,
                    "update_lsb": update_lsb,
                    "lattice_ratio_r": lattice_ratio,
                    **stats,
                    "first_layer_cosine_to_bp": cosine,
                    "first_layer_grad_norm_ratio_to_bp": ratio,
                    "first_layer_relative_error_to_bp": relative_error,
                    "useful_first_layer_credit": finite
                    and cosine >= 0.9
                    and 0.5 <= ratio <= 2.0
                    and relative_error <= 0.6,
                }
            )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
