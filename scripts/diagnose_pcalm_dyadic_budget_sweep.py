from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from diagnose_pcalm_width64_update_lattice_alignment import run_alignment
from predcircuit.pcalm import (
    ResidualMLP,
    Schedule,
    gradient_cosine,
    gradient_relative_error,
    method_grad,
)

BUDGETS = (64, 96, 128, 160, 192, 224, 256)
STATE_LR = 15.0 / 64.0
ALPHA = 59.0 / 64.0
DUAL_LEAK = 5.0 / 256.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find the minimum useful relaxation budget for the width-64 dyadic PC-ALM candidate."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
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
    bp = method_grad(
        bp_model,
        x,
        y,
        Schedule("bp", budget=0),
        state_lr=STATE_LR,
        rho=1.0,
    )
    bp_first = bp[0]
    bp_norm = float(bp_first.norm())

    rows: list[dict[str, object]] = []
    for budget in BUDGETS:
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
            update_precision="fixed14_i1",
            state_precision="fixed16_i3",
            dual_precision="fixed12_i1",
            budget=budget,
            state_lr=STATE_LR,
            dual_leak=DUAL_LEAK,
            alpha=ALPHA,
        )
        first = grads[0]
        cosine = gradient_cosine([first], [bp_first])
        ratio = float(first.norm()) / bp_norm if bp_norm else float("nan")
        relative_error = gradient_relative_error([first], [bp_first])
        finite = bool(stats["finite"])
        rows.append(
            {
                "seed": args.seed,
                "budget": budget,
                "finite": finite,
                "useful_first_layer_credit": finite
                and cosine >= 0.9
                and 0.5 <= ratio <= 2.0
                and relative_error <= 0.6,
                "first_layer_cosine_to_bp": cosine,
                "first_layer_grad_norm_ratio_to_bp": ratio,
                "first_layer_relative_error_to_bp": relative_error,
                **stats,
            }
        )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
