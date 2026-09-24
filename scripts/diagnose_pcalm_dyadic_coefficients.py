from __future__ import annotations

import argparse
import math
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

PRECISIONS = {
    "state14_update14_dual12": ("fixed14_i3", "fixed14_i1", "fixed12_i1"),
    "state15_update13_dual12": ("fixed15_i3", "fixed13_i1", "fixed12_i1"),
    "state16_update14_dual12": ("fixed16_i3", "fixed14_i1", "fixed12_i1"),
}
COEFFICIENTS = {
    "original": (0.234285, 0.925, 0.02),
    "dyadic": (15.0 / 64.0, 59.0 / 64.0, 5.0 / 256.0),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate multiplierless PC-ALM coefficients on width-64 fixed-point holdouts."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=256)
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
        state_lr=COEFFICIENTS["original"][0],
        rho=1.0,
    )
    bp_first = bp[0]
    bp_norm = float(bp_first.norm())

    rows: list[dict[str, object]] = []
    for precision_name, (state_precision, update_precision, dual_precision) in PRECISIONS.items():
        reference_grads: list[torch.Tensor] | None = None
        for coefficient_name, (state_lr, alpha, dual_leak) in COEFFICIENTS.items():
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
                dual_precision=dual_precision,
                budget=args.budget,
                state_lr=state_lr,
                dual_leak=dual_leak,
                alpha=alpha,
            )
            if reference_grads is None:
                reference_grads = grads
            first = grads[0]
            ratio = float(first.norm()) / bp_norm if bp_norm else math.nan
            cosine = gradient_cosine([first], [bp_first])
            relative_error = gradient_relative_error([first], [bp_first])
            finite = bool(stats["finite"])
            rows.append(
                {
                    "seed": args.seed,
                    "config": f"{precision_name}_{coefficient_name}",
                    "precision": precision_name,
                    "coefficients": coefficient_name,
                    "state_lr": state_lr,
                    "alpha": alpha,
                    "dual_leak": dual_leak,
                    "finite": finite,
                    "useful_first_layer_credit": finite
                    and cosine >= 0.9
                    and 0.5 <= ratio <= 2.0
                    and relative_error <= 0.6,
                    "first_layer_cosine_to_bp": cosine,
                    "first_layer_grad_norm_ratio_to_bp": ratio,
                    "first_layer_relative_error_to_bp": relative_error,
                    "all_gradient_relative_error_to_original": gradient_relative_error(
                        grads, reference_grads
                    ),
                    **stats,
                }
            )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
