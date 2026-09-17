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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test whether reallocating 14-bit state/update range to fractional precision closes the leaky PC-ALM gap."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--state-lr", type=float, default=0.234285)
    parser.add_argument("--dual-leak", type=float, default=0.02)
    args = parser.parse_args()

    depth = 32
    width = 64
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

    # Keep total width fixed at 14 bits.  The baseline has no observed saturation,
    # so move one range bit at a time into the fractional field before considering
    # a wider datapath.  The 12-bit dual format is intentionally unchanged.
    configs = {
        "fp32_leaky": ("fp32", "fp32", "fp32"),
        "baseline_u14i2_s14i3_d12i1": ("fixed14_i2", "fixed14_i3", "fixed12_i1"),
        "state_frac_plus1": ("fixed14_i2", "fixed14_i2", "fixed12_i1"),
        "update_frac_plus1": ("fixed14_i1", "fixed14_i3", "fixed12_i1"),
        "both_frac_plus1": ("fixed14_i1", "fixed14_i2", "fixed12_i1"),
    }

    rows: list[dict[str, object]] = []
    fp32_grads: list[torch.Tensor] | None = None
    for name, (update_precision, state_precision, dual_precision) in configs.items():
        model = ResidualMLP(
            depth=depth, width=width, input_dim=8, output_dim=4, activation="relu", seed=model_seed
        )
        grads, stats = run(
            model,
            x,
            y,
            update_precision=update_precision,
            state_precision=state_precision,
            dual_precision=dual_precision,
            budget=args.budget,
            state_lr=args.state_lr,
            dual_leak=args.dual_leak,
        )
        if fp32_grads is None:
            fp32_grads = grads
        first = grads[0]
        ratio = float(first.norm()) / bp_norm if bp_norm else math.nan
        cosine = gradient_cosine([first], [bp_first])
        relative_error = gradient_relative_error([first], [bp_first])
        finite = bool(stats["finite"])
        rows.append(
            {
                "seed": args.seed,
                "config": name,
                "budget": args.budget,
                "state_lr": args.state_lr,
                "dual_leak": args.dual_leak,
                "update_precision": update_precision,
                "state_precision": state_precision,
                "dual_precision": dual_precision,
                **stats,
                "useful_first_layer_credit": finite
                and cosine >= 0.9
                and 0.5 <= ratio <= 2.0
                and relative_error <= 0.6,
                "first_layer_cosine_to_bp": cosine,
                "first_layer_grad_norm_ratio_to_bp": ratio,
                "first_layer_relative_error_to_bp": relative_error,
                "all_gradient_relative_error_to_fp32": gradient_relative_error(grads, fp32_grads),
            }
        )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
