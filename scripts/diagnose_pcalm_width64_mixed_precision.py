from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from diagnose_pcalm_width64_state_ablation import run_config
from predcircuit.pcalm import ResidualMLP, Schedule, gradient_cosine, gradient_relative_error, method_grad

CONFIGS = {
    "state12_update14": ("fixed14_i2", "fixed12_i3"),
    "state14_update12": ("fixed12_i2", "fixed14_i3"),
    "state14_update14": ("fixed14_i2", "fixed14_i3"),
    "state16_update12": ("fixed12_i2", "fixed16_i3"),
    "state12_update16": ("fixed16_i2", "fixed12_i3"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe mixed state/update precision with 12-bit duals at width 64."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=128)
    args = parser.parse_args()

    depth = 32
    width = 64
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
        state_lr=0.25,
        rho=1.0,
    )
    bp_first = bp[0]
    bp_norm = float(bp_first.norm())

    rows: list[dict[str, object]] = []
    for name, (update_precision, state_precision) in CONFIGS.items():
        model = ResidualMLP(
            depth=depth,
            width=width,
            input_dim=8,
            output_dim=4,
            activation="relu",
            seed=model_seed,
        )
        grads, residual, finite = run_config(
            model,
            x,
            y,
            update_precision=update_precision,
            state_precision=state_precision,
            dual_precision="fixed12_i1",
            budget=args.budget,
        )
        first = grads[0]
        ratio = float(first.norm()) / bp_norm if bp_norm else math.nan
        cosine = gradient_cosine([first], [bp_first])
        relative_error = gradient_relative_error([first], [bp_first])
        rows.append(
            {
                "seed": args.seed,
                "config": name,
                "update_precision": update_precision,
                "state_precision": state_precision,
                "dual_precision": "fixed12_i1",
                "finite": finite,
                "useful_first_layer_credit": finite
                and cosine >= 0.9
                and 0.5 <= ratio <= 2.0
                and relative_error <= 0.6,
                "first_layer_cosine_to_bp": cosine,
                "first_layer_grad_norm_ratio_to_bp": ratio,
                "first_layer_relative_error_to_bp": relative_error,
                "residual_total": residual,
            }
        )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
