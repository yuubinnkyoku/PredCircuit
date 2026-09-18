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

# Quantization-free worst-seed refinement of the useful-credit stability ceiling.
# The full 15-seed third refinement showed that seeds 856 and 857 already lose
# useful credit at 0.3162; the preceding 0.005-spaced sweep shows both are useful
# at 0.310 and fail at 0.315. Do not repeat those endpoints: resolve the interior.
STATE_LRS = [0.311, 0.312, 0.313, 0.314, 0.3145]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve the worst-seed continuous PC-ALM useful-credit ceiling between 0.310 and 0.315."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--dual-leak", type=float, default=0.02)
    args = parser.parse_args()

    depth, width, batch = 32, 64, 4
    model_seed = args.seed + 1000 * width + depth
    data_seed = args.seed + 10_000 + 1000 * width + depth
    generator = torch.Generator().manual_seed(data_seed)
    x = torch.randn(batch, 8, generator=generator)
    y = torch.randn(batch, 4, generator=generator)

    bp_model = ResidualMLP(
        depth=depth, width=width, input_dim=8, output_dim=4, activation="relu", seed=model_seed
    )
    bp = method_grad(bp_model, x, y, Schedule("bp", budget=0), state_lr=0.234285, rho=1.0)
    bp_first = bp[0]
    bp_norm = float(bp_first.norm())

    rows: list[dict[str, object]] = []
    for state_lr in STATE_LRS:
        model = ResidualMLP(
            depth=depth, width=width, input_dim=8, output_dim=4, activation="relu", seed=model_seed
        )
        grads, stats = run_alignment(
            model,
            x,
            y,
            update_precision="fp32",
            state_precision="fp32",
            dual_precision="fp32",
            budget=args.budget,
            state_lr=state_lr,
            dual_leak=args.dual_leak,
        )
        first = grads[0]
        cosine = gradient_cosine([first], [bp_first])
        relative_error = gradient_relative_error([first], [bp_first])
        ratio = float(first.norm()) / bp_norm if bp_norm else float("nan")
        finite = bool(stats["finite"])
        rows.append(
            {
                "seed": args.seed,
                "state_lr": state_lr,
                "effective_lr": state_lr * batch,
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
