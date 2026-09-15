from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from predcircuit.epc import epc_grad
from predcircuit.pcalm import (
    ResidualMLP,
    Schedule,
    gradient_cosine,
    gradient_relative_error,
    method_grad,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure deep ePC credit against BP on the shared ResidualMLP."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--output-dim", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--budgets", default="1,2,4,8,16,32,64,96,128")
    parser.add_argument("--error-lrs", default="0.03,0.1,0.3")
    parser.add_argument("--activation", choices=["linear", "tanh", "relu"], default="relu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    budgets = [int(x) for x in args.budgets.split(",")]
    error_lrs = [float(x) for x in args.error_lrs.split(",")]
    model = ResidualMLP(
        depth=args.depth,
        width=args.width,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        activation=args.activation,
        seed=args.seed + args.depth,
    )
    gen = torch.Generator().manual_seed(args.seed + 10_000 + args.depth)
    x = torch.randn(args.batch_size, args.input_dim, generator=gen)
    y = torch.randn(args.batch_size, args.output_dim, generator=gen)
    bp = method_grad(model, x, y, Schedule("bp", budget=0), state_lr=0.25, rho=1.0)
    bp_first = bp[0]
    bp_norm = float(bp_first.norm())

    rows = []
    for error_lr in error_lrs:
        for budget in budgets:
            grads, finite = epc_grad(model, x, y, error_lr=error_lr, steps=budget)
            first = grads[0]
            norm = float(first.norm())
            ratio = norm / bp_norm if bp_norm > 0 else math.nan
            cosine = gradient_cosine([first], [bp_first])
            rel = gradient_relative_error([first], [bp_first])
            useful = finite and cosine >= 0.9 and 0.5 <= ratio <= 2.0 and rel <= 0.6
            rows.append(
                {
                    "seed": args.seed,
                    "budget": budget,
                    "error_lr": error_lr,
                    "finite": finite,
                    "useful_first_layer_credit": useful,
                    "first_layer_cosine_to_bp": cosine,
                    "first_layer_grad_norm_ratio_to_bp": ratio,
                    "first_layer_relative_error_to_bp": rel,
                }
            )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
