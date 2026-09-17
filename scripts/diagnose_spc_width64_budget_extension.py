from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from diagnose_pcalm_matched_credit_budget import credit_metrics
from diagnose_pcalm_width64_budget import parse_ints
from predcircuit.pcalm import ResidualMLP, Schedule, method_grad


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find the width-64 sPC relaxation budget needed for useful deep credit."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--budgets", default="320,384,512,768,1024")
    parser.add_argument("--state-lr", type=float, default=0.25)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    input_dim, output_dim, batch_size = 8, 4, 4
    model_seed = args.seed + 1000 * args.width + args.depth
    data_seed = args.seed + 10_000 + 1000 * args.width + args.depth
    generator = torch.Generator().manual_seed(data_seed)
    x = torch.randn(batch_size, input_dim, generator=generator)
    y = torch.randn(batch_size, output_dim, generator=generator)

    model = ResidualMLP(
        depth=args.depth,
        width=args.width,
        input_dim=input_dim,
        output_dim=output_dim,
        activation="relu",
        seed=model_seed,
    )
    bp = method_grad(
        model,
        x,
        y,
        Schedule("bp", budget=0),
        state_lr=args.state_lr,
        rho=args.rho,
    )

    rows: list[dict[str, object]] = []
    for budget in parse_ints(args.budgets):
        grad = method_grad(
            model,
            x,
            y,
            Schedule("pc", budget=budget),
            state_lr=args.state_lr,
            rho=args.rho,
        )
        finite = all(bool(torch.isfinite(g).all()) for g in grad)
        cosine, ratio, relerr, useful = credit_metrics(grad, bp, finite=finite)
        rows.append(
            {
                "seed": args.seed,
                "width": args.width,
                "depth": args.depth,
                "method": "spc",
                "budget": budget,
                "finite": finite,
                "useful_first_layer_credit": useful,
                "first_layer_cosine_to_bp": cosine,
                "first_layer_grad_norm_ratio_to_bp": ratio,
                "first_layer_relative_error_to_bp": relerr,
            }
        )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
