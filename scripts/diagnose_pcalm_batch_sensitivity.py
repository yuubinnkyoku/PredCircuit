from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from predcircuit.pcalm import (
    ResidualMLP,
    Schedule,
    gradient_cosine,
    method_grad,
    run_pcalm,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--state-lr", type=float, default=0.30)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    depth, width, input_dim, output_dim = 32, 8, 8, 4
    batch_size, train_size = 4, 64
    budgets = (32, 56, 80, 112)
    gen = torch.Generator().manual_seed(args.seed + 90_000)
    teacher = torch.randn(input_dim, output_dim, generator=gen) / input_dim**0.5
    train_x = torch.randn(train_size, input_dim, generator=gen)
    train_y = torch.tanh(train_x @ teacher)
    model = ResidualMLP(
        depth=depth,
        width=width,
        input_dim=input_dim,
        output_dim=output_dim,
        activation="relu",
        seed=args.seed + 1000 * width + depth,
    )

    rows: list[dict[str, object]] = []
    for batch_ix, start in enumerate(range(0, train_size, batch_size)):
        x = train_x[start : start + batch_size]
        y = train_y[start : start + batch_size]
        bp = method_grad(
            model,
            x,
            y,
            Schedule("bp", budget=0),
            state_lr=args.state_lr,
            rho=1.0,
        )
        for budget in budgets:
            schedule = Schedule("pcalm", budget=budget, alpha=0.925)
            pc = method_grad(model, x, y, schedule, state_lr=args.state_lr, rho=1.0)
            _, _, trace = run_pcalm(
                model,
                x,
                y,
                state_lr=args.state_lr,
                rho=1.0,
                alpha=0.925,
                budget=budget,
                record_trace=True,
            )
            assert trace is not None
            residual_final = trace.residual_norms[-1]
            dual_final = trace.dual_norms[-1]
            rows.append(
                {
                    "seed": args.seed,
                    "batch": batch_ix,
                    "budget": budget,
                    "global_grad_cosine_bp": gradient_cosine(pc, bp),
                    "layer0_grad_cosine_bp": gradient_cosine([pc[0]], [bp[0]]),
                    "layer0_grad_norm_ratio_bp": float(
                        pc[0].norm() / bp[0].norm().clamp_min(torch.finfo(bp[0].dtype).eps)
                    ),
                    "residual_l2_sum": sum(residual_final),
                    "residual_l2_max": max(residual_final),
                    "dual_l2_sum": sum(dual_final),
                    "max_abs_dual": trace.max_abs_dual[-1],
                    "finite": trace.finite,
                }
            )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
