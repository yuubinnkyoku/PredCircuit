from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from predcircuit.pcalm import (
    ResidualMLP,
    Schedule,
    al_energy_shifted,
    constraint_residuals,
    gradient_cosine,
    method_grad,
    run_pcalm,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--checkpoints", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 56, 80, 112])
    p.add_argument("--depth", type=int, default=32)
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--state-lr", type=float, default=0.25)
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.925)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    input_dim, output_dim = 8, 4
    gen = torch.Generator().manual_seed(args.seed + 90_000)
    teacher = torch.randn(input_dim, output_dim, generator=gen) / input_dim**0.5
    x = torch.randn(args.batch_size, input_dim, generator=gen)
    y = torch.tanh(x @ teacher)
    model = ResidualMLP(
        depth=args.depth,
        width=args.width,
        input_dim=input_dim,
        output_dim=output_dim,
        activation="relu",
        seed=args.seed + 1000 * args.width + args.depth,
    )
    bp = method_grad(model, x, y, Schedule("bp", budget=0), state_lr=args.state_lr, rho=args.rho)

    rows: list[dict[str, object]] = []
    for budget in args.checkpoints:
        free, duals, trace = run_pcalm(
            model,
            x,
            y,
            state_lr=args.state_lr,
            rho=args.rho,
            alpha=args.alpha,
            budget=budget,
            record_trace=True,
        )
        assert trace is not None
        detached_free = [z.detach() for z in free]
        detached_duals = [d.detach() for d in duals]
        loss = al_energy_shifted(model, x, y, detached_free, detached_duals, rho=args.rho)
        grads = [g.detach() for g in torch.autograd.grad(loss, tuple(model.weights))]
        residuals = constraint_residuals(model, x, detached_free)
        for layer, (residual, dual) in enumerate(zip(residuals, detached_duals, strict=True)):
            rows.append(
                {
                    "seed": args.seed,
                    "budget": budget,
                    "layer": layer,
                    "residual_norm": float(residual.norm()),
                    "dual_norm": float(dual.norm()),
                    "max_abs_dual": float(dual.abs().max()),
                    "layer_grad_cosine_bp": gradient_cosine([grads[layer]], [bp[layer]]),
                    "layer_grad_norm_ratio_bp": float(grads[layer].norm() / bp[layer].norm().clamp_min(torch.finfo(bp[layer].dtype).eps)),
                    "global_grad_cosine_bp": gradient_cosine(grads, bp),
                    "finite": trace.finite,
                }
            )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    first = frame[frame["layer"] == 0][["budget", "residual_norm", "dual_norm", "layer_grad_cosine_bp", "layer_grad_norm_ratio_bp", "global_grad_cosine_bp"]]
    print(first.to_string(index=False))


if __name__ == "__main__":
    main()
