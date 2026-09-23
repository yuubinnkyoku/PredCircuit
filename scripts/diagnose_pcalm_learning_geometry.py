from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from predcircuit.pcalm import ResidualMLP, Schedule, gradient_cosine, method_grad


def apply_grads(model: ResidualMLP, grads: list[torch.Tensor], lr: float) -> None:
    with torch.no_grad():
        for weight, grad in zip(model.weights, grads, strict=True):
            weight.add_(grad, alpha=-lr)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--state-lr", type=float, default=0.30)
    p.add_argument("--budget", type=int, default=56)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    depth, width, input_dim, output_dim = 32, 8, 8, 4
    batch_size, train_size, updates = 4, 64, 24
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
    schedule = Schedule("pcalm", budget=args.budget, alpha=0.925)
    rows: list[dict[str, object]] = []

    # Keep the diagnostic batch fixed across checkpoints. The previous version
    # diagnosed each checkpoint on its current training minibatch, confounding
    # learning-time drift with minibatch-to-minibatch gradient variation.
    probe_x = train_x[:batch_size]
    probe_y = train_y[:batch_size]

    for update in range(updates + 1):
        start = (update * batch_size) % train_size
        indices = torch.arange(start, start + batch_size) % train_size
        x, y = train_x[indices], train_y[indices]
        if update in {0, 4, 8, 16, 24}:
            bp = method_grad(
                model,
                probe_x,
                probe_y,
                Schedule("bp", budget=0),
                state_lr=args.state_lr,
                rho=1.0,
            )
            pc = method_grad(
                model,
                probe_x,
                probe_y,
                schedule,
                state_lr=args.state_lr,
                rho=1.0,
            )
            global_cosine = gradient_cosine(pc, bp)
            for layer, (pg, bg) in enumerate(zip(pc, bp, strict=True)):
                rows.append(
                    {
                        "seed": args.seed,
                        "update": update,
                        "layer": layer,
                        "layer_grad_cosine_bp": gradient_cosine([pg], [bg]),
                        "layer_grad_norm_ratio_bp": float(
                            pg.norm() / bg.norm().clamp_min(torch.finfo(bg.dtype).eps)
                        ),
                        "global_grad_cosine_bp": global_cosine,
                    }
                )
        if update == updates:
            break
        grads = method_grad(model, x, y, schedule, state_lr=args.state_lr, rho=1.0)
        apply_grads(model, grads, 0.01)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame[frame["layer"] == 0].to_string(index=False))


if __name__ == "__main__":
    main()
