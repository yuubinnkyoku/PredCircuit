from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from predcircuit.magnitude_control import pcalm_leak_grad
from predcircuit.pcalm import ResidualMLP, Schedule, gradient_cosine, gradient_relative_error, method_grad


def clone_model(model: ResidualMLP) -> ResidualMLP:
    clone = ResidualMLP(
        depth=32,
        width=8,
        input_dim=8,
        output_dim=4,
        activation="relu",
        seed=0,
    )
    with torch.no_grad():
        for dst, src in zip(clone.weights, model.weights, strict=True):
            dst.copy_(src)
    return clone


def apply_grads(model: ResidualMLP, grads: list[torch.Tensor], lr: float) -> None:
    with torch.no_grad():
        for weight, grad in zip(model.weights, grads, strict=True):
            weight.add_(grad, alpha=-lr)


def mse(model: ResidualMLP, x: torch.Tensor, y: torch.Tensor) -> float:
    pred = model.forward_free(x)[-1]
    return float((0.5 * (pred - y).square().sum(dim=1)).mean())


def metrics(grads: list[torch.Tensor], bp: list[torch.Tensor]) -> tuple[float, float, float]:
    eps = torch.finfo(bp[0].dtype).eps
    ratio = float(grads[0].norm() / bp[0].norm().clamp_min(eps))
    return (
        gradient_cosine(grads, bp),
        ratio,
        gradient_relative_error([grads[0]], [bp[0]]),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--updates", type=int, default=64)
    p.add_argument("--budget", type=int, default=112)
    p.add_argument("--state-lr", type=float, default=0.25)
    p.add_argument("--weight-lr", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=0.925)
    p.add_argument("--dual-leak", type=float, default=0.01)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    depth, width, input_dim, output_dim = 32, 8, 8, 4
    batch_size, train_size, eval_size = 4, 64, 256
    gen = torch.Generator().manual_seed(args.seed + 90_000)
    teacher = torch.randn(input_dim, output_dim, generator=gen) / input_dim**0.5
    train_x = torch.randn(train_size, input_dim, generator=gen)
    train_y = torch.tanh(train_x @ teacher)
    eval_x = torch.randn(eval_size, input_dim, generator=gen)
    eval_y = torch.tanh(eval_x @ teacher)

    base = ResidualMLP(
        depth=depth,
        width=width,
        input_dim=input_dim,
        output_dim=output_dim,
        activation="relu",
        seed=args.seed + 1000 * width + depth,
    )
    models = {name: clone_model(base) for name in ("bp", "pcalm", "leaky")}
    initial_loss = mse(base, eval_x, eval_y)
    probe_x, probe_y = train_x[:batch_size], train_y[:batch_size]
    checkpoints = {0, 8, 16, 24, 32, 48, args.updates}
    rows: list[dict[str, object]] = []

    for update in range(args.updates + 1):
        if update in checkpoints:
            for name, model in models.items():
                bp = method_grad(model, probe_x, probe_y, Schedule("bp", budget=0), state_lr=0.0, rho=1.0)
                max_abs_dual = 0.0
                finite = True
                if name == "bp":
                    grads = bp
                elif name == "pcalm":
                    grads = method_grad(
                        model,
                        probe_x,
                        probe_y,
                        Schedule("pcalm", budget=args.budget, alpha=args.alpha),
                        state_lr=args.state_lr,
                        rho=1.0,
                    )
                else:
                    grads, max_abs_dual, finite = pcalm_leak_grad(
                        model,
                        probe_x,
                        probe_y,
                        state_lr=args.state_lr,
                        rho=1.0,
                        alpha=args.alpha,
                        dual_leak=args.dual_leak,
                        budget=args.budget,
                    )
                cosine, first_ratio, first_rel = metrics(grads, bp)
                loss = mse(model, eval_x, eval_y)
                rows.append(
                    {
                        "seed": args.seed,
                        "update": update,
                        "method": name,
                        "eval_loss": loss,
                        "relative_loss_reduction": (initial_loss - loss) / initial_loss,
                        "global_grad_cosine_bp": cosine,
                        "first_layer_grad_norm_ratio_bp": first_ratio,
                        "first_layer_relative_error_bp": first_rel,
                        "max_abs_dual": max_abs_dual,
                        "finite": finite and all(bool(torch.isfinite(g).all()) for g in grads),
                    }
                )
        if update == args.updates:
            break

        start = (update * batch_size) % train_size
        indices = torch.arange(start, start + batch_size) % train_size
        x, y = train_x[indices], train_y[indices]
        for name, model in models.items():
            if name == "bp":
                grads = method_grad(model, x, y, Schedule("bp", budget=0), state_lr=0.0, rho=1.0)
            elif name == "pcalm":
                grads = method_grad(
                    model,
                    x,
                    y,
                    Schedule("pcalm", budget=args.budget, alpha=args.alpha),
                    state_lr=args.state_lr,
                    rho=1.0,
                )
            else:
                grads, _max_abs_dual, _finite = pcalm_leak_grad(
                    model,
                    x,
                    y,
                    state_lr=args.state_lr,
                    rho=1.0,
                    alpha=args.alpha,
                    dual_leak=args.dual_leak,
                    budget=args.budget,
                )
            apply_grads(model, grads, args.weight_lr)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
