from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from predcircuit.epc import epc_grad
from predcircuit.pcalm import ResidualMLP, Schedule, bp_loss, method_grad


def apply_grads(model: ResidualMLP, grads: list[torch.Tensor], lr: float) -> None:
    with torch.no_grad():
        for weight, grad in zip(model.weights, grads, strict=True):
            weight.add_(grad, alpha=-lr)


def apply_epc_t1_grads(
    model: ResidualMLP, grads: list[torch.Tensor], *, lr: float, error_lr: float
) -> None:
    if error_lr <= 0.0:
        raise ValueError("T=1 ePC compensation requires error_lr > 0")
    with torch.no_grad():
        for layer_ix, (weight, grad) in enumerate(zip(model.weights, grads, strict=True)):
            layer_lr = lr / error_lr if layer_ix < model.depth - 1 else lr
            weight.add_(grad, alpha=-layer_lr)


def eval_loss(model: ResidualMLP, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float(bp_loss(model, x, y))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32, 56, 112])
    p.add_argument("--depth", type=int, default=32)
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--updates", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--train-size", type=int, default=64)
    p.add_argument("--eval-size", type=int, default=256)
    p.add_argument("--weight-lr", type=float, default=0.01)
    p.add_argument("--state-lr", type=float, default=0.25)
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.925)
    p.add_argument("--epc-error-lr", type=float, default=0.3)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if any(budget <= 0 for budget in args.budgets):
        raise ValueError("all PC-ALM budgets must be positive")

    input_dim, output_dim = 8, 4
    data_gen = torch.Generator().manual_seed(args.seed + 90_000)
    teacher = torch.randn(input_dim, output_dim, generator=data_gen) / input_dim**0.5
    train_x = torch.randn(args.train_size, input_dim, generator=data_gen)
    eval_x = torch.randn(args.eval_size, input_dim, generator=data_gen)
    train_y = torch.tanh(train_x @ teacher)
    eval_y = torch.tanh(eval_x @ teacher)

    base = ResidualMLP(
        depth=args.depth,
        width=args.width,
        input_dim=input_dim,
        output_dim=output_dim,
        activation="relu",
        seed=args.seed + 1000 * args.width + args.depth,
    )
    names = ["bp", "epc", *[f"pcalm_t{budget}" for budget in args.budgets]]
    models: dict[str, ResidualMLP] = {}
    for name in names:
        model = ResidualMLP(
            depth=args.depth,
            width=args.width,
            input_dim=input_dim,
            output_dim=output_dim,
            activation="relu",
            seed=0,
        )
        model.load_state_dict(base.state_dict())
        models[name] = model

    rows: list[dict[str, object]] = []
    for update in range(args.updates + 1):
        for name, model in models.items():
            budget = 0 if name == "bp" else 1 if name == "epc" else int(name.removeprefix("pcalm_t"))
            rows.append(
                {
                    "seed": args.seed,
                    "method": name,
                    "update": update,
                    "eval_loss": eval_loss(model, eval_x, eval_y),
                    "budget_per_update": budget,
                    "cumulative_relaxation_steps": update * budget,
                }
            )
        if update == args.updates:
            break

        start = (update * args.batch_size) % args.train_size
        indices = torch.arange(start, start + args.batch_size) % args.train_size
        x = train_x[indices]
        y = train_y[indices]

        bp = method_grad(
            models["bp"], x, y, Schedule("bp", budget=0), state_lr=args.state_lr, rho=args.rho
        )
        apply_grads(models["bp"], bp, args.weight_lr)

        epc, epc_finite = epc_grad(models["epc"], x, y, error_lr=args.epc_error_lr, steps=1)
        if not epc_finite:
            raise RuntimeError("ePC became non-finite")
        apply_epc_t1_grads(models["epc"], epc, lr=args.weight_lr, error_lr=args.epc_error_lr)

        for budget in args.budgets:
            name = f"pcalm_t{budget}"
            grads = method_grad(
                models[name],
                x,
                y,
                Schedule("pcalm", budget=budget, alpha=args.alpha),
                state_lr=args.state_lr,
                rho=args.rho,
            )
            if not all(torch.isfinite(grad).all() for grad in grads):
                raise RuntimeError(f"PC-ALM T={budget} became non-finite")
            apply_grads(models[name], grads, args.weight_lr)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    final = frame[frame["update"] == args.updates].copy()
    initial = frame[frame["update"] == 0].set_index("method")["eval_loss"]
    final["loss_ratio_to_initial"] = [
        row.eval_loss / initial[row.method] for row in final.itertuples()
    ]
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()
