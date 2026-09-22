from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from predcircuit.epc import epc_grad
from predcircuit.magnitude_control import pcalm_leak_grad, spc_grad
from predcircuit.pcalm import ResidualMLP, Schedule, bp_loss, method_grad


def apply_grads(model: ResidualMLP, grads: list[torch.Tensor], lr: float) -> None:
    with torch.no_grad():
        for weight, grad in zip(model.weights, grads, strict=True):
            weight.add_(grad, alpha=-lr)


def eval_loss(model: ResidualMLP, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float(bp_loss(model, x, y))


def main() -> None:
    p = argparse.ArgumentParser(description="Compare actual multi-update learning, not only one-shot credit geometry.")
    p.add_argument("--seed", type=int, required=True)
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
    p.add_argument("--dual-leak", type=float, default=0.01)
    p.add_argument("--pcalm-budget", type=int, default=112)
    p.add_argument("--spc-budget", type=int, default=1024)
    p.add_argument("--epc-budget", type=int, default=1)
    p.add_argument("--epc-error-lr", type=float, default=0.3)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

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
    methods = ("bp", "spc", "pcalm", "pcalm_leak", "epc")
    models: dict[str, ResidualMLP] = {}
    for name in methods:
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

    budgets = {
        "bp": 0,
        "spc": args.spc_budget,
        "pcalm": args.pcalm_budget,
        "pcalm_leak": args.pcalm_budget,
        "epc": args.epc_budget,
    }
    rows: list[dict[str, object]] = []

    for update in range(args.updates + 1):
        for name, model in models.items():
            rows.append(
                {
                    "seed": args.seed,
                    "method": name,
                    "update": update,
                    "eval_loss": eval_loss(model, eval_x, eval_y),
                    "budget_per_update": budgets[name],
                    "cumulative_relaxation_steps": update * budgets[name],
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

        spc, _residuals, spc_finite = spc_grad(
            models["spc"], x, y, state_lr=args.state_lr, rho=args.rho, budget=args.spc_budget
        )
        if not spc_finite:
            raise RuntimeError("sPC became non-finite")
        apply_grads(models["spc"], spc, args.weight_lr)

        pcalm = method_grad(
            models["pcalm"],
            x,
            y,
            Schedule("pcalm", budget=args.pcalm_budget, alpha=args.alpha),
            state_lr=args.state_lr,
            rho=args.rho,
        )
        apply_grads(models["pcalm"], pcalm, args.weight_lr)

        leak, _max_dual, leak_finite = pcalm_leak_grad(
            models["pcalm_leak"],
            x,
            y,
            state_lr=args.state_lr,
            rho=args.rho,
            alpha=args.alpha,
            dual_leak=args.dual_leak,
            budget=args.pcalm_budget,
        )
        if not leak_finite:
            raise RuntimeError("leaky PC-ALM became non-finite")
        apply_grads(models["pcalm_leak"], leak, args.weight_lr)

        epc, epc_finite = epc_grad(
            models["epc"], x, y, error_lr=args.epc_error_lr, steps=args.epc_budget
        )
        if not epc_finite:
            raise RuntimeError("ePC became non-finite")
        # At T=1, ePC's first-layer credit is scaled by error_lr relative to BP.
        # Compensate the global weight step so this baseline is not penalized by that trivial scale.
        apply_grads(models["epc"], epc, args.weight_lr / args.epc_error_lr)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    final = frame[frame["update"] == args.updates].copy()
    initial = frame[frame["update"] == 0].set_index("method")["eval_loss"]
    final["loss_ratio_to_initial"] = [row.eval_loss / initial[row.method] for row in final.itertuples()]
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()
