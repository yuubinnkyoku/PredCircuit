from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from predcircuit.magnitude_control import credit_metrics, pcalm_leak_grad
from predcircuit.pcalm import ResidualMLP, Schedule, method_grad


def parse_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part]


def parse_floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part]


def main() -> None:
    p = argparse.ArgumentParser(
        description="PC-ALM dual-leak under official Sakana depth-specific state_lr."
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--depth", type=int, default=32)
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--state-lrs",
        default="0.234285,0.25",
        help="Official Sakana depth-32 state_lr first, project default second.",
    )
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.925)
    p.add_argument("--dual-leaks", default="0,0.005,0.01,0.02")
    p.add_argument("--budgets", default="128,256")
    p.add_argument("--activation", choices=["linear", "tanh", "relu"], default="relu")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    state_lrs = parse_floats(a.state_lrs)
    leaks = parse_floats(a.dual_leaks)
    budgets = parse_ints(a.budgets)

    model = ResidualMLP(
        depth=a.depth,
        width=a.width,
        input_dim=8,
        output_dim=4,
        activation=a.activation,
        seed=a.seed + a.depth,
    )
    gen = torch.Generator().manual_seed(a.seed + 10_000 + a.depth)
    x = torch.randn(a.batch_size, 8, generator=gen)
    y = torch.randn(a.batch_size, 4, generator=gen)
    # BP is independent of state_lr for the supervised loss path.
    bp = method_grad(model, x, y, Schedule("bp", budget=0), state_lr=0.25, rho=a.rho)

    rows: list[dict[str, float | int | bool]] = []
    for state_lr in state_lrs:
        for dual_leak in leaks:
            for budget in budgets:
                grads, max_abs_dual, finite = pcalm_leak_grad(
                    model,
                    x,
                    y,
                    state_lr=state_lr,
                    rho=a.rho,
                    alpha=a.alpha,
                    dual_leak=dual_leak,
                    budget=budget,
                )
                metrics = credit_metrics(grads, bp)
                metrics["finite"] = bool(metrics["finite"] and finite)
                if not metrics["finite"]:
                    metrics["useful_first_layer_credit"] = False
                rows.append(
                    {
                        "seed": a.seed,
                        "state_lr": state_lr,
                        "state_lr_label": (
                            "sakana_depth32" if abs(state_lr - 0.234285) < 1e-6 else "project_0.25"
                        ),
                        "dual_leak": dual_leak,
                        "budget": budget,
                        "max_abs_dual": max_abs_dual,
                        **metrics,
                    }
                )

    frame = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(a.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
