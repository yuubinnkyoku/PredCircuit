from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from predcircuit.pcalm import (
    ResidualMLP,
    Schedule,
    _solve_inner,
    al_energy_shifted,
    constraint_residuals,
    free_init,
    gradient_cosine,
    gradient_relative_error,
    method_grad,
    zero_duals_like,
)


def parse_floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part]


def parse_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part]


def leaked_pcalm_grad(
    model: ResidualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    state_lr: float,
    rho: float,
    alpha: float,
    dual_leak: float,
    budget: int,
) -> tuple[list[torch.Tensor], float, float, bool]:
    if not 0.0 <= dual_leak < 1.0:
        raise ValueError("dual_leak must satisfy 0 <= dual_leak < 1")

    free = free_init(model, x)
    duals_before = zero_duals_like(constraint_residuals(model, x, free))
    max_abs_dual = 0.0
    finite = True

    for outer_ix in range(budget):
        free = _solve_inner(
            model,
            x,
            y,
            free,
            duals_before,
            state_lr=state_lr,
            rho=rho,
            steps=1,
        )
        residuals = constraint_residuals(model, x, free)
        duals_after = [
            ((1.0 - dual_leak) * lam + alpha * residual).detach()
            for lam, residual in zip(duals_before, residuals, strict=True)
        ]
        max_abs_dual = max(
            max_abs_dual,
            max((float(d.abs().max()) for d in duals_after), default=0.0),
        )
        finite = finite and all(
            bool(torch.isfinite(t).all()) for t in [*free, *residuals, *duals_after]
        )
        if outer_ix == budget - 1:
            credit_duals = duals_before
            break
        duals_before = duals_after
    else:
        raise AssertionError("unreachable")

    free_detached = [z.detach() for z in free]
    credit_detached = [d.detach() for d in credit_duals]
    loss = al_energy_shifted(model, x, y, free_detached, credit_detached, rho=rho)
    grads = torch.autograd.grad(loss, tuple(model.weights), allow_unused=False)
    grad_list = [g.detach().clone() for g in grads]
    residual_total = sum(float(r.norm()) for r in constraint_residuals(model, x, free_detached))
    finite = finite and all(bool(torch.isfinite(g).all()) for g in grad_list)
    return grad_list, residual_total, max_abs_dual, finite


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test whether dual leak broadens the useful-credit window in deep PC-ALM."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--output-dim", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--budgets", default="80,96,112,120,128")
    parser.add_argument("--dual-leaks", default="0,0.01,0.025,0.05,0.1,0.2")
    parser.add_argument("--state-lr", type=float, default=0.25)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.925)
    parser.add_argument("--activation", choices=["linear", "tanh", "relu"], default="relu")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/pcalm_dual_leak.csv"),
    )
    args = parser.parse_args()

    budgets = parse_ints(args.budgets)
    leaks = parse_floats(args.dual_leaks)
    if budgets != sorted(set(budgets)):
        raise ValueError("budgets must be strictly increasing and unique")
    if any(not 0.0 <= leak < 1.0 for leak in leaks):
        raise ValueError("all dual leaks must satisfy 0 <= leak < 1")

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
    bp = method_grad(
        model,
        x,
        y,
        Schedule("bp", budget=0),
        state_lr=args.state_lr,
        rho=args.rho,
    )
    bp_first = bp[0]
    bp_first_norm = float(bp_first.norm())

    rows: list[dict[str, float | int | bool]] = []
    for leak in leaks:
        for budget in budgets:
            grad, residual_total, max_abs_dual, finite = leaked_pcalm_grad(
                model,
                x,
                y,
                state_lr=args.state_lr,
                rho=args.rho,
                alpha=args.alpha,
                dual_leak=leak,
                budget=budget,
            )
            first = grad[0]
            first_norm = float(first.norm())
            norm_ratio = first_norm / bp_first_norm if bp_first_norm > 0.0 else math.nan
            cosine = gradient_cosine([first], [bp_first])
            relative_error = gradient_relative_error([first], [bp_first])
            useful = (
                finite
                and math.isfinite(cosine)
                and cosine >= 0.9
                and 0.5 <= norm_ratio <= 2.0
                and relative_error <= 0.6
            )

            if leak == 0.0:
                reference = method_grad(
                    model,
                    x,
                    y,
                    Schedule("pcalm", budget=budget, alpha=args.alpha),
                    state_lr=args.state_lr,
                    rho=args.rho,
                )
                max_reference_error = max(
                    float((a - b).abs().max()) for a, b in zip(grad, reference, strict=True)
                )
                if max_reference_error > 1e-6:
                    raise RuntimeError(
                        f"leak=0 parity failed at budget={budget}: {max_reference_error}"
                    )
            else:
                max_reference_error = math.nan

            rows.append(
                {
                    "seed": args.seed,
                    "budget": budget,
                    "state_lr": args.state_lr,
                    "rho": args.rho,
                    "alpha": args.alpha,
                    "dual_leak": leak,
                    "finite": finite,
                    "useful_first_layer_credit": useful,
                    "first_layer_cosine_to_bp": cosine,
                    "first_layer_grad_norm_ratio_to_bp": norm_ratio,
                    "first_layer_relative_error_to_bp": relative_error,
                    "residual_total": residual_total,
                    "max_abs_dual_seen": max_abs_dual,
                    "leak_zero_max_grad_error": max_reference_error,
                }
            )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
