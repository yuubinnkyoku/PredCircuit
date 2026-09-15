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


def parse_ints(text: str) -> list[int]:
    values = [int(part) for part in text.split(",") if part]
    if values != sorted(set(values)):
        raise ValueError("budgets must be strictly increasing and unique")
    return values


def pcalm_grad_with_leak(
    model: ResidualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    state_lr: float,
    rho: float,
    alpha: float,
    dual_leak: float,
    budget: int,
) -> tuple[list[torch.Tensor], bool]:
    if budget < 1:
        raise ValueError("budget must be positive")
    if not 0.0 <= dual_leak < 1.0:
        raise ValueError("dual_leak must satisfy 0 <= dual_leak < 1")

    free = free_init(model, x)
    duals_before = zero_duals_like(constraint_residuals(model, x, free))
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
        finite = finite and all(
            bool(torch.isfinite(t).all()) for t in [*free, *residuals, *duals_after]
        )
        if outer_ix == budget - 1:
            credit_duals = duals_before
            break
        duals_before = duals_after
    else:
        raise AssertionError("unreachable")

    loss = al_energy_shifted(
        model,
        x,
        y,
        [z.detach() for z in free],
        [d.detach() for d in credit_duals],
        rho=rho,
    )
    grads = torch.autograd.grad(loss, tuple(model.weights), allow_unused=False)
    result = [g.detach().clone() for g in grads]
    finite = finite and all(bool(torch.isfinite(g).all()) for g in result)
    return result, finite


def credit_metrics(
    candidate: list[torch.Tensor],
    bp: list[torch.Tensor],
    *,
    finite: bool,
) -> tuple[float, float, float, bool]:
    candidate_first = candidate[0]
    bp_first = bp[0]
    bp_norm = float(bp_first.norm())
    candidate_norm = float(candidate_first.norm())
    norm_ratio = candidate_norm / bp_norm if bp_norm > 0.0 else math.nan
    cosine = gradient_cosine([candidate_first], [bp_first])
    relative_error = gradient_relative_error([candidate_first], [bp_first])
    useful = (
        finite
        and math.isfinite(cosine)
        and cosine >= 0.9
        and 0.5 <= norm_ratio <= 2.0
        and relative_error <= 0.6
    )
    return cosine, norm_ratio, relative_error, useful


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure minimum relaxation budget for matched first-layer credit quality."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--output-dim", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--spc-budgets",
        default="32,48,64,80,96,112,128,144,160,192,224,256",
    )
    parser.add_argument(
        "--pcalm-budgets",
        default="48,64,80,96,104,112,120,128,144,160",
    )
    parser.add_argument("--state-lr", type=float, default=0.25)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.925)
    parser.add_argument("--dual-leak", type=float, default=0.01)
    parser.add_argument("--activation", choices=["linear", "tanh", "relu"], default="relu")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/pcalm_matched_credit_budget.csv"),
    )
    args = parser.parse_args()

    spc_budgets = parse_ints(args.spc_budgets)
    pcalm_budgets = parse_ints(args.pcalm_budgets)

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

    rows: list[dict[str, float | int | str | bool]] = []

    for budget in spc_budgets:
        grad = method_grad(
            model,
            x,
            y,
            Schedule("pc", budget=budget),
            state_lr=args.state_lr,
            rho=args.rho,
        )
        finite = all(bool(torch.isfinite(g).all()) for g in grad)
        cosine, norm_ratio, relative_error, useful = credit_metrics(grad, bp, finite=finite)
        rows.append(
            {
                "seed": args.seed,
                "method": "spc",
                "budget": budget,
                "alpha": 0.0,
                "dual_leak": 0.0,
                "finite": finite,
                "useful_first_layer_credit": useful,
                "first_layer_cosine_to_bp": cosine,
                "first_layer_grad_norm_ratio_to_bp": norm_ratio,
                "first_layer_relative_error_to_bp": relative_error,
            }
        )

    for method, leak in (("pcalm", 0.0), ("pcalm_leak", args.dual_leak)):
        for budget in pcalm_budgets:
            grad, finite = pcalm_grad_with_leak(
                model,
                x,
                y,
                state_lr=args.state_lr,
                rho=args.rho,
                alpha=args.alpha,
                dual_leak=leak,
                budget=budget,
            )
            cosine, norm_ratio, relative_error, useful = credit_metrics(
                grad,
                bp,
                finite=finite,
            )
            rows.append(
                {
                    "seed": args.seed,
                    "method": method,
                    "budget": budget,
                    "alpha": args.alpha,
                    "dual_leak": leak,
                    "finite": finite,
                    "useful_first_layer_credit": useful,
                    "first_layer_cosine_to_bp": cosine,
                    "first_layer_grad_norm_ratio_to_bp": norm_ratio,
                    "first_layer_relative_error_to_bp": relative_error,
                }
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
                max_error = max(
                    float((a - b).abs().max()) for a, b in zip(grad, reference, strict=True)
                )
                if max_error > 1e-6:
                    raise RuntimeError(
                        f"standard PC-ALM parity failed at budget={budget}: {max_error}"
                    )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
