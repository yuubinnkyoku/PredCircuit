from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from predcircuit.pcalm import (
    ResidualMLP,
    Schedule,
    gradient_cosine,
    gradient_relative_error,
    method_grad,
    run_pc,
    run_pcalm,
)


def parse_floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part]


def parse_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part]


def late_residual_cv(residual_norms: list[list[float]]) -> float:
    totals = torch.tensor([sum(row) for row in residual_norms], dtype=torch.float64)
    if totals.numel() <= 1:
        return 0.0
    tail = totals[-min(8, totals.numel()) :]
    mean = float(tail.mean())
    if mean == 0.0:
        return 0.0
    return float(tail.std(unbiased=False) / abs(mean))


def row_for(
    *,
    seed: int,
    method: str,
    depth: int,
    width: int,
    budget: int,
    state_lr: float,
    rho: float,
    alpha: float,
    grad: list[torch.Tensor],
    bp: list[torch.Tensor],
    trace_finite: bool,
    final_residual_total: float,
    late_residual_cv_value: float,
    max_abs_post_dual_over_trace: float,
    max_abs_weight_credit_dual: float,
) -> dict[str, float | int | str | bool]:
    first_grad = grad[0]
    first_bp = bp[0]
    bp_norm = float(first_bp.norm())
    grad_norm = float(first_grad.norm())
    norm_ratio = grad_norm / bp_norm if bp_norm > 0.0 else math.nan
    first_cosine = gradient_cosine([first_grad], [first_bp])
    first_relative_error = gradient_relative_error([first_grad], [first_bp])
    finite = trace_finite and all(bool(torch.isfinite(g).all()) for g in grad)
    useful_credit = (
        finite
        and math.isfinite(first_cosine)
        and first_cosine >= 0.9
        and 0.5 <= norm_ratio <= 2.0
        and first_relative_error <= 0.6
    )
    return {
        "seed": seed,
        "method": method,
        "depth": depth,
        "width": width,
        "budget": budget,
        "state_lr": state_lr,
        "rho": rho,
        "alpha": alpha,
        "global_gradient_cosine_to_bp": gradient_cosine(grad, bp),
        "global_gradient_relative_error_to_bp": gradient_relative_error(grad, bp),
        "first_layer_gradient_cosine_to_bp": first_cosine,
        "first_layer_gradient_relative_error_to_bp": first_relative_error,
        "first_layer_grad_norm": grad_norm,
        "bp_first_layer_grad_norm": bp_norm,
        "first_layer_grad_norm_ratio_to_bp": norm_ratio,
        "final_residual_total": final_residual_total,
        "late_residual_cv": late_residual_cv_value,
        "max_abs_post_dual_over_trace": max_abs_post_dual_over_trace,
        "max_abs_weight_credit_dual": max_abs_weight_credit_dual,
        "finite": finite,
        "useful_first_layer_credit": useful_credit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep PC-ALM coefficients for stable first-layer credit at depth 32."
    )
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--output-dim", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--budgets", default="16,32,48,64")
    parser.add_argument("--state-lrs", default="0.125,0.25")
    parser.add_argument("--rhos", default="0.5,1.0,2.0")
    parser.add_argument("--alphas", default="0.5,1.0")
    parser.add_argument("--activation", choices=["linear", "tanh", "relu"], default="relu")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/pcalm_stability_sweep.csv"),
    )
    args = parser.parse_args()

    budgets = parse_ints(args.budgets)
    state_lrs = parse_floats(args.state_lrs)
    rhos = parse_floats(args.rhos)
    alphas = parse_floats(args.alphas)

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
        state_lr=state_lrs[0],
        rho=rhos[0],
    )

    rows: list[dict[str, float | int | str | bool]] = []
    for state_lr in state_lrs:
        for rho in rhos:
            for budget in budgets:
                _, pc_duals, pc_trace = run_pc(
                    model,
                    x,
                    y,
                    state_lr=state_lr,
                    rho=rho,
                    steps=budget,
                    record_trace=True,
                )
                assert pc_trace is not None
                pc_grad = method_grad(
                    model,
                    x,
                    y,
                    Schedule("pc", budget=budget),
                    state_lr=state_lr,
                    rho=rho,
                )
                rows.append(
                    row_for(
                        seed=args.seed,
                        method="pc",
                        depth=args.depth,
                        width=args.width,
                        budget=budget,
                        state_lr=state_lr,
                        rho=rho,
                        alpha=0.0,
                        grad=pc_grad,
                        bp=bp,
                        trace_finite=pc_trace.finite,
                        final_residual_total=sum(pc_trace.residual_norms[-1]),
                        late_residual_cv_value=late_residual_cv(pc_trace.residual_norms),
                        max_abs_post_dual_over_trace=max(pc_trace.max_abs_dual),
                        max_abs_weight_credit_dual=max(
                            (float(d.abs().max()) for d in pc_duals), default=0.0
                        ),
                    )
                )

                for alpha in alphas:
                    _, credit_duals, trace = run_pcalm(
                        model,
                        x,
                        y,
                        state_lr=state_lr,
                        rho=rho,
                        alpha=alpha,
                        budget=budget,
                        inner_steps=1,
                        record_trace=True,
                    )
                    assert trace is not None
                    grad = method_grad(
                        model,
                        x,
                        y,
                        Schedule("pcalm", budget=budget, alpha=alpha),
                        state_lr=state_lr,
                        rho=rho,
                    )
                    rows.append(
                        row_for(
                            seed=args.seed,
                            method="pcalm",
                            depth=args.depth,
                            width=args.width,
                            budget=budget,
                            state_lr=state_lr,
                            rho=rho,
                            alpha=alpha,
                            grad=grad,
                            bp=bp,
                            trace_finite=trace.finite,
                            final_residual_total=sum(trace.residual_norms[-1]),
                            late_residual_cv_value=late_residual_cv(trace.residual_norms),
                            max_abs_post_dual_over_trace=max(trace.max_abs_dual),
                            max_abs_weight_credit_dual=max(
                                (float(d.abs().max()) for d in credit_duals), default=0.0
                            ),
                        )
                    )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
