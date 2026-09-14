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
    run_pcalm,
)


def parse_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part]


def late_cv(values: list[float]) -> float:
    tail = torch.tensor(values[-min(8, len(values)) :], dtype=torch.float64)
    if tail.numel() <= 1:
        return 0.0
    mean = float(tail.mean())
    return 0.0 if mean == 0.0 else float(tail.std(unbiased=False) / abs(mean))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test whether hardware-local PC-ALM signals identify the useful-credit time window."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--output-dim", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--budgets", default="64,72,80,88,96,104,112,120,128")
    parser.add_argument("--state-lr", type=float, default=0.25)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.925)
    parser.add_argument("--activation", choices=["linear", "tanh", "relu"], default="relu")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/pcalm_local_stop.csv"),
    )
    args = parser.parse_args()

    budgets = parse_ints(args.budgets)
    if budgets != sorted(set(budgets)):
        raise ValueError("budgets must be strictly increasing and unique")

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
    previous_residual = math.nan
    previous_dual = math.nan
    for budget in budgets:
        _, credit_duals, trace = run_pcalm(
            model,
            x,
            y,
            state_lr=args.state_lr,
            rho=args.rho,
            alpha=args.alpha,
            budget=budget,
            inner_steps=1,
            record_trace=True,
        )
        assert trace is not None
        grad = method_grad(
            model,
            x,
            y,
            Schedule("pcalm", budget=budget, alpha=args.alpha),
            state_lr=args.state_lr,
            rho=args.rho,
        )
        first = grad[0]
        first_norm = float(first.norm())
        norm_ratio = first_norm / bp_first_norm if bp_first_norm > 0.0 else math.nan
        cosine = gradient_cosine([first], [bp_first])
        relative_error = gradient_relative_error([first], [bp_first])
        residual_series = [sum(layer_norms) for layer_norms in trace.residual_norms]
        residual = residual_series[-1]
        dual_max = max((float(d.abs().max()) for d in credit_duals), default=0.0)
        finite = trace.finite and all(bool(torch.isfinite(g).all()) for g in grad)
        useful = (
            finite
            and math.isfinite(cosine)
            and cosine >= 0.9
            and 0.5 <= norm_ratio <= 2.0
            and relative_error <= 0.6
        )
        rows.append(
            {
                "seed": args.seed,
                "budget": budget,
                "state_lr": args.state_lr,
                "rho": args.rho,
                "alpha": args.alpha,
                "finite": finite,
                "useful_first_layer_credit": useful,
                "first_layer_cosine_to_bp": cosine,
                "first_layer_grad_norm_ratio_to_bp": norm_ratio,
                "first_layer_relative_error_to_bp": relative_error,
                "residual_total": residual,
                "residual_delta_from_previous_checkpoint": (
                    residual - previous_residual if math.isfinite(previous_residual) else math.nan
                ),
                "residual_late_cv": late_cv(residual_series),
                "max_abs_weight_credit_dual": dual_max,
                "dual_delta_from_previous_checkpoint": (
                    dual_max - previous_dual if math.isfinite(previous_dual) else math.nan
                ),
            }
        )
        previous_residual = residual
        previous_dual = dual_max

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
