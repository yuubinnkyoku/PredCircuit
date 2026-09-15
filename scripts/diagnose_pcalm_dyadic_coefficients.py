from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from diagnose_pcalm_matched_credit_budget import credit_metrics, pcalm_grad_with_leak
from predcircuit.pcalm import ResidualMLP, Schedule, gradient_relative_error, method_grad


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare tuned PC-ALM dual coefficients with shift/add dyadic approximations."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--budget", type=int, default=128)
    args = parser.parse_args()

    state_lr = 0.25
    rho = 1.0
    input_dim = 8
    output_dim = 4
    batch_size = 4

    model = ResidualMLP(
        depth=args.depth,
        width=args.width,
        input_dim=input_dim,
        output_dim=output_dim,
        activation="relu",
        seed=args.seed + args.depth,
    )
    gen = torch.Generator().manual_seed(args.seed + 10_000 + args.depth)
    x = torch.randn(batch_size, input_dim, generator=gen)
    y = torch.randn(batch_size, output_dim, generator=gen)
    bp = method_grad(
        model,
        x,
        y,
        Schedule("bp", budget=0),
        state_lr=state_lr,
        rho=rho,
    )

    configs = [
        ("tuned", 0.925, 0.01),
        # 237/256 = 1 - 1/16 - 1/128 - 1/256
        # 253/256 = 1 - 1/128 - 1/256, so leak = 3/256.
        ("dyadic_8bit", 237.0 / 256.0, 3.0 / 256.0),
    ]

    rows: list[dict[str, float | int | str | bool]] = []
    reference_grad: list[torch.Tensor] | None = None
    for name, alpha, dual_leak in configs:
        grad, finite = pcalm_grad_with_leak(
            model,
            x,
            y,
            state_lr=state_lr,
            rho=rho,
            alpha=alpha,
            dual_leak=dual_leak,
            budget=args.budget,
        )
        if reference_grad is None:
            reference_grad = grad
        cosine, norm_ratio, relative_error, useful = credit_metrics(grad, bp, finite=finite)
        error_to_tuned = gradient_relative_error(grad, reference_grad)
        rows.append(
            {
                "seed": args.seed,
                "config": name,
                "alpha": alpha,
                "retain": 1.0 - dual_leak,
                "dual_leak": dual_leak,
                "budget": args.budget,
                "finite": finite,
                "useful_first_layer_credit": useful,
                "first_layer_cosine_to_bp": cosine,
                "first_layer_grad_norm_ratio_to_bp": norm_ratio,
                "first_layer_relative_error_to_bp": relative_error,
                "all_gradient_relative_error_to_tuned": error_to_tuned,
            }
        )

    tuned = rows[0]
    dyadic = rows[1]
    if not math.isfinite(float(dyadic["all_gradient_relative_error_to_tuned"])):
        raise RuntimeError("non-finite dyadic coefficient comparison")
    if not bool(tuned["finite"]) or not bool(dyadic["finite"]):
        raise RuntimeError("coefficient diagnostic became non-finite")

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
