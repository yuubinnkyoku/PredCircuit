from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from diagnose_pcalm_accumulator_precision import AccumulatorQuantizedResidualMLP
from diagnose_pcalm_update_precision import run_precision
from predcircuit.pcalm import (
    ResidualMLP,
    Schedule,
    gradient_cosine,
    gradient_relative_error,
    method_grad,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check accumulator fixed-point range as residual-MLP width grows."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument(
        "--precisions",
        default="fp32,fixed16_i6,fixed12_i5",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    input_dim = 8
    output_dim = 4
    batch_size = 4
    state_lr = 0.25
    rho = 1.0
    alpha = 0.925
    dual_leak = 0.01

    model_seed = args.seed + 1000 * args.width + args.depth
    data_seed = args.seed + 10_000 + 1000 * args.width + args.depth
    generator = torch.Generator().manual_seed(data_seed)
    x = torch.randn(batch_size, input_dim, generator=generator)
    y = torch.randn(batch_size, output_dim, generator=generator)

    bp_model = ResidualMLP(
        depth=args.depth,
        width=args.width,
        input_dim=input_dim,
        output_dim=output_dim,
        activation="relu",
        seed=model_seed,
    )
    bp = method_grad(
        bp_model,
        x,
        y,
        Schedule("bp", budget=0),
        state_lr=state_lr,
        rho=rho,
    )
    bp_first = bp[0]
    bp_norm = float(bp_first.norm())

    rows: list[dict[str, object]] = []
    reference: list[torch.Tensor] | None = None
    for precision in args.precisions.split(","):
        model = AccumulatorQuantizedResidualMLP(
            depth=args.depth,
            width=args.width,
            input_dim=input_dim,
            output_dim=output_dim,
            activation="relu",
            seed=model_seed,
            accumulator_precision=precision,
        )
        grads, metrics = run_precision(
            model,
            x,
            y,
            state_lr=state_lr,
            rho=rho,
            alpha=alpha,
            dual_leak=dual_leak,
            budget=args.budget,
            update_precision="fixed12_i2",
        )
        first = grads[0]
        ratio = float(first.norm()) / bp_norm if bp_norm else math.nan
        cosine = gradient_cosine([first], [bp_first])
        relative_error = gradient_relative_error([first], [bp_first])
        if reference is None:
            reference = grads
        finite = bool(metrics["finite"])
        useful = finite and cosine >= 0.9 and 0.5 <= ratio <= 2.0 and relative_error <= 0.6
        rows.append(
            {
                "seed": args.seed,
                "depth": args.depth,
                "width": args.width,
                "budget": args.budget,
                "operand_precision": "fixed12_i3",
                "accumulator_precision": precision,
                "finite": finite,
                "useful_first_layer_credit": useful,
                "first_layer_cosine_to_bp": cosine,
                "first_layer_grad_norm_ratio_to_bp": ratio,
                "first_layer_relative_error_to_bp": relative_error,
                "all_gradient_relative_error_to_fp32_accumulator": gradient_relative_error(
                    grads, reference
                ),
                "operand_saturation_rate": (
                    model.operand_saturated / model.operand_total if model.operand_total else 0.0
                ),
                "accumulator_saturation_rate": (
                    model.accumulator_saturated / model.accumulator_total
                    if model.accumulator_total
                    else 0.0
                ),
                "max_abs_operand_pre_quant": model.max_abs_operand_pre_quant,
                "max_abs_accumulator_pre_quant": model.max_abs_accumulator_pre_quant,
                **metrics,
            }
        )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
