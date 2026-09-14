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


def quantize_dual(
    value: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, int, int, float]:
    """Quantize dual storage while keeping the arithmetic datapath in FP32."""
    max_pre = float(value.abs().max()) if value.numel() else 0.0
    if mode == "fp32":
        return value.detach(), 0, value.numel(), max_pre
    if mode == "fp16":
        return value.to(torch.float16).to(torch.float32).detach(), 0, value.numel(), max_pre
    if mode == "bf16":
        return value.to(torch.bfloat16).to(torch.float32).detach(), 0, value.numel(), max_pre

    if not mode.startswith("fixed"):
        raise ValueError(f"unknown precision mode: {mode}")
    bits_text, int_text = mode.removeprefix("fixed").split("_i", maxsplit=1)
    bits = int(bits_text)
    integer_bits = int(int_text)
    frac_bits = bits - 1 - integer_bits
    if bits < 2 or frac_bits < 0:
        raise ValueError(f"invalid fixed-point mode: {mode}")
    step = 2.0 ** (-frac_bits)
    lo = -(2.0**integer_bits)
    hi = (2.0**integer_bits) - step
    saturated = int(((value < lo) | (value > hi)).sum())
    clipped = value.clamp(lo, hi)
    quantized = torch.round(clipped / step) * step
    return quantized.detach(), saturated, value.numel(), max_pre


def precision_pcalm_grad(
    model: ResidualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    state_lr: float,
    rho: float,
    alpha: float,
    dual_leak: float,
    budget: int,
    precision: str,
) -> tuple[list[torch.Tensor], float, float, float, int, int, bool]:
    free = free_init(model, x)
    duals_before = zero_duals_like(constraint_residuals(model, x, free))
    max_abs_dual_seen = 0.0
    max_abs_pre_quant = 0.0
    saturated_total = 0
    quantized_total = 0
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
        duals_after: list[torch.Tensor] = []
        for lam, residual in zip(duals_before, residuals, strict=True):
            updated = (1.0 - dual_leak) * lam + alpha * residual
            quantized, saturated, elements, max_pre = quantize_dual(updated, precision)
            duals_after.append(quantized)
            saturated_total += saturated
            quantized_total += elements
            max_abs_pre_quant = max(max_abs_pre_quant, max_pre)
        max_abs_dual_seen = max(
            max_abs_dual_seen,
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
    return (
        grad_list,
        residual_total,
        max_abs_dual_seen,
        max_abs_pre_quant,
        saturated_total,
        quantized_total,
        finite,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure sensitivity of leaked PC-ALM to low-precision dual storage."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--output-dim", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--budget", type=int, default=112)
    parser.add_argument("--state-lr", type=float, default=0.25)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.925)
    parser.add_argument("--dual-leak", type=float, default=0.01)
    parser.add_argument("--activation", choices=["linear", "tanh", "relu"], default="relu")
    parser.add_argument(
        "--precisions",
        default=(
            "fp32,fp16,bf16,"
            "fixed16_i0,fixed16_i1,fixed16_i2,"
            "fixed12_i0,fixed12_i1,fixed12_i2,"
            "fixed8_i0,fixed8_i1,fixed8_i2"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/pcalm_dual_precision.csv"),
    )
    args = parser.parse_args()

    precisions = [part for part in args.precisions.split(",") if part]
    if not precisions or precisions[0] != "fp32":
        raise ValueError("precision list must start with fp32")

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

    rows: list[dict[str, float | int | bool | str]] = []
    fp32_grad: list[torch.Tensor] | None = None
    for precision in precisions:
        (
            grad,
            residual_total,
            max_abs_dual,
            max_abs_pre_quant,
            saturated_total,
            quantized_total,
            finite,
        ) = precision_pcalm_grad(
            model,
            x,
            y,
            state_lr=args.state_lr,
            rho=args.rho,
            alpha=args.alpha,
            dual_leak=args.dual_leak,
            budget=args.budget,
            precision=precision,
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
        if precision == "fp32":
            fp32_grad = grad
            precision_grad_error = 0.0
        else:
            if fp32_grad is None:
                raise RuntimeError("fp32 reference was not produced")
            precision_grad_error = gradient_relative_error(grad, fp32_grad)

        rows.append(
            {
                "seed": args.seed,
                "budget": args.budget,
                "state_lr": args.state_lr,
                "rho": args.rho,
                "alpha": args.alpha,
                "dual_leak": args.dual_leak,
                "precision": precision,
                "finite": finite,
                "useful_first_layer_credit": useful,
                "first_layer_cosine_to_bp": cosine,
                "first_layer_grad_norm_ratio_to_bp": norm_ratio,
                "first_layer_relative_error_to_bp": relative_error,
                "all_gradient_relative_error_to_fp32": precision_grad_error,
                "residual_total": residual_total,
                "max_abs_dual_seen": max_abs_dual,
                "max_abs_dual_pre_quant": max_abs_pre_quant,
                "saturated_values": saturated_total,
                "quantized_values": quantized_total,
                "saturation_rate": (saturated_total / quantized_total if quantized_total else 0.0),
            }
        )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
