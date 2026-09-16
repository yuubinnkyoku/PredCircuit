from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from diagnose_pcalm_matched_credit_budget import credit_metrics
from predcircuit.pcalm import (
    ResidualMLP,
    Schedule,
    _solve_inner,
    al_energy_shifted,
    constraint_residuals,
    free_init,
    method_grad,
    zero_duals_like,
)


def parse_floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part]


def parse_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part]


def quantize_signed(x: torch.Tensor, *, bits: int, integer_bits: int) -> tuple[torch.Tensor, int]:
    """Saturating signed fixed point; integer_bits includes the sign bit."""
    frac_bits = bits - integer_bits
    if frac_bits < 0:
        raise ValueError("integer_bits must not exceed total bits")
    scale = float(1 << frac_bits)
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    scaled = torch.round(x * scale)
    saturated = int(((scaled < qmin) | (scaled > qmax)).sum().item())
    return torch.clamp(scaled, qmin, qmax) / scale, saturated


def main() -> None:
    p = argparse.ArgumentParser(
        description="Measure isolated fixed-point quantization of PC-ALM dual state."
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--depth", type=int, default=32)
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--state-lr", type=float, default=0.25)
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.925)
    p.add_argument("--dual-leaks", default="0,0.005,0.01,0.02")
    p.add_argument("--bits", default="16,12,8")
    p.add_argument(
        "--integer-bits", type=int, default=2, help="Includes sign bit; 2 gives range [-2, 2)."
    )
    p.add_argument("--budget", type=int, default=128)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    leaks = parse_floats(a.dual_leaks)
    bit_widths = parse_ints(a.bits)
    if any(not 0.0 <= leak < 1.0 for leak in leaks):
        raise ValueError("all dual leaks must satisfy 0 <= leak < 1")
    if any(bits < a.integer_bits for bits in bit_widths):
        raise ValueError("all bit widths must be >= integer-bits")

    model = ResidualMLP(
        depth=a.depth,
        width=a.width,
        input_dim=8,
        output_dim=4,
        activation="relu",
        seed=a.seed + a.depth,
    )
    gen = torch.Generator().manual_seed(a.seed + 10_000 + a.depth)
    x = torch.randn(a.batch_size, 8, generator=gen)
    y = torch.randn(a.batch_size, 4, generator=gen)
    bp = method_grad(model, x, y, Schedule("bp", budget=0), state_lr=a.state_lr, rho=a.rho)

    rows: list[dict[str, float | int | bool | str]] = []
    for leak in leaks:
        for bits in [0, *bit_widths]:
            free = free_init(model, x)
            duals = zero_duals_like(constraint_residuals(model, x, free))
            saturation_count = 0
            max_abs_prequant = 0.0
            max_abs_dual = 0.0
            dual_step_sq = 0.0

            for _ in range(a.budget):
                duals_before = duals
                free = _solve_inner(
                    model, x, y, free, duals_before, state_lr=a.state_lr, rho=a.rho, steps=1
                )
                residuals = constraint_residuals(model, x, free)
                prequant = [
                    ((1.0 - leak) * lam + a.alpha * r).detach()
                    for lam, r in zip(duals_before, residuals, strict=True)
                ]
                max_abs_prequant = max(
                    max_abs_prequant,
                    max((float(d.abs().max()) for d in prequant), default=0.0),
                )
                if bits == 0:
                    duals = prequant
                else:
                    quantized: list[torch.Tensor] = []
                    for d in prequant:
                        q, sat = quantize_signed(d, bits=bits, integer_bits=a.integer_bits)
                        quantized.append(q.detach())
                        saturation_count += sat
                    duals = quantized
                max_abs_dual = max(
                    max_abs_dual,
                    max((float(d.abs().max()) for d in duals), default=0.0),
                )
                dual_step_sq += sum(
                    float((da - db).square().sum())
                    for da, db in zip(duals, duals_before, strict=True)
                )

            residuals = constraint_residuals(model, x, free)
            loss = al_energy_shifted(
                model,
                x,
                y,
                [z.detach() for z in free],
                [d.detach() for d in duals],
                rho=a.rho,
            )
            grads = [g.detach() for g in torch.autograd.grad(loss, tuple(model.weights))]
            finite = all(bool(torch.isfinite(q).all()) for q in [*free, *duals, *residuals, *grads])
            cosine, ratio, relerr, useful = credit_metrics(grads, bp, finite=finite)
            rows.append(
                {
                    "seed": a.seed,
                    "dual_leak": leak,
                    "format": "fp32" if bits == 0 else f"q{bits}.{bits - a.integer_bits}",
                    "bits": bits,
                    "integer_bits": 32 if bits == 0 else a.integer_bits,
                    "fractional_bits": 0 if bits == 0 else bits - a.integer_bits,
                    "budget": a.budget,
                    "finite": finite,
                    "useful_first_layer_credit": useful,
                    "cosine_to_bp": cosine,
                    "grad_norm_ratio_to_bp": ratio,
                    "relative_error_to_bp": relerr,
                    "saturation_count": saturation_count,
                    "max_abs_prequant_dual": max_abs_prequant,
                    "max_abs_dual": max_abs_dual,
                    "dual_step_rms_accum": math.sqrt(dual_step_sq / max(a.budget, 1)),
                }
            )

    a.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
