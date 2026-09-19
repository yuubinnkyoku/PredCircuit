from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

import diagnose_pcalm_width64_update_lattice_alignment as alignment
from diagnose_pcalm_width64_leaky_mixed_precision import quantize as base_quantize
from predcircuit.pcalm import (
    ResidualMLP,
    Schedule,
    gradient_cosine,
    gradient_relative_error,
    method_grad,
)

STATE_PRECISION = "fixed15_i3"
UPDATE_PRECISION = "fixed14_i1"
DUAL_PRECISION = "fixed12_i1"
MODES = ("nearest", "stochastic")
RNG_SHARING_MODES = ("element", "tensor")


def stochastic_quantize_fixed(
    value: torch.Tensor,
    precision: str,
    *,
    generator: torch.Generator,
    rng_sharing: str = "element",
) -> tuple[torch.Tensor, int, int]:
    if precision != UPDATE_PRECISION:
        return base_quantize(value, precision)
    if rng_sharing not in RNG_SHARING_MODES:
        raise ValueError(f"unsupported rng_sharing={rng_sharing!r}")

    bits_text, integer_text = precision.removeprefix("fixed").split("_i", 1)
    bits = int(bits_text)
    integer_bits = int(integer_text)
    frac_bits = bits - 1 - integer_bits
    scale = float(2**frac_bits)
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1

    scaled = value * scale
    clipped = scaled.clamp(qmin, qmax)
    lower = torch.floor(clipped)
    probability_up = clipped - lower
    random_shape = probability_up.shape if rng_sharing == "element" else ()
    random = torch.rand(
        random_shape,
        dtype=probability_up.dtype,
        device=probability_up.device,
        generator=generator,
    )
    rounded = lower + (random < probability_up).to(probability_up.dtype)
    saturated = int(((scaled < qmin) | (scaled > qmax)).sum())
    return rounded / scale, saturated, value.numel()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test unbiased stochastic rounding only at the PC-ALM update quantizer."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--state-lr", type=float, default=0.234285)
    parser.add_argument("--dual-leak", type=float, default=0.02)
    parser.add_argument(
        "--rounding-seed",
        type=int,
        default=None,
        help="Explicit stochastic-rounding RNG seed; defaults to seed + 90000.",
    )
    parser.add_argument(
        "--rng-sharing",
        choices=RNG_SHARING_MODES,
        default="element",
        help=(
            "Randomness sharing inside each update-quantizer invocation: "
            "element draws one variate per tensor element; tensor broadcasts one variate "
            "over the whole tensor."
        ),
    )
    parser.add_argument(
        "--stochastic-only",
        action="store_true",
        help="Skip the deterministic nearest baseline when only RNG sensitivity is needed.",
    )
    args = parser.parse_args()

    depth, width = 32, 64
    model_seed = args.seed + 1000 * width + depth
    data_seed = args.seed + 10_000 + 1000 * width + depth
    generator = torch.Generator().manual_seed(data_seed)
    x = torch.randn(4, 8, generator=generator)
    y = torch.randn(4, 4, generator=generator)

    bp_model = ResidualMLP(
        depth=depth,
        width=width,
        input_dim=8,
        output_dim=4,
        activation="relu",
        seed=model_seed,
    )
    bp = method_grad(
        bp_model,
        x,
        y,
        Schedule("bp", budget=0),
        state_lr=args.state_lr,
        rho=1.0,
    )
    bp_first = bp[0]
    bp_norm = float(bp_first.norm())

    rounding_seed = args.rounding_seed if args.rounding_seed is not None else args.seed + 90_000
    modes = ("stochastic",) if args.stochastic_only else MODES
    rows: list[dict[str, object]] = []
    original_quantize = alignment.quantize
    try:
        for mode in modes:
            rounding_generator = torch.Generator().manual_seed(rounding_seed)

            def experiment_quantize(
                value: torch.Tensor, precision: str
            ) -> tuple[torch.Tensor, int, int]:
                if mode != "stochastic" or precision != UPDATE_PRECISION:
                    return base_quantize(value, precision)
                return stochastic_quantize_fixed(
                    value,
                    precision,
                    generator=rounding_generator,
                    rng_sharing=args.rng_sharing,
                )

            setattr(alignment, "quantize", experiment_quantize)
            model = ResidualMLP(
                depth=depth,
                width=width,
                input_dim=8,
                output_dim=4,
                activation="relu",
                seed=model_seed,
            )
            grads, stats = alignment.run_alignment(
                model,
                x,
                y,
                update_precision=UPDATE_PRECISION,
                state_precision=STATE_PRECISION,
                dual_precision=DUAL_PRECISION,
                budget=args.budget,
                state_lr=args.state_lr,
                dual_leak=args.dual_leak,
            )
            first = grads[0]
            ratio = float(first.norm()) / bp_norm if bp_norm else math.nan
            cosine = gradient_cosine([first], [bp_first])
            rel = gradient_relative_error([first], [bp_first])
            finite = bool(stats["finite"])
            rows.append(
                {
                    "seed": args.seed,
                    "rounding": mode,
                    "rounding_seed": rounding_seed if mode == "stochastic" else -1,
                    "rng_sharing": args.rng_sharing if mode == "stochastic" else "none",
                    "state_lr": args.state_lr,
                    "lattice_ratio_r": args.state_lr * x.shape[0],
                    "state_precision": STATE_PRECISION,
                    "update_precision": UPDATE_PRECISION,
                    "dual_precision": DUAL_PRECISION,
                    **stats,
                    "first_layer_cosine_to_bp": cosine,
                    "first_layer_grad_norm_ratio_to_bp": ratio,
                    "first_layer_relative_error_to_bp": rel,
                    "useful_first_layer_credit": finite
                    and cosine >= 0.9
                    and 0.5 <= ratio <= 2.0
                    and rel <= 0.6,
                }
            )
    finally:
        setattr(alignment, "quantize", original_quantize)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
