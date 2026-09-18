from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import cast

import pandas as pd
import torch

import diagnose_pcalm_width64_update_lattice_alignment as alignment
from diagnose_pcalm_width64_leaky_mixed_precision import quantize as base_quantize
from predcircuit.pcalm import ResidualMLP, Schedule, gradient_cosine, gradient_relative_error, method_grad

STATE_PRECISION = "fixed15_i3"
UPDATE_PRECISION = "fixed14_i1"
DUAL_PRECISION = "fixed12_i1"
TARGETS = ("none", "update", "dual")


def main() -> None:
    parser = argparse.ArgumentParser(description="Discriminate update-grid from dual-grid phase sensitivity.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--state-lr", type=float, default=0.234285)
    parser.add_argument("--dual-leak", type=float, default=0.02)
    args = parser.parse_args()

    depth, width = 32, 64
    model_seed = args.seed + 1000 * width + depth
    data_seed = args.seed + 10_000 + 1000 * width + depth
    generator = torch.Generator().manual_seed(data_seed)
    x = torch.randn(4, 8, generator=generator)
    y = torch.randn(4, 4, generator=generator)

    bp_model = ResidualMLP(depth=depth, width=width, input_dim=8, output_dim=4, activation="relu", seed=model_seed)
    bp = method_grad(bp_model, x, y, Schedule("bp", budget=0), state_lr=args.state_lr, rho=1.0)
    bp_first = bp[0]
    bp_norm = float(bp_first.norm())

    update_lsb = alignment.fixed_lsb(UPDATE_PRECISION)
    dual_lsb = alignment.fixed_lsb(DUAL_PRECISION)
    assert update_lsb is not None and dual_lsb is not None
    lsbs: dict[str, float] = {"update": float(update_lsb), "dual": float(dual_lsb)}

    rows: list[dict[str, object]] = []
    original_quantize = alignment.quantize
    try:
        for target in TARGETS:
            precision_target = UPDATE_PRECISION if target == "update" else DUAL_PRECISION if target == "dual" else ""
            offset = 0.0 if target == "none" else 0.5 * lsbs[cast(str, target)]

            def phase_quantize(value: torch.Tensor, precision: str) -> tuple[torch.Tensor, int, int]:
                if precision != precision_target or offset == 0.0:
                    return base_quantize(value, precision)
                shifted, saturated, total = base_quantize(value - offset, precision)
                return shifted + offset, saturated, total

            setattr(alignment, "quantize", phase_quantize)
            model = ResidualMLP(depth=depth, width=width, input_dim=8, output_dim=4, activation="relu", seed=model_seed)
            grads, stats = alignment.run_alignment(
                model, x, y,
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
            rows.append({
                "seed": args.seed,
                "phase_target": target,
                "phase_lsb": 0.0 if target == "none" else 0.5,
                "phase_offset": offset,
                "state_lr": args.state_lr,
                "lattice_ratio_r": args.state_lr * x.shape[0],
                "state_precision": STATE_PRECISION,
                "update_precision": UPDATE_PRECISION,
                "dual_precision": DUAL_PRECISION,
                **stats,
                "first_layer_cosine_to_bp": cosine,
                "first_layer_grad_norm_ratio_to_bp": ratio,
                "first_layer_relative_error_to_bp": rel,
                "useful_first_layer_credit": finite and cosine >= 0.9 and 0.5 <= ratio <= 2.0 and rel <= 0.6,
            })
    finally:
        setattr(alignment, "quantize", original_quantize)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
