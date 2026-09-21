from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Literal

import pandas as pd
import torch

from diagnose_pcalm_dual_precision import quantize_dual
from diagnose_pcalm_update_precision import run_precision
from predcircuit.pcalm import (
    ResidualMLP,
    Schedule,
    gradient_cosine,
    gradient_relative_error,
    method_grad,
)


class WeightQuantizedResidualMLP(ResidualMLP):
    """Quantize only stored weights; keep activations and MAC arithmetic in FP32."""

    def __init__(
        self,
        *,
        depth: int,
        width: int,
        input_dim: int,
        output_dim: int,
        activation: Literal["linear", "tanh", "relu"] = "tanh",
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
        weight_precision: str = "fp32",
    ) -> None:
        super().__init__(
            depth=depth,
            width=width,
            input_dim=input_dim,
            output_dim=output_dim,
            activation=activation,
            seed=seed,
            dtype=dtype,
        )
        self.weight_precision = weight_precision
        self.weight_saturated = 0
        self.weight_total = 0
        self.max_abs_weight_pre_quant = 0.0

    def _quantized_weight_ste(self, weight: torch.Tensor) -> torch.Tensor:
        quantized, saturated, total, max_pre = quantize_dual(weight, self.weight_precision)
        self.weight_saturated += saturated
        self.weight_total += total
        self.max_abs_weight_pre_quant = max(self.max_abs_weight_pre_quant, max_pre)
        return weight + (quantized - weight.detach())

    def block_pred(self, layer_ix: int, z_prev: torch.Tensor) -> torch.Tensor:
        inp = z_prev if layer_ix == 0 else self.activation(z_prev)
        weight = self._quantized_weight_ste(self.weights[layer_ix])
        pred = self.scales[layer_ix] * (inp @ weight.T)
        if self.skips[layer_ix]:
            pred = pred + z_prev
        return pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--precisions",
        # Keep the representable range fixed at [-8, 8) across fixed-point
        # formats so the 12/10/9/8-bit comparison measures quantization
        # resolution rather than clipping.  The preceding 20-seed i1 run
        # observed |W|max up to 4.319 and ~4.5% saturation, so i1 was
        # confounded by range clipping.
        default="fp32,fixed12_i3,fixed10_i3,fixed9_i3,fixed8_i3",
    )
    args = parser.parse_args()

    depth = 32
    width = 8
    state_lr = 0.25
    rho = 1.0
    alpha = 0.925
    dual_leak = 0.01
    budget = 112
    generator = torch.Generator().manual_seed(args.seed + 10000 + depth)
    x = torch.randn(4, 8, generator=generator)
    y = torch.randn(4, 4, generator=generator)

    bp_model = ResidualMLP(
        depth=depth,
        width=width,
        input_dim=8,
        output_dim=4,
        activation="relu",
        seed=args.seed + depth,
    )
    bp = method_grad(bp_model, x, y, Schedule("bp", budget=0), state_lr=state_lr, rho=rho)
    bp_first = bp[0]
    bp_norm = float(bp_first.norm())

    rows: list[dict[str, object]] = []
    reference: list[torch.Tensor] | None = None
    for precision in args.precisions.split(","):
        model = WeightQuantizedResidualMLP(
            depth=depth,
            width=width,
            input_dim=8,
            output_dim=4,
            activation="relu",
            seed=args.seed + depth,
            weight_precision=precision,
        )
        grads, metrics = run_precision(
            model,
            x,
            y,
            state_lr=state_lr,
            rho=rho,
            alpha=alpha,
            dual_leak=dual_leak,
            budget=budget,
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
                "weight_precision": precision,
                "finite": finite,
                "useful_first_layer_credit": useful,
                "first_layer_cosine_to_bp": cosine,
                "first_layer_grad_norm_ratio_to_bp": ratio,
                "first_layer_relative_error_to_bp": relative_error,
                "all_gradient_relative_error_to_fp32_weights": gradient_relative_error(
                    grads, reference
                ),
                "weight_saturation_rate": (
                    model.weight_saturated / model.weight_total if model.weight_total else 0.0
                ),
                "max_abs_weight_pre_quant": model.max_abs_weight_pre_quant,
                **metrics,
            }
        )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
