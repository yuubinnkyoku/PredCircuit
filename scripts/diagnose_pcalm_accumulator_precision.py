from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Literal

import pandas as pd
import torch

from diagnose_pcalm_dual_precision import quantize_dual
from diagnose_pcalm_mac_operand_precision import OperandQuantizedResidualMLP
from diagnose_pcalm_update_precision import run_precision
from predcircuit.pcalm import (
    ResidualMLP,
    Schedule,
    gradient_cosine,
    gradient_relative_error,
    method_grad,
)


class AccumulatorQuantizedResidualMLP(OperandQuantizedResidualMLP):
    """Quantize the running dot-product accumulator while keeping 12-bit operands.

    Products are formed from fixed12_i3 operands. Because those operands are
    binary fixed-point values, the FP32 product represents the corresponding
    12x12 product exactly at the tested magnitudes. The diagnostic then rounds
    the running accumulator after every multiply-accumulate operation, isolating
    accumulator precision from operand precision.
    """

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
        accumulator_precision: str,
    ) -> None:
        super().__init__(
            depth=depth,
            width=width,
            input_dim=input_dim,
            output_dim=output_dim,
            activation=activation,
            seed=seed,
            dtype=dtype,
            operand_precision="fixed12_i3",
        )
        self.accumulator_precision = accumulator_precision
        self.accumulator_saturated = 0
        self.accumulator_total = 0
        self.max_abs_accumulator_pre_quant = 0.0

    def _quantize_accumulator_ste(self, value: torch.Tensor) -> torch.Tensor:
        quantized, saturated, total, max_pre = quantize_dual(
            value, self.accumulator_precision
        )
        self.accumulator_saturated += saturated
        self.accumulator_total += total
        self.max_abs_accumulator_pre_quant = max(
            self.max_abs_accumulator_pre_quant, max_pre
        )
        return value + (quantized - value.detach())

    def block_pred(self, layer_ix: int, z_prev: torch.Tensor) -> torch.Tensor:
        inp = z_prev if layer_ix == 0 else self.activation(z_prev)
        q_inp = self._quantize_ste(inp)
        q_weight = self._quantize_ste(self.weights[layer_ix])
        accumulator = torch.zeros(
            (q_inp.shape[0], q_weight.shape[0]),
            dtype=q_inp.dtype,
            device=q_inp.device,
        )
        for feature_ix in range(q_inp.shape[1]):
            product = q_inp[:, feature_ix : feature_ix + 1] * q_weight[
                :, feature_ix
            ].unsqueeze(0)
            accumulator = self._quantize_accumulator_ste(accumulator + product)
        pred = self.scales[layer_ix] * accumulator
        if self.skips[layer_ix]:
            pred = pred + z_prev
        return pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--precisions",
        default="fp32,fixed24_i6,fixed20_i6,fixed16_i6,fixed12_i6,fixed12_i5",
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
            depth=depth,
            width=width,
            input_dim=8,
            output_dim=4,
            activation="relu",
            seed=args.seed + depth,
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
                    model.operand_saturated / model.operand_total
                    if model.operand_total
                    else 0.0
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
