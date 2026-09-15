from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.input.rglob("pcalm_mac_operand_precision_seed*.csv"))
    if not files:
        raise FileNotFoundError("no MAC operand precision seed CSVs found")
    frame = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    summary = (
        frame.groupby("operand_precision", sort=False)
        .agg(
            seeds=("seed", "nunique"),
            finite_rate=("finite", "mean"),
            useful_rate=("useful_first_layer_credit", "mean"),
            cosine_mean=("first_layer_cosine_to_bp", "mean"),
            norm_ratio_mean=("first_layer_grad_norm_ratio_to_bp", "mean"),
            bp_relative_error_mean=("first_layer_relative_error_to_bp", "mean"),
            fp32_operand_error_mean=("all_gradient_relative_error_to_fp32_operands", "mean"),
            operand_saturation_rate_mean=("operand_saturation_rate", "mean"),
            max_abs_operand=("max_abs_operand_pre_quant", "max"),
        )
        .reset_index()
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
