from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate accumulator width-scaling holdout.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-files", type=int, default=15)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.input.rglob("pcalm_accumulator_width_w*_seed*.csv"))
    if len(paths) != args.expected_files:
        raise RuntimeError(f"expected {args.expected_files} files, found {len(paths)}")

    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    summary = (
        frame.groupby(["width", "accumulator_precision"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            finite_rate=("finite", "mean"),
            useful_rate=("useful_first_layer_credit", "mean"),
            cosine_mean=("first_layer_cosine_to_bp", "mean"),
            norm_ratio_mean=("first_layer_grad_norm_ratio_to_bp", "mean"),
            bp_relative_error_mean=("first_layer_relative_error_to_bp", "mean"),
            fp32_accumulator_error_mean=(
                "all_gradient_relative_error_to_fp32_accumulator",
                "mean",
            ),
            operand_saturation_rate_mean=("operand_saturation_rate", "mean"),
            accumulator_saturation_rate_mean=("accumulator_saturation_rate", "mean"),
            max_abs_operand=("max_abs_operand_pre_quant", "max"),
            max_abs_accumulator=("max_abs_accumulator_pre_quant", "max"),
        )
        .sort_values(["width", "accumulator_precision"])
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
