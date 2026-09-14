from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.input.glob("pcalm_update_precision_seed*.csv"))
    if len(files) != 20:
        raise RuntimeError(f"expected 20 seed files, found {len(files)}")

    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    summary = (
        frame.groupby("update_precision", sort=False)
        .agg(
            seeds=("seed", "nunique"),
            finite_rate=("finite", "mean"),
            useful_rate=("useful_first_layer_credit", "mean"),
            cosine=("first_layer_cosine_to_bp", "mean"),
            norm_ratio=("first_layer_grad_norm_ratio_to_bp", "mean"),
            relative_error=("first_layer_relative_error_to_bp", "mean"),
            gradient_error_to_fp32=(
                "all_gradient_relative_error_to_fp32_update",
                "mean",
            ),
            residual=("residual_total", "mean"),
            max_abs_update_grad=("max_abs_update_grad_pre_quant", "max"),
            max_abs_update_residual=("max_abs_update_residual_pre_quant", "max"),
            max_abs_state=("max_abs_state_pre_quant", "max"),
            max_abs_dual=("max_abs_dual_pre_quant", "max"),
            update_grad_saturation_rate=("update_grad_saturation_rate", "mean"),
            update_residual_saturation_rate=(
                "update_residual_saturation_rate",
                "mean",
            ),
            state_saturation_rate=("state_saturation_rate", "mean"),
            dual_saturation_rate=("dual_saturation_rate", "mean"),
        )
        .reset_index()
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out.with_name("pcalm_update_precision_all.csv"), index=False)
    summary.to_csv(args.out, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
