from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate dyadic coefficient holdout.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, default=15)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.input.rglob("pcalm_dyadic_coefficients_seed*.csv"))
    if len(paths) != args.expected_seeds:
        raise RuntimeError(f"expected {args.expected_seeds} files, found {len(paths)}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if frame["seed"].nunique() != args.expected_seeds:
        raise RuntimeError("unexpected unique seed count")

    summary = (
        frame.groupby(["precision", "coefficients"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            finite_rate=("finite", "mean"),
            useful_rate=("useful_first_layer_credit", "mean"),
            cosine_mean=("first_layer_cosine_to_bp", "mean"),
            norm_ratio_mean=("first_layer_grad_norm_ratio_to_bp", "mean"),
            bp_relative_error_mean=("first_layer_relative_error_to_bp", "mean"),
            gradient_error_to_original_mean=("all_gradient_relative_error_to_original", "mean"),
            gradient_error_to_original_max=("all_gradient_relative_error_to_original", "max"),
            late_update_zero_fraction_mean=("late_update_zero_fraction", "mean"),
            late_state_zero_step_fraction_mean=("late_state_zero_step_fraction", "mean"),
            state_saturation_rate_max=("state_saturation_rate", "max"),
            update_saturation_rate_max=("update_saturation_rate", "max"),
            dual_saturation_rate_max=("dual_saturation_rate", "max"),
        )
        .sort_values(["precision", "coefficients"])
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
