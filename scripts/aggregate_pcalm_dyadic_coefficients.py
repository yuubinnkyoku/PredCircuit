from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate dyadic coefficient holdout.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.input.rglob("pcalm_dyadic_coefficients_seed*.csv"))
    if len(paths) != args.expected_seeds:
        raise RuntimeError(f"expected {args.expected_seeds} files, found {len(paths)}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if frame["seed"].nunique() != args.expected_seeds:
        raise RuntimeError("unexpected unique seed count")

    summary = (
        frame.groupby("config", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            finite_rate=("finite", "mean"),
            useful_rate=("useful_first_layer_credit", "mean"),
            cosine_mean=("first_layer_cosine_to_bp", "mean"),
            norm_ratio_mean=("first_layer_grad_norm_ratio_to_bp", "mean"),
            bp_relative_error_mean=("first_layer_relative_error_to_bp", "mean"),
            tuned_gradient_error_mean=("all_gradient_relative_error_to_tuned", "mean"),
            tuned_gradient_error_max=("all_gradient_relative_error_to_tuned", "max"),
        )
        .sort_values("config")
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
