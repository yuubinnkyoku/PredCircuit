from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate PC-ALM dual leak holdout results.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=Path("results/generated"))
    args = parser.parse_args()

    files = sorted(args.input.glob("pcalm_dual_leak_seed*.csv"))
    if len(files) != args.expected_seeds:
        raise RuntimeError(f"expected {args.expected_seeds} seed files, found {len(files)}")
    frame = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    if frame["seed"].nunique() != args.expected_seeds:
        raise RuntimeError("seed count does not match expected holdout size")

    group = frame.groupby(["dual_leak", "budget"], as_index=False)
    summary = group.agg(
        seed_count=("seed", "nunique"),
        finite_rate=("finite", "mean"),
        useful_count=("useful_first_layer_credit", "sum"),
        useful_rate=("useful_first_layer_credit", "mean"),
        mean_cosine=("first_layer_cosine_to_bp", "mean"),
        mean_norm_ratio=("first_layer_grad_norm_ratio_to_bp", "mean"),
        mean_relative_error=("first_layer_relative_error_to_bp", "mean"),
        mean_residual=("residual_total", "mean"),
        mean_max_abs_dual=("max_abs_dual_seen", "mean"),
    )

    per_seed_leak = frame.groupby(["seed", "dual_leak"], as_index=False).agg(
        any_useful=("useful_first_layer_credit", "max"),
        useful_budget_count=("useful_first_layer_credit", "sum"),
    )
    leak_summary = per_seed_leak.groupby("dual_leak", as_index=False).agg(
        oracle_seed_count=("any_useful", "sum"),
        oracle_seed_rate=("any_useful", "mean"),
        mean_useful_budget_count=("useful_budget_count", "mean"),
        median_useful_budget_count=("useful_budget_count", "median"),
    )

    best_fixed = summary.loc[summary.groupby("dual_leak")["useful_count"].idxmax()].copy()
    best_fixed = best_fixed.sort_values("dual_leak")

    leak_zero_error = frame.loc[frame["dual_leak"] == 0.0, "leak_zero_max_grad_error"].max()
    if leak_zero_error > 1e-6:
        raise RuntimeError(f"leak=0 reference parity failed: {leak_zero_error}")
    if float(summary["finite_rate"].min()) < 1.0:
        raise RuntimeError("non-finite dual-leak condition observed")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "pcalm_dual_leak_all.csv", index=False)
    summary.to_csv(args.out_dir / "pcalm_dual_leak_summary.csv", index=False)
    leak_summary.to_csv(args.out_dir / "pcalm_dual_leak_window_summary.csv", index=False)
    best_fixed.to_csv(args.out_dir / "pcalm_dual_leak_best_fixed.csv", index=False)
    print("Window summary")
    print(leak_summary.to_string(index=False))
    print("\nBest fixed budget per leak")
    print(best_fixed.to_string(index=False))


if __name__ == "__main__":
    main()
