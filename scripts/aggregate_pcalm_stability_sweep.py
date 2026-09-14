from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import cast

import pandas as pd


KEYS = ["method", "state_lr", "rho", "alpha", "budget"]
METRICS = [
    "global_gradient_cosine_to_bp",
    "global_gradient_relative_error_to_bp",
    "first_layer_gradient_cosine_to_bp",
    "first_layer_gradient_relative_error_to_bp",
    "first_layer_grad_norm_ratio_to_bp",
    "final_residual_total",
    "late_residual_cv",
    "max_abs_post_dual_over_trace",
    "max_abs_weight_credit_dual",
]


def ci95(series: pd.Series) -> float:
    values = series.dropna()
    if len(values) < 2:
        return math.nan
    return 1.96 * float(values.std(ddof=1)) / math.sqrt(len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate PC-ALM stability sweep seeds.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=Path("results/generated"))
    args = parser.parse_args()

    files = sorted(args.input.rglob("pcalm_stability_sweep_seed*.csv"))
    if len(files) != args.expected_seeds:
        raise RuntimeError(f"expected {args.expected_seeds} seed files, found {len(files)}")
    all_rows = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    seed_count = all_rows["seed"].nunique()
    if seed_count != args.expected_seeds:
        raise RuntimeError(f"expected {args.expected_seeds} distinct seeds, found {seed_count}")

    records: list[dict[str, float | int | str]] = []
    for key, group in all_rows.groupby(KEYS, dropna=False, sort=True):
        method, state_lr, rho, alpha, budget = cast(tuple[str, float, float, float, int], key)
        record: dict[str, float | int | str] = {
            "method": method,
            "state_lr": float(state_lr),
            "rho": float(rho),
            "alpha": float(alpha),
            "budget": int(budget),
            "n": len(group),
            "finite_rate": float(group["finite"].mean()),
            "useful_credit_rate": float(group["useful_first_layer_credit"].mean()),
        }
        finite_group = group[group["finite"]]
        for metric in METRICS:
            record[f"{metric}_mean"] = float(finite_group[metric].mean())
            record[f"{metric}_ci95"] = ci95(finite_group[metric])
        records.append(record)
    summary = pd.DataFrame(records)

    pcalm = summary[summary["method"] == "pcalm"].copy()
    qualifying = pcalm[
        (pcalm["finite_rate"] == 1.0) & (pcalm["useful_credit_rate"] >= 0.8)
    ].copy()
    if not qualifying.empty:
        qualifying = qualifying.sort_values(
            [
                "budget",
                "useful_credit_rate",
                "first_layer_gradient_relative_error_to_bp_mean",
                "max_abs_weight_credit_dual_mean",
            ],
            ascending=[True, False, True, True],
        )

    minima_rows: list[pd.Series] = []
    for _, group in pcalm.groupby(["state_lr", "rho", "alpha"], sort=True):
        passed = group[
            (group["finite_rate"] == 1.0) & (group["useful_credit_rate"] >= 0.8)
        ].sort_values("budget")
        if not passed.empty:
            minima_rows.append(passed.iloc[0])
    minima = pd.DataFrame(minima_rows, columns=summary.columns)
    if not minima.empty:
        minima = minima.sort_values(
            [
                "budget",
                "useful_credit_rate",
                "first_layer_gradient_relative_error_to_bp_mean",
            ],
            ascending=[True, False, True],
        )

    pc = all_rows[all_rows["method"] == "pc"].copy()
    pa = all_rows[all_rows["method"] == "pcalm"].copy()
    paired = pa.merge(
        pc,
        on=["seed", "depth", "width", "budget", "state_lr", "rho"],
        suffixes=("_pcalm", "_pc"),
        validate="many_to_one",
    )
    paired["delta_first_layer_cosine"] = (
        paired["first_layer_gradient_cosine_to_bp_pcalm"]
        - paired["first_layer_gradient_cosine_to_bp_pc"]
    )
    paired["delta_first_layer_relative_error"] = (
        paired["first_layer_gradient_relative_error_to_bp_pcalm"]
        - paired["first_layer_gradient_relative_error_to_bp_pc"]
    )
    paired["delta_global_cosine"] = (
        paired["global_gradient_cosine_to_bp_pcalm"]
        - paired["global_gradient_cosine_to_bp_pc"]
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(args.out_dir / "pcalm_stability_all.csv", index=False)
    summary.to_csv(args.out_dir / "pcalm_stability_summary.csv", index=False)
    qualifying.to_csv(args.out_dir / "pcalm_stability_qualifying.csv", index=False)
    minima.to_csv(args.out_dir / "pcalm_stability_minimum_t.csv", index=False)
    paired.to_csv(args.out_dir / "pcalm_stability_paired.csv", index=False)

    print("Top qualifying operating points:")
    if qualifying.empty:
        print("none")
    else:
        cols = [
            "state_lr",
            "rho",
            "alpha",
            "budget",
            "finite_rate",
            "useful_credit_rate",
            "first_layer_gradient_cosine_to_bp_mean",
            "first_layer_grad_norm_ratio_to_bp_mean",
            "first_layer_gradient_relative_error_to_bp_mean",
            "late_residual_cv_mean",
            "max_abs_weight_credit_dual_mean",
        ]
        print(qualifying[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
