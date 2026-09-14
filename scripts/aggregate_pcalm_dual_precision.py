from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = [
    "first_layer_cosine_to_bp",
    "first_layer_grad_norm_ratio_to_bp",
    "first_layer_relative_error_to_bp",
    "all_gradient_relative_error_to_fp32",
    "residual_total",
    "max_abs_dual_seen",
    "max_abs_dual_pre_quant",
    "saturation_rate",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate PC-ALM dual precision holdout results.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=Path("results/generated"))
    args = parser.parse_args()

    paths = sorted(args.input.rglob("pcalm_dual_precision_seed*.csv"))
    if len(paths) != args.expected_seeds:
        raise RuntimeError(f"expected {args.expected_seeds} seed CSVs, found {len(paths)}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    seed_count = frame["seed"].nunique()
    if seed_count != args.expected_seeds:
        raise RuntimeError(f"expected {args.expected_seeds} unique seeds, found {seed_count}")

    expected_rows_per_seed = frame["precision"].nunique()
    counts = frame.groupby("seed").size()
    if not bool((counts == expected_rows_per_seed).all()):
        raise RuntimeError("seed CSVs do not contain the same number of precision modes")

    summary_rows: list[dict[str, float | int | str]] = []
    fp32 = frame[frame["precision"] == "fp32"].set_index("seed")
    if len(fp32) != args.expected_seeds:
        raise RuntimeError("fp32 reference row missing for one or more seeds")

    for precision_key, group in frame.groupby("precision", sort=False):
        precision = str(precision_key)
        aligned = group.set_index("seed").loc[fp32.index]
        row: dict[str, float | int | str] = {
            "precision": precision,
            "seed_count": len(aligned),
            "finite_count": int(aligned["finite"].astype(bool).sum()),
            "useful_count": int(aligned["useful_first_layer_credit"].astype(bool).sum()),
            "saturated_seed_count": int((aligned["saturated_values"] > 0).sum()),
            "saturated_values_total": int(aligned["saturated_values"].sum()),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(aligned[metric].mean())
            row[f"{metric}_max"] = float(aligned[metric].max())
        for metric in (
            "first_layer_cosine_to_bp",
            "first_layer_grad_norm_ratio_to_bp",
            "first_layer_relative_error_to_bp",
            "residual_total",
        ):
            row[f"delta_vs_fp32_{metric}_mean"] = float(
                (aligned[metric] - fp32[metric]).mean()
            )

        reference_useful = fp32["useful_first_layer_credit"].astype(bool)
        current_useful = aligned["useful_first_layer_credit"].astype(bool)
        row["useful_gained_vs_fp32"] = int((~reference_useful & current_useful).sum())
        row["useful_lost_vs_fp32"] = int((reference_useful & ~current_useful).sum())
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "pcalm_dual_precision_all.csv", index=False)
    summary.to_csv(args.out_dir / "pcalm_dual_precision_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
