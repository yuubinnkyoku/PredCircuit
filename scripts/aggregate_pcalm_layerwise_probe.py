from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy import stats


def mean_ci95(values: np.ndarray) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan, math.nan
    mean = float(values.mean())
    if len(values) == 1:
        return mean, math.nan, math.nan
    sem = float(stats.sem(values))
    half = float(stats.t.ppf(0.975, len(values) - 1) * sem)
    return mean, mean - half, mean + half


def paired_effect(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return math.nan
    sd = float(values.std(ddof=1))
    return float(values.mean() / sd) if sd > 0.0 else math.inf


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate PC-ALM layerwise probe artifacts.")
    parser.add_argument("--input", type=Path, default=Path("artifacts"))
    parser.add_argument("--expected-seeds", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=Path("results/generated"))
    args = parser.parse_args()

    paths = sorted(args.input.rglob("pcalm_layerwise_probe_seed*.csv"))
    if len(paths) != args.expected_seeds:
        raise RuntimeError(f"expected {args.expected_seeds} seed files, found {len(paths)}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    seeds = sorted(frame["seed"].unique())
    if len(seeds) != args.expected_seeds:
        raise RuntimeError(f"expected {args.expected_seeds} unique seeds, found {len(seeds)}")
    if not bool(frame["trace_finite"].all()):
        raise RuntimeError("non-finite inference trace found")

    pair_rows: list[dict[str, float | int]] = []
    keys = ["seed", "depth", "budget", "layer", "distance_from_output"]
    paired = frame.pivot(index=keys, columns="method", values="layer_gradient_cosine_to_bp")
    paired = paired.dropna(subset=["pc", "pcalm"]).reset_index()
    paired["cosine_diff_pcalm_minus_pc"] = paired["pcalm"] - paired["pc"]
    for key, group in paired.groupby(
        ["depth", "budget", "layer", "distance_from_output"], sort=True
    ):
        depth, budget, layer, distance = cast(tuple[int, int, int, int], key)
        diff = group["cosine_diff_pcalm_minus_pc"].to_numpy(dtype=float)
        mean, low, high = mean_ci95(diff)
        finite_diff = diff[np.isfinite(diff)]
        pvalue = (
            float(stats.ttest_1samp(finite_diff, 0.0).pvalue) if len(finite_diff) >= 2 else math.nan
        )
        pair_rows.append(
            {
                "depth": int(depth),
                "budget": int(budget),
                "layer": int(layer),
                "distance_from_output": int(distance),
                "n": len(finite_diff),
                "mean_cosine_pc": float(group["pc"].mean()),
                "mean_cosine_pcalm": float(group["pcalm"].mean()),
                "mean_diff": mean,
                "ci95_low": low,
                "ci95_high": high,
                "paired_dz": paired_effect(diff),
                "paired_p": pvalue,
            }
        )

    front_rows: list[dict[str, float | int | str]] = []
    for key, group in frame.groupby(["seed", "method", "depth", "budget"], sort=True):
        seed, method, depth, budget = cast(tuple[int, str, int, int], key)
        row: dict[str, float | int | str] = {
            "seed": int(seed),
            "method": str(method),
            "depth": int(depth),
            "budget": int(budget),
        }
        for threshold in (0.8, 0.9):
            reached = group[group["layer_gradient_cosine_to_bp"] >= threshold]
            suffix = str(threshold).replace(".", "p")
            row[f"layers_ge_{suffix}"] = len(reached)
            row[f"max_distance_ge_{suffix}"] = (
                int(reached["distance_from_output"].max()) if len(reached) else -1
            )
            first = group[group["layer"] == 0]
            row[f"first_layer_ge_{suffix}"] = int(
                bool(
                    len(first) and float(first.iloc[0]["layer_gradient_cosine_to_bp"]) >= threshold
                )
            )
        front_rows.append(row)

    front = pd.DataFrame(front_rows)
    threshold_rows: list[dict[str, float | int | str]] = []
    for key, group in frame.groupby(["seed", "method", "depth"], sort=True):
        seed, method, depth = cast(tuple[int, str, int], key)
        first = group[group["layer"] == 0].sort_values("budget")
        row: dict[str, float | int | str] = {
            "seed": int(seed),
            "method": str(method),
            "depth": int(depth),
        }
        for threshold in (0.8, 0.9):
            reached = first[first["layer_gradient_cosine_to_bp"] >= threshold]
            suffix = str(threshold).replace(".", "p")
            row[f"first_layer_min_budget_ge_{suffix}"] = (
                int(reached.iloc[0]["budget"]) if len(reached) else -1
            )
        threshold_rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pair_rows).to_csv(args.out_dir / "pcalm_layerwise_paired.csv", index=False)
    front.to_csv(args.out_dir / "pcalm_layerwise_fronts.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(
        args.out_dir / "pcalm_layerwise_first_layer_thresholds.csv", index=False
    )
    frame.to_csv(args.out_dir / "pcalm_layerwise_all.csv", index=False)

    first_layer = pd.DataFrame(pair_rows)
    first_layer = first_layer[first_layer["layer"] == 0]
    print("Paired first-layer cosine summary:")
    print(first_layer.to_string(index=False))
    print("\nFirst-layer threshold budgets:")
    print(pd.DataFrame(threshold_rows).to_string(index=False))


if __name__ == "__main__":
    main()
