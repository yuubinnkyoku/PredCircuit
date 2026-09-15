from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate matched-credit minimum-budget holdout.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, default=20)
    parser.add_argument(
        "--detail-out",
        type=Path,
        default=Path("results/generated/pcalm_matched_credit_minimums.csv"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("results/generated/pcalm_matched_credit_summary.csv"),
    )
    args = parser.parse_args()

    paths = sorted(args.input.rglob("pcalm_matched_credit_budget_seed*.csv"))
    if len(paths) != args.expected_seeds:
        raise RuntimeError(f"expected {args.expected_seeds} seed files, found {len(paths)}")

    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    seeds = sorted(frame["seed"].unique())
    if len(seeds) != args.expected_seeds:
        raise RuntimeError(f"expected {args.expected_seeds} unique seeds, found {len(seeds)}")

    minimum_rows: list[dict[str, float | int | str | bool]] = []
    for seed in seeds:
        seed_frame = frame[frame["seed"] == seed]
        for method in ("spc", "pcalm", "pcalm_leak"):
            method_frame = seed_frame[seed_frame["method"] == method].sort_values("budget")
            if method_frame.empty:
                raise RuntimeError(f"missing method={method} for seed={seed}")
            useful = method_frame[method_frame["useful_first_layer_credit"].astype(bool)]
            reached = not useful.empty
            minimum_budget = float(useful.iloc[0]["budget"]) if reached else float("nan")
            minimum_rows.append(
                {
                    "seed": int(seed),
                    "method": method,
                    "reached": reached,
                    "minimum_budget": minimum_budget,
                    "max_budget_tested": int(method_frame["budget"].max()),
                    "all_finite": bool(method_frame["finite"].astype(bool).all()),
                }
            )

    minimums = pd.DataFrame(minimum_rows)
    summary_rows: list[dict[str, float | int | str]] = []
    for method in ("spc", "pcalm", "pcalm_leak"):
        subset = minimums[minimums["method"] == method]
        reached = subset[subset["reached"].astype(bool)]
        summary_rows.append(
            {
                "method": method,
                "seeds": len(subset),
                "reach_count": len(reached),
                "reach_rate": float(len(reached) / len(subset)),
                "minimum_budget_median_reached": (
                    float(reached["minimum_budget"].median()) if not reached.empty else float("nan")
                ),
                "minimum_budget_q25_reached": (
                    float(reached["minimum_budget"].quantile(0.25))
                    if not reached.empty
                    else float("nan")
                ),
                "minimum_budget_q75_reached": (
                    float(reached["minimum_budget"].quantile(0.75))
                    if not reached.empty
                    else float("nan")
                ),
                "all_finite_rate": float(subset["all_finite"].astype(bool).mean()),
                "max_budget_tested": int(subset["max_budget_tested"].max()),
            }
        )

    wide = minimums.pivot(index="seed", columns="method", values="minimum_budget")
    reached_wide = minimums.pivot(index="seed", columns="method", values="reached").astype(bool)
    spc_max = int(
        minimums.loc[minimums["method"] == "spc", "max_budget_tested"].max()
    )

    paired = wide.dropna(subset=["spc", "pcalm_leak"])
    if not paired.empty:
        ratios = paired["pcalm_leak"] / paired["spc"]
        paired_count = len(paired)
        ratio_median = float(ratios.median())
        overlap_win_rate = float((ratios < 1.0).mean())
        serialized_win_rate = float((ratios < 0.80255).mean())
        traffic_win_rate = float((ratios < 0.5).mean())
    else:
        paired_count = 0
        ratio_median = float("nan")
        overlap_win_rate = float("nan")
        serialized_win_rate = float("nan")
        traffic_win_rate = float("nan")

    pcalm_reached_spc_censored = reached_wide["pcalm_leak"] & ~reached_wide["spc"]
    censored = wide.loc[pcalm_reached_spc_censored, "pcalm_leak"]
    proven_traffic_wins_censored = int((censored / spc_max < 0.5).sum())

    summary_rows.append(
        {
            "method": "paired_pcalm_leak_vs_spc",
            "seeds": len(seeds),
            "reach_count": paired_count,
            "reach_rate": float(paired_count / len(seeds)),
            "minimum_budget_median_reached": ratio_median,
            "minimum_budget_q25_reached": overlap_win_rate,
            "minimum_budget_q75_reached": serialized_win_rate,
            "all_finite_rate": traffic_win_rate,
            "max_budget_tested": spc_max,
        }
    )
    summary_rows.append(
        {
            "method": "censored_spc_break_even",
            "seeds": len(seeds),
            "reach_count": int(pcalm_reached_spc_censored.sum()),
            "reach_rate": float(pcalm_reached_spc_censored.mean()),
            "minimum_budget_median_reached": float(proven_traffic_wins_censored),
            "minimum_budget_q25_reached": float("nan"),
            "minimum_budget_q75_reached": float("nan"),
            "all_finite_rate": float("nan"),
            "max_budget_tested": spc_max,
        }
    )

    summary = pd.DataFrame(summary_rows)
    args.detail_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    minimums.to_csv(args.detail_out, index=False)
    summary.to_csv(args.summary_out, index=False)
    print("minimums")
    print(minimums.to_string(index=False))
    print("\nsummary")
    print(summary.to_string(index=False))
    print(
        "\npaired row encodes: median T_ratio, overlap win rate, serialized win rate, "
        "traffic-proxy win rate in the median/q25/q75/all_finite columns respectively."
    )
    print(
        "censored row reach_count = PC-ALM-leak reached while sPC missed by its max budget; "
        "median column = subset already proving T_pcalm/T_spc < 0.5 from censoring alone."
    )


if __name__ == "__main__":
    main()
