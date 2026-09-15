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
                "row_type": "method",
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
                "paired_count": float("nan"),
                "paired_t_ratio_median": float("nan"),
                "paired_overlap_cycle_win_rate": float("nan"),
                "paired_serial_cycle_win_rate": float("nan"),
                "paired_state_traffic_proxy_win_rate": float("nan"),
                "pcalm_reached_spc_censored_count": float("nan"),
                "censored_proven_state_traffic_win_count": float("nan"),
            }
        )

    wide = minimums.pivot(index="seed", columns="method", values="minimum_budget")
    reached_wide = minimums.pivot(index="seed", columns="method", values="reached").astype(bool)
    spc_max = int(minimums.loc[minimums["method"] == "spc", "max_budget_tested"].max())

    paired = wide.dropna(subset=["spc", "pcalm_leak"])
    if not paired.empty:
        ratios = paired["pcalm_leak"] / paired["spc"]
        ratio_median = float(ratios.median())
        overlap_win_rate = float((ratios < 1.0).mean())
        serialized_win_rate = float((ratios < 0.80255).mean())
        traffic_win_rate = float((ratios < 0.5).mean())
    else:
        ratios = pd.Series(dtype=float)
        ratio_median = float("nan")
        overlap_win_rate = float("nan")
        serialized_win_rate = float("nan")
        traffic_win_rate = float("nan")

    pcalm_reached_spc_censored = reached_wide["pcalm_leak"] & ~reached_wide["spc"]
    censored_pcalm_t = wide.loc[pcalm_reached_spc_censored, "pcalm_leak"]
    proven_traffic_wins_censored = int((censored_pcalm_t / spc_max < 0.5).sum())

    summary_rows.append(
        {
            "row_type": "comparison",
            "method": "pcalm_leak_vs_spc",
            "seeds": len(seeds),
            "reach_count": float("nan"),
            "reach_rate": float("nan"),
            "minimum_budget_median_reached": float("nan"),
            "minimum_budget_q25_reached": float("nan"),
            "minimum_budget_q75_reached": float("nan"),
            "all_finite_rate": float("nan"),
            "max_budget_tested": spc_max,
            "paired_count": len(ratios),
            "paired_t_ratio_median": ratio_median,
            "paired_overlap_cycle_win_rate": overlap_win_rate,
            "paired_serial_cycle_win_rate": serialized_win_rate,
            "paired_state_traffic_proxy_win_rate": traffic_win_rate,
            "pcalm_reached_spc_censored_count": int(pcalm_reached_spc_censored.sum()),
            "censored_proven_state_traffic_win_count": proven_traffic_wins_censored,
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
        "\nCycle thresholds use the current 128-MAC/32-dual-lane model: overlap wins at "
        "T_pcalm/T_spc < 1; serialized wins below 0.80255. The pessimistic state-traffic "
        "proxy wins below 0.5."
    )


if __name__ == "__main__":
    main()
