from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.input.glob("pcalm_weight_precision_seed*.csv"))
    if not files:
        raise FileNotFoundError(f"no seed CSVs under {args.input}")
    frame = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    numeric = [
        "first_layer_cosine_to_bp",
        "first_layer_grad_norm_ratio_to_bp",
        "first_layer_relative_error_to_bp",
        "all_gradient_relative_error_to_fp32_weights",
        "weight_saturation_rate",
        "max_abs_weight_pre_quant",
        "dead_update_rate_lr_0p001",
        "dead_update_rate_lr_0p01",
        "dead_update_rate_lr_0p1",
    ]
    summary = (
        frame.groupby("weight_precision", sort=False)
        .agg(
            seeds=("seed", "nunique"),
            finite_rate=("finite", "mean"),
            useful_rate=("useful_first_layer_credit", "mean"),
            **{f"mean_{name}": (name, "mean") for name in numeric},
            **{f"max_{name}": (name, "max") for name in numeric},
        )
        .reset_index()
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
