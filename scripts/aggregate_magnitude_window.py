from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def aggregate_spc_magnitude(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    if not frames:
        raise ValueError("no input CSVs")
    df = pd.concat(frames, ignore_index=True)
    grouped = (
        df.groupby(["budget", "control"], as_index=False)
        .agg(
            n_seeds=("seed", "count"),
            useful_rate=("useful_first_layer_credit", "mean"),
            finite_rate=("finite", "mean"),
            mean_cosine=("first_layer_cosine_to_bp", "mean"),
            mean_norm_ratio=("first_layer_grad_norm_ratio_to_bp", "mean"),
            mean_rel_error=("first_layer_relative_error_to_bp", "mean"),
            mean_grad_norm=("first_layer_grad_norm", "mean"),
            mean_residual=("mean_hidden_residual_norm", "mean"),
        )
        .sort_values(["budget", "control"])
    )
    return grouped


def aggregate_epc_equilibrium(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    if not frames:
        raise ValueError("no input CSVs")
    df = pd.concat(frames, ignore_index=True)
    grouped = (
        df.groupby(["error_lr", "budget"], as_index=False)
        .agg(
            n_seeds=("seed", "count"),
            finite_rate=("finite", "mean"),
            stationarity_ok_rate=("stationarity_ok", "mean"),
            useful_bp_rate=("useful_first_layer_credit", "mean"),
            mean_cosine_bp=("first_layer_cosine_to_bp", "mean"),
            mean_cosine_spc=("first_layer_cosine_to_spc_ref", "mean"),
            mean_norm_ratio=("first_layer_grad_norm_ratio_to_bp", "mean"),
            mean_rel_energy_step=("energy_relative_last_step", "mean"),
            mean_stationarity=("stationarity_error_grad_norm", "mean"),
            epc_macs=("epc_macs_total", "mean"),
            spc_macs=("spc_macs_total", "mean"),
            pcalm_macs=("pcalm_macs_total", "mean"),
        )
        .sort_values(["error_lr", "budget"])
    )
    return grouped


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate magnitude-window holdout CSVs.")
    p.add_argument(
        "--kind",
        choices=["spc", "epc"],
        required=True,
    )
    p.add_argument("--inputs", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    if a.kind == "spc":
        frame = aggregate_spc_magnitude(a.inputs)
    else:
        frame = aggregate_epc_equilibrium(a.inputs)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(a.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
