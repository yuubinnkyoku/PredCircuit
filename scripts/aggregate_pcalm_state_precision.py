from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,required=True); p.add_argument("--out",type=Path,required=True); a=p.parse_args()
    files=sorted(a.input.glob("pcalm_state_precision_seed*.csv"))
    if len(files)!=20: raise RuntimeError(f"expected 20 seed files, found {len(files)}")
    frame=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
    summary=(frame.groupby("state_precision",sort=False).agg(seeds=("seed","nunique"),finite_rate=("finite","mean"),useful_rate=("useful_first_layer_credit","mean"),cosine=("first_layer_cosine_to_bp","mean"),norm_ratio=("first_layer_grad_norm_ratio_to_bp","mean"),relative_error=("first_layer_relative_error_to_bp","mean"),gradient_error_to_fp32=("all_gradient_relative_error_to_fp32_state","mean"),residual=("residual_total","mean"),max_abs_state=("max_abs_state_pre_quant","max"),saturation_rate=("state_saturation_rate","mean")).reset_index())
    a.out.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(a.out.with_name("pcalm_state_precision_all.csv"),index=False); summary.to_csv(a.out,index=False); print(summary.to_string(index=False))

if __name__ == "__main__": main()
