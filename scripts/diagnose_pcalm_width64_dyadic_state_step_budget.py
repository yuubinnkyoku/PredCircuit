from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from diagnose_pcalm_width64_update_lattice_alignment import run_alignment
from predcircuit.pcalm import (
    ResidualMLP,
    Schedule,
    gradient_cosine,
    gradient_relative_error,
    method_grad,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the multiplier-free eta_eff=1 PC-ALM point at reduced relaxation budgets."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    depth, width, batch = 32, 64, 4
    state_lr = 0.25  # eta_eff = batch * state_lr = 1 exactly
    model_seed = args.seed + 1000 * width + depth
    data_seed = args.seed + 10_000 + 1000 * width + depth
    generator = torch.Generator().manual_seed(data_seed)
    x = torch.randn(batch, 8, generator=generator)
    y = torch.randn(batch, 4, generator=generator)

    bp_model = ResidualMLP(
        depth=depth,
        width=width,
        input_dim=8,
        output_dim=4,
        activation="relu",
        seed=model_seed,
    )
    bp = method_grad(bp_model, x, y, Schedule("bp", budget=0), state_lr=state_lr, rho=1.0)
    bp_first = bp[0]
    bp_norm = float(bp_first.norm())

    model = ResidualMLP(
        depth=depth,
        width=width,
        input_dim=8,
        output_dim=4,
        activation="relu",
        seed=model_seed,
    )
    grads, stats = run_alignment(
        model,
        x,
        y,
        update_precision="fixed14_i1",
        state_precision="fixed15_i3",
        dual_precision="fixed12_i1",
        budget=args.budget,
        state_lr=state_lr,
        dual_leak=0.02,
    )
    first = grads[0]
    ratio = float(first.norm()) / bp_norm if bp_norm else math.nan
    cosine = gradient_cosine([first], [bp_first])
    relative_error = gradient_relative_error([first], [bp_first])
    finite = bool(stats["finite"])
    row = {
        "seed": args.seed,
        "budget": args.budget,
        "state_lr": state_lr,
        "effective_lr": batch * state_lr,
        **stats,
        "first_layer_cosine_to_bp": cosine,
        "first_layer_grad_norm_ratio_to_bp": ratio,
        "first_layer_relative_error_to_bp": relative_error,
        "useful_first_layer_credit": finite
        and cosine >= 0.9
        and 0.5 <= ratio <= 2.0
        and relative_error <= 0.6,
    }
    frame = pd.DataFrame([row])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
