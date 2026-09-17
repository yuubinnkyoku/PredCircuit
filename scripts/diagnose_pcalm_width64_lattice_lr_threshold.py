from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from diagnose_pcalm_width64_update_lattice_alignment import (
    fixed_lsb,
    predicted_gradient_deadzone,
    run_alignment,
)
from predcircuit.pcalm import (
    ResidualMLP,
    Schedule,
    gradient_cosine,
    gradient_relative_error,
    method_grad,
)

STATE_LRS = [0.22, 0.234285, 0.245, 0.25, 0.255, 0.27]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe the predicted state/update lattice transition near eta_eff=1."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--dual-leak", type=float, default=0.02)
    args = parser.parse_args()

    depth, width, batch = 32, 64, 4
    state_precision = "fixed15_i3"
    update_precision = "fixed14_i1"
    dual_precision = "fixed12_i1"
    state_lsb = fixed_lsb(state_precision)
    update_lsb = fixed_lsb(update_precision)
    assert state_lsb is not None and update_lsb is not None

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
    bp = method_grad(
        bp_model,
        x,
        y,
        Schedule("bp", budget=0),
        state_lr=STATE_LRS[0],
        rho=1.0,
    )
    bp_first = bp[0]
    bp_norm = float(bp_first.norm())

    rows: list[dict[str, object]] = []
    for state_lr in STATE_LRS:
        effective_lr = state_lr * batch
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
            update_precision=update_precision,
            state_precision=state_precision,
            dual_precision=dual_precision,
            budget=args.budget,
            state_lr=state_lr,
            dual_leak=args.dual_leak,
        )
        first = grads[0]
        ratio = float(first.norm()) / bp_norm if bp_norm else math.nan
        cosine = gradient_cosine([first], [bp_first])
        relative_error = gradient_relative_error([first], [bp_first])
        finite = bool(stats["finite"])
        deadzone, minimum_quanta = predicted_gradient_deadzone(
            update_precision,
            state_precision=state_precision,
            effective_lr=effective_lr,
        )
        one_quantum_to_half_state_lsb = (
            effective_lr * update_lsb / (0.5 * state_lsb)
        )
        rows.append(
            {
                "seed": args.seed,
                "state_lr": state_lr,
                "effective_lr": effective_lr,
                "one_quantum_to_half_state_lsb": one_quantum_to_half_state_lsb,
                "predicted_gradient_deadzone": deadzone,
                "minimum_update_quanta_to_move_state": minimum_quanta,
                **stats,
                "first_layer_cosine_to_bp": cosine,
                "first_layer_grad_norm_ratio_to_bp": ratio,
                "first_layer_relative_error_to_bp": relative_error,
                "useful_first_layer_credit": finite
                and cosine >= 0.9
                and 0.5 <= ratio <= 2.0
                and relative_error <= 0.6,
            }
        )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
