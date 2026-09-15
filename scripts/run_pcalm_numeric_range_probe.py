from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from predcircuit.pcalm import ResidualMLP, run_pcalm

# SakanaAI/pc-alm tuned activity step sizes for the reference ResidualMLP.
REFERENCE_STATE_LR = {8: 0.209541, 16: 0.221921, 32: 0.234285, 64: 0.242954, 128: 0.247725}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    ix = round((len(ordered) - 1) * q)
    return ordered[ix]


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure real PC-ALM lambda/update ranges.")
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--output-dim", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--state-lr", type=float, default=None)
    parser.add_argument("--activation", choices=["linear", "tanh", "relu"], default="relu")
    parser.add_argument("--seed", type=int, default=940)
    parser.add_argument("--out", type=Path, default=Path("results/generated/pcalm_numeric_range.json"))
    args = parser.parse_args()

    state_lr = args.state_lr
    if state_lr is None:
        if args.depth not in REFERENCE_STATE_LR:
            raise ValueError("provide --state-lr for depths without a reference value")
        state_lr = REFERENCE_STATE_LR[args.depth]

    model = ResidualMLP(
        depth=args.depth,
        width=args.width,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        activation=args.activation,
        seed=args.seed,
    )
    gen = torch.Generator().manual_seed(args.seed + 10_000)
    x = torch.randn(args.batch_size, args.input_dim, generator=gen)
    y = torch.randn(args.batch_size, args.output_dim, generator=gen)
    _, _, trace = run_pcalm(
        model,
        x,
        y,
        state_lr=state_lr,
        rho=args.rho,
        alpha=args.alpha,
        budget=args.budget,
        record_trace=True,
    )
    assert trace is not None
    assert trace.max_abs_residuals is not None
    assert trace.max_abs_dual_updates is not None

    per_layer_max_residual = [
        max(step[layer] for step in trace.max_abs_residuals) for layer in range(args.depth - 1)
    ]
    per_layer_max_update = [
        max(step[layer] for step in trace.max_abs_dual_updates)
        for layer in range(args.depth - 1)
    ]
    updates = [value for step in trace.max_abs_dual_updates for value in step]
    result = {
        "depth": args.depth,
        "width": args.width,
        "seed": args.seed,
        "budget": args.budget,
        "state_lr": state_lr,
        "rho": args.rho,
        "alpha": args.alpha,
        "finite": trace.finite,
        "global_max_abs_dual": max(trace.max_abs_dual),
        "global_max_abs_residual": max(per_layer_max_residual),
        "global_max_abs_dual_update": max(per_layer_max_update),
        "dual_update_p50": percentile(updates, 0.50),
        "dual_update_p10": percentile(updates, 0.10),
        "dual_update_p01": percentile(updates, 0.01),
        "per_layer_max_abs_residual": per_layer_max_residual,
        "per_layer_max_abs_dual_update": per_layer_max_update,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
