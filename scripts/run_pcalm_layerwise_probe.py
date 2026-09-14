from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from predcircuit.pcalm import (
    ResidualMLP,
    Schedule,
    gradient_cosine,
    gradient_relative_error,
    method_grad,
    run_pc,
    run_pcalm,
)


def parse_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure layerwise sPC/PC-ALM credit propagation against BP."
    )
    parser.add_argument("--depths", default="16,32")
    parser.add_argument("--budgets", default="8,16,32,64,128")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--output-dim", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--state-lr", type=float, default=0.25)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--activation", choices=["linear", "tanh", "relu"], default="relu")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/pcalm_layerwise_probe.csv"),
    )
    args = parser.parse_args()

    rows: list[dict[str, float | int | str | bool]] = []
    for depth in parse_ints(args.depths):
        model = ResidualMLP(
            depth=depth,
            width=args.width,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            activation=args.activation,
            seed=args.seed + depth,
        )
        gen = torch.Generator().manual_seed(args.seed + 10_000 + depth)
        x = torch.randn(args.batch_size, args.input_dim, generator=gen)
        y = torch.randn(args.batch_size, args.output_dim, generator=gen)
        bp = method_grad(
            model,
            x,
            y,
            Schedule("bp", budget=0),
            state_lr=args.state_lr,
            rho=args.rho,
        )

        for budget in parse_ints(args.budgets):
            total_primal_steps = budget * args.inner_steps
            for method in ("pc", "pcalm"):
                if method == "pc":
                    schedule = Schedule("pc", budget=total_primal_steps)
                    _, _, trace = run_pc(
                        model,
                        x,
                        y,
                        state_lr=args.state_lr,
                        rho=args.rho,
                        steps=total_primal_steps,
                        record_trace=True,
                    )
                else:
                    schedule = Schedule(
                        "pcalm",
                        budget=budget,
                        alpha=args.alpha,
                        inner_steps=args.inner_steps,
                    )
                    _, _, trace = run_pcalm(
                        model,
                        x,
                        y,
                        state_lr=args.state_lr,
                        rho=args.rho,
                        alpha=args.alpha,
                        budget=budget,
                        inner_steps=args.inner_steps,
                        record_trace=True,
                    )

                grad = method_grad(
                    model,
                    x,
                    y,
                    schedule,
                    state_lr=args.state_lr,
                    rho=args.rho,
                )
                assert trace is not None
                final_residuals = trace.residual_norms[-1]
                final_duals = trace.dual_norms[-1]
                max_abs_dual = max(trace.max_abs_dual)

                for layer_ix, (layer_grad, layer_bp) in enumerate(
                    zip(grad, bp, strict=True)
                ):
                    hidden_layer = layer_ix < depth - 1
                    residual_norm = final_residuals[layer_ix] if hidden_layer else math.nan
                    dual_norm = final_duals[layer_ix] if hidden_layer else math.nan
                    rows.append(
                        {
                            "seed": args.seed,
                            "method": method,
                            "depth": depth,
                            "width": args.width,
                            "budget": budget,
                            "total_primal_steps": total_primal_steps,
                            "layer": layer_ix,
                            "distance_from_output": depth - 1 - layer_ix,
                            "layer_gradient_cosine_to_bp": gradient_cosine(
                                [layer_grad], [layer_bp]
                            ),
                            "layer_gradient_relative_error_to_bp": gradient_relative_error(
                                [layer_grad], [layer_bp]
                            ),
                            "layer_grad_norm": float(layer_grad.norm()),
                            "bp_layer_grad_norm": float(layer_bp.norm()),
                            "final_residual_norm": residual_norm,
                            "final_dual_norm": dual_norm,
                            "max_abs_dual_over_trace": max_abs_dual,
                            "trace_finite": trace.finite,
                        }
                    )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
