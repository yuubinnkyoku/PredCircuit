from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Compare sPC and PC-ALM credit with BP.")
    parser.add_argument("--depths", default="8,16,32")
    parser.add_argument("--budgets", default="4,8,16,32")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--output-dim", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--state-lr", type=float, default=0.01)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--activation", choices=["linear", "tanh", "relu"], default="tanh")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/generated/pcalm_gradient_probe.csv"),
    )
    args = parser.parse_args()

    rows: list[dict[str, float | int | str | bool]] = []
    depths = parse_ints(args.depths)
    budgets = parse_ints(args.budgets)

    for depth in depths:
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

        weight_scalars = sum(weight.numel() for weight in model.weights)
        hidden_scalars = args.batch_size * (depth - 1) * args.width
        scalar_bytes = x.element_size()

        for budget in budgets:
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
                    persistent_state_scalars = hidden_scalars
                    dual_scalar_updates = 0
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
                    persistent_state_scalars = 2 * hidden_scalars
                    dual_scalar_updates = budget * hidden_scalars

                grad = method_grad(
                    model,
                    x,
                    y,
                    schedule,
                    state_lr=args.state_lr,
                    rho=args.rho,
                )
                assert trace is not None
                rows.append(
                    {
                        "method": method,
                        "depth": depth,
                        "width": args.width,
                        "budget": budget,
                        "inner_steps": args.inner_steps,
                        "total_primal_steps": total_primal_steps,
                        "gradient_cosine_to_bp": gradient_cosine(grad, bp),
                        "gradient_relative_error_to_bp": gradient_relative_error(grad, bp),
                        "finite": trace.finite,
                        "max_abs_dual": max(trace.max_abs_dual),
                        "final_mean_residual_norm": (
                            sum(trace.residual_norms[-1]) / len(trace.residual_norms[-1])
                        ),
                        "weight_scalars": weight_scalars,
                        "persistent_state_scalars_min": persistent_state_scalars,
                        "persistent_state_bytes_min": persistent_state_scalars * scalar_bytes,
                        # Each primal step needs prediction and reverse credit matvecs.
                        # This is a bookkeeping estimate, not a cycle-accurate hardware count.
                        "local_matvec_macs_estimate": (
                            2 * weight_scalars * args.batch_size * total_primal_steps
                        ),
                        "dual_scalar_updates": dual_scalar_updates,
                    }
                )

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
