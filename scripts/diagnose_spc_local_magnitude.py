from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from predcircuit.magnitude_control import (
    credit_metrics,
    oracle_norm_match,
    pcalm_leak_grad,
    spc_grad,
)
from predcircuit.pcalm import ResidualMLP, Schedule, method_grad


def parse_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part]


def local_residual_match(
    grads: list[torch.Tensor],
    residual_norms: list[float],
    *,
    target: float,
) -> list[torch.Tensor]:
    """Rescale the whole credit vector using only residual statistics.

    target is a desired first-layer credit norm in residual-derived units.
    Uses no BP quantities. Zero first-layer credit cannot be repaired.
    """
    if not grads:
        raise ValueError("grads must be non-empty")
    first_norm = float(grads[0].norm())
    if first_norm <= 0.0:
        return [g.detach().clone() for g in grads]
    mean_res = sum(residual_norms) / len(residual_norms) if residual_norms else 0.0
    scale = (target * mean_res + 1e-12) / first_norm
    return [g.detach().clone() * scale for g in grads]


def local_layer_rms(
    grads: list[torch.Tensor],
    residual_norms: list[float],
    *,
    floor: float = 1e-6,
) -> list[torch.Tensor]:
    """Layerwise credit scaling by inverse residual RMS proxy (local only)."""
    out: list[torch.Tensor] = []
    for i, g in enumerate(grads):
        if i == 0 or i - 1 >= len(residual_norms):
            out.append(g.detach().clone())
            continue
        r = residual_norms[i - 1]
        out.append(g.detach().clone() / max(r, floor))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Local-only (non-BP) sPC credit magnitude controllers vs oracle/PC-ALM."
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--depth", type=int, default=32)
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--state-lr", type=float, default=0.25)
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.925)
    p.add_argument("--dual-leak", type=float, default=0.01)
    p.add_argument("--budgets", default="128,256")
    p.add_argument("--local-targets", default="0.5,1.0,2.0")
    p.add_argument("--activation", choices=["linear", "tanh", "relu"], default="relu")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    budgets = parse_ints(a.budgets)
    targets = [float(t) for t in a.local_targets.split(",") if t]
    model = ResidualMLP(
        depth=a.depth,
        width=a.width,
        input_dim=8,
        output_dim=4,
        activation=a.activation,
        seed=a.seed + a.depth,
    )
    gen = torch.Generator().manual_seed(a.seed + 10_000 + a.depth)
    x = torch.randn(a.batch_size, 8, generator=gen)
    y = torch.randn(a.batch_size, 4, generator=gen)
    bp = method_grad(model, x, y, Schedule("bp", budget=0), state_lr=a.state_lr, rho=a.rho)
    bp_first_norm = float(bp[0].norm())

    rows: list[dict[str, float | int | bool | str]] = []
    for budget in budgets:
        spc_grads, residual_norms, spc_finite = spc_grad(
            model, x, y, state_lr=a.state_lr, rho=a.rho, budget=budget
        )
        pcalm_grads, max_abs_dual, pcalm_finite = pcalm_leak_grad(
            model,
            x,
            y,
            state_lr=a.state_lr,
            rho=a.rho,
            alpha=a.alpha,
            dual_leak=a.dual_leak,
            budget=budget,
        )

        controls: list[tuple[str, list[torch.Tensor], bool]] = [
            ("spc_raw", spc_grads, spc_finite),
            (
                "spc_oracle_norm_match",
                oracle_norm_match(spc_grads, bp_first_norm),
                spc_finite,
            ),
            (
                "spc_local_layer_rms",
                local_layer_rms(spc_grads, residual_norms),
                spc_finite,
            ),
        ]
        for target in targets:
            controls.append(
                (
                    f"spc_local_residual_match_t{target:g}",
                    local_residual_match(spc_grads, residual_norms, target=target),
                    spc_finite,
                )
            )
        controls.append(("pcalm_dual_leak", pcalm_grads, pcalm_finite))

        for name, grads, finite_flag in controls:
            metrics = credit_metrics(grads, bp)
            metrics["finite"] = bool(metrics["finite"] and finite_flag)
            if not metrics["finite"]:
                metrics["useful_first_layer_credit"] = False
            rows.append(
                {
                    "seed": a.seed,
                    "budget": budget,
                    "control": name,
                    "max_abs_dual": max_abs_dual,
                    "mean_hidden_residual_norm": (
                        sum(residual_norms) / len(residual_norms)
                        if residual_norms
                        else float("nan")
                    ),
                    **metrics,
                }
            )

    frame = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(a.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
