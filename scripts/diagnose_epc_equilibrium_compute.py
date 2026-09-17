from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from predcircuit.epc import local_weight_energy, run_epc
from predcircuit.magnitude_control import epc_stationarity, mac_accounting, spc_credit
from predcircuit.pcalm import (
    ResidualMLP,
    Schedule,
    gradient_cosine,
    gradient_relative_error,
    method_grad,
)


def parse_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part]


def parse_floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part]


def main() -> None:
    p = argparse.ArgumentParser(
        description="ePC equilibrium/stationarity and compute accounting vs sPC/BP."
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--depth", type=int, default=32)
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--state-lr", type=float, default=0.25)
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--error-lrs", default="0.1,0.3,0.8,1.0")
    p.add_argument("--budgets", default="8,16,32,64,128")
    p.add_argument("--spc-reference-budget", type=int, default=256)
    p.add_argument("--stationarity-tol", type=float, default=1e-3)
    p.add_argument("--activation", choices=["linear", "tanh", "relu"], default="relu")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    error_lrs = parse_floats(a.error_lrs)
    budgets = parse_ints(a.budgets)

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
    spc_ref = spc_credit(
        model,
        x,
        y,
        state_lr=a.state_lr,
        rho=a.rho,
        budget=a.spc_reference_budget,
    )

    rows: list[dict[str, float | int | bool]] = []
    for error_lr in error_lrs:
        for budget in budgets:
            errors, trace = run_epc(
                model,
                x,
                y,
                error_lr=error_lr,
                steps=budget,
                record_trace=True,
            )
            assert trace is not None
            energy = local_weight_energy(model, x, y, errors)
            grads = torch.autograd.grad(energy, tuple(model.weights), allow_unused=False)
            grads = [g.detach().clone() for g in grads]
            stationarity = epc_stationarity(model, x, y, errors)

            e0 = trace.energy[0] if trace.energy else math.nan
            eT = trace.energy[-1] if trace.energy else math.nan
            if e0 is not None and abs(e0) > 1e-12:
                rel_energy_drop = float((e0 - eT) / abs(e0))
            else:
                rel_energy_drop = math.nan
            if len(trace.energy) >= 2:
                prev, last = trace.energy[-2], trace.energy[-1]
                denom = max(abs(prev), 1e-12)
                rel_energy_step = float(abs(last - prev) / denom)
            else:
                rel_energy_step = math.nan

            bp_cos = gradient_cosine([grads[0]], [bp[0]])
            bp_rel = gradient_relative_error([grads[0]], [bp[0]])
            spc_cos = gradient_cosine([grads[0]], [spc_ref[0]])
            bp_norm = float(bp[0].norm())
            grad_norm = float(grads[0].norm())
            norm_ratio = grad_norm / bp_norm if bp_norm > 0 else math.nan
            finite = trace.finite and all(bool(torch.isfinite(g).all()) for g in grads)
            useful_bp = bool(
                finite and bp_cos >= 0.9 and (0.5 <= norm_ratio <= 2.0) and bp_rel <= 0.6
            )
            stationarity_ok = bool(rel_energy_step <= a.stationarity_tol)

            epc_cost = mac_accounting(
                family="epc",
                depth=a.depth,
                width=a.width,
                batch_size=a.batch_size,
                budget=budget,
            )
            spc_cost = mac_accounting(
                family="spc",
                depth=a.depth,
                width=a.width,
                batch_size=a.batch_size,
                budget=budget,
            )
            pcalm_cost = mac_accounting(
                family="pcalm",
                depth=a.depth,
                width=a.width,
                batch_size=a.batch_size,
                budget=budget,
            )
            bp_cost = mac_accounting(
                family="bp",
                depth=a.depth,
                width=a.width,
                batch_size=a.batch_size,
                budget=1,
            )

            rows.append(
                {
                    "seed": a.seed,
                    "error_lr": error_lr,
                    "budget": budget,
                    "finite": finite,
                    "energy_final": float(eT),
                    "energy_relative_drop_from_init": rel_energy_drop,
                    "energy_relative_last_step": rel_energy_step,
                    "stationarity_error_grad_norm": stationarity,
                    "stationarity_tol": a.stationarity_tol,
                    "stationarity_ok": stationarity_ok,
                    "first_layer_cosine_to_bp": bp_cos,
                    "first_layer_grad_norm_ratio_to_bp": norm_ratio,
                    "first_layer_relative_error_to_bp": bp_rel,
                    "first_layer_cosine_to_spc_ref": spc_cos,
                    "useful_first_layer_credit": useful_bp,
                    "epc_macs_total": epc_cost["total_macs_estimate"],
                    "spc_macs_total": spc_cost["total_macs_estimate"],
                    "pcalm_macs_total": pcalm_cost["total_macs_estimate"],
                    "bp_macs_once": bp_cost["total_macs_estimate"],
                    "epc_persistent_state_bits": epc_cost["persistent_state_bits_estimate"],
                    "spc_persistent_state_bits": spc_cost["persistent_state_bits_estimate"],
                    "pcalm_persistent_state_bits": pcalm_cost["persistent_state_bits_estimate"],
                }
            )

    frame = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(a.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
