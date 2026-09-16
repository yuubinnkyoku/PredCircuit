from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from diagnose_pcalm_matched_credit_budget import credit_metrics
from predcircuit.pcalm import (
    ResidualMLP, Schedule, _solve_inner, al_energy_shifted, constraint_residuals,
    free_init, method_grad, zero_duals_like,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Trace BP-blind stopping observables for pure PC-ALM.")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--depth", type=int, default=32)
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--state-lr", type=float, default=0.25)
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.925)
    p.add_argument("--max-budget", type=int, default=160)
    p.add_argument("--min-budget", type=int, default=48)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    model = ResidualMLP(depth=a.depth, width=a.width, input_dim=8, output_dim=4,
                        activation="relu", seed=a.seed + a.depth)
    gen = torch.Generator().manual_seed(a.seed + 10_000 + a.depth)
    x = torch.randn(a.batch_size, 8, generator=gen)
    y = torch.randn(a.batch_size, 4, generator=gen)
    bp = method_grad(model, x, y, Schedule("bp", budget=0), state_lr=a.state_lr, rho=a.rho)

    free = free_init(model, x)
    duals = zero_duals_like(constraint_residuals(model, x, free))
    prev_free = [z.clone() for z in free]
    rows = []
    for t in range(1, a.max_budget + 1):
        duals_before = duals
        free = _solve_inner(model, x, y, free, duals_before,
                            state_lr=a.state_lr, rho=a.rho, steps=1)
        residuals = constraint_residuals(model, x, free)
        duals_after = [(lam + a.alpha * r).detach()
                       for lam, r in zip(duals_before, residuals, strict=True)]
        if t >= a.min_budget:
            # All stopping observables below are available from PC-ALM state alone.
            residual_norm = math.sqrt(sum(float(r.square().sum()) for r in residuals))
            dual_norm = math.sqrt(sum(float(d.square().sum()) for d in duals_after))
            dual_step_norm = math.sqrt(sum(float((da-db).square().sum())
                                           for da, db in zip(duals_after, duals_before, strict=True)))
            state_step_norm = math.sqrt(sum(float((z-pz).square().sum())
                                            for z, pz in zip(free, prev_free, strict=True)))
            energy = float(al_energy_shifted(model, x, y, free, duals_before, rho=a.rho).detach())
            loss = al_energy_shifted(model, x, y, [z.detach() for z in free],
                                     [d.detach() for d in duals_before], rho=a.rho)
            grads = [g.detach() for g in torch.autograd.grad(loss, tuple(model.weights))]
            finite = all(bool(torch.isfinite(q).all()) for q in [*free, *duals_after, *residuals, *grads])
            cosine, ratio, relerr, useful = credit_metrics(grads, bp, finite=finite)
            rows.append({"seed": a.seed, "budget": t, "residual_norm": residual_norm,
                         "dual_norm": dual_norm, "dual_step_norm": dual_step_norm,
                         "state_step_norm": state_step_norm, "al_energy": energy,
                         "finite": finite, "cosine_to_bp": cosine,
                         "grad_norm_ratio_to_bp": ratio, "relative_error_to_bp": relerr,
                         "useful_first_layer_credit": useful})
        prev_free = [z.clone() for z in free]
        duals = duals_after

    a.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
