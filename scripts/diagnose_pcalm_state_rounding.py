from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from diagnose_pcalm_dual_precision import quantize_dual
from predcircuit.pcalm import (
    ResidualMLP, Schedule, al_energy_shifted, constraint_residuals, free_init,
    gradient_cosine, gradient_relative_error, method_grad, zero_duals_like,
)


def fixed_params(mode: str) -> tuple[float, float, float]:
    bits_text, int_text = mode.removeprefix("fixed").split("_i", maxsplit=1)
    bits, integer_bits = int(bits_text), int(int_text)
    frac_bits = bits - 1 - integer_bits
    step = 2.0 ** (-frac_bits)
    return step, -(2.0 ** integer_bits), (2.0 ** integer_bits) - step


def q_nearest(x: torch.Tensor, mode: str) -> tuple[torch.Tensor, int]:
    q, sat, _, _ = quantize_dual(x, mode)
    return q, sat


def q_stochastic(x: torch.Tensor, mode: str, gen: torch.Generator) -> tuple[torch.Tensor, int]:
    step, lo, hi = fixed_params(mode)
    sat = int(((x < lo) | (x > hi)).sum())
    v = x.clamp(lo, hi) / step
    lower = torch.floor(v)
    prob = v - lower
    q = (lower + (torch.rand(prob.shape, generator=gen, device=prob.device) < prob).to(prob.dtype)) * step
    return q.detach(), sat


def run(model, x, y, *, rounding: str, seed: int, budget: int, state_lr: float, dual_leak: float):
    rho, alpha = 1.0, 0.925
    effective_lr = state_lr * x.shape[0]
    free = free_init(model, x)
    duals = zero_duals_like(constraint_residuals(model, x, free))
    feedback = [torch.zeros_like(z) for z in free]
    gen = torch.Generator().manual_seed(seed + 900_000)
    state_sat = state_total = 0
    zero_moves = move_total = 0
    finite = True

    for outer_ix in range(budget):
        variables = [z.detach().requires_grad_(True) for z in free]
        energy = al_energy_shifted(model, x, y, variables, [d.detach() for d in duals], rho=rho)
        raw_grads = torch.autograd.grad(energy, variables)
        new_free, new_feedback = [], []
        for z, grad, err in zip(variables, raw_grads, feedback, strict=True):
            q_grad, _, _, _ = quantize_dual(grad, "fixed14_i1")
            raw_target = z.detach() - effective_lr * q_grad
            target = raw_target + err if rounding == "error_feedback" else raw_target
            if rounding == "stochastic":
                q_state, sat = q_stochastic(target, "fixed15_i3", gen)
            else:
                q_state, sat = q_nearest(target, "fixed15_i3")
            state_sat += sat
            state_total += q_state.numel()
            zero_moves += int((q_state == z.detach()).sum())
            move_total += q_state.numel()
            new_free.append(q_state)
            new_feedback.append((target - q_state).detach() if rounding == "error_feedback" else torch.zeros_like(q_state))
        free, feedback = new_free, new_feedback

        residuals = constraint_residuals(model, x, free)
        q_residuals = [quantize_dual(r, "fixed14_i1")[0] for r in residuals]
        duals_after = [quantize_dual((1.0-dual_leak)*lam + alpha*r, "fixed12_i1")[0]
                       for lam, r in zip(duals, q_residuals, strict=True)]
        finite = finite and all(bool(torch.isfinite(t).all()) for t in [*free, *duals_after, *feedback])
        if outer_ix == budget - 1:
            credit_duals = duals
            break
        duals = duals_after

    loss = al_energy_shifted(model, x, y, [z.detach() for z in free], [d.detach() for d in credit_duals], rho=rho)
    grads = [g.detach() for g in torch.autograd.grad(loss, tuple(model.weights))]
    return grads, {
        "finite": finite and all(bool(torch.isfinite(g).all()) for g in grads),
        "residual_total": sum(float(r.norm()) for r in constraint_residuals(model, x, free)),
        "state_saturation_rate": state_sat/state_total if state_total else 0.0,
        "state_zero_move_rate": zero_moves/move_total if move_total else 0.0,
        "feedback_max_abs": max(float(e.abs().max()) for e in feedback) if feedback else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=860)
    p.add_argument("--budget", type=int, default=128)
    p.add_argument("--state-lr", type=float, default=0.25)
    p.add_argument("--dual-leak", type=float, default=0.02)
    p.add_argument("--stochastic-repeats", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    depth, width = 32, 64
    model_seed = a.seed + 1000*width + depth
    data_seed = a.seed + 10_000 + 1000*width + depth
    gen = torch.Generator().manual_seed(data_seed)
    x, y = torch.randn(4, 8, generator=gen), torch.randn(4, 4, generator=gen)
    bp_model = ResidualMLP(depth=depth, width=width, input_dim=8, output_dim=4, activation="relu", seed=model_seed)
    bp = method_grad(bp_model, x, y, Schedule("bp", budget=0), state_lr=a.state_lr, rho=1.0)
    bp_first, bp_norm = bp[0], float(bp[0].norm())
    rows = []
    trials = [("nearest", 0), ("error_feedback", 0)] + [("stochastic", i) for i in range(a.stochastic_repeats)]
    for rounding, trial in trials:
        model = ResidualMLP(depth=depth, width=width, input_dim=8, output_dim=4, activation="relu", seed=model_seed)
        grads, stats = run(model, x, y, rounding=rounding, seed=a.seed+trial, budget=a.budget, state_lr=a.state_lr, dual_leak=a.dual_leak)
        first = grads[0]
        cosine = gradient_cosine([first], [bp_first])
        ratio = float(first.norm())/bp_norm if bp_norm else math.nan
        rel = gradient_relative_error([first], [bp_first])
        rows.append({"seed":a.seed,"rounding":rounding,"trial":trial,**stats,
                     "first_layer_cosine_to_bp":cosine,"first_layer_grad_norm_ratio_to_bp":ratio,
                     "first_layer_relative_error_to_bp":rel,
                     "useful_first_layer_credit":bool(stats["finite"]) and cosine>=0.9 and 0.5<=ratio<=2.0 and rel<=0.6})
    frame = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(a.out, index=False)
    print(frame.to_string(index=False))

if __name__ == "__main__":
    main()
