from __future__ import annotations

import math

import torch

from .epc import error_energy
from .hardware import estimate_local_learning_cost
from .pcalm import (
    ResidualMLP,
    Schedule,
    _solve_inner,
    al_energy_shifted,
    constraint_residuals,
    free_init,
    gradient_cosine,
    gradient_relative_error,
    method_grad,
    zero_duals_like,
)

EPS = 1e-12


def spc_grad(
    model: ResidualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    state_lr: float,
    rho: float,
    budget: int,
) -> tuple[list[torch.Tensor], list[float], bool]:
    free = free_init(model, x)
    duals = zero_duals_like(constraint_residuals(model, x, free))
    finite = True
    for _ in range(budget):
        free = _solve_inner(model, x, y, free, duals, state_lr=state_lr, rho=rho, steps=1)
        residuals = constraint_residuals(model, x, free)
        finite = finite and all(bool(torch.isfinite(z).all()) for z in [*free, *residuals])
    free_det = [z.detach() for z in free]
    duals_det = [d.detach() for d in duals]
    residuals = constraint_residuals(model, x, free_det)
    residual_norms = [float(r.detach().norm()) for r in residuals]
    loss = al_energy_shifted(model, x, y, free_det, duals_det, rho=rho)
    grads = torch.autograd.grad(loss, tuple(model.weights), allow_unused=False)
    grads = [g.detach().clone() for g in grads]
    finite = finite and all(bool(torch.isfinite(g).all()) for g in grads)
    return grads, residual_norms, finite


def pcalm_leak_grad(
    model: ResidualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    state_lr: float,
    rho: float,
    alpha: float,
    dual_leak: float,
    budget: int,
) -> tuple[list[torch.Tensor], float, bool]:
    if not 0.0 <= dual_leak < 1.0:
        raise ValueError("dual_leak must satisfy 0 <= dual_leak < 1")
    free = free_init(model, x)
    duals_before = zero_duals_like(constraint_residuals(model, x, free))
    max_abs_dual = 0.0
    finite = True
    for outer_ix in range(budget):
        free = _solve_inner(model, x, y, free, duals_before, state_lr=state_lr, rho=rho, steps=1)
        residuals = constraint_residuals(model, x, free)
        duals_after = [
            ((1.0 - dual_leak) * lam + alpha * residual).detach()
            for lam, residual in zip(duals_before, residuals, strict=True)
        ]
        max_abs_dual = max(
            max_abs_dual,
            max((float(d.abs().max()) for d in duals_after), default=0.0),
        )
        finite = finite and all(
            bool(torch.isfinite(t).all()) for t in [*free, *residuals, *duals_after]
        )
        if outer_ix == budget - 1:
            credit_duals = duals_before
            break
        duals_before = duals_after
    free_det = [z.detach() for z in free]
    duals_det = [d.detach() for d in credit_duals]
    loss = al_energy_shifted(model, x, y, free_det, duals_det, rho=rho)
    grads = torch.autograd.grad(loss, tuple(model.weights), allow_unused=False)
    grads = [g.detach().clone() for g in grads]
    finite = finite and all(bool(torch.isfinite(g).all()) for g in grads)
    return grads, max_abs_dual, finite


def oracle_norm_match(grads: list[torch.Tensor], bp_first_norm: float) -> list[torch.Tensor]:
    if not grads:
        raise ValueError("grads must be non-empty")
    first_norm = float(grads[0].norm())
    if first_norm <= 0.0 or bp_first_norm <= 0.0:
        return [g.detach().clone() for g in grads]
    scale = bp_first_norm / first_norm
    out = [g.detach().clone() for g in grads]
    out[0] = out[0] * scale
    return out


def residual_norm_gain(
    grads: list[torch.Tensor],
    residual_norms: list[float],
) -> list[torch.Tensor]:
    if not residual_norms:
        return [g.detach().clone() for g in grads]
    mean_res = sum(residual_norms) / len(residual_norms)
    gain = 1.0 / (mean_res + EPS)
    return [g.detach().clone() * gain for g in grads]


def credit_metrics(
    grads: list[torch.Tensor],
    bp: list[torch.Tensor],
) -> dict[str, float | bool]:
    bp_first_norm = float(bp[0].norm())
    first_norm = float(grads[0].norm())
    ratio = first_norm / bp_first_norm if bp_first_norm > 0 else math.nan
    cosine = gradient_cosine([grads[0]], [bp[0]])
    rel = gradient_relative_error([grads[0]], [bp[0]])
    finite = all(bool(torch.isfinite(g).all()) for g in grads)
    useful = bool(finite and cosine >= 0.9 and (0.5 <= ratio <= 2.0) and rel <= 0.6)
    return {
        "finite": finite,
        "useful_first_layer_credit": useful,
        "first_layer_cosine_to_bp": cosine,
        "first_layer_grad_norm_ratio_to_bp": ratio,
        "first_layer_relative_error_to_bp": rel,
        "first_layer_grad_norm": first_norm,
    }


def spc_credit(
    model: ResidualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    state_lr: float,
    rho: float,
    budget: int,
) -> list[torch.Tensor]:
    grads, _residual_norms, _finite = spc_grad(
        model, x, y, state_lr=state_lr, rho=rho, budget=budget
    )
    return grads


def epc_stationarity(
    model: ResidualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    errors: list[torch.Tensor],
) -> float:
    variables = [e.detach().requires_grad_(True) for e in errors]
    energy = error_energy(model, x, y, variables)
    grads = torch.autograd.grad(energy, variables, allow_unused=False)
    total = 0.0
    for g in grads:
        total += float(g.detach().square().sum())
    return math.sqrt(total)


def mac_accounting(
    *,
    family: str,
    depth: int,
    width: int,
    batch_size: int,
    budget: int,
) -> dict[str, float]:
    if family == "epc":
        cost = estimate_local_learning_cost(
            family="spc",
            depth=depth,
            width=width,
            input_dim=8,
            output_dim=4,
            batch_size=batch_size,
            state_bits=32,
            weight_bits=32,
        )
        per_step = float(cost.macs_per_relaxation_step) * 2.0
        persistent_bits = float((depth - 1) * batch_size * width * 32)
    elif family in {"spc", "pcalm"}:
        dual_bits = 32 if family == "pcalm" else None
        cost = estimate_local_learning_cost(
            family=family,
            depth=depth,
            width=width,
            input_dim=8,
            output_dim=4,
            batch_size=batch_size,
            state_bits=32,
            weight_bits=32,
            dual_bits=dual_bits,
        )
        per_step = float(cost.macs_per_relaxation_step)
        if family == "pcalm":
            per_step += float(cost.dual_scalar_updates_per_step)
        persistent_bits = float(cost.persistent_state_bits)
    elif family == "bp":
        cost = estimate_local_learning_cost(
            family="spc",
            depth=depth,
            width=width,
            input_dim=8,
            output_dim=4,
            batch_size=batch_size,
            state_bits=32,
            weight_bits=32,
        )
        per_step = float(cost.macs_per_relaxation_step) * 2.0
        persistent_bits = float(cost.free_state_scalars * 32)
    else:
        raise ValueError(f"unknown family: {family}")

    return {
        "macs_per_step_estimate": per_step,
        "total_macs_estimate": per_step * budget,
        "persistent_state_bits_estimate": persistent_bits,
    }


def bp_grad(model: ResidualMLP, x: torch.Tensor, y: torch.Tensor) -> list[torch.Tensor]:
    return method_grad(model, x, y, Schedule("bp", budget=0), state_lr=0.0, rho=1.0)
