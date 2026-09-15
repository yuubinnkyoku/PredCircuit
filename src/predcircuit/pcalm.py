from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch


ActivationName = Literal["linear", "tanh", "relu"]
WeightCreditTiming = Literal["pre_dual_energy", "post_dual_energy"]
ScheduleFamily = Literal["pc", "pcalm", "pcalm_leak"]


@dataclass(frozen=True)
class PCALMSchedule:
    family: ScheduleFamily
    rho: float = 1.0
    alpha: float = 1.0
    dual_leak: float = 1.0
    weight_credit_timing: WeightCreditTiming = "pre_dual_energy"


@dataclass
class PCALMTrace:
    finite: bool
    residual_norm: list[float]
    dual_norm: list[float]
    max_abs_dual: list[float]
    max_abs_residuals: list[list[float]] | None = None
    max_abs_dual_updates: list[list[float]] | None = None


class ResidualMLP:
    def __init__(
        self,
        *,
        depth: int,
        width: int,
        input_dim: int,
        output_dim: int,
        activation: ActivationName = "relu",
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if depth < 2:
            raise ValueError("depth must be at least 2")
        self.depth = depth
        self.width = width
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        generator = torch.Generator().manual_seed(seed)

        weights: list[torch.Tensor] = []
        weights.append(torch.randn(width, input_dim, generator=generator, dtype=dtype))
        for _ in range(1, depth - 1):
            weights.append(torch.randn(width, width, generator=generator, dtype=dtype))
        weights.append(torch.randn(output_dim, width, generator=generator, dtype=dtype))
        self.weights = [w.requires_grad_(True) for w in weights]

        self.scales = [1.0 / math.sqrt(input_dim)]
        self.scales.extend([1.0 / math.sqrt(width * depth)] * (depth - 2))
        self.scales.append(1.0 / width)
        self.skips = [False] + [True] * (depth - 2) + [False]

    def activate(self, value: torch.Tensor) -> torch.Tensor:
        if self.activation == "linear":
            return value
        if self.activation == "tanh":
            return torch.tanh(value)
        if self.activation == "relu":
            return torch.relu(value)
        raise ValueError(f"unknown activation: {self.activation}")

    def block_pred(self, layer_ix: int, state: torch.Tensor) -> torch.Tensor:
        pred = self.scales[layer_ix] * (state @ self.weights[layer_ix].T)
        pred = self.activate(pred)
        if self.skips[layer_ix]:
            pred = pred + state
        return pred

    def forward_hidden(self, x: torch.Tensor) -> list[torch.Tensor]:
        hidden: list[torch.Tensor] = []
        state = x
        for layer_ix in range(self.depth - 1):
            state = self.block_pred(layer_ix, state)
            hidden.append(state)
        return hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.forward_hidden(x)
        return self.block_pred(self.depth - 1, hidden[-1])


def free_init(model: ResidualMLP, x: torch.Tensor) -> list[torch.Tensor]:
    return [state.detach().clone() for state in model.forward_hidden(x)]


def constraint_residuals(
    model: ResidualMLP, x: torch.Tensor, free: list[torch.Tensor]
) -> list[torch.Tensor]:
    residuals: list[torch.Tensor] = []
    prev = x
    for layer_ix, state in enumerate(free):
        residuals.append(state - model.block_pred(layer_ix, prev))
        prev = state
    return residuals


def supervised_loss(
    model: ResidualMLP, y: torch.Tensor, free: list[torch.Tensor]
) -> torch.Tensor:
    prediction = model.block_pred(model.depth - 1, free[-1])
    return 0.5 * ((prediction - y) ** 2).sum(dim=1).mean()


def al_energy_shifted(
    model: ResidualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    free: list[torch.Tensor],
    duals: list[torch.Tensor],
    *,
    rho: float,
) -> torch.Tensor:
    residuals = constraint_residuals(model, x, free)
    batch_size = x.shape[0]
    penalty = torch.zeros((), dtype=x.dtype, device=x.device)
    for residual, dual in zip(residuals, duals, strict=True):
        shifted = residual + dual / rho
        penalty = penalty + 0.5 * rho * (shifted**2).sum() / batch_size
    return supervised_loss(model, y, free) + penalty


def zero_duals_like(residuals: list[torch.Tensor]) -> list[torch.Tensor]:
    return [torch.zeros_like(residual) for residual in residuals]


def _solve_inner(
    model: ResidualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    free: list[torch.Tensor],
    duals: list[torch.Tensor],
    *,
    state_lr: float,
    rho: float,
    steps: int,
) -> list[torch.Tensor]:
    batch_size = x.shape[0]
    for _ in range(steps):
        vars_ = [state.detach().requires_grad_(True) for state in free]
        loss = al_energy_shifted(model, x, y, vars_, duals, rho=rho)
        grads = torch.autograd.grad(loss, tuple(vars_), allow_unused=False)
        free = [
            (state - state_lr * batch_size * grad).detach()
            for state, grad in zip(vars_, grads, strict=True)
        ]
    return free


def run_pc(
    model: ResidualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    state_lr: float,
    rho: float,
    budget: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    free = free_init(model, x)
    residuals = constraint_residuals(model, x, free)
    duals = zero_duals_like(residuals)
    free = _solve_inner(
        model,
        x,
        y,
        free,
        duals,
        state_lr=state_lr,
        rho=rho,
        steps=budget,
    )
    return free, duals


def run_pcalm(
    model: ResidualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    state_lr: float,
    rho: float,
    alpha: float,
    budget: int,
    dual_leak: float = 1.0,
    weight_credit_timing: WeightCreditTiming = "pre_dual_energy",
    record_trace: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor], PCALMTrace | None]:
    free = free_init(model, x)
    residuals = constraint_residuals(model, x, free)
    duals = zero_duals_like(residuals)
    residual_norms: list[float] = []
    dual_norms: list[float] = []
    max_abs_dual: list[float] = []
    max_abs_residuals: list[list[float]] | None = [] if record_trace else None
    max_abs_dual_updates: list[list[float]] | None = [] if record_trace else None
    finite = True
    duals_weight = [dual.detach().clone() for dual in duals]

    for _ in range(budget):
        free = _solve_inner(
            model,
            x,
            y,
            free,
            duals,
            state_lr=state_lr,
            rho=rho,
            steps=1,
        )
        residuals = constraint_residuals(model, x, free)
        duals_weight = [dual.detach().clone() for dual in duals]
        dual_updates = [alpha * residual for residual in residuals]
        duals = [
            (dual_leak * dual + update).detach()
            for dual, update in zip(duals, dual_updates, strict=True)
        ]
        if weight_credit_timing == "post_dual_energy":
            duals_weight = [dual.detach().clone() for dual in duals]
        elif weight_credit_timing != "pre_dual_energy":
            raise ValueError(f"unknown weight credit timing: {weight_credit_timing}")

        if record_trace:
            residual_norms.append(
                math.sqrt(sum(float((residual**2).sum()) for residual in residuals))
            )
            dual_norms.append(math.sqrt(sum(float((dual**2).sum()) for dual in duals)))
            max_abs_dual.append(max(float(dual.abs().max()) for dual in duals))
            assert max_abs_residuals is not None
            assert max_abs_dual_updates is not None
            max_abs_residuals.append([float(residual.abs().max()) for residual in residuals])
            max_abs_dual_updates.append([float(update.abs().max()) for update in dual_updates])
            finite = finite and all(bool(torch.isfinite(state).all()) for state in free)
            finite = finite and all(bool(torch.isfinite(dual).all()) for dual in duals)

    trace = None
    if record_trace:
        trace = PCALMTrace(
            finite=finite,
            residual_norm=residual_norms,
            dual_norm=dual_norms,
            max_abs_dual=max_abs_dual,
            max_abs_residuals=max_abs_residuals,
            max_abs_dual_updates=max_abs_dual_updates,
        )
    return free, duals_weight, trace


def bp_loss(model: ResidualMLP, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    prediction = model.forward(x)
    return 0.5 * ((prediction - y) ** 2).sum(dim=1).mean()


def bp_grad(model: ResidualMLP, x: torch.Tensor, y: torch.Tensor) -> list[torch.Tensor]:
    loss = bp_loss(model, x, y)
    grads = torch.autograd.grad(loss, tuple(model.weights), allow_unused=False)
    return [grad.detach().clone() for grad in grads]


def method_grad(
    model: ResidualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    schedule: PCALMSchedule,
    state_lr: float,
    budget: int,
) -> list[torch.Tensor]:
    if schedule.family == "pc":
        free, duals = run_pc(
            model,
            x,
            y,
            state_lr=state_lr,
            rho=schedule.rho,
            budget=budget,
        )
        free = [z.detach() for z in free]
        duals = [d.detach() for d in duals]
        loss = al_energy_shifted(model, x, y, free, duals, rho=schedule.rho)
    elif schedule.family in ("pcalm", "pcalm_leak"):
        dual_leak = schedule.dual_leak if schedule.family == "pcalm_leak" else 1.0
        free, duals, _ = run_pcalm(
            model,
            x,
            y,
            state_lr=state_lr,
            rho=schedule.rho,
            alpha=schedule.alpha,
            budget=budget,
            dual_leak=dual_leak,
            weight_credit_timing=schedule.weight_credit_timing,
            record_trace=False,
        )
        free = [z.detach() for z in free]
        duals = [d.detach() for d in duals]
        loss = al_energy_shifted(model, x, y, free, duals, rho=schedule.rho)
    else:
        raise ValueError(f"unknown schedule family: {schedule.family}")

    grads = torch.autograd.grad(loss, tuple(model.weights), allow_unused=False)
    return [g.detach().clone() for g in grads]


def flatten_grads(grads: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.reshape(-1) for g in grads])


def gradient_cosine(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    va = flatten_grads(a)
    vb = flatten_grads(b)
    denom = va.norm() * vb.norm()
    if float(denom) == 0.0:
        return float("nan")
    return float(torch.dot(va, vb) / denom)


def gradient_relative_error(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    va = flatten_grads(a)
    vb = flatten_grads(b)
    denom = vb.norm().clamp_min(torch.finfo(vb.dtype).eps)
    return float((va - vb).norm() / denom)
