from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

WeightCreditTiming = Literal["pre_dual_energy", "post_dual_energy"]
MethodFamily = Literal["bp", "pc", "pcalm"]


@dataclass(frozen=True)
class Schedule:
    family: MethodFamily
    budget: int
    alpha: float = 0.0
    inner_steps: int = 1
    weight_credit_timing: WeightCreditTiming = "pre_dual_energy"


@dataclass
class InferenceTrace:
    residual_norms: list[list[float]]
    dual_norms: list[list[float]]
    max_abs_dual: list[float]
    finite: bool


class ResidualMLP(torch.nn.Module):
    """Residual MLP matching SakanaAI/pc-alm's reference parameterization."""

    def __init__(
        self,
        *,
        depth: int,
        width: int,
        input_dim: int,
        output_dim: int,
        activation: Literal["linear", "tanh", "relu"] = "tanh",
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must include at least one hidden layer and one output layer")
        if width < 1 or input_dim < 1 or output_dim < 1:
            raise ValueError("width, input_dim, and output_dim must be positive")
        if activation not in {"linear", "tanh", "relu"}:
            raise ValueError(f"unknown activation: {activation}")

        self.depth = depth
        self.width = width
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = activation
        self.scales = tuple(
            [1.0 / math.sqrt(input_dim)]
            + [1.0 / math.sqrt(width * depth)] * (depth - 2)
            + [1.0 / width]
        )
        self.skips = tuple([False] + [True] * (depth - 2) + [False])

        gen = torch.Generator().manual_seed(seed)
        weights: list[torch.nn.Parameter] = []
        for layer_ix in range(depth):
            in_dim = input_dim if layer_ix == 0 else width
            out_dim = output_dim if layer_ix == depth - 1 else width
            value = torch.randn(out_dim, in_dim, generator=gen, dtype=dtype)
            weights.append(torch.nn.Parameter(value))
        self.weights = torch.nn.ParameterList(weights)

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "linear":
            return x
        if self.activation_name == "tanh":
            return torch.tanh(x)
        return torch.relu(x)

    def block_pred(self, layer_ix: int, z_prev: torch.Tensor) -> torch.Tensor:
        inp = z_prev if layer_ix == 0 else self.activation(z_prev)
        pred = self.scales[layer_ix] * (inp @ self.weights[layer_ix].T)
        if self.skips[layer_ix]:
            pred = pred + z_prev
        return pred

    def forward_activations(self, x: torch.Tensor) -> list[torch.Tensor]:
        acts: list[torch.Tensor] = []
        z_prev = x
        for layer_ix in range(self.depth):
            z_prev = self.block_pred(layer_ix, z_prev)
            acts.append(z_prev)
        return acts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_activations(x)[-1]


def free_init(model: ResidualMLP, x: torch.Tensor) -> list[torch.Tensor]:
    return [z.detach().clone() for z in model.forward_activations(x)[:-1]]


def constraint_residuals(
    model: ResidualMLP,
    x: torch.Tensor,
    free: list[torch.Tensor],
) -> list[torch.Tensor]:
    residuals: list[torch.Tensor] = []
    for layer_ix, z_l in enumerate(free):
        z_prev = x if layer_ix == 0 else free[layer_ix - 1]
        residuals.append(z_l - model.block_pred(layer_ix, z_prev))
    return residuals


def zero_duals_like(residuals: list[torch.Tensor]) -> list[torch.Tensor]:
    return [torch.zeros_like(r) for r in residuals]


def supervised_loss(
    model: ResidualMLP,
    y: torch.Tensor,
    free: list[torch.Tensor],
) -> torch.Tensor:
    y_pred = model.block_pred(model.depth - 1, free[-1])
    return 0.5 * (y_pred - y).square().sum(dim=-1).mean()


def bp_loss(model: ResidualMLP, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y_pred = model(x)
    return 0.5 * (y_pred - y).square().sum(dim=-1).mean()


def al_energy_shifted(
    model: ResidualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    free: list[torch.Tensor],
    duals: list[torch.Tensor],
    *,
    rho: float,
) -> torch.Tensor:
    if rho <= 0.0:
        raise ValueError("rho must be positive")
    residuals = constraint_residuals(model, x, free)
    if len(residuals) != len(duals):
        raise ValueError("free and duals must describe the same hidden layers")
    total = supervised_loss(model, y, free)
    batch_size = x.shape[0]
    for residual, dual in zip(residuals, duals, strict=True):
        shifted = residual + dual / rho
        total = total + 0.5 * rho * shifted.square().sum() / batch_size
    return total


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
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if state_lr < 0.0:
        raise ValueError("state_lr must be non-negative")
    if steps == 0:
        return [z.detach().clone() for z in free]

    effective_lr = state_lr * x.shape[0]
    current = [z.detach().clone() for z in free]
    fixed_duals = [d.detach() for d in duals]
    for _ in range(steps):
        variables = [z.detach().requires_grad_(True) for z in current]
        energy = al_energy_shifted(model, x, y, variables, fixed_duals, rho=rho)
        grads = torch.autograd.grad(energy, variables)
        current = [
            (z - effective_lr * g).detach()
            for z, g in zip(variables, grads, strict=True)
        ]
    return current


def _trace_snapshot(
    model: ResidualMLP,
    x: torch.Tensor,
    free: list[torch.Tensor],
    duals: list[torch.Tensor],
) -> tuple[list[float], list[float], float, bool]:
    with torch.no_grad():
        residuals = constraint_residuals(model, x, free)
        residual_norms = [float(r.norm()) for r in residuals]
        dual_norms = [float(d.norm()) for d in duals]
        max_abs_dual = max((float(d.abs().max()) for d in duals), default=0.0)
        tensors = [*free, *duals, *residuals]
        finite = all(bool(torch.isfinite(t).all()) for t in tensors)
    return residual_norms, dual_norms, max_abs_dual, finite


def run_pc(
    model: ResidualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    state_lr: float,
    rho: float,
    steps: int,
    record_trace: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor], InferenceTrace | None]:
    free = free_init(model, x)
    duals = zero_duals_like(constraint_residuals(model, x, free))
    trace = InferenceTrace([], [], [], True) if record_trace else None

    if record_trace:
        assert trace is not None
        residual_norms, dual_norms, max_abs_dual, finite = _trace_snapshot(
            model, x, free, duals
        )
        trace.residual_norms.append(residual_norms)
        trace.dual_norms.append(dual_norms)
        trace.max_abs_dual.append(max_abs_dual)
        trace.finite = trace.finite and finite

    for _ in range(steps):
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
        if record_trace:
            assert trace is not None
            residual_norms, dual_norms, max_abs_dual, finite = _trace_snapshot(
                model, x, free, duals
            )
            trace.residual_norms.append(residual_norms)
            trace.dual_norms.append(dual_norms)
            trace.max_abs_dual.append(max_abs_dual)
            trace.finite = trace.finite and finite
    return free, duals, trace


def run_pcalm(
    model: ResidualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    state_lr: float,
    rho: float,
    alpha: float,
    budget: int,
    inner_steps: int = 1,
    weight_credit_timing: WeightCreditTiming = "pre_dual_energy",
    record_trace: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor], InferenceTrace | None]:
    if budget < 1:
        raise ValueError("PC-ALM budget must be at least 1")
    if inner_steps < 1:
        raise ValueError("PC-ALM inner_steps must be at least 1")
    if weight_credit_timing not in {"pre_dual_energy", "post_dual_energy"}:
        raise ValueError("weight_credit_timing must be pre_dual_energy or post_dual_energy")

    free = free_init(model, x)
    duals = zero_duals_like(constraint_residuals(model, x, free))
    trace = InferenceTrace([], [], [], True) if record_trace else None

    if record_trace:
        assert trace is not None
        residual_norms, dual_norms, max_abs_dual, finite = _trace_snapshot(
            model, x, free, duals
        )
        trace.residual_norms.append(residual_norms)
        trace.dual_norms.append(dual_norms)
        trace.max_abs_dual.append(max_abs_dual)
        trace.finite = trace.finite and finite

    duals_before = duals
    for outer_ix in range(budget):
        free = _solve_inner(
            model,
            x,
            y,
            free,
            duals_before,
            state_lr=state_lr,
            rho=rho,
            steps=inner_steps,
        )
        residuals = constraint_residuals(model, x, free)
        duals_after = [
            (lam + alpha * residual).detach()
            for lam, residual in zip(duals_before, residuals, strict=True)
        ]

        if record_trace:
            assert trace is not None
            residual_norms, dual_norms, max_abs_dual, finite = _trace_snapshot(
                model, x, free, duals_after
            )
            trace.residual_norms.append(residual_norms)
            trace.dual_norms.append(dual_norms)
            trace.max_abs_dual.append(max_abs_dual)
            trace.finite = trace.finite and finite

        if outer_ix == budget - 1:
            duals_weight = (
                duals_before if weight_credit_timing == "pre_dual_energy" else duals_after
            )
            return free, duals_weight, trace
        duals_before = duals_after

    raise AssertionError("unreachable")


def method_grad(
    model: ResidualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    schedule: Schedule,
    *,
    state_lr: float,
    rho: float,
) -> list[torch.Tensor]:
    if schedule.family == "bp":
        loss = bp_loss(model, x, y)
    elif schedule.family == "pc":
        free, duals, _ = run_pc(
            model,
            x,
            y,
            state_lr=state_lr,
            rho=rho,
            steps=schedule.budget,
        )
        free = [z.detach() for z in free]
        duals = [d.detach() for d in duals]
        loss = al_energy_shifted(model, x, y, free, duals, rho=rho)
    elif schedule.family == "pcalm":
        free, duals, _ = run_pcalm(
            model,
            x,
            y,
            state_lr=state_lr,
            rho=rho,
            alpha=schedule.alpha,
            budget=schedule.budget,
            inner_steps=schedule.inner_steps,
            weight_credit_timing=schedule.weight_credit_timing,
        )
        free = [z.detach() for z in free]
        duals = [d.detach() for d in duals]
        loss = al_energy_shifted(model, x, y, free, duals, rho=rho)
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
