from __future__ import annotations

from dataclasses import dataclass

import torch

from .pcalm import ResidualMLP


@dataclass
class EPCTrace:
    energy: list[float]
    max_abs_error: list[float]
    finite: bool


def zero_errors(model: ResidualMLP, x: torch.Tensor) -> list[torch.Tensor]:
    """Initialize ePC error variables to zero with hidden-layer shapes."""
    with torch.no_grad():
        hidden = model.forward_activations(x)[:-1]
    return [torch.zeros_like(z) for z in hidden]


def states_from_errors(
    model: ResidualMLP,
    x: torch.Tensor,
    errors: list[torch.Tensor],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Reconstruct hidden states and output from ePC error variables."""
    if len(errors) != model.depth - 1:
        raise ValueError("ePC needs one error variable per hidden layer")

    hidden: list[torch.Tensor] = []
    state = x
    for layer_ix, error in enumerate(errors):
        state = model.block_pred(layer_ix, state) + error
        hidden.append(state)
    output = model.block_pred(model.depth - 1, state)
    return hidden, output


def error_energy(
    model: ResidualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    errors: list[torch.Tensor],
) -> torch.Tensor:
    """Official ePC error-coordinate energy, using sum reduction for inference."""
    _, output = states_from_errors(model, x, errors)
    error_term = torch.zeros((), dtype=x.dtype, device=x.device)
    for error in errors:
        error_term = error_term + 0.5 * error.square().sum()
    supervised = 0.5 * (output - y).square().sum()
    return error_term + supervised


def local_weight_energy(
    model: ResidualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    errors: list[torch.Tensor],
) -> torch.Tensor:
    """State-form energy with detached reconstructed states for local weight credit.

    This follows the official ePC implementation: reconstruct each state as
    prediction + error, detach that state, and evaluate the ordinary local PC
    prediction error. Its numerical value equals ``error_energy / batch_size``
    while its autograd graph enforces local weight updates.
    """
    if len(errors) != model.depth - 1:
        raise ValueError("ePC needs one error variable per hidden layer")

    batch_size = x.shape[0]
    if batch_size < 1:
        raise ValueError("batch must be non-empty")

    total = torch.zeros((), dtype=x.dtype, device=x.device)
    state = x
    for layer_ix, error in enumerate(errors):
        prediction = model.block_pred(layer_ix, state)
        next_state = (prediction + error.detach()).detach()
        total = total + 0.5 * (prediction - next_state).square().sum() / batch_size
        state = next_state

    output = model.block_pred(model.depth - 1, state)
    total = total + 0.5 * (output - y).square().sum() / batch_size
    return total


def run_epc(
    model: ResidualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    error_lr: float,
    steps: int,
    record_trace: bool = False,
) -> tuple[list[torch.Tensor], EPCTrace | None]:
    """Minimize the ePC energy in error coordinates with plain SGD."""
    if error_lr < 0.0:
        raise ValueError("error_lr must be non-negative")
    if steps < 0:
        raise ValueError("steps must be non-negative")

    current = zero_errors(model, x)
    trace = EPCTrace([], [], True) if record_trace else None

    for _ in range(steps):
        variables = [error.detach().requires_grad_(True) for error in current]
        energy = error_energy(model, x, y, variables)
        grads = torch.autograd.grad(energy, variables)
        current = [
            (error - error_lr * grad).detach()
            for error, grad in zip(variables, grads, strict=True)
        ]

        if trace is not None:
            trace.energy.append(float(energy.detach()))
            trace.max_abs_error.append(
                max((float(error.abs().max()) for error in current), default=0.0)
            )
            trace.finite = trace.finite and bool(torch.isfinite(energy)) and all(
                bool(torch.isfinite(error).all()) for error in current
            )

    return current, trace


def epc_grad(
    model: ResidualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    error_lr: float,
    steps: int,
) -> tuple[list[torch.Tensor], bool]:
    """Return local ePC weight credit after error-coordinate inference."""
    errors, trace = run_epc(
        model,
        x,
        y,
        error_lr=error_lr,
        steps=steps,
        record_trace=True,
    )
    energy = local_weight_energy(model, x, y, errors)
    grads = torch.autograd.grad(energy, tuple(model.weights), allow_unused=False)
    result = [grad.detach().clone() for grad in grads]
    finite = (trace.finite if trace is not None else True) and bool(torch.isfinite(energy))
    finite = finite and all(bool(torch.isfinite(grad).all()) for grad in result)
    return result, finite
