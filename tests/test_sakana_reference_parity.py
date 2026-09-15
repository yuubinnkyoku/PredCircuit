from __future__ import annotations

import math

import torch

from predcircuit.pcalm import ResidualMLP, constraint_residuals, run_pcalm


def reference_scales(width: int, depth: int, input_dim: int) -> tuple[float, ...]:
    return tuple(
        [1.0 / math.sqrt(input_dim)]
        + [1.0 / math.sqrt(width * depth)] * (depth - 2)
        + [1.0 / width]
    )


def reference_block_pred(
    weight: torch.Tensor,
    scale: float,
    skip: bool,
    z_prev: torch.Tensor,
    *,
    is_first: bool,
) -> torch.Tensor:
    inp = z_prev if is_first else torch.tanh(z_prev)
    pred = scale * (inp @ weight.T)
    return pred + z_prev if skip else pred


def reference_forward(model: ResidualMLP, x: torch.Tensor) -> list[torch.Tensor]:
    acts: list[torch.Tensor] = []
    z = x
    scales = reference_scales(model.width, model.depth, model.input_dim)
    skips = tuple([False] + [True] * (model.depth - 2) + [False])
    for layer_ix, weight in enumerate(model.weights):
        z = reference_block_pred(
            weight,
            scales[layer_ix],
            skips[layer_ix],
            z,
            is_first=layer_ix == 0,
        )
        acts.append(z)
    return acts


def reference_pcalm(
    model: ResidualMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    state_lr: float,
    rho: float,
    alpha: float,
    budget: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    forward = reference_forward(model, x)
    free = [z.detach().clone() for z in forward[:-1]]
    duals = [torch.zeros_like(z) for z in free]
    scales = reference_scales(model.width, model.depth, model.input_dim)
    skips = tuple([False] + [True] * (model.depth - 2) + [False])

    for _ in range(budget):
        variables = [z.detach().requires_grad_(True) for z in free]
        residuals: list[torch.Tensor] = []
        for layer_ix, z_l in enumerate(variables):
            z_prev = x if layer_ix == 0 else variables[layer_ix - 1]
            pred = reference_block_pred(
                model.weights[layer_ix],
                scales[layer_ix],
                skips[layer_ix],
                z_prev,
                is_first=layer_ix == 0,
            )
            residuals.append(z_l - pred)
        y_pred = reference_block_pred(
            model.weights[-1], scales[-1], False, variables[-1], is_first=False
        )
        energy = 0.5 * (y_pred - y).square().sum(dim=-1).mean()
        for residual, dual in zip(residuals, duals, strict=True):
            shifted = residual + dual / rho
            energy = energy + 0.5 * rho * shifted.square().sum() / x.shape[0]
        grads = torch.autograd.grad(energy, variables)
        free = [
            (z - state_lr * x.shape[0] * grad).detach()
            for z, grad in zip(variables, grads, strict=True)
        ]

        residuals = []
        for layer_ix, z_l in enumerate(free):
            z_prev = x if layer_ix == 0 else free[layer_ix - 1]
            pred = reference_block_pred(
                model.weights[layer_ix],
                scales[layer_ix],
                skips[layer_ix],
                z_prev,
                is_first=layer_ix == 0,
            )
            residuals.append(z_l - pred)
        duals = [
            (dual + alpha * residual).detach()
            for dual, residual in zip(duals, residuals, strict=True)
        ]
    return free, duals


def test_residual_mlp_matches_sakana_parameterization() -> None:
    model = ResidualMLP(
        depth=8,
        width=5,
        input_dim=3,
        output_dim=2,
        activation="tanh",
        seed=17,
        dtype=torch.float64,
    )
    x = torch.randn(4, 3, generator=torch.Generator().manual_seed(18), dtype=torch.float64)
    ours = model.forward_activations(x)
    ref = reference_forward(model, x)
    for a, b in zip(ours, ref, strict=True):
        assert torch.allclose(a, b, atol=1e-12, rtol=1e-12)


def test_one_step_pcalm_matches_sakana_update_order() -> None:
    model = ResidualMLP(
        depth=4,
        width=5,
        input_dim=3,
        output_dim=2,
        activation="tanh",
        seed=19,
        dtype=torch.float64,
    )
    gen = torch.Generator().manual_seed(20)
    x = torch.randn(7, 3, generator=gen, dtype=torch.float64)
    y = torch.randn(7, 2, generator=gen, dtype=torch.float64)
    state_lr = 0.1
    rho = 1.0
    alpha = 0.7
    budget = 1

    ours_free, _, _ = run_pcalm(
        model,
        x,
        y,
        state_lr=state_lr,
        rho=rho,
        alpha=alpha,
        budget=budget,
        inner_steps=1,
        weight_credit_timing="post_dual_energy",
    )
    ours_residuals = constraint_residuals(model, x, ours_free)
    ours_duals = [alpha * residual for residual in ours_residuals]
    ref_free, ref_duals = reference_pcalm(
        model,
        x,
        y,
        state_lr=state_lr,
        rho=rho,
        alpha=alpha,
        budget=budget,
    )

    for a, b in zip(ours_free, ref_free, strict=True):
        assert torch.allclose(a, b, atol=1e-12, rtol=1e-12)
    for a, b in zip(ours_duals, ref_duals, strict=True):
        assert torch.allclose(a, b, atol=1e-12, rtol=1e-12)
