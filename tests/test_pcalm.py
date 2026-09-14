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


def test_alpha_zero_matches_pc_same_primal_budget() -> None:
    model = ResidualMLP(depth=5, width=4, input_dim=3, output_dim=2, seed=2)
    x = torch.tensor([[0.3, -0.2, 0.1], [-0.1, 0.4, 0.2]])
    y = torch.tensor([[0.1, -0.3], [0.2, 0.05]])

    pc_free, _, _ = run_pc(model, x, y, state_lr=0.02, rho=1.0, steps=12)
    alm_free, alm_duals, _ = run_pcalm(
        model,
        x,
        y,
        state_lr=0.02,
        rho=1.0,
        alpha=0.0,
        budget=4,
        inner_steps=3,
    )

    for pc_state, alm_state in zip(pc_free, alm_free, strict=True):
        torch.testing.assert_close(pc_state, alm_state, atol=1e-7, rtol=1e-6)
    for dual in alm_duals:
        assert torch.count_nonzero(dual) == 0


def test_alpha_zero_matches_pc_gradient() -> None:
    model = ResidualMLP(
        depth=4,
        width=5,
        input_dim=3,
        output_dim=2,
        activation="tanh",
        seed=0,
    )
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(7, 3, generator=gen)
    y = torch.nn.functional.one_hot(torch.arange(7) % 2, 2).float()

    pc_grad = method_grad(
        model,
        x,
        y,
        Schedule("pc", budget=3),
        state_lr=0.1,
        rho=1.0,
    )
    alm_grad = method_grad(
        model,
        x,
        y,
        Schedule("pcalm", budget=3, alpha=0.0),
        state_lr=0.1,
        rho=1.0,
    )

    for pc_layer, alm_layer in zip(pc_grad, alm_grad, strict=True):
        torch.testing.assert_close(pc_layer, alm_layer, atol=1e-5, rtol=1e-5)


def test_batch_duplication_preserves_per_sample_state_step() -> None:
    model = ResidualMLP(depth=4, width=3, input_dim=2, output_dim=1, seed=4)
    x = torch.tensor([[0.25, -0.5]])
    y = torch.tensor([[0.15]])
    x_dup = x.repeat(4, 1)
    y_dup = y.repeat(4, 1)

    single, _, _ = run_pc(model, x, y, state_lr=0.03, rho=1.3, steps=5)
    duplicated, _, _ = run_pc(model, x_dup, y_dup, state_lr=0.03, rho=1.3, steps=5)

    for single_state, duplicated_state in zip(single, duplicated, strict=True):
        torch.testing.assert_close(
            duplicated_state,
            single_state.repeat(4, 1),
            atol=2e-7,
            rtol=1e-6,
        )


def test_pcalm_sample_is_invariant_to_other_batch_members() -> None:
    model = ResidualMLP(
        depth=4,
        width=5,
        input_dim=3,
        output_dim=2,
        activation="tanh",
        seed=0,
    )
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(7, 3, generator=gen)
    y = torch.nn.functional.one_hot(torch.arange(7) % 2, 2).float()

    single, _, _ = run_pcalm(
        model,
        x[:1],
        y[:1],
        state_lr=0.1,
        rho=1.0,
        alpha=1.0,
        budget=3,
        inner_steps=1,
    )
    batch, _, _ = run_pcalm(
        model,
        x,
        y,
        state_lr=0.1,
        rho=1.0,
        alpha=1.0,
        budget=3,
        inner_steps=1,
    )

    for single_state, batch_state in zip(single, batch, strict=True):
        torch.testing.assert_close(single_state[0], batch_state[0], atol=1e-5, rtol=1e-5)


def test_pre_and_post_dual_credit_are_distinct() -> None:
    model = ResidualMLP(depth=4, width=3, input_dim=2, output_dim=1, seed=5)
    x = torch.tensor([[0.2, -0.4]])
    y = torch.tensor([[0.7]])

    _, pre_duals, _ = run_pcalm(
        model,
        x,
        y,
        state_lr=0.04,
        rho=1.0,
        alpha=0.2,
        budget=1,
        inner_steps=2,
        weight_credit_timing="pre_dual_energy",
    )
    _, post_duals, _ = run_pcalm(
        model,
        x,
        y,
        state_lr=0.04,
        rho=1.0,
        alpha=0.2,
        budget=1,
        inner_steps=2,
        weight_credit_timing="post_dual_energy",
    )

    assert all(torch.count_nonzero(dual) == 0 for dual in pre_duals)
    assert any(torch.count_nonzero(dual) > 0 for dual in post_duals)


def test_trace_records_finite_and_dual_growth() -> None:
    model = ResidualMLP(depth=6, width=4, input_dim=3, output_dim=2, seed=7)
    x = torch.randn(2, 3, generator=torch.Generator().manual_seed(1))
    y = torch.randn(2, 2, generator=torch.Generator().manual_seed(2))

    _, _, trace = run_pcalm(
        model,
        x,
        y,
        state_lr=0.01,
        rho=1.0,
        alpha=0.1,
        budget=3,
        inner_steps=2,
        record_trace=True,
    )

    assert trace is not None
    assert len(trace.residual_norms) == 4
    assert len(trace.dual_norms) == 4
    assert trace.max_abs_dual[0] == 0.0
    assert trace.max_abs_dual[-1] > 0.0
    assert trace.finite


def test_linear_pcalm_gradient_moves_toward_bp() -> None:
    model = ResidualMLP(
        depth=4,
        width=2,
        input_dim=2,
        output_dim=1,
        activation="linear",
        seed=11,
        dtype=torch.float64,
    )
    x = torch.tensor([[0.2, -0.1]], dtype=torch.float64)
    y = torch.tensor([[0.35]], dtype=torch.float64)

    bp = method_grad(model, x, y, Schedule("bp", budget=0), state_lr=0.01, rho=1.0)
    short = method_grad(
        model,
        x,
        y,
        Schedule("pcalm", budget=2, alpha=0.2, inner_steps=2),
        state_lr=0.01,
        rho=1.0,
    )
    converged = method_grad(
        model,
        x,
        y,
        Schedule("pcalm", budget=200, alpha=0.2, inner_steps=2),
        state_lr=0.01,
        rho=1.0,
    )

    short_cosine = gradient_cosine(short, bp)
    converged_cosine = gradient_cosine(converged, bp)
    short_error = gradient_relative_error(short, bp)
    converged_error = gradient_relative_error(converged, bp)

    assert converged_cosine > short_cosine
    assert converged_cosine > 0.98
    assert converged_error < short_error
