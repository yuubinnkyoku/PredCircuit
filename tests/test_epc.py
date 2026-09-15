import torch

from predcircuit.epc import (
    epc_grad,
    error_energy,
    local_weight_energy,
    run_epc,
    states_from_errors,
    zero_errors,
)
from predcircuit.pcalm import ResidualMLP


def test_zero_errors_reconstruct_forward_pass() -> None:
    model = ResidualMLP(depth=5, width=4, input_dim=3, output_dim=2, seed=3)
    x = torch.randn(2, 3, generator=torch.Generator().manual_seed(4))

    errors = zero_errors(model, x)
    hidden, output = states_from_errors(model, x, errors)
    forward = model.forward_activations(x)

    assert len(errors) == model.depth - 1
    for reconstructed, reference in zip(hidden, forward[:-1], strict=True):
        torch.testing.assert_close(reconstructed, reference)
    torch.testing.assert_close(output, forward[-1])


def test_error_and_local_energy_have_same_value() -> None:
    model = ResidualMLP(depth=4, width=3, input_dim=2, output_dim=1, seed=7)
    gen = torch.Generator().manual_seed(8)
    x = torch.randn(3, 2, generator=gen)
    y = torch.randn(3, 1, generator=gen)
    errors = zero_errors(model, x)
    errors = [0.1 * torch.randn(error.shape, generator=gen, dtype=error.dtype) for error in errors]

    by_errors = error_energy(model, x, y, errors) / x.shape[0]
    local = local_weight_energy(model, x, y, errors)
    torch.testing.assert_close(local, by_errors, atol=1e-6, rtol=1e-6)


def test_epc_error_step_is_invariant_to_batch_duplication() -> None:
    model = ResidualMLP(depth=4, width=3, input_dim=2, output_dim=1, seed=9)
    x = torch.tensor([[0.25, -0.5]])
    y = torch.tensor([[0.15]])

    single, _ = run_epc(model, x, y, error_lr=0.1, steps=5)
    duplicated, _ = run_epc(model, x.repeat(4, 1), y.repeat(4, 1), error_lr=0.1, steps=5)

    for single_error, duplicated_error in zip(single, duplicated, strict=True):
        torch.testing.assert_close(
            duplicated_error,
            single_error.repeat(4, 1),
            atol=2e-7,
            rtol=1e-6,
        )


def test_epc_gradient_is_finite_and_matches_weight_shapes() -> None:
    model = ResidualMLP(
        depth=6,
        width=4,
        input_dim=3,
        output_dim=2,
        activation="relu",
        seed=10,
    )
    gen = torch.Generator().manual_seed(11)
    x = torch.randn(4, 3, generator=gen)
    y = torch.randn(4, 2, generator=gen)

    grads, finite = epc_grad(model, x, y, error_lr=0.1, steps=8)

    assert finite
    assert len(grads) == len(model.weights)
    for grad, weight in zip(grads, model.weights, strict=True):
        assert grad.shape == weight.shape
        assert torch.isfinite(grad).all()
