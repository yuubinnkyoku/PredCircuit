from __future__ import annotations

import math

import torch

from predcircuit.magnitude_control import (
    credit_metrics,
    epc_stationarity,
    mac_accounting,
    oracle_norm_match,
    residual_norm_gain,
    spc_credit,
    spc_grad,
)
from predcircuit.pcalm import ResidualMLP, Schedule, method_grad
from predcircuit.epc import run_epc


def _tiny_model(seed: int = 0) -> ResidualMLP:
    return ResidualMLP(
        depth=4,
        width=4,
        input_dim=3,
        output_dim=2,
        activation="relu",
        seed=seed,
    )


def test_oracle_norm_match_sets_first_layer_norm() -> None:
    g0 = torch.tensor([[1.0, 2.0], [0.0, 0.5]])
    g1 = torch.tensor([[3.0]])
    matched = oracle_norm_match([g0, g1], bp_first_norm=2.5)
    assert torch.isclose(matched[0].norm(), torch.tensor(2.5))
    assert torch.allclose(matched[1], g1)


def test_residual_norm_gain_scales_inversely() -> None:
    grads = [torch.ones(2, 2), torch.ones(2, 2)]
    out = residual_norm_gain(grads, [2.0, 2.0])
    assert torch.allclose(out[0], grads[0] * 0.5)


def test_credit_metrics_detects_pure_scale() -> None:
    bp = [torch.ones(2, 2) * 2.0]
    scaled = [torch.ones(2, 2) * 0.004]
    metrics = credit_metrics(scaled, bp)
    assert metrics["first_layer_cosine_to_bp"] > 0.99
    assert metrics["first_layer_grad_norm_ratio_to_bp"] < 0.01
    assert not metrics["useful_first_layer_credit"]
    matched = oracle_norm_match(scaled, float(bp[0].norm()))
    metrics2 = credit_metrics(matched, bp)
    assert metrics2["useful_first_layer_credit"]
    assert metrics2["first_layer_relative_error_to_bp"] < 1e-5


def test_spc_grad_runs_and_finite() -> None:
    model = _tiny_model()
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(2, 3, generator=gen)
    y = torch.randn(2, 2, generator=gen)
    grads, residual_norms, finite = spc_grad(model, x, y, state_lr=0.1, rho=1.0, budget=3)
    assert finite
    assert len(grads) == model.depth
    assert len(residual_norms) == model.depth - 1
    assert all(math.isfinite(r) for r in residual_norms)


def test_epc_stationarity_and_spc_credit() -> None:
    model = _tiny_model(seed=2)
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(2, 3, generator=gen)
    y = torch.randn(2, 2, generator=gen)
    spc_ref = spc_credit(model, x, y, state_lr=0.1, rho=1.0, budget=4)
    assert len(spc_ref) == model.depth

    errors, _trace = run_epc(model, x, y, error_lr=0.1, steps=3, record_trace=True)
    stat = epc_stationarity(model, x, y, errors)
    assert math.isfinite(stat)

    bp = method_grad(model, x, y, Schedule("bp", budget=0), state_lr=0.1, rho=1.0)
    metrics = credit_metrics(spc_ref, bp)
    assert metrics["finite"]


def test_mac_accounting_families() -> None:
    for family in ("epc", "spc", "pcalm", "bp"):
        costs = mac_accounting(
            family=family,
            depth=4,
            width=4,
            batch_size=2,
            budget=5,
        )
        assert costs["macs_per_step_estimate"] > 0
        if family != "bp":
            assert costs["total_macs_estimate"] == costs["macs_per_step_estimate"] * 5
