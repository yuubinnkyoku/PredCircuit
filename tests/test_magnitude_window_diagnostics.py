from __future__ import annotations

import math

import pytest
import torch

from predcircuit.magnitude_control import (
    credit_metrics,
    epc_stationarity,
    mac_accounting,
    oracle_norm_match,
    residual_norm_gain,
    residual_norm_gain_then_unit_first,
    spc_credit,
    spc_grad,
    unit_first_layer_norm,
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


def test_residual_gain_then_unit_first_matches_spec() -> None:
    grads = [torch.ones(2, 2) * 0.01, torch.ones(3, 2) * 0.2]
    residual_norms = [4.0, 4.0]
    stage1 = residual_norm_gain(grads, residual_norms)
    stage2 = residual_norm_gain_then_unit_first(grads, residual_norms)
    assert torch.allclose(stage1[0], grads[0] * 0.25)
    assert torch.isclose(stage2[0].norm(), torch.tensor(1.0))
    # Stage 2 only rescales layer 0; deeper layers keep stage-1 gain.
    assert torch.allclose(stage2[1], stage1[1])
    unit = unit_first_layer_norm(grads)
    assert torch.isclose(unit[0].norm(), torch.tensor(1.0))
    assert torch.allclose(unit[1], grads[1])


def test_credit_metrics_unit_first_layer_geometry() -> None:
    bp = [torch.ones(2, 2) * 2.0, torch.ones(2, 2)]
    raw = [torch.ones(2, 2) * 0.01, torch.ones(2, 2) * 0.5]
    unit = unit_first_layer_norm(raw)
    m = credit_metrics(unit, bp)
    assert m["first_layer_cosine_to_bp"] > 0.99
    assert m["first_layer_grad_norm"] == pytest.approx(1.0, rel=1e-5)
    # Unit first-layer is not automatically useful vs BP-norm criterion.
    assert not m["useful_first_layer_credit"] or m["first_layer_grad_norm_ratio_to_bp"] <= 2.0


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
