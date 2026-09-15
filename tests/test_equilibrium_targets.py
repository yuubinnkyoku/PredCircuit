import pytest

from scripts.diagnose_equilibrium_targets import (
    exact_pc,
    exact_pcalm,
    run_pcalm,
    run_spc,
)


def test_spc_and_pcalm_converge_to_distinct_exact_targets() -> None:
    a, b, y = 0.7, 1.2, -0.4
    pc_star = exact_pc(a, b, y)
    pcalm_h_star, pcalm_lam_star = exact_pcalm(a, b, y)

    assert pc_star == pytest.approx(0.0901639344262295)
    assert pcalm_h_star == pytest.approx(0.7)
    assert pcalm_lam_star == pytest.approx(-1.488)
    assert abs(pc_star - pcalm_h_star) > 0.6

    spc_h = run_spc(a, b, y, lr=0.1, steps=100)
    pcalm_h, pcalm_lam = run_pcalm(a, b, y, lr=0.1, alpha=1.0, rho=1.0, steps=100)

    assert spc_h == pytest.approx(pc_star, abs=1e-9)
    assert pcalm_h == pytest.approx(pcalm_h_star, abs=1e-6)
    assert pcalm_lam == pytest.approx(pcalm_lam_star, abs=2e-6)
