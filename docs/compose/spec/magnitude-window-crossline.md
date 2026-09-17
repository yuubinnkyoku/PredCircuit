---
feature: magnitude-window-crossline
status: designed
updated: 2026-09-18
branch: main
commits: # filled at delivery
---

# Magnitude-window cross-line conclusion

## Report

## [S1] Problem

PredCircuit has two mature but separate research lines:

1. FlyVis local temporal PC / type-pair shared credit: topology interaction, residual credit attenuation, late 4m+r recovery, Goldilocks init band, late-gate brake.
2. Residual-MLP credit methods (sPC / PC-ALM / ePC): deep first-layer credit geometry, dual-leak windows, fixed-point formats, RTL cost.

Each line already contains magnitude/dynamics failures that look like "the learning rule is wrong" when read alone. There is no durable synthesis that states the cross-line mechanism, and the residual-MLP line still lacks experiments that *separate direction quality from magnitude control* on sPC and that evaluate ePC on an equilibrium axis rather than BP geometry alone.

Without that separation, the hardware claim ("local dynamics can beat global credit") rests on dual-leak PC-ALM numbers that critics can attribute to a one-off engineering tweak.

## [S2] Design

### Claim under test (non-trivial conclusion candidate)

> Local credit methods often already carry useful **direction** information; they fail when **credit magnitude / temporal synchronization** leaves a usable window. Stabilizers that only control magnitude or timing (dual leak, residual attenuation, ungated late shared credit, Goldilocks init, residual-norm gates) recover performance without inventing a new biological learning rule. Exact gradients win because they couple global magnitude control to the descent direction.

This claim must be **falsifiable** by the new diagnostics below, not asserted from cherry-picked tables.

### Experiment contracts

#### E1 — sPC magnitude-control diagnostic

Network: `ResidualMLP(depth=32, width=8, input_dim=8, output_dim=4, activation=relu)`, batch 4, seeds 960–979, budgets T ∈ {32,64,128,256}.

Baseline: `method_grad(..., Schedule("pc", budget=T), state_lr=0.25, rho=1.0)`.

Controls (diagnostic only; never claimed as deployable rules unless local):

1. **oracle_norm_match**: first-layer credit rescaled to BP first-layer norm (layer-0 only; other layers untouched).
2. **residual_norm_gain**: multiply the whole credit list by `1 / (mean_hidden_residual_norm + eps)` then rescale to unit first-layer cosine-preserving magnitude — reports both raw and norm-matched geometry.
3. **pcalm_leak_ref**: PC-ALM `dual_leak=0.01`, `alpha=0.925`, same shape/T, as the magnitude-stabilized local reference.

Record per seed/T/control: finite, first-layer cosine to BP, norm ratio, relative error, residual norms, dual max |λ| for PC-ALM, useful predicate (cosine≥0.9, norm∈[0.5,2], rel≤0.6).

**Acceptance (for the claim):** after oracle_norm_match, if mean relative error remains ≫ PC-ALM leak useful regime while cosine stays high, magnitude is necessary but not sufficient; if rel error collapses toward PC-ALM levels, sPC failure is predominantly scale/synchronization. Either outcome is informative and must be recorded.

#### E2 — ePC equilibrium + compute diagnostic

Same network family. ePC error_lrs ∈ {0.1,0.3,0.8,1.0}, T ∈ {8,16,32,64,128}.

Record:

- energy at each step; relative energy decrease; error-variable grad norm (stationarity);
- first-layer BP geometry metrics (reported, not used for stopping);
- agreement with long-run sPC credit (cosine after T=256 sPC) as an equilibrium-oriented proxy;
- analytical compute accounting: ePC reverse-mode-like MAC lower bound per step vs sPC/PC-ALM matrix-MAC model in `predcircuit.hardware`.

**Acceptance:** a documented comparison table at matched stationarity (relative error-energy change ≤1e-3) and/or matched useful-credit quality, with MAC/state estimates on both axes. Do not tune ePC solely to cross BP norm ratio at T=1.

#### E3 — Cross-line synthesis document

Write `docs/magnitude-window-crossline.md` using only:

- committed results under `results/` and research logs;
- E1/E2 outputs when present;
- explicit caveats (gradient-geometry ≠ training accuracy; dual leak ≠ Sakana pure PC-ALM; FlyVis residual attenuation used an oracle projection).

Structure: shared mechanism table → line-specific evidence → hardware implication → falsifiers → next experiment.

### Out of scope

- Full training accuracy for PC-ALM/ePC on FlyVis (separate feature).
- Claiming biological learning uses dual leak.
- Extending RTL beyond existing measured engines.
- Re-running completed dual-leak / Q3.9 holdouts.

## Tasks

- [ ] T1: Add `scripts/diagnose_spc_magnitude_control.py` — acceptance: runs one seed locally, writes CSV with E1 columns (covers: S2 E1)
- [ ] T2: Add `scripts/diagnose_epc_equilibrium_compute.py` — acceptance: runs one seed locally, writes energy/stationarity/compute CSV (covers: S2 E2)
- [ ] T3: Unit tests for new diagnostics' pure helpers — acceptance: `uv run pytest` passes (covers: S2 E1,E2)
- [ ] T4: GitHub Actions workflows for E1/E2 holdouts — acceptance: workflow files trigger on path push (covers: S2 E1,E2)
- [ ] T5: Local smoke + ruff/ty — acceptance: lint/typecheck clean on new scripts (covers: S2 E1,E2)
- [ ] T6: Commit and push to trigger CI — acceptance: Actions runs queued (covers: S2 E1,E2)
- [ ] T7: Synthesize `docs/magnitude-window-crossline.md` from existing + CI evidence — acceptance: falsifiable claim stated with evidence table (covers: S2 E3)
- [ ] T8: Overnight loop: collect CI artifacts, update research log, refine conclusion — acceptance: results recorded or blocked reasons listed (covers: S2 E3)
