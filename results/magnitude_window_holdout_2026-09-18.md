# Magnitude-window holdout: sPC scale control and ePC equilibrium compute

Date: 2026-09-18
Source: Actions runs 35244069612 (sPC magnitude) and 35244061762 (ePC equilibrium), commit family `df81f35`..`9d2aaef`.
Downloaded artifacts: **20/20 seeds** for sPC (final aggregate), 19/20 for ePC.
Network: ResidualMLP depth=32, width=8, batch=4, ReLU, state_lr=0.25, rho=1.0.
Controls: raw sPC; oracle first-layer BP-norm match; residual-norm gain; PC-ALM dual_leak=0.01 alpha=0.925.

Useful credit: first-layer cosine to BP ≥ 0.9, norm ratio in [0.5, 2], relative error ≤ 0.6, finite.

## E1 — sPC magnitude control (20 seeds)

| T | control | useful | mean cosine | mean norm ratio | mean relative error |
|---:|---|---:|---:|---:|---:|
| 32 | spc_raw | 0/20 | — | 0 | 1.000 |
| 32 | pcalm_dual_leak | 0/20 | 0.083 | ~0 | 1.000 |
| 64 | spc_raw | 0/20 | 0.222 | ~0 | 1.000 |
| 64 | spc_oracle_norm_match | 0/20 | 0.222 | 0.650* | 1.150 |
| 64 | pcalm_dual_leak | 1/20 | 0.883 | 0.409 | 0.666 |
| 128 | spc_raw | 0/20 | 0.884 | 0.000082 | 0.9999 |
| 128 | spc_oracle_norm_match | **9/20 (45%)** | 0.884 | 1.000 | 0.466 |
| 128 | spc_residual_norm_gain | 0/20 | 0.884 | 0.001 | 0.999 |
| 128 | pcalm_dual_leak | **20/20** | 0.948 | 1.009 | 0.330 |
| 256 | spc_raw | 0/20 | 0.830 | 0.0044 | 0.996 |
| 256 | spc_oracle_norm_match | **4/20 (20%)** | 0.830 | 1.000 | 0.565 |
| 256 | pcalm_dual_leak | **20/20** | 0.956 | 1.021 | 0.317 |

\*Oracle mean norm ratio at T=64 is diluted by seeds whose first-layer credit is exactly zero (cannot be rescaled).

### E1 findings

1. **Magnitude collapse is the dominant raw sPC failure mode.** At T=128 the mean first-layer cosine is already 0.884 while the credit norm is ~0.008% of BP.
2. **Oracle scale restore is necessary but not sufficient.** It recovers 45% of seeds at T=128 and only 20% at T=256. Direction residual remains after scale is fixed; longer sPC relaxation slightly *worsens* mean cosine (0.884 → 0.830).
3. **Naive residual-norm gain does not rescue deep sPC credit.**
4. **Dual-leak PC-ALM succeeds on every seed at T∈{128,256}**, with higher cosine than sPC *and* unit-scale credit. The advantage is not “just a better stopping time” — it is a usable fixed-budget credit window.

## E2 — ePC equilibrium + compute (19 seeds)

| error_lr | best stationarity_ok rate | useful BP rate | energy behavior |
|---:|---:|---:|---|
| 0.1 | 0.21 (T=64) | 0/19 all T | energy decreases mildly |
| 0.3 | 0.11 (T=16) | 0/19 all T | noisy, not stationary |
| 0.8 | 0 | 0/19 all T | energy often increases |
| 1.0 | 0 | 0/19 all T | diverges |

Mean first-layer cosine to BP stays ≤0.74 even at error_lr=0.1; cosine to long-run sPC credit is ~0.67–0.81, not equilibrium agreement. Analytical compute: ePC step MAC lower bound is 2× sPC at equal T in this accounting model; persistent error-state bits equal sPC free-state bits.

### E2 findings

Under the shared ResidualMLP diagnostic, **ePC is not a competitive deep-credit method** at these coefficients: it neither reaches BP-like useful first-layer credit nor a clean stationary point in the tested grid. This does **not** refute the PC-ALM paper’s training results; it freezes a negative for this *gradient-geometry / equilibrium* gate and keeps ePC out of the current hardware shortlist unless official training-scale dynamics change the picture.

## Cross-line non-trivial conclusion (holdout-supported)

> Local credit methods fail when the **usable magnitude window** desynchronizes from the update scale. On deep residual MLPs, raw sPC already carries substantial BP-direction information but dies in magnitude; restoring scale alone only works for a subset of seeds because direction residual remains. Dual-leak PC-ALM is the only tested local method that reliably opens a **fixed-budget, unit-scale, high-cosine** credit window. On the FlyVis line, residual credit attenuation and late-gate braking independently show that *direction residual* and *late magnitude growth* are separate failure channels. Exact gradients win by coupling both controls globally — not because local rules are information-free.

Hardware implication (strengthened): the FPGA path should bank on **magnitude-stabilized dual dynamics (leak + Q3.9 dual state + 14/14/12 mixed precision)** rather than on pure sPC with more T or on ePC T=1 collinearity. sPC remains the honest baseline that fails the useful-credit gate even at T=1024 (width-64 note) and at T≤256 here.

## Remaining CI

- sPC holdout completed success with all 20 seeds.
- ePC holdout 19/20 at first download; rates unchanged by missing seed on inspected aggregates.
- Unrelated FlyVis workflows triggered by `src/predcircuit/**` were cancelled.

## Artifacts

- `results/artifacts/magnitude-window-spc/`
- `results/artifacts/magnitude-window-epc/`
- Aggregates: `results/generated/spc_magnitude_holdout_agg.csv`, `results/generated/epc_equilibrium_holdout_agg.csv` (gitignored generated/)
