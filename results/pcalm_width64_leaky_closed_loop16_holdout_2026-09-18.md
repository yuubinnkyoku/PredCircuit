# Leaky PC-ALM 16/16/12 closed-loop holdout (2026-09-18)

Source: GitHub Actions run `35343063671`, commit `9d5dd76b2e4675adf0fab96d81b7194676b7c7ab`. All 15 seed jobs (845--859) completed successfully and all 15 artifacts were recovered and aggregated without rerunning the experiment. CI run `35343063677` also passed.

## Question

Does moving the closed loop to one finer fractional bit preserve the previously observed dimensionless lattice rule

\[
R = \eta_{\mathrm{eff}}\Delta_u/(\Delta_h/2),
\]

with a sharp useful-credit transition at `R=1`?

Configuration: depth 32, width 64, batch 4, ReLU, T=256, dual leak 0.02, state `fixed16_i3`, update `fixed16_i2`, dual `fixed12_i1`. Here state half-LSB and update LSB are both `2^-13`, so `R=4*state_lr`. The three preregistered points are official Sakana `state_lr=0.234285` (`R=0.93714`), the predicted boundary 0.25 (`R=1`), and 0.26 (`R=1.04`).

## Aggregate results

| state_lr | R | useful credit | BP cosine mean | min cosine | grad-norm ratio mean | BP relative error mean | residual mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.234285 | 0.93714 | **15/15** | 0.952242 | 0.912375 | 0.846224 | 0.321261 | 0.291654 |
| 0.250000 | 1.00000 | **15/15** | 0.979336 | 0.966681 | 0.970709 | 0.201018 | 0.288446 |
| 0.260000 | 1.04000 | **15/15** | 0.993011 | 0.986107 | 1.025953 | 0.121967 | 0.287839 |

All 45 rows are finite. State/update/dual saturation is zero throughout.

## Matched-ratio falsification of an R-only law

The earlier `fixed15_i3` state + `fixed14_i1` update experiment has the *same state/update LSB ratio* and therefore the same `R=4*state_lr`, but both state and update grids are one fractional bit coarser. At the same official learning rate (`R=0.93714`) it produced only **8/15 useful**, cosine `0.882994`, and relative error `0.508913`. At `R=1` it jumped to 15/15 useful, cosine `0.988561`, relative error `0.151970`.

The new one-bit-finer pair is already **15/15 useful below R=1**. Therefore `R>=1` is **not a necessary condition for useful credit in general**, and the fixed-point dynamics are not scale invariant under a common one-bit refinement of the state and update grids.

This does not erase the earlier lattice-boundary evidence: for the coarser pair, crossing R=1 caused a reproducible 15-seed discontinuity, and changing the LSB ratio shifted that discontinuity as predicted. Instead, the new result qualifies the design rule: `R` captures a one-update-quantum reachability mechanism, but **absolute grid resolution is a second control variable** that can preserve useful aggregate credit even when a single update quantum cannot move a state by half an LSB.

A better hardware hypothesis is therefore two-dimensional:

1. `R` controls discrete one-quantum state reachability and predicts lock risk for a fixed absolute grid regime.
2. Absolute state/update resolution controls whether sub-threshold multi-quantum / accumulated dynamics can still approximate the floating-point credit trajectory.
3. Dynamic stability still supplies the upper learning-rate constraint; finer update precision cannot be rescued indefinitely by increasing `state_lr`.

At `state_lr=0.26`, the 16/16/12 pair reaches cosine `0.9930` and relative error `0.1220`, close to the FP32 leaky geometry previously measured around cosine 0.994 / norm ratio 1.02 / relative error 0.108. Thus the extra fractional bit is useful, but only when interpreted together with learning-rate margin rather than through `R>=1` as a binary validity test.

## Highest-value next test

Do not spend more runs merely sharpening the R=1 boundary. The strongest next falsification is an **absolute-scale matched family**: hold `R`, network, dual precision, and coefficients fixed while jointly shifting state and update fractional precision by +/-1 bit. Record late zero-step rate and realized/requested state motion in addition to gradient geometry. This directly measures the second dimension suggested here and can turn the former binary lattice rule into a hardware design surface `(R, absolute LSB, stability)`.
