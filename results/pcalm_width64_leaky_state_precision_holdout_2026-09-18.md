# Width-64 leaky PC-ALM state-precision holdout

Source: GitHub Actions run `35274752473`, commit `5d4e1d8d`, seeds 845--859. All 15 seed jobs completed successfully and all artifacts were recovered without rerunning the experiment.

Configuration: ResidualMLP depth 32, width 64, batch 4, ReLU, `T=256`, official Sakana depth-32 `state_lr=0.234285`, `rho=1`, `alpha=0.925`, `dual_leak=0.02`. The update/residual datapath is fixed at `fixed14_i1`; the dual remains `fixed12_i1`. The experiment varies only hidden-state precision while preserving the `i3` state range.

Useful first-layer credit means finite, cosine to BP >= 0.9, norm ratio in [0.5, 2.0], and BP-relative error <= 0.6.

## Holdout aggregate

| state precision | useful | mean cosine | mean norm/BP | mean BP-relative error | mean all-gradient error vs FP32 | mean residual | state saturation |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 15/15 | 0.994314 | 1.020262 | 0.107833 | 0 | 0.245252 | 0 |
| `fixed14_i3` | **15/15** | 0.968459 | 0.902834 | 0.256994 | 0.191110 | 0.315150 | 0 |
| `fixed15_i3` | **8/15** | 0.882994 | 0.675067 | 0.508913 | 0.378972 | 0.270198 | 0 |
| `fixed16_i3` | **15/15** | **0.983727** | **0.975926** | **0.179098** | **0.121683** | 0.290507 | 0 |

Dual and update saturation were also zero in every fixed-point condition. The maximum observed quantized dual magnitude was below 0.18, so the non-monotone state-precision result is not explained by dual range pressure.

## Non-monotone precision effect

The surprising result is systematic, not a single threshold-crossing seed:

- `fixed15_i3` has larger BP-relative error than `fixed14_i3` on **15/15 seeds**. The paired mean increase is +0.251920.
- `fixed16_i3` improves on `fixed15_i3` on **15/15 seeds**. The paired mean decrease is -0.329815.
- `fixed16_i3` also improves on `fixed14_i3` on **15/15 seeds**, with paired mean decrease -0.077896.

All three formats use the same state integer range and have zero state saturation. Therefore the ordering cannot be attributed to clipping. The quantizer is the same saturating round-to-nearest implementation for every fixed-point width; only the state lattice spacing changes (`2^-10`, `2^-11`, `2^-12` respectively).

This falsifies a naive monotone-width assumption for the iterative state dynamics: adding one fractional state bit from 14 to 15 can move the closed-loop trajectory farther from the FP32/BP-credit operating point, while adding the next bit restores and improves it.

## Interpretation

Treat state precision as a discrete dynamical-system parameter, not only as an approximation-error budget. A plausible mechanism is a quantization-induced change of trajectory / limit-cycle structure: the hidden state is fed back for 256 outer updates, so a different rounding lattice can change subsequent residuals and dual updates even without saturation.

The result does **not** yet identify the mechanism. In particular, it is premature to call it stochastic regularization, damping, or resonance without a trajectory-level measurement.

## Hardware implication

Two currently defensible operating points are:

- compact: `state=fixed14_i3`, `update=fixed14_i1`, `dual=fixed12_i1` -- 15/15 useful, lower state storage;
- higher-fidelity: `state=fixed16_i3`, `update=fixed14_i1`, `dual=fixed12_i1` -- 15/15 useful and materially closer to FP32.

`fixed15_i3` should **not** be selected merely because it lies numerically between 14 and 16 bits.

## Next diagnostic

Measure trajectories for state 14/15/16 bits on the same held-out seeds: per-step residual norm, dual norm, state-change norm, zero-update fraction, and checkpoint gradient geometry. The target is to determine whether 15-bit state enters an oscillatory/limit-cycle regime or simply follows a shifted convergence window. Only after that should the RTL default choose between 14- and 16-bit hidden state.
