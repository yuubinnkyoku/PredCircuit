# Shifted lattice-boundary holdout (2026-09-18)

Source: GitHub Actions run 35297891359, commit `baf181eafd9913c01704a61b18cbe7250734f4cb`. All 15 ratio-shift artifacts (seeds 845--859) were recovered; 210 rows were aggregated without rerunning the experiment.

## Pre-registered boundary test: `fixed13_i1`

With state precision fixed at `fixed15_i3`, changing update precision from the earlier `fixed14_i1` case to `fixed13_i1` doubles the update LSB. The predicted lattice boundary therefore shifts from `state_lr = 0.250` to `0.125`, according to

`R = effective_lr * update_LSB / state_half_LSB`, with the predicted transition at `R = 1`.

| state_lr | R | BP cosine mean ± sd | BP relative error mean | late state zero-step | useful credit |
|---:|---:|---:|---:|---:|---:|
| 0.110 | 0.880 | 0.5484 ± 0.1963 | 0.8815 | 99.9791% | 0/15 |
| 0.120 | 0.960 | 0.5488 ± 0.1989 | 0.8809 | 99.9783% | 0/15 |
| 0.124 | 0.992 | 0.5479 ± 0.2024 | 0.8798 | 99.9885% | 0/15 |
| **0.125** | **1.000** | **0.9324 ± 0.0213** | **0.4129** | **93.1391%** | **14/15** |
| 0.126 | 1.008 | 0.9641 ± 0.0169 | 0.2719 | 99.3032% | 15/15 |
| 0.130 | 1.040 | 0.9645 ± 0.0157 | 0.2695 | 99.2783% | 15/15 |
| 0.140 | 1.120 | 0.9647 ± 0.0164 | 0.2698 | 99.2270% | 15/15 |

Paired `0.124 -> 0.125` changes across all 15 seeds:

- BP cosine improved in **15/15** seeds; mean delta `+0.38449`, range `+0.18103 .. +0.82460`.
- BP relative error improved in **15/15** seeds; mean delta `-0.46688`.
- late state zero-step fraction decreased in **15/15** seeds; mean delta `-0.06849`.
- useful first-layer credit changed from **0/15** to **14/15** at the exact predicted boundary, and to **15/15** at `0.126`.
- update/state/dual saturation is zero throughout this `fixed13_i1` sweep, so the boundary recovery is not explained by clipping or overflow.

This independently shifts the observed gradient-geometry transition from the earlier `0.250` boundary to the pre-predicted `0.125` boundary when the update LSB doubles. It strongly disfavors the alternative explanation that `state_lr ≈ 0.25` is merely a favorable continuous-valued PC-ALM learning rate.

## Finer update precision: `fixed15_i1`

The same formula predicts a boundary at `state_lr = 0.500`, but this test is confounded by dynamical instability before the lattice boundary can be isolated.

| state_lr | BP cosine mean ± sd | BP relative error mean | update saturation | residual total mean | useful credit |
|---:|---:|---:|---:|---:|---:|
| 0.440 | -0.1678 ± 0.1323 | 32.86 | 79.87% | 299.62 | 0/15 |
| 0.480 | -0.1657 ± 0.1436 | 41.29 | 83.45% | 329.75 | 0/15 |
| 0.495 | -0.1513 ± 0.1382 | 45.32 | 84.61% | 341.17 | 0/15 |
| 0.500 | -0.0029 ± 0.0542 | 47.49 | 84.82% | 342.52 | 0/15 |
| 0.505 | -0.0218 ± 0.0828 | 47.63 | 85.51% | 347.06 | 0/15 |
| 0.520 | -0.0253 ± 0.0866 | 49.69 | 86.74% | 362.83 | 0/15 |
| 0.560 | 0.0089 ± 0.0786 | 58.51 | 88.32% | 376.37 | 0/15 |

This is not evidence against the lattice rule: the continuous PC-ALM dynamics are already in a saturated/high-residual regime. It instead yields a second hardware constraint: a usable fixed-point format must make the `R >= 1` lattice condition reachable *inside* the dynamical stability region.

## Current design-rule hypothesis

A fixed-point PC-ALM state/update pair is viable only if there exists a state learning rate satisfying both:

1. `effective_lr * update_LSB >= state_LSB / 2` (avoid one-quantum lattice lock), and
2. the corresponding PC-ALM relaxation remains dynamically stable without material saturation.

The first condition now has two boundary locations consistent with the same dimensionless rule (`0.250` for the earlier update LSB, `0.125` after doubling it). The second condition prevents arbitrarily fine update quantization from being rescued by increasing the learning rate.

## Highest-value next test

Measure the upper stable `state_lr` boundary in floating point / sufficiently fine non-saturating precision under the same depth-32 width-64 leaky-dual configuration. Combining that empirical stability ceiling with the lattice inequality will give a direct upper bound on the allowable `state_LSB / update_LSB` ratio for RTL numeric-format selection, rather than another isolated bit-width table.
