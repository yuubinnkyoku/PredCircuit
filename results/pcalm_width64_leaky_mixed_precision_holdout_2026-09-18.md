# Width-64 leaky PC-ALM mixed-precision holdout (2026-09-18)

## Setup

- depth 32, width 64, batch 4
- holdout seeds 845--859 (15 seeds)
- T = 256
- state_lr = 0.234285
- rho = 1.0, alpha = 0.925
- dual_leak = 0.02
- FP32 leaky PC-ALM compared against fixed-point `update/residual=fixed14_i2`, `state=fixed14_i3`, `dual=fixed12_i1`
- useful first-layer credit: finite, cosine >= 0.9, norm ratio in [0.5, 2.0], BP-relative error <= 0.6

## Aggregate results

| configuration | finite | useful | residual total (mean) | max |lambda| (mean / max) | first-layer cosine | norm / BP | BP-relative error | all-gradient error vs FP32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 leaky | 15/15 | 15/15 | 0.24525 | 0.13403 / 0.17780 | 0.99431 | 1.02026 | 0.10783 | 0 |
| fixed 14/14/12 leaky | 15/15 | 15/15 | 0.30223 | 0.11406 / 0.16992 | 0.93521 | 0.78046 | 0.38413 | 0.28695 |

All state, update/residual, and dual saturation rates were exactly 0 across all 15 seeds for both configurations. The fixed-point maximum observed dual magnitude was 0.169921875, far from the `fixed12_i1` representable boundary; the observed failure mode is therefore not saturation or overflow.

## Interpretation

The fixed 14/14/12 design point is robust under the leaky-dual dynamics by the pre-registered useful-credit criterion (15/15 finite and useful), and 12-bit dual storage has ample dynamic-range margin in this experiment. However, quantization still materially degrades gradient geometry relative to FP32 leaky PC-ALM: mean cosine falls from 0.9943 to 0.9352, norm ratio from 1.020 to 0.780, and BP-relative error rises from 0.108 to 0.384. Thus `dual_leak=0.02` solves the dual-range concern but does not make 14/14/12 numerically equivalent to FP32.

The next precision experiment should not widen the dual: dual saturation is zero and its range is small. The information-rich follow-up is to widen the closed-loop state/update path (e.g. 16/16/12, plus asymmetric 16/14/12 and 14/16/12) under the same leaky holdout, or to test finer fractional scaling at 14 bits. This directly targets the remaining quantization error while preserving the demonstrated 12-bit dual-memory advantage.
