# PC-ALM width-64 numeric-range result (2026-09-22)

Configuration: depth=32, width=64, seed=940, relaxation budget=128, state_lr=0.234285, rho=1, alpha=1. Source: successful `PC-ALM numeric range` Actions run for commit `e29d34c`.

## Observed FP32 trajectory

- finite: true
- max |lambda|: 0.1861758232
- max |residual| = max |alpha r|: 0.0262741484
- max |W^T r| final sum: 0.0101073040
- max absolute partial sum while accumulating W^T r: 0.0103478888
- partial/final range ratio: 1.023803
- required signed integer bits for the observed W^T r range: 1 (including sign in the probe convention)

The important result is that widening from N=8 to N=64 did **not** produce a large hidden accumulation excursion in this reference trajectory: the worst partial sum was only 2.38% above the worst final sum. Thus the earlier provisional 14-bit accumulator is not range-limited here; fractional precision, not overflow range, is the next question. This is one seed/configuration and is not a universal bound.

## Dual-update resolution

Among nonzero FP32 dual updates, p50=1.5234947e-4, p10=1.6689301e-5, p01=2.3841858e-7. The fraction that would round to zero under a global nearest fixed-point format sized only from the observed lambda range is:

- 16-bit, Q1.15-like: 9.37%
- 12-bit, Q1.11-like: 65.99%
- 8-bit, Q1.7-like: 99.68%

This is substantially harsher than the earlier width-8 probe. Therefore a 12-bit lambda path can have ample dynamic range while still discarding most small dual increments. Any claim that lambda12 is a faithful numerical approximation needs trajectory-level validation; range safety alone is insufficient.

## Hardware implication

For the P=32, N=64 core, accumulator width should no longer be selected from worst-case `64 * max(product)` growth. The measured partial-sum range is tiny in this trajectory. Candidate widths such as 14/16/18 bits should instead be compared by quantizing the *accumulation operation itself* and measuring PC-ALM convergence/gradient geometry. Lambda needs the complementary experiment: 12-bit nearest versus 16-bit nearest and, only if necessary, an unbiased low-cost update rule.
