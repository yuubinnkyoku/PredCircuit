# PC-ALM 9-bit stochastic-weight training: 128-step screening

Date: 2026-09-21
Commit under test: `921f537f15835c56d180c9fd908e0fe82c688c6f`

## Setup

Four seeds (800--803), 128 training steps. The four configurations are the same as the short-horizon diagnostic:

- `fp32_lr0p01`
- `fp32_lr0p1`
- `q9_det_lr0p1`
- `q9_sr12_lr0p1`: 9-bit stored weights, 12-bit stochastic rounding, weight LR 0.1

All four workflow jobs completed successfully and all rows were finite.

## Aggregate results

Mean evaluation MSE across the four seeds:

| step | FP32 lr=.01 | FP32 lr=.1 | q9 deterministic lr=.1 | q9 stochastic-12 lr=.1 |
|---:|---:|---:|---:|---:|
| 1 | 1.057643 | 1.057572 | 1.058108 | 1.058341 |
| 16 | 1.057071 | 1.052517 | 1.056535 | 1.055822 |
| 32 | 1.055923 | 1.044938 | 1.053610 | 1.047177 |
| 64 | 1.055331 | 1.043369 | 1.052419 | 1.047971 |
| 128 | 1.052168 | 1.032030 | 1.045484 | 1.034532 |

The stochastic-9-bit minus FP32-lr=.1 MSE gap was:

- step 16: +0.003306
- step 32: +0.002239
- step 64: +0.004601
- step 128: +0.002502 (SD across seeds 0.001321)

At step 128 every seed had a positive but small stochastic-vs-FP32 gap: 0.000944, 0.002538, 0.002356, 0.004172 for seeds 800--803 respectively.

Mean step-1 to step-128 MSE improvement:

- FP32 lr=.1: 0.025542
- q9 deterministic lr=.1: 0.012624
- q9 stochastic-12 lr=.1: 0.023809

Thus the 9-bit stochastic configuration retained about 93.2% of the FP32-lr=.1 MSE improvement over this horizon, whereas deterministic 9-bit retained about 49.4%.

Mean logical weight move rate over all 128 steps:

- FP32 lr=.01: 90.12%
- FP32 lr=.1: 90.79%
- q9 deterministic lr=.1: 0.0424%
- q9 stochastic-12 lr=.1: 1.0092%

The stochastic configuration therefore remained strongly write-sparse while tracking the FP32 trajectory much more closely than deterministic rounding.

## Interpretation

The main failure mode being screened for was cumulative stochastic-rounding drift. It is not visible over 128 steps: the stochastic-vs-FP32 MSE gap does not grow monotonically from the 16-step value and is only +0.00250 on average at step 128. All four seeds remain finite.

This strengthens the case for 9-bit stored weights + 12-bit stochastic rounding as a hardware candidate. The result is still a small-network, four-seed screening result and is not yet sufficient to claim task-level convergence equivalence or an energy advantage. In particular, the observed ~1% logical write rate should be converted into an explicit BRAM read/write and RNG/comparator energy model before making a power claim.

## Next discriminating experiment

The next highest-value step is no longer another short trajectory. Quantify the hardware consequence of the observed write sparsity: compare per-training-step memory transactions and estimated dynamic memory energy for FP32/BP-like dense writes, deterministic q9, and stochastic q9, while including the 12-bit RNG/comparator overhead. In parallel, extend only the surviving stochastic candidate to a task-level training horizon if compute budget permits.
