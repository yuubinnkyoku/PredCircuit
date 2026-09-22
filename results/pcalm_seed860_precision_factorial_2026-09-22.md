# PC-ALM seed 860 precision factorial diagnostic (2026-09-22)

## Setup

- depth=32, width=64, seed=860, T=128
- state_lr=0.25, dual_leak=0.02
- factors: update=fixed14_i1, state=fixed15_i3, dual=fixed12_i1 versus FP32
- Reused the completed `pcalm-seed860-T128-state-precision` artifact from Actions run 35685496330; no experiment was rerun.

## Results

| config | BP cosine | grad norm / BP | BP relative error | residual total | all-grad error vs FP32 | useful |
|---|---:|---:|---:|---:|---:|:---:|
| FP32 | 0.967973 | 0.944652 | 0.252133 | 0.245340 | 0.000000 | yes |
| update14 only | 0.967457 | 0.946716 | 0.253884 | 0.254458 | 0.018773 | yes |
| state15 only | 0.589989 | 1.533182 | 1.241582 | 0.373965 | 1.136245 | no |
| dual12 only | 0.954491 | 0.890513 | 0.305025 | 0.281749 | 0.096071 | yes |
| update14 + state15 | 0.583934 | 1.530523 | 1.247018 | 0.390046 | 1.138751 | no |
| update14 + dual12 | 0.947221 | 0.881620 | 0.327224 | 0.285923 | 0.104044 | yes |
| state15 + dual12 | 0.555207 | 1.505509 | 1.262861 | 0.373212 | 1.140721 | no |
| update14 + state15 + dual12 | 0.564590 | 1.507472 | 1.253102 | 0.396882 | 1.138368 | no |

All eight trajectories are finite. Update and dual saturation are zero in every row. Every row containing fixed15_i3 state has the same small state saturation rate, 1.4668e-4; rows with FP32 state have zero state saturation.

## Interpretation

The factorial isolates the seed-860 failure to **state quantization itself**, not update or dual quantization. `state15 only` already reproduces almost the entire failure: first-layer cosine falls from 0.968 to 0.590 and all-gradient relative error rises to 1.136. In contrast, `update14 only` is essentially FP32 (cosine 0.967), and `dual12 only` remains useful (cosine 0.954). Combining update14 and dual12 while keeping state FP32 also remains useful at cosine 0.947.

Interactions with state quantization are secondary. Adding dual12 to state15 moves cosine 0.590 -> 0.555; adding update14 alone moves it 0.590 -> 0.584. The deployed three-quantizer point is 0.565. Thus the earlier state-width sweep should be reinterpreted: the problem is not simply "too few state fractional bits", because 14/15/16-bit formats all failed, but the act of deterministic fixed-grid state quantization is the dominant trigger for this seed.

This also explains why widening lambda is unlikely to fix seed 860: dual12 without state quantization is already well behaved. The next hardware-relevant question is therefore whether the state-grid failure can be removed without widening the state memory, e.g. stochastic/error-feedback state rounding, or by changing the state scale/step so that late requested updates do not collapse into the quantization dead zone.

## Hardware consequence

Do not increase dual precision or update precision as the first fix for seed 860. Also do not assume 16-bit state solves it: the prior 14/15/16-bit sweep did not recover the gradient geometry. The highest-value next experiment is a rounding-rule / quantization-error-feedback comparison at fixed15_i3 state, because that can preserve the current 15-bit BRAM footprint while attacking the identified failure mechanism directly.
