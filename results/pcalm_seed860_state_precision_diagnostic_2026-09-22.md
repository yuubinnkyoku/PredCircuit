# PC-ALM seed 860 state-precision diagnostic (2026-09-22)

## Setup

- depth=32, width=64, seed=860, T=128
- state_lr=0.25, dual_leak=0.02
- deployed low-precision point: update=fixed14_i1, state=fixed15_i3, dual=fixed12_i1
- The state-only sweep keeps update and dual quantization fixed while changing state precision.

## Recovered artifact

The completed `pcalm-seed860-T128-state-precision` artifact was reused rather than rerun.

| config | BP cosine | grad norm / BP | BP relative error | residual total | max |lambda| | state saturation | useful |
|---|---:|---:|---:|---:|---:|---:|:---:|
| all FP32 | 0.967973 | 0.944652 | 0.252133 | 0.245340 | 0.107161 | 0 | yes |
| u14 / s14 / d12 | 0.572869 | 1.517578 | 1.250718 | 0.443917 | 0.479492 | 1.4668e-4 | no |
| u14 / s15 / d12 | 0.564590 | 1.507472 | 1.253102 | 0.396882 | 0.479492 | 1.4668e-4 | no |
| u14 / s16 / d12 | 0.566156 | 1.505975 | 1.250091 | 0.377104 | 0.480469 | 1.4668e-4 | no |

## Interpretation

Adding state fractional bits from fixed14_i3 through fixed16_i3 does **not** recover seed 860.  The first-layer BP cosine remains about 0.565--0.573 and the BP-relative error remains about 1.25.  Residual total improves monotonically as state precision increases, but the credit direction does not.  Therefore the seed-860 failure cannot be attributed primarily to insufficient state fractional precision.

The all-FP32 reference recovers strongly (cosine 0.968), but that row also removes update and dual quantization.  It therefore does not identify which remaining quantizer causes the failure.  A full 2^3 factorial over update=fixed14_i1, state=fixed15_i3, and dual=fixed12_i1 has been launched to separate main effects and interactions.

## Consequence for hardware work

Do not spend state BRAM/bit width on 16-bit state as a fix for this failure: it gives essentially no gradient-geometry recovery in this case.  The next decision should be based on whether update quantization, dual quantization, or their interaction is responsible.  This is especially important because a dual-precision fix would directly change the cost of PC-ALM relative to sPC, while an update-only fix may be cheaper in state memory.
