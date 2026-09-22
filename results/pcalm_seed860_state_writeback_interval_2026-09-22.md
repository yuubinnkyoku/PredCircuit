# PC-ALM seed 860 state writeback interval diagnostic (2026-09-22)

Condition: depth=32, width=64, seed=860, T=128, state_lr=0.25, dual_leak=0.02, update=`fixed14_i1`, state writeback=`fixed15_i3`, dual=`fixed12_i1`.

| interval | cosine to BP | norm ratio | relative error | residual | useful |
|---:|---:|---:|---:|---:|:---:|
| 1 | 0.564590 | 1.507472 | 1.253102 | 0.396882 | no |
| 2 | 0.631938 | 1.411363 | 1.099162 | 0.430110 | no |
| 4 | 0.718714 | 1.209488 | 0.851063 | 0.496668 | no |
| 8 | 0.835639 | 1.028245 | 0.582069 | 0.567238 | no |
| 16 | 0.898789 | 0.941182 | 0.440426 | 0.625784 | no |
| 32 | 0.921132 | 0.920206 | 0.389252 | 0.650673 | yes |
| 64 | 0.931302 | 0.903759 | 0.365288 | 0.687847 | yes |
| 128 | 0.947188 | 0.881734 | 0.327289 | 0.701335 | yes |

All trajectories were finite. State saturation remained very small (0.0147% at interval 1 and 0.0252% at intervals 16--128).

## Interpretation

The previous hypothesis that increasing the high-precision holding interval would fail is rejected by the artifact. Reducing the frequency of `fixed15_i3` projection improves first-layer gradient geometry monotonically. The first tested interval satisfying the current useful-credit criterion is 32 steps. This points to repeated state-grid projection, rather than dynamic-range saturation, as the dominant cause of the seed-860 failure under this configuration.

This is not yet a low-cost hardware solution: interval > 1 requires higher-precision state to persist between quantized writebacks. The result instead sets a concrete architectural target: determine the minimum guard/accumulator precision and retention scheme that reproduces the interval-32 recovery without storing every state in FP32.
