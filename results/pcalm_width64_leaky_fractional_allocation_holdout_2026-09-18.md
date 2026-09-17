# Width-64 leaky PC-ALM fractional-allocation holdout (2026-09-18)

Configuration: depth 32, width 64, T=256, state_lr=0.234285, dual_leak=0.02, seeds 845-859. The dual remains fixed12_i1 in every fixed-point condition.

| config | useful | cosine to BP | norm/BP | first-layer rel. error | all-grad rel. error to FP32 | residual | max |lambda| | mean state saturation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 leaky | 15/15 | 0.994314 | 1.020262 | 0.107833 | 0 | 0.245252 | 0.177797 | 0 |
| baseline u14_i2 / s14_i3 | 15/15 | 0.935208 | 0.780462 | 0.384130 | 0.286950 | 0.302231 | 0.169922 | 0 |
| state +1 frac: u14_i2 / s14_i2 | 8/15 | 0.612139 | 4.264244 | 3.853352 | 2.643582 | 0.718206 | 2.000000 | 3.42e-4 |
| update +1 frac: u14_i1 / s14_i3 | 15/15 | 0.968459 | 0.902834 | 0.256994 | 0.191110 | 0.315150 | 0.176758 | 0 |
| both +1 frac: u14_i1 / s14_i2 | 4/15 | 0.553848 | 4.109493 | 3.988210 | 2.747910 | 0.694730 | 2.000000 | 3.42e-4 |

## Interpretation

The previous zero-saturation result for fixed14_i3 state does **not** imply that one integer bit is spare. Narrowing the state range from fixed14_i3 to fixed14_i2 causes rare saturation in 7/15 seeds; those rare events are enough to destabilize the recurrent relaxation loop, drive the dual to its 12-bit limit in several seeds, and destroy gradient geometry. State range is therefore a hard constraint, not merely a precision-allocation choice.

In contrast, narrowing the update range from fixed14_i2 to fixed14_i1 remains saturation-free on all 15 holdout seeds and improves cosine from 0.9352 to 0.9685, norm ratio from 0.7805 to 0.9028, first-layer relative error from 0.3841 to 0.2570, and all-gradient error to FP32 from 0.2870 to 0.1911. Thus one additional update fractional bit is genuinely useful at no storage-width cost.

The next precision experiment should not reduce state range again. The high-value question is how many extra *state fractional* bits are needed while preserving the fixed14_i3 dynamic range. Test fixed15_i3 and fixed16_i3 state, paired with the successful fixed14_i1 update and unchanged fixed12_i1 dual. This distinguishes a one-bit state-precision deficit from a broader recurrent quantization floor without wasting bits on dual range.