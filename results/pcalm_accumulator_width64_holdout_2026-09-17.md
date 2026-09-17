# PC-ALM width-64 accumulator holdout (2026-09-17)

Commit evaluated: `c78cf6f4422a86a0cd7d5529f6a37670c1eec463`

Configuration: residual MLP, depth 32, width 64, T=128, state_lr=0.25, rho=1.0, alpha=0.925, dual_leak=0.01. Holdout seeds are 845--859 and use the same width-dependent model/data seeding as the accumulator-width probe.

The corrected holdout workflow completed successfully. All 15 seed jobs and the aggregate job passed.

| accumulator | useful / 15 | cosine mean | norm ratio mean | BP relative error mean | FP32-acc error mean | acc saturation mean | max abs acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 4/15 | 0.879204 | 1.294746 | 0.633308 | 0 | 0 | 41.8408 |
| fixed12_i5 | 6/15 | 0.888114 | 1.298892 | 0.617179 | 0.446141 | 9.77e-6 | 36.2602 |
| fixed12_i6 | 4/15 | 0.866390 | 1.295273 | 0.657384 | 0.475085 | 0 | 41.8899 |
| fixed12_i7 | 1/15 | 0.839938 | 1.368978 | 0.757579 | 0.527094 | 0 | 42.2930 |
| fixed13_i6 | 6/15 | 0.887321 | 1.298197 | 0.618013 | 0.450334 | 0 | 41.8118 |
| fixed14_i6 | 5/15 | 0.883111 | 1.291676 | 0.623000 | 0.453422 | 0 | 41.8279 |
| fixed15_i6 | 6/15 | 0.879515 | 1.279022 | 0.623462 | 0.444475 | 0 | 41.8593 |
| fixed16_i6 | 5/15 | 0.884548 | 1.299648 | 0.626946 | 0.449438 | 0 | 41.8493 |
| fixed16_i7 | 6/15 | 0.879515 | 1.279022 | 0.623462 | 0.444475 | 0 | 41.8593 |
| fixed16_i8 | 5/15 | 0.883111 | 1.291676 | 0.623000 | 0.453422 | 0 | 41.8279 |

## Interpretation

The discovery-set observation that width 64 degrades FP32 PC-ALM credit quality is reproduced: FP32 is useful on only 4/15 independent holdout seeds at T=128, with mean first-layer cosine 0.879 and mean relative error 0.633. Therefore the dominant width-64 issue is not accumulator precision. 14-bit i6 has zero accumulator saturation but is useful on only 5/15 seeds, close to the FP32 baseline.

The next experiment should test whether width 64 simply needs more relaxation steps before tuning fixed-point formats further. In particular, sweep T at fixed FP32 PC-ALM and retain sPC as a comparator on exactly the same width-dependent model/data seeds.
