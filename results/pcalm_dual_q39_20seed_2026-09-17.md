# PC-ALM 12-bit Q3.9 dual-state check (20 seeds)

Date: 2026-09-17
Source workflow run: 35138464996, commit `9879255229ef0d3d9f5043e0151792d0e70e848a`.
Seeds 920--939, depth 32, width 8, T=128. Only the PC-ALM dual state is quantized; other state and arithmetic remain FP32. Q3.9 is signed 12-bit with range [-4,4) and step 2^-9.

All 20 workflow jobs completed successfully and all 20 artifacts were recovered. There were no Q3.9 saturation events in any seed for any tested leak.

| dual leak | FP32 useful | Q3.9 useful | FP32 mean cosine | Q3.9 mean cosine | FP32 mean norm ratio | Q3.9 mean norm ratio | FP32 mean relative error | Q3.9 mean relative error | max pre-quant | Q3.9 saturated seeds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.005 | 18/20 | 18/20 | 0.961714 | 0.961991 | 1.259001 | 1.257012 | 0.440793 | 0.438465 | 3.1566 | 0/20 |
| 0.010 | 19/20 | 19/20 | 0.958541 | 0.958425 | 1.075138 | 1.072338 | 0.352887 | 0.351545 | 2.6488 | 0/20 |
| 0.020 | 16/20 | 17/20 | 0.947059 | 0.948952 | 0.827991 | 0.825376 | 0.364013 | 0.362259 | 1.9736 | 0/20 |

At leak=0.01, Q3.9 exactly preserves the FP32 useful/non-useful classification for all 20 seeds. In particular seed 930, which had failed under the previous Q2.x range-limited formats because of saturation, is useful again: FP32 cosine 0.900908 / norm ratio 0.826079 / relative error 0.440413 versus Q3.9 0.901054 / 0.826148 / 0.440129, with zero saturation.

The previous Q2.10/Q2.14 loss was therefore a range-allocation problem rather than evidence that 12-bit dual precision is insufficient. For this configuration, 12-bit Q3.9 is a strong provisional dual-state format. This does not yet establish that the complete PC-ALM datapath can be 12-bit, because h, residuals, weights, products and accumulators are still FP32.

The next high-value test is whole-datapath fixed-point sensitivity, beginning with measured dynamic ranges for h/residual/product/accumulator and a mixed-precision baseline that keeps accumulators wider while using Q3.9 for lambda.