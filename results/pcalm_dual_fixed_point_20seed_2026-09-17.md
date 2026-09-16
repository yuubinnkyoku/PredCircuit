# PC-ALM dual fixed-point sweep: 20-seed result (2026-09-17)

Source: GitHub Actions run `35126348881`, commit `5ac79cae`, seeds 920--939. All 20 matrix jobs completed successfully. Network: depth 32, width 8, batch 4, ReLU, `state_lr=0.25`, `rho=1`, `alpha=0.925`, budget `T=128`. Only the PC-ALM dual state is quantized after every dual update; states, residuals, weights, and MAC arithmetic remain FP32. Fixed-point formats all use two integer bits including sign, hence range `[-2,2)` and steps 2^-14, 2^-10, and 2^-6 for Q2.14, Q2.10, and Q2.6.

`useful` means first-layer credit satisfies cosine >= 0.9, BP gradient norm ratio in [0.5,2], relative error <= 0.6, and all values are finite.

| dual leak | format | useful / 20 | seeds with saturation | saturation events | mean cosine | mean norm ratio | mean relative error | median max pre-quant | max max pre-quant |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | FP32 | 7 | 0 | 0 | 0.9635 | 1.5071 | 0.6219 | 1.290 | 3.847 |
| 0 | Q2.14 | 6 | 4 | 871 | 0.9579 | 1.4687 | 0.6230 | 1.290 | 2.233 |
| 0 | Q2.10 | 6 | 4 | 865 | 0.9582 | 1.4682 | 0.6218 | 1.291 | 2.233 |
| 0 | Q2.6 | 7 | 4 | 877 | 0.9414 | 1.3902 | 0.5943 | 1.261 | 2.231 |
| 0.005 | FP32 | 18 | 0 | 0 | 0.9617 | 1.2590 | 0.4408 | 1.076 | 3.156 |
| 0.005 | Q2.14 | 17 | 3 | 502 | 0.9589 | 1.2410 | 0.4484 | 1.076 | 2.182 |
| 0.005 | Q2.10 | 17 | 3 | 502 | 0.9588 | 1.2405 | 0.4481 | 1.076 | 2.182 |
| 0.005 | Q2.6 | 16 | 3 | 456 | 0.9454 | 1.1518 | 0.4297 | 1.030 | 2.171 |
| 0.01 | FP32 | **19** | 0 | 0 | 0.9585 | 1.0751 | **0.3529** | 0.917 | 2.649 |
| 0.01 | Q2.14 | 18 | 2 | 196 | 0.9580 | 1.0683 | 0.3567 | 0.917 | 2.140 |
| 0.01 | Q2.10 | **18** | 2 | 196 | **0.9581** | **1.0681** | **0.3563** | 0.916 | 2.140 |
| 0.01 | Q2.6 | 17 | 2 | 175 | 0.9423 | 0.9695 | 0.3729 | 0.874 | 2.142 |
| 0.02 | FP32 | 16 | 0 | 0 | 0.9471 | 0.8280 | 0.3640 | 0.726 | 1.973 |
| 0.02 | Q2.14 | 16 | **0** | **0** | 0.9471 | 0.8280 | 0.3640 | 0.726 | 1.973 |
| 0.02 | Q2.10 | **17** | **0** | **0** | **0.9482** | **0.8264** | **0.3629** | 0.726 | 1.973 |
| 0.02 | Q2.6 | 16 | **0** | **0** | 0.9300 | 0.7275 | 0.4388 | 0.656 | 1.994 |

## Main observations

1. **Twelve fractional-precision bits are not needed for the dual state.** Q2.10 tracks Q2.14 almost exactly at every leak. Their useful counts differ from FP32 only when the common `[-2,2)` range saturates (except a one-seed threshold crossing at leak 0.02). This separates precision from dynamic range: adding fractional bits beyond ten gives no visible benefit here.

2. **Leak reduces actual fixed-point saturation, not only the FP32 median dual magnitude.** With the same Q2 range, saturation affects 4/20 seeds at leak 0, 3/20 at 0.005, 2/20 at 0.01, and 0/20 at 0.02. Thus the earlier range reduction has a direct hardware consequence.

3. **At leak 0.01, Q2.10 retains 18/20 useful seeds versus 19/20 in FP32.** The single lost seed is seed 930, which is one of the two saturated seeds. Seed 938 also saturates but remains useful. Q2.14 loses the same seed, so this loss is caused by the Q2 dynamic range rather than insufficient fractional precision.

4. **Leak 0.02 is the first tested setting that fits Q2 without any saturation over all 20 seeds.** Q2.10 gives 17/20 useful versus 16/20 FP32; the extra seed is a boundary crossing rather than evidence that quantization intrinsically improves the method. Its mean gradient norm ratio is only 0.826, however, whereas leak 0.01 is much closer to BP scale (1.068 in Q2.10). Therefore choosing 0.02 solely to eliminate saturation would trade away gradient-scale fidelity.

5. **Eight-bit Q2.6 is plausible but not transparent.** At leak 0.01 it still gives 17/20 useful, but mean cosine falls from 0.958 (Q2.10) to 0.942 and the norm ratio from 1.068 to 0.970. At leak 0.02 it remains 16/20 but mean cosine falls to 0.930 and norm ratio to 0.727. This is enough to justify further study, not enough to select 8-bit as the default RTL format.

## Interpretation for hardware

The most informative failure is seed 930 at leak 0.01: both Q2.14 and Q2.10 fail while FP32 succeeds. Increasing total width while keeping the same two integer bits does not repair it. Therefore the next numerical-format question is not "12 or 16 bits?" but **whether one more integer bit is sufficient while retaining a 12-bit word** (e.g. Q3.9, range `[-4,4)`). The FP32 maximum over the current 20-seed leak-0.01 sweep is 2.649, so Q3.9 would cover every observed FP32 dual value without saturation while still using only 12 bits.

This also sharpens the role of dual leak. Leak 0.01 is currently the better algorithmic operating point at T=128 (19/20 FP32, gradient norm ratio near one), while leak 0.02 is the safer Q2 numerical operating point (zero saturation). A Q3.9 test can determine whether the algorithmic optimum and numerical optimum can be reunited without increasing state width.

## Next highest-value experiment

Run the same seeds with Q3.9 at leaks 0.005, 0.01, and 0.02. The decisive prediction is that leak 0.01 Q3.9 should remove the two Q2 saturation cases and recover the FP32 19/20 result if dynamic range is the only remaining 12-bit limitation. If it does, 12-bit dual state becomes a strong RTL candidate; only then should whole-state/MAC quantization be added.