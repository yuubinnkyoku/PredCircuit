# PC-ALM seed 860 state accumulator range diagnostic (2026-09-22)

Configuration: depth=32, width=64, seed=860, T=128, state_lr=0.25, dual_leak=0.02, writeback interval=32. Update quantization is `fixed14_i1`, dual quantization is `fixed12_i1`, and stored state writeback is `fixed15_i3`. Only the internal state accumulator format is varied.

| accumulator | BP cosine | norm ratio | relative error | accumulator saturation | useful |
|---|---:|---:|---:|---:|---|
| fp32 | 0.921132 | 0.920206 | 0.389252 | 0 | yes |
| fixed24_i3 | 0.567844 | 1.507782 | 1.249414 | 1.4668e-4 | no |
| fixed24_i4 | 0.920897 | 0.919719 | 0.389807 | 0 | yes |
| fixed24_i5 | 0.923213 | 0.921446 | 0.384293 | 0 | yes |
| fixed24_i6 | 0.921315 | 0.917972 | 0.388832 | 0 | yes |
| fixed20_i4 | 0.924795 | 0.919570 | 0.380501 | 0 | yes |
| fixed20_i5 | 0.924411 | 0.920684 | 0.381415 | 0 | yes |
| fixed18_i4 | 0.923862 | 0.918595 | 0.382763 | 0 | yes |
| fixed18_i5 | 0.924031 | 0.923397 | 0.382319 | 0 | yes |
| fixed16_i4 | 0.926050 | 0.923605 | 0.377409 | 0 | yes |
| fixed16_i5 | 0.927111 | 0.949547 | 0.375457 | 0 | yes |

The result sharply separates range from fractional precision. `fixed24_i3` still fails despite nine extra fractional bits relative to the 15-bit stored state, while moving one integer bit into the internal accumulator (`fixed24_i4`) removes accumulator saturation and recovers the FP32-like gradient geometry. The recovery persists down to `fixed16_i4`: one extra total bit plus one extra integer-range bit is sufficient for seed 860 at interval 32, and `fixed16_i4` slightly exceeds the FP32-accumulator cosine in this run.

This does **not** show that 16-bit accumulation is universally sufficient. Seed 860 was selected because it is a known difficult trajectory. The next high-value check is a multi-seed comparison of `fp32`, `fixed16_i4`, and `fixed16_i5` at interval 32, with the existing every-step `fixed15_i3` configuration retained as the failure/control condition. If the 16-bit formats remain useful across seeds, the hardware candidate becomes a 15-bit stored state plus a 16-bit wider-range local accumulator rather than a wide floating-point state path.
