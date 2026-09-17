# PC-ALM width-32 accumulator 13–15 bit boundary (20 seeds)

Date: 2026-09-17

Configuration: depth 32, width 32, T=128, dual leak gamma=0.01. Seeds 840–844 are the existing width sweep; seeds 845–859 are the independent holdout. No seed was rerun for this summary.

## Useful counts

| accumulator format | seeds 840–844 | seeds 845–859 | total |
|---|---:|---:|---:|
| FP32 | 5/5 | 14/15 | 19/20 |
| fixed12_i5 | 4/5 | 14/15 | 18/20 |
| fixed12_i6 | 3/5 | 12/15 | 15/20 |
| fixed12_i7 | 3/5 | 8/15 | 11/20 |
| fixed13_i6 | 4/5 | 14/15 | 18/20 |
| fixed14_i6 | 5/5 | 15/15 | 20/20 |
| fixed15_i6 | 4/5 | 14/15 | 18/20 |
| fixed16_i6 | 4/5 | 15/15 | 19/20 |
| fixed16_i7 | 4/5 | 14/15 | 18/20 |
| fixed16_i8 | 5/5 | 15/15 | 20/20 |

All formats were finite for all seeds. On the holdout, fixed13_i6, fixed14_i6, fixed15_i6 and all 16-bit formats had zero accumulator saturation. fixed12_i5 had a tiny nonzero mean saturation rate (1.45e-5), with observed max |accumulator| 38.783; sufficiently ranged formats observed maxima around 46.2.

## Key interpretation

The decisive variable is fractional precision once range is sufficient. fixed13_i6 and fixed12_i5 both have 7 fractional bits and are nearly/effectively the same where fixed12_i5 does not saturate. fixed14_i6 and fixed16_i8 both have 8 fractional bits and produced identical aggregate metrics in the 5-seed sweep and the holdout; both reach 20/20 useful. fixed15_i6 and fixed16_i7 both have 9 fractional bits and likewise match each other, but reach only 18/20. Therefore useful count is not monotonic in total bit width or fractional bits at this small sample size; the robust hardware conclusion is narrower: 14-bit i6 is sufficient to eliminate the 12-bit range/precision tradeoff at width 32, while 16 bits is not required by these experiments.

For a conservative mixed-precision candidate, retain lambda as 12-bit Q3.9 and use a 14-bit i6 sequential MAC accumulator. This is a candidate, not yet a universal minimum: it still needs validation at wider networks / different depths and, ideally, end-to-end training rather than gradient-geometry diagnostics alone.
