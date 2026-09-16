# PC-ALM width-32 accumulator precision: 20-seed validation

Date: 2026-09-17

This combines the original seeds 840--844 (5 seeds) with the preregistered holdout seeds 845--859 (15 seeds). The holdout workflow completed successfully as GitHub Actions run 35156132628. No original seeds were rerun.

## Result

| accumulator | useful | cosine mean | norm ratio mean | BP relative error mean | error vs FP32-acc mean | accumulator saturation rate mean | max abs accumulator |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 19/20 | 0.932555 | 1.136691 | 0.415962 | 0 | 0 | 46.2014 |
| fixed16_i6 | 19/20 | 0.934223 | 1.131839 | 0.409991 | 0.218888 | 0 | 46.2254 |
| fixed16_i7 | 18/20 | 0.930867 | 1.125304 | 0.414112 | 0.213654 | 0 | 46.1827 |
| fixed16_i8 | 20/20 | 0.930597 | 1.135573 | 0.419765 | 0.224399 | 0 | 46.2137 |
| fixed12_i5 | 18/20 | 0.934970 | 1.137507 | 0.409949 | 0.222996 | 1.09e-5 | 38.7831 |
| fixed12_i6 | 15/20 | 0.926259 | 1.146664 | 0.437251 | 0.243373 | 0 | 46.2101 |
| fixed12_i7 | 11/20 | 0.908844 | 1.165294 | 0.486072 | 0.287376 | 0 | 46.3003 |

All formats remained finite on all 20 seeds. Operand saturation was zero for every format.

## Interpretation

The initial 5-seed impression that width 32 makes a 12-bit accumulator clearly fail did not replicate strongly on the 15-seed holdout. `fixed12_i5` reaches 18/20 useful seeds versus FP32's 19/20, while `fixed16_i6` also reaches 19/20 and `fixed16_i8` reaches 20/20. Thus the earlier 4/5 versus 5/5 split was mostly seed noise, not evidence that 16 bits are mandatory at width 32.

However, the integer/fraction allocation effect replicates strongly: at a fixed 12-bit total width, increasing integer bits from i5 to i6 to i7 decreases useful rate 18/20 -> 15/20 -> 11/20 and increases mean error versus the FP32-accumulator trajectory 0.2230 -> 0.2434 -> 0.2874. Since i6 and i7 have zero accumulator saturation, this degradation is attributable to lost fractional precision rather than insufficient range.

`fixed12_i5` is close to FP32 but is not range-safe: the holdout includes accumulator excursions above the nominal i5 range, with a nonzero mean saturation rate and an observed pre-quantization maximum of 38.783. Therefore 12-bit i5 is a useful aggressive operating point, not yet a safe default.

The current hardware-oriented conclusion is mixed precision rather than a blanket 12-bit datapath: the dual state can remain 12-bit Q3.9, while a 16-bit accumulator is the conservative width-32 choice. Importantly, the reason is not a large accuracy gap at 12 bits; it is the inability to simultaneously preserve i5-like fractional precision and safely cover the observed accumulator tail within only 12 total bits.

## Next discriminating experiment

Test a 13--15-bit accumulator family that keeps approximately the i5 fractional resolution while adding one or more range bits (for example 13-bit i6 and 14-bit i7), alongside 16-bit controls, on the same width-32 setup. This directly measures the minimum extra bits needed to cover the tail without sacrificing the fractional precision that the 12-bit i6/i7 results show is important. The goal is a minimum-safe accumulator format, not simply proving that 16 bits works.
