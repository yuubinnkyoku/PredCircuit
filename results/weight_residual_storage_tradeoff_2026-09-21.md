# 9-bit weight + residual storage trade-off (2026-09-21)

## Question

The weight dead-zone experiment showed that direct fixed-point updates are effectively frozen: for `fixed9_i3`, the mean dead-update rate across seeds 800--819 was 100% at learning rate 0.01 and 99.8214% even at learning rate 0.1. A natural repair is to keep a sub-LSB residual accumulator per weight. This note asks whether that repair preserves the BRAM advantage that motivated 9-bit weights.

## Existing design point

For 30 hidden 64x64 matrices there are 122,880 weights. With P=32 cyclic banks, each bank stores 3,840 weights. A 9-bit weight fits the RAMB36E1 4096x9 organization, so one RAMB36 per bank is sufficient: 32 RAMB36 total.

## Per-weight residual changes the physical RAM mode

Let each stored weight have a residual of R bits. If weight and residual are stored together, the logical word is `9 + R` bits. For every R >= 1 this exceeds the 9-bit RAMB36E1 organization, so it must use at least the 2048x18 organization (for total width <=18). Because each bank has depth 3,840, two RAMB36 are then required per bank:

| format | logical bits/weight | RAMB36/bank | P=32 total |
|---|---:|---:|---:|
| 9-bit weight only | 9 | 1 | 32 |
| 9-bit + 1-bit residual | 10 | 2 | 64 |
| 9-bit + 4-bit residual | 13 | 2 | 64 |
| 9-bit + 6-bit residual | 15 | 2 | 64 |
| 9-bit + 8-bit residual | 17 | 2 | 64 |

Storing the residual in a separate BRAM array does not rescue the BRAM count for ordinary direct banking. For residual widths up to 9 bits, each of the same 32 banks still needs one 4096x9 RAMB36, so weight + residual again totals 64 RAMB36.

Thus the BRAM discontinuity is even sharper than the numerical 10->9-bit boundary:

> **At P=32, any ordinary per-weight residual state stored in BRAM removes the entire 32-RAMB36 saving of the 9-bit weight representation.**

This does not prove residual accumulation is useless. It means it must earn its cost through a different implementation: LUTRAM/register residuals, a shared/sparse accumulator, stochastic rounding, blockwise error feedback, or an update rule that avoids persistent per-weight sub-LSB state.

## Area scale

A dense R-bit residual has `122,880 * R` logical bits. This is 491,520 bits at R=4, 737,280 bits at R=6, and 983,040 bits at R=8. Those are too large to call a negligible side state on the target class of FPGA; mapping them to LUTRAM would trade BRAM savings for substantial LUT cost and therefore needs synthesis rather than assumption.

## Consequence for the next experiment

A dense residual-accumulator training run is no longer the highest-value next step by itself: even if it fixes learning, a conventional BRAM implementation gives back the memory win. The more discriminating next experiment is to test **stateless stochastic rounding of weight updates** (and, if needed, a sparse/shared residual scheme) against deterministic round-to-nearest. Stochastic rounding can make sub-LSB updates unbiased without storing a residual for every weight, so it directly tests whether P=32 / 9-bit / 32-RAMB36 can remain a viable learning design point.

This is a resource-model conclusion, not yet a claim about stochastic-rounding learning quality or RNG cost; those remain to be measured.
