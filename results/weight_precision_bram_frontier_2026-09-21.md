# Weight precision × banking BRAM frontier (2026-09-21)

This note combines the corrected legal RAMB36E1 TDP shape model with the existing 30-layer, 64x64 depth-packed cyclic weight store. It is an architectural capacity analysis, not a numerical-accuracy result.

Assumptions: 30 hidden matrices, each 64x64; cyclic bank `(r+c) mod P`; depth packing across layers; writable weights use RAMB36E1 true-dual-port shapes. Bank occupancies are 15360/7680/3840/1920 words for P=8/16/32/64.

| logical weight bits | physical RAM word | P=8 | P=16 | P=32 | P=64 |
|---:|---:|---:|---:|---:|---:|
| 16 | 18 | 64 | 64 | 64 | 64 |
| 14 | 18 | 64 | 64 | 64 | 64 |
| 12 | 18 | 64 | 64 | 64 | 64 |
| 10 | 18 | 64 | 64 | 64 | 64 |
| 9 | 9 | 32 | 32 | 32 | 64 |
| 8 | 9 | 32 | 32 | 32 | 64 |

## Main result

The 9-bit RAMB36 mode creates a discontinuity, but **only up to P=32**. At P=64 each bank contains 1920 words, so one 4096x9 RAMB36 is still required per bank: 64 banks imply 64 RAMB36. Thus reducing weights from 14 to 9 bits halves weight BRAM at P=8/16/32, but gives no BRAM-count reduction at P=64.

This changes the earlier interpretation that low precision would simply make P=64 more attractive. Under 9-bit weights, P=32 becomes a new Pareto point: it uses 32 RAMB36 and has ideal matrix-cycle count 30,848 per step in the existing cost model. P=64 halves the ideal matrix cycles again to 15,424, but requires 64 RAMB36. P=16 also uses 32 RAMB36 while taking 61,696 ideal matrix cycles, so absent a frequency/routing advantage it is dominated by P=32 in BRAM-versus-cycle space.

The 10->9 bit transition is therefore architecturally meaningful in two dimensions at once: it changes the legal RAM primitive width/depth mode and moves the likely spatial-parallelism sweet spot from P=64 toward P=32 if BRAM pressure matters.

## Scientific caveat

No claim is made here that PC-ALM tolerates 9-bit **weights**. Existing fixed-point evidence establishes 12-bit Q3.9 only for the dual variable lambda while states, residuals, weights and MAC arithmetic remain FP32. A weight-quantization experiment is required before the 9-bit row can be treated as a realizable PC-ALM design point.

## Highest-value follow-up

Test weight-only quantization at 12/10/9/8 bits while keeping lambda at the already-supported Q3.9 and the rest of the datapath high precision. The decisive comparison is 10 versus 9 bits: if 9-bit weights preserve convergence/gradient geometry, the hardware model predicts a 64->32 RAMB36 drop at P=32; if 9-bit fails but 10-bit succeeds, no weight-BRAM saving is obtained from that precision reduction on RAMB36E1.