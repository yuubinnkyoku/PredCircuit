# PC-ALM dual engine area-time product (2026-09-16)

This note derives one additional architectural result from the already measured XC7 8-lane and 16-lane dual-engine points. No synthesis is rerun.

Source measurements are the fixed-capacity 992 x 12-bit dual-state comparison in `dual_engine_8v16_xc7_2026-09-16.md`:

- 8 lanes: 1,408 estimated LCs, 125 estimated sweep cycles, 3 RAMB18E1.
- 16 lanes: 2,826 estimated LCs, 63 estimated sweep cycles, 64 RAM64M and no block RAM.

## Area-time product

Using estimated LCs times sweep cycles as a first-order arithmetic area-time proxy:

- 8 lanes: `1408 * 125 = 176,000 LC-cycles`
- 16 lanes: `2826 * 63 = 178,038 LC-cycles`
- ratio: `178038 / 176000 = 1.01158`

Thus doubling the arithmetic lanes from 8 to 16 changes this proxy by only about +1.16%. The measured logic scaling and cycle scaling almost exactly cancel.

This is evidence that, over these two points, lane replication primarily trades area for latency rather than improving the intrinsic area-time efficiency of the dual recurrence. The 16-lane design therefore should not be selected on throughput alone when the 8-lane sweep can already be hidden under the MAC path.

## Memory breaks the apparent symmetry

The nearly constant LC-cycle product does not mean the two designs are equivalent. The storage mapping changes qualitatively:

- 8 lanes: 124 x 96-bit logical memory -> 3 RAMB18E1.
- 16 lanes: 62 x 192-bit logical memory -> 64 RAM64M distributed-memory primitives.

So the 16-lane point preserves arithmetic area-time efficiency but consumes a less attractive memory regime under the current XC7/Yosys mapping. This makes the 8-lane point preferable unless timing forces more parallelism.

## Timing decision boundary

For the existing 126-cycle MAC overlap budget, the dual engine can be hidden when

`C_dual / f_dual <= 126 / f_MAC`.

Therefore:

- 8 lanes require `f_dual / f_MAC >= 125/126 = 0.99206`.
- 16 lanes require `f_dual / f_MAC >= 63/126 = 0.5`.

The correct interpretation is therefore conditional:

1. If post-route 8-lane timing reaches at least 99.21% of the MAC clock, 16 lanes buys no relaxation-step throughput while paying ~2x logic and moving the state store from BRAM to LUTRAM.
2. If 8-lane timing misses that boundary, extra lanes are useful as timing insurance, but the next design should seek the smallest parallelism/banking change that restores overlap while preserving a compact BRAM mapping.

## Updated design rule

For the PC-ALM dual path, choose the minimum lane count satisfying the real-time overlap constraint after timing, not the maximum affordable lane count. At the measured 8/16 points, arithmetic area-time is essentially flat; the decisive quantities are therefore timing closure and the discrete FPGA memory-mapping regime.

This strengthens the case for measuring post-place-and-route timing before adding more dual parallelism. It also suggests that future resource models should track at least `(LCs, cycles, memory primitive class, memory primitive count)` rather than reducing a design point to LUT count alone.
