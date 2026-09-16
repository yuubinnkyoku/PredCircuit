# PC-ALM dual engine: 8 vs 16 lanes on XC7 (2026-09-16)

Source: RTL smoke run 35054514003 at commit `5493f3d5e44c40a0b9e7116b7d70d546e7d90ca9`, Yosys 0.33 `synth_xilinx -family xc7`.

Both points hold the logical dual-state capacity fixed at 992 signed 12-bit values (11,904 bits). The 8-lane point uses 124 words x 96 bits; the 16-lane point uses 62 words x 192 bits. These are synthesis/mapping results, not post-place-and-route timing results.

| metric | 8 lanes | 16 lanes | 16/8 |
|---|---:|---:|---:|
| logical dual state | 11,904 bit | 11,904 bit | 1.00x |
| issue words | 124 | 62 | 0.50x |
| estimated sweep cycles incl. one drain | 125 | 63 | 0.504x |
| Estimated LCs | 1,408 | 2,826 | 2.007x |
| CARRY4 | 266 | 530 | 1.992x |
| FDRE | 129 | 432 | 3.349x |
| LUT2 | 264 | 538 | 2.038x |
| LUT3 | 411 | 797 | 1.939x |
| LUT4 | 225 | 374 | 1.662x |
| LUT5 | 104 | 248 | 2.385x |
| LUT6 | 668 | 1,407 | 2.106x |
| MUXF7 | 141 | 323 | 2.291x |
| MUXF8 | 22 | 58 | 2.636x |
| RAMB18E1 | 3 | 0 | -- |
| RAM64M | 0 | 64 | -- |
| DSP48 | 0 | 0 | -- |
| Yosys peak memory | 169.98 MB | 228.97 MB | 1.347x |

Both final XC7 checks report 0 problems.

## Main observation

Doubling lane count almost exactly doubles mapped logic while halving sweep cycles, but it also changes the inferred storage architecture qualitatively. The 124x96 8-lane memory maps to 3 RAMB18E1 blocks. The shallower 62x192 16-lane memory maps to 64 RAM64M distributed-RAM primitives instead of block RAM.

Therefore the 16-lane point is not a free timing-margin upgrade. Under this RTL/mapping flow it pays approximately 2.01x estimated LCs and loses the compact 3-BRAM state store in exchange for reducing the estimated sweep from 125 to 63 cycles.

For the existing 126-cycle MAC budget, the clock-ratio condition for hiding the dual sweep is approximately:

- 8 lanes: `f_dual / f_MAC >= 125/126 = 0.9921`
- 16 lanes: `f_dual / f_MAC >= 63/126 = 0.5`

Thus 8 lanes is the area/memory-efficient boundary point, while 16 lanes buys large timing slack at substantial logic and distributed-memory cost. Neither dominates until post-place-and-route timing is measured.

## Research interpretation

The important design variable is not lane count alone. Lane count changes memory depth and word width, which can cross an FPGA memory-inference boundary. A resource model for PC-ALM dual state must therefore include the discrete RAM mapping regime, not only `state_bits` and arithmetic lanes.

This also means interpolating linearly between the 8- and 16-lane points is unsafe. A 9/10/12-lane design may have a different memory mapping again, and non-divisibility of 992 states introduces padding or banking choices.

## Next high-value measurement

Measure post-place-and-route `Fmax` (or at minimum a timing estimate from a real XC7 implementation flow) for the 8-lane engine and the MAC datapath. If the 8-lane dual engine reaches at least 0.992x the MAC clock, 16 lanes has no throughput advantage for this 126-cycle overlap target and 8 lanes is strongly preferable. If not, test a banked intermediate design chosen to preserve BRAM inference before accepting the 16-lane LUTRAM cost.