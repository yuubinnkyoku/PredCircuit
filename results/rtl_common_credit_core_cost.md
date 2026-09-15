# Fair common-credit-core XC7 cost

Source: GitHub Actions run `34992856505`, artifact `rtl-synthesis-smoke`, commit `419d76a382f8718cfaaffd35c23fee9eb30b2935`.

Both variants are generated from `rtl/credit_core_common.sv` with identical ports and the same nominal credit expression `credit = residual + effective_dual`. The sPC specialization sets `effective_dual = 0`; the PC-ALM specialization keeps the 12-bit dual recurrence `lambda' = round((253 lambda + 237 r)/256)`.

## XC7 synthesis

| Primitive | sPC | PC-ALM | Increment |
|---|---:|---:|---:|
| CARRY4 | 0 | 31 | +31 |
| FDCE | 0 | 13 | +13 |
| LUT2 | 0 | 38 | +38 |
| LUT3 | 0 | 54 | +54 |
| LUT4 | 0 | 12 | +12 |
| LUT5 | 0 | 16 | +16 |
| LUT6 | 0 | 101 | +101 |
| DSP48 | 0 | 0 | 0 |
| internal logic cells (excluding I/O buffers/BUFG) | 0 | 352 | +352 |

The PC-ALM top additionally has one BUFG. Both tops expose 16 IBUFs and 26 OBUFs, so I/O is matched.

## Important interpretation

The sPC specialization optimizes to wires only: with rho=1 and a DATA_W-wide residual, `sat(residual + 0)` is exactly `residual`, so Yosys removes the whole internal datapath. This is not a synthesis failure; it is the correct optimized implementation of this one-element sPC credit operation.

Therefore an area *ratio* between these two tiny tops is not meaningful (the sPC internal-logic denominator is zero). The useful quantity is the **incremental cost of one 12-bit PC-ALM dual element** under this fixed-coefficient design: 13 FF-class cells, 31 CARRY4s, 221 LUT2--LUT6 cells in total, and zero DSP48s. The LUT count includes rounding, saturation, recurrence, and credit addition, not just storage.

This result strengthens the earlier conclusion that the dyadic coefficients avoid DSP usage, but weakens any claim that the dual recurrence is "almost free": its combinational logic is substantial at one-element granularity. A network-level comparison must amortize this cost against the MAC datapath and the observed reduction in required relaxation steps; a standalone sPC-vs-PC-ALM area ratio would be misleading.
