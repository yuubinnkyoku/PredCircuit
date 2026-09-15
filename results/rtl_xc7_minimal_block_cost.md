# XC7 minimal-block mapping: sPC vs PC-ALM

Source: GitHub Actions RTL synthesis smoke run 34973169562 at commit `5cc2a35eec5ad1f8e174ab5a7728a19dfd958451` (Yosys 0.33, `synth_xilinx` mapping). The arithmetic exhaustive test passed before synthesis/mapping, so these reports correspond to the verified 12-bit dual-update block.

## Mapped cell counts

| Cell | sPC credit | PC-ALM dual update | Delta |
|---|---:|---:|---:|
| CARRY4 | 28 | 31 | +3 |
| FDCE | 0 | 13 | +13 |
| LUT1 | 0 | 1 | +1 |
| LUT2 | 16 | 55 | +39 |
| LUT3 | 22 | 33 | +11 |
| LUT4 | 1 | 17 | +16 |
| LUT5 | 2 | 9 | +7 |
| LUT6 | 21 | 141 | +120 |
| MUXF7 | 0 | 40 | +40 |
| MUXF8 | 0 | 12 | +12 |
| DSP48 | 0 | 0 | 0 |

The raw total-cell counts are 162 for `spc_credit` and 453 for `pcalm_dual_update`, but those totals include top-level I/O cells and therefore should not be interpreted as an area ratio for a replicated internal datapath. The useful result is the primitive delta above, especially `DSP48 = 0` for both designs.

## Interpretation

The fixed coefficients in the PC-ALM dual recurrence map to LUT/carry logic rather than DSP48s, as intended by the shift/add implementation. The added registered dual state costs 13 FDCEs in this standalone 12-bit block. The combinational cost is not negligible: the current standalone PC-ALM block adds 3 CARRY4s and substantial LUT/mux logic over the sPC credit-only block.

This is not yet a full-network area or timing comparison. The two top modules expose different state/control/I/O and duplicate saturation/credit logic, so a raw 453/162 cell ratio would overstate the incremental cost in a shared PC/sPC-PC-ALM layer engine. No placement-and-route Fmax or power result is available from this Yosys-only mapping.

## Next measurement

Build a common 12-bit layer-local credit core with identical ports and shared saturation/credit datapath, switching only the dual recurrence on/off. Synthesize both configurations from the same parameterized top. This will isolate the true incremental cost of lambda storage/update and permit a fair critical-path comparison before scaling to the full layer engine.
