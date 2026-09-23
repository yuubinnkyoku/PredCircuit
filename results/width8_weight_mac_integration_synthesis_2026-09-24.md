# Width-8 transpose-friendly weight source + P=8 MAC synthesis (2026-09-24)

## Purpose

Close the gap between the earlier standalone weight-source synthesis and the actual arithmetic path used by PC-ALM. Both alternatives drive the same `shared_mac_array` with `P=8`; only the weight-storage/orientation strategy differs.

Source commit: `52389b6f336e75c59ae12c459c0cf26f19c1409d`.
GitHub Actions run: `35920301843` (`Weight MAC integration synthesis`). Both synthesis artifacts completed successfully.

## XC7 Yosys results

| integrated top | cells | estimated LCs | DSP48E1 | FDRE | LUT2 | LUT3 | RAM64M |
|---|---:|---:|---:|---:|---:|---:|---:|
| single-copy W + transpose select + P=8 MAC | 735 | 110 | 8 | 166 | 53 | 110 | 42 |
| dual-orientation W/W^T + P=8 MAC | 807 | 152 | 8 | 166 | 41 | 152 | 84 |

The common P=8 MAC submodule itself synthesized to 8 DSP48E1, 166 FDRE, 41 LUT2, 40 LUT3 and an estimated 41 LCs. The remaining difference is therefore dominated by the weight source.

## Interpretation

The single-copy transpose-select organization remains smaller after integration with the actual P=8 arithmetic path. Relative to it, dual orientation uses 42 extra estimated LCs (+38.2%) and doubles the distributed-memory primitive count from 42 to 84 RAM64M, while DSP and accumulator/register cost are identical.

Importantly, integrating the transpose-select path with the multipliers and reduction/accumulation logic does not cause an unexpected LUT explosion in synthesis. This supports keeping the optimistic width-8 arithmetic/operand-delivery cycle model (8 cycles for Wh and 8 cycles for W^T c per layer) as a resource-feasible lower bound, rather than forcing the pessimistic serialized transpose-bank model.

This does **not** establish equal Fmax. Yosys `synth_xilinx` cell counts do not include placement/routing timing, and the single-copy column-select network can still create a longer physical path into the DSP inputs. The result is therefore a resource result, not a timing result.

## Consequence for the PC-ALM/ePC crossover

For depth 32 (31 hidden transitions), if both orientations sustain the intended 8 weights/cycle, the ideal fully layer-parallel latency ratio remains approximately

`C_PC-ALM / C_reverse = 2T / 31`.

Thus the arithmetic/operand-delivery crossover still requires roughly `T <= 15`. Current PC-ALM behavior around T=77 (oracle-like adaptive lower bound) to T=112 (realistic fixed budget) remains the dominant gap; transpose storage is not presently the dominant resource obstacle at width 8.

## Next discriminating measurement

Run implementation/timing (or, if vendor place-and-route is unavailable in CI, a timing-aware FPGA flow) for the two integrated tops under the same target/clock constraint. The useful question is whether single-copy transpose selection loses enough Fmax to justify the +42 LC/+42 RAM64M dual-orientation cost. If not, standardize on single-copy storage and move the main investigation back to reducing/understanding T and low-precision PC-ALM stability.
