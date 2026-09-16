# 8-lane PC-ALM dual engine: measured XC7 synthesis

Date: 2026-09-16
Source commit: `085d99e6a3e47284f1584222449b42134a0171e6`
Actions run: `35046710107` (`RTL smoke`, success)
Artifact: `rtl-synthesis-smoke` (`10426924068`)
Tool: Yosys 0.33, `synth_xilinx -family xc7`

## Configuration

- 8 dual-update lanes
- `DATA_W=12`
- 124 addresses
- packed dual memory: 124 x 96 bit = 11,904 bit = 992 x 12-bit dual states
- synchronous read/write memory
- one address issued per cycle; expected steady sweep = 124 issue cycles + 1 pipeline drain ~= 125 cycles
- zero fill = 124 cycles

## XC7 mapped result

| resource / statistic | count |
|---|---:|
| Estimated LCs | 1,408 |
| RAMB18E1 | 3 |
| CARRY4 | 266 |
| FDRE | 129 |
| LUT2 | 264 |
| LUT3 | 411 |
| LUT4 | 225 |
| LUT5 | 104 |
| LUT6 | 668 |
| MUXF7 | 141 |
| MUXF8 | 22 |
| DSP48 | 0 |
| BUFG | 1 |
| IBUF | 107 |
| OBUF | 210 |

The important memory result is that the 11,904-bit dual state maps to **3 RAMB18E1** rather than being expanded into ~11.9k flip-flops. The fixed-coefficient 8-lane dual arithmetic still maps with **0 DSP48**.

## Comparison with the earlier scalar common-core measurement

The earlier scalar PC-ALM-vs-sPC common-core delta was 221 LUT primitives + 31 CARRY4 + 13 FF per scalar element, with 0 DSP48. Multiplying the scalar LUT-primitive count by eight gives 1,768, but that is not directly comparable to the mapped 8-lane `Estimated LCs=1,408`: the latter includes the integrated engine, memory-facing/control logic and uses Xilinx packing/mux resources, while the former was a primitive-count delta against a degenerate sPC common core. Therefore the scalar x8 estimate should not be treated as an area prediction.

## Generic-synthesis caveat

The generic `proc; memory; opt` path lowered the memory to 11,914 `$dffe`-class cells and emitted many undriven-wire warnings. This is **not** the FPGA implementation result. The XC7 technology-mapped path completed with `check` reporting 0 problems and inferred 3 RAMB18E1. Do not interpret the generic lowering as evidence that PC-ALM requires 11.9k FFs.

## Hardware interpretation

For the existing depth=32, width=8, batch=4 cycle model, the major MAC path is 126 cycles per relaxation step. This 8-lane synchronous dual engine is expected to need about 125 cycles per full 992-state sweep, so it remains just inside that budget in cycle count. The margin is only one cycle; this result does **not** establish post-place-and-route timing closure or equal clock frequency.

The next high-value measurement is therefore timing-aware rather than another scalar synthesis: either obtain post-synthesis/place-and-route Fmax for this engine and the MAC datapath, or parameterize the engine at 8/9/10/16 lanes and compare area versus timing/cycle slack. 8 lanes is now a credible area-minimal point, not yet a proven throughput-optimal point.
