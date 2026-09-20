# Shared MAC parallelism sweep (XC7)

Date: 2026-09-20
Source commit: `25b84db0b209836335581e39a5af5887ca13d506`
Workflow run: `35509635400` (`Shared MAC Array Synthesis`), conclusion: success.

## Purpose

Measure the arithmetic-only area scaling of the shared `W h` / `W^T c` MAC array before adding weight/state banking. This deliberately isolates compute replication from memory-system cost. The element/dual update width remains fixed at V=8 elsewhere in the design.

## Yosys `synth_xilinx -family xc7` results

| P | DSP48E1 | Estimated LC | Matrix cycles / relaxation (`ceil(987136/P)`) |
|---:|---:|---:|---:|
| 8  | 8  | 41 | 123392 |
| 16 | 16 | 41 | 61696 |
| 32 | 32 | 41 | 30848 |
| 64 | 64 | 41 | 15424 |

The reported LC count stays at 41 across this sweep because the 14x14 multipliers (and much of the lane-local post-add arithmetic) map into DSP48E1 blocks; the remaining LUT logic is dominated by shared accumulator/counter/control logic. This is an arithmetic-only result and must not be interpreted as the area of a usable matrix engine: input fanout, weight storage/banking, routing, placement, and timing are not represented by the LC count alone.

## Immediate implications

1. Arithmetic replication itself is DSP-linear: `DSP48E1 = P` for P=8..64.
2. There is no observed LUT/LC explosion from the reduction/accumulation structure in synthesis at these P values.
3. Therefore the next informative boundary is memory delivery and physical timing, not another arithmetic-only P point.
4. The previous LC-only break-even expression is insufficient for choosing P, because the shared datapath's principal scalable resource is DSP, while the measured PC-ALM frontend adds 0 DSP but 1739 LC + 3 RAMB36. Resource classes should remain separate rather than being converted into an arbitrary LUT-equivalent.
5. The next implementation should add a banked weight source (and later state access) and measure memory-stall cycles plus post-place timing/Fmax. Only then can P=8/16/32/64 be compared as realistic accelerator design points.

## Scientific caution

This sweep does not establish wall-clock speedup or energy efficiency. `synth_xilinx` is synthesis, not placement/routing; no Fmax or routing congestion is available here. The ideal cycle counts above assume P operands can be supplied every active cycle. The result is useful precisely because it narrows the next hypothesis: if scaling degrades after adding storage, the degradation can be attributed to memory/physical implementation rather than the arithmetic RTL measured here.
