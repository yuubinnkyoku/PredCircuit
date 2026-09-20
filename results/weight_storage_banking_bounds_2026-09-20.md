# Weight-storage and banking bounds for the width-64 core

Date: 2026-09-20

## Motivation

The arithmetic-only shared-MAC sweep is now measured: P=8/16/32/64 maps to 8/16/32/64 DSP48E1 while estimated LC stays at 41. The next question is therefore not another multiplier point but whether the fixed weights can feed both `W h` and `W^T c` without making the memory system dominate.

This note establishes capacity and port-width lower bounds before writing a banked RTL memory. It deliberately does **not** claim a realizable conflict-free schedule yet.

## Matched network weight capacity

For the current B=4, L=32, N=64, I=8, O=4 residual MLP, the unique dense weights are

`W = I*N + (L-2)*N^2 + N*O`

`  = 8*64 + 30*64^2 + 64*4`

`  = 123,648 weights`.

At the current 14-bit datapath width this is

`123,648 * 14 = 1,731,072 bit = 211.3125 KiB`.

Using a Xilinx-style RAMB36 capacity of 36,864 bit, the pure-capacity lower bound is

`ceil(1,731,072 / 36,864) = 47 RAMB36`.

This is already much larger than the measured PC-ALM dual increment of 3 RAMB36. Thus for a fully on-chip implementation, **weight storage, not lambda storage, is the dominant BRAM-capacity term**: the dual adds only 3/47 = 6.38% relative to the weight-capacity lower bound (before hidden-state storage and before banking overhead).

## Per-cycle width lower bound

A P-lane MAC array consumes P 14-bit weights per active cycle, so the logical read width is

`W_read(P) = 14 P bit/cycle`.

For P=8/16/32/64 this is 112/224/448/896 bit/cycle.

A RAMB36 port is at most 72 bits wide in the simple width-bound model. Therefore a one-port wide-word implementation needs at least

`ceil(14 P / 72)`

RAMB36 primitives in parallel purely for width:

| P | weight bits/cycle | 72-bit width lower bound |
|---:|---:|---:|
| 8 | 112 | 2 |
| 16 | 224 | 4 |
| 32 | 448 | 7 |
| 64 | 896 | 13 |

These width bounds are all below the 47-block capacity bound. Therefore **P=64 does not, by itself, force more than the capacity-minimum number of BRAM36 blocks**. With an appropriate striped layout, enough aggregate on-chip bit capacity exists to expose a 896-bit logical word without duplicating the whole weight set.

This corrects a tempting but misleading interpretation of the 89.6-Gbit/s-at-100-MHz figure: the aggregate internal bandwidth is large, but it is not automatically an additional BRAM-capacity cost.

## The real unresolved problem: W versus W^T

Capacity is not the hard part. Access geometry is.

`W h` naturally streams groups of columns from one output row. `W^T c` naturally needs the transposed access geometry. A layout that makes P same-row weights conflict-free does not automatically make P same-column weights conflict-free on the same physical banks.

There are three qualitatively different implementation choices:

1. **single copy, conflict-aware schedule**: preserve the 47-RAMB36 capacity floor but accept/reorder cycles when transpose accesses conflict;
2. **bank mapping supporting both directions**: search for a deterministic 2-D/cyclic mapping and schedule that sustains the required P for both directions without full duplication;
3. **duplicated/transposed copy**: simplest bandwidth story but raises the weight-capacity floor from 47 to 94 RAMB36 before state/dual storage.

The third option is an important upper reference, not the default design. Duplicating weights would cost an extra 47 RAMB36, dwarfing PC-ALM's measured +3-RAMB36 dual memory. A poor `W^T` memory design can therefore erase far more FPGA area than the algorithmic PC-ALM overhead under study.

## Consequence for the next RTL experiment

Do not implement a generic `P`-wide row-major ROM and call its synthesis result the memory cost. That would test only `W h` and could hide the transpose penalty.

The next memory experiment should use one 64x64 hidden-layer matrix as the minimal hard case and exercise **both** traces:

- forward: row-major dot-product groups for `W h`;
- transpose: column-equivalent groups for `W^T c`.

For P in {8,16,32,64}, report separately:

- realized RAMB36 count;
- forward conflict/stall cycles;
- transpose conflict/stall cycles;
- useful weight reads per cycle;
- whether weights are duplicated;
- post-place Fmax when available.

A design is not considered P-lane merely because it instantiates P DSPs; it must sustain P useful weight deliveries per active matrix cycle in both directions, or expose the lost cycles as `C_mem`.

## Current interpretation

The arithmetic sweep showed no LUT-side scaling wall through P=64. This bound shows that raw BRAM **capacity** also need not be the first P=64 wall: 47 blocks are required by the weights themselves, while only 13 parallel 72-bit slices are required by width. The likely first memory wall is therefore the bidirectional access pattern (`W` versus `W^T`) and its banking/routing cost.

That distinction matters scientifically. If PredCircuit gains from spatial locality, it must come from a layout/schedule that keeps these local matrices on chip and reuses them in both directions. Comparing against a deliberately duplicated or conflict-heavy layout would overstate PC's hardware cost and obscure the actual architectural question.