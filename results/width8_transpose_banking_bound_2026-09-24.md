# Width-8 transpose-banking bound for PC-ALM (2026-09-24)

## Why this matters

The current width-8 latency lower bound assumes the existing `P=8` MAC array receives eight weights and eight data operands every active cycle. That is sufficient for a length-8 dot product per cycle, but it does not prove that the same physical weight memory can sustain both PC-ALM matrix phases without stalls.

PC-ALM repeatedly needs both `W h` and `W^T c`. This note isolates the banking consequence before building a larger RTL datapath.

Matched point: `H=31`, `N=8`, hardware batch `B=4`. For the compact hardware candidate, use 16-bit stored weights when quoting capacity. The existing arithmetic-only `shared_mac_array` currently defaults to 14-bit operands; the banking argument is independent of that small width difference.

## Conflict in a single orientation

Let a dense 8x8 matrix be stored in eight single-read banks by column:

`bank[k][r] = W[r,k]`.

Then one row for `W h` is ideal: at output row `r`, read address `r` from all eight banks and obtain

`W[r,0..7]`

in one cycle. A P=8 engine can consume the complete dot product.

The transpose phase asks for one row of `W^T`, equivalently one column of `W`:

`W[0..7,r]`.

Under the same banking, all eight required values reside in `bank[r]` at eight different addresses. A conventional one-read-per-bank organization therefore cannot deliver the transpose dot product in one cycle. It has an 8-way bank conflict.

Row banking simply exchanges the problem: `W^T c` becomes conflict-free and `W h` conflicts. Therefore one conventional orientation cannot make both phases conflict-free at P=N.

## Three implementation choices

### A. Keep one copy and serialize the conflicting phase

One phase remains 8 cycles per 8x8 matvec; the conflicting phase becomes 64 cycles if only one weight from the conflicted bank can be read per cycle. A PC-ALM layer relaxation changes from the arithmetic-only lower bound

`8 + 8 = 16 cycles`

to

`8 + 64 = 72 cycles`.

With one such source per layer, the ideal PC-ALM latency becomes `72T`, while the earlier comparison reverse sweep remains `8H = 248` cycles if its chosen orientation is conflict-free. The latency ratio is

`72T / (8H) = 9T/H`.

At the current budgets:

- `T=77`: `22.35x` the reverse-sweep latency;
- `T=112`: `32.52x`;
- crossover requires `9T < H`, hence `T < 31/9 = 3.44`.

So accepting the transpose bank conflict is not a plausible route for the present PC-ALM budget.

### B. Store both W and W^T

Maintain two physical orientations:

- row-oriented copy for `W h`;
- transpose/column-oriented copy for `W^T c`.

Both phases can then supply eight weights per cycle to P=8, preserving the 16-cycle/layer arithmetic lower bound. The cost is approximately 2x weight storage plus update coherence whenever a weight changes.

At 16 bits/weight:

- one 8x8 layer: `64 * 16 = 1,024 bits = 128 B`;
- dual orientation per layer: `2,048 bits = 256 B`;
- all 31 transitions, one orientation: `31,744 bits = 3.875 KiB`;
- all 31 transitions, dual orientation: `63,488 bits = 7.750 KiB`.

The extra transpose copy is therefore only another 3.875 KiB at this narrow point. Raw capacity is small; write/update duplication and physical banking are the real costs.

Relative to the previously estimated PC-ALM dynamic state (`z+lambda`, B=4) of 3.270 KiB, dual-orientation weights plus dynamic state total about

`7.750 + 3.270 = 11.020 KiB`.

That is still small in absolute capacity on an FPGA, although a layer-parallel design may consume far more physical memory primitives than capacity alone suggests because each layer needs independent read bandwidth.

### C. Register/LUTRAM matrix with transpose-select routing

Because each width-8 layer contains only 64 weights, a small register/LUTRAM structure can expose either a row or a column through muxing without storing two logical copies. This trades duplicated storage for LUTs, routing and fanout. It may be attractive at N=8 but scales poorly: the crossbar/routing cost grows with matrix width and the fully layer-parallel design replicates it 31 times.

This option therefore needs synthesis rather than an analytical claim.

## Consequence for the previous latency bound

The earlier `2T/H` layer-parallel latency ratio is achievable only if operand delivery sustains both orientations without stalls. A naïve single-orientation BRAM/LUTRAM bank does not satisfy that assumption.

For width 8, dual-orientation storage is the cleanest lower-bound implementation because its absolute capacity is tiny. Under that assumption the previous ideal ratios remain valid:

- oracle `T=77`: `4.97x` reverse-sweep latency;
- realistic `T=112`: `7.23x`;
- arithmetic crossover: `T<=15` approximately.

Without transpose-friendly storage, the bound degrades to roughly `9T/H`, making the current algorithm much less competitive.

This also sharpens the memory-access accounting. Weight residency alone is not enough; **transpose-friendly residency** is required. A future hardware comparison that counts only one stored W matrix but assumes one-cycle P=8 delivery for both `W h` and `W^T c` is internally inconsistent unless it explicitly pays for a transpose-capable register/crossbar structure.

## Design decision

For the next measured width-8 datapath, use one of two explicit alternatives rather than hiding transpose delivery behind the MAC interface:

1. dual-orientation W/W^T storage, which preserves the 16-cycle/layer lower bound and costs 7.750 KiB total weights at H=31; or
2. a single-copy transpose-select register/LUTRAM source, synthesized to measure the LUT/routing/Fmax penalty.

Do not use a naïve one-orientation bank as the performance candidate: its 8-way transpose conflict changes the algorithm/hardware crossover from about `T<=15` to about `T<=3`.

The result strengthens, rather than relaxes, the current Go/No-Go condition. PC-ALM still needs a major reduction in relaxation count to compete with ePC/BP on latency, and any claimed FPGA advantage must include the cost of making both matrix orientations locally available.
