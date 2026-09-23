# Width-8 PC-ALM vs ePC/BP cost bound (2026-09-24)

## Purpose

Update the earlier first-order hardware crossover with the current end-to-end/gradient evidence rather than the older illustrative `T=96` point. This is an analytical lower-bound model, not a measured FPGA speedup.

Current PC-ALM operating assumptions from the batch-sensitivity work are:

- realistic high-quality relaxation budget: `T_realistic = 112`;
- optimistic oracle adaptive budget under the loose quality gate: mean `T_oracle = 76.9` (rounded to 77 for integer work counts);
- simple implementable residual/dual stopping signals do not reliably attain that oracle, so `T=77` is an optimistic bound, not a deployable result;
- ePC is retained as a `T=1` digital comparison and BP as the reference reverse-credit baseline.

The matched narrow model is depth 32, hence `H=31` hidden-state transitions, width `N=8`, hardware batch `B=4`.

## Arithmetic lower bound

For dense equal-width layers, one PC-ALM relaxation step requires approximately two dense matvec passes per hidden transition:

`M_step = 2 H N^2 = 2 * 31 * 8^2 = 3,968 MACs`.

A single BP/ePC reverse-credit sweep requires approximately

`M_reverse = H N^2 = 1,984 MACs`

for credit propagation alone. Parameter-gradient outer products are omitted symmetrically.

Therefore:

| method / budget | credit/dynamics MACs | ratio to one reverse sweep |
| --- | ---: | ---: |
| BP/ePC reverse sweep | 1,984 | 1x |
| PC-ALM optimistic oracle `T=77` | 305,536 | 154x |
| PC-ALM realistic `T=112` | 444,416 | 224x |

This is the central arithmetic bound. The lambda update itself is not the problem; repeated matrix work is. Any FPGA advantage over ePC/BP must therefore come from a very large difference in realized operator cost, locality, concurrency, or from a future algorithmic reduction in `T`. Spatial parallelism alone cannot erase a 154--224x work ratio.

## Shared-MAC cycle bound using measured RTL structure

The repository already contains a synthesized `shared_mac_array`: its P=8 point maps to 8 DSP48E1 blocks, and the arithmetic-only P sweep shows linear DSP scaling with no LUT explosion. For the width-8 network, P=8 is enough to consume one complete length-8 dot product per active cycle, assuming the operands are supplied without stalls.

Therefore a single shared P=8 matrix engine needs, ideally:

- one `8x8` matvec: `8` active cycles;
- one PC-ALM layer relaxation (`W h` plus `W^T c`): `16` active cycles;
- all `H=31` layers for one PC-ALM step: `496` active cycles;
- one BP/ePC reverse sweep: `31 * 8 = 248` active cycles.

Thus a single shared P=8 engine gives exactly the arithmetic ratios again:

- PC-ALM T=77: `496 * 77 = 38,192` active matrix cycles = `154x` BP/ePC;
- PC-ALM T=112: `496 * 112 = 55,552` active matrix cycles = `224x` BP/ePC.

This is an operand-delivery lower bound. Banking stalls, state traffic and control can only increase it.

## Corrected layer-parallel dependency/cycle bound

An earlier version of this note compared `T` PC-ALM waves directly with `H` BP waves and stated a `T/H` latency ratio. That omitted the two matrix phases (`W h` and `W^T c`) inside each PC-ALM relaxation step. The arithmetic model above already counted both, so the old dependency-wave paragraph was inconsistent with the MAC accounting.

With **one P=8 matrix engine per layer** (31 engines, 248 DSP48E1 total), all layers may be spatialized, but each layer still needs ideally 16 active matrix cycles per PC-ALM relaxation step. A conventional reverse sweep on one P=8 engine needs 8 cycles per layer across 31 ordered layers, or 248 cycles total.

Hence the corrected ideal latency ratios are

`C_PC / C_reverse = (2 N T) / (N H) = 2T/H`:

- optimistic oracle T=77: `154/31 = 4.97x`;
- realistic T=112: `224/31 = 7.23x`.

The pure one-engine-per-layer latency crossover is therefore

`2T < H`, i.e. `T < 15.5` for H=31,

not `T < 31`.

This is a stricter and more useful target. From the realistic T=112 point, algorithmic relaxation must fall by more than `112/15.5 = 7.23x` to beat a single reverse sweep on latency using 31 layer-local P=8 engines. Even the unattainable oracle mean T=77 remains about `4.97x` above that boundary.

A design with two independent matrix engines per layer could in principle reduce the local two-phase arithmetic bottleneck, but it doubles the already large layer-spatialized DSP budget and does not automatically remove data dependencies between prediction, residual/dual and transpose-feedback phases. Such a design must be measured rather than assumed.

## Persistent state capacity at the compact fixed-point candidate

Use the existing compact candidate widths: state `z=15` bits, dual `lambda=12` bits, weight `W=16` bits.

Single-copy dense weight storage is

`B_W = H N^2 * 16 = 31,744 bits = 3.875 KiB`.

PC-ALM persistent dynamic state at `B=4` is

`B_dyn = B H N (15+12) = 26,784 bits = 3.270 KiB`.

So dynamic state is `84.4%` of the single-copy weight capacity at width 8. Lambda is therefore a first-order BRAM-capacity concern in the narrow regime even though its arithmetic update is cheap.

Under the optimistic one-read/one-write-per-state-element lower bound, PC-ALM dynamic-state traffic is

`Q_step >= 2 B H N (15+12) = 53,568 bits = 6.539 KiB/step`.

Accumulated lower bounds are:

- oracle `T=77`: `503.5 KiB` local state traffic;
- realistic `T=112`: `732.4 KiB` local state traffic.

These are on-chip traffic lower bounds when state is BRAM/register resident; banking conflicts and extra reads can only increase them.

## Weight residency is mandatory

If weights are streamed for both forward and transpose passes, PC-ALM needs approximately `2T` weight-matrix streams versus one reverse-credit stream for BP/ePC. At the current budgets that reproduces the same `154x` or `224x` traffic ratio and leaves no bandwidth crossover.

Therefore a plausible PC-ALM FPGA must keep/bank weights locally enough to amortize repeated reuse. At width 8 the weights fit easily in capacity, but the dynamic state is already 84.4% of one weight copy, so banking/port structure rather than raw capacity may dominate.

## What the sPC advantage does and does not prove

Previous width-64 results show PC-ALM can beat sPC strongly in required relaxation count and can amortize the lambda frontend relative to sPC. That remains useful evidence for PC-ALM versus state-based PC.

It does **not** establish an advantage over ePC/BP. Against the digital `T=1` comparison, the current width-8 PC-ALM point carries a 154--224x dense credit/dynamics MAC disadvantage. With a concrete P=8 matrix engine, the ideal one-engine-per-layer latency disadvantage is still 4.97--7.23x before memory stalls and physical implementation effects.

Hence the hardware-entry criterion should be interpreted in two stages:

1. PC-ALM versus sPC: already promising enough to justify keeping the switchable-core design alive;
2. PC-ALM versus ePC/BP: not yet cleared for a performance/energy claim. A full RTL accelerator should not be justified from the sPC crossover alone.

## Consequence

The existing arithmetic-only shared-MAC synthesis is sufficient to expose a stronger no-free-lunch bound than the previous wave count: at width 8, P=8 already consumes a full dot product each active cycle, so merely increasing multiplier parallelism inside a shared engine cannot remove the 154--224x total-work gap. Replicating one P=8 engine per layer costs 248 DSP48E1 and still leaves a corrected 4.97--7.23x ideal latency gap.

The next hardware measurement should therefore be a **banked weight/state source attached to the existing P=8 engine**, not another arithmetic-only P sweep. It should measure operand stalls and storage cost; those numbers can only worsen the lower bound above, but quantify by how much.

In parallel, the algorithmic target is now sharper: useful PC-ALM relaxation must approach `T<=15` to cross the ideal one-engine-per-layer latency boundary for depth 32. Until then, any claimed advantage over ePC/BP must come from measured energy-per-operation/locality benefits large enough to compensate a several-fold latency disadvantage, not from layer parallelism by itself.
