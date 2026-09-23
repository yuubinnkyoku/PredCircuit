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

## Dependency-wave lower bound

With one dedicated engine per layer, PC-ALM can overlap layer-local dynamics and needs approximately `T` global relaxation waves. A conventional reverse-credit pass has approximately `H=31` ordered layer waves.

Thus:

- optimistic PC-ALM: `77/31 = 2.48x` as many dependency waves;
- realistic PC-ALM: `112/31 = 3.61x` as many dependency waves.

Even with ideal one-engine-per-layer spatialization, the current PC-ALM budget does not beat a single reverse sweep on dependency depth. To reach the pure layer-parallel latency crossover requires `T < H = 31`, i.e. at least a `112/31 = 3.61x` reduction from the realistic budget (or `77/31 = 2.48x` from the unattainable oracle mean).

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

It does **not** establish an advantage over ePC/BP. Against the digital `T=1` comparison, the current width-8 PC-ALM point carries a 154--224x dense credit/dynamics MAC disadvantage and a 2.48--3.61x ideal dependency-wave disadvantage before implementation effects.

Hence the hardware-entry criterion should be interpreted in two stages:

1. PC-ALM versus sPC: already promising enough to justify keeping the switchable-core design alive;
2. PC-ALM versus ePC/BP: not yet cleared for a performance/energy claim. A full RTL accelerator should not be justified from the sPC crossover alone.

## Consequence

The most informative next software/hardware experiment is no longer another simple stopping heuristic. It is a measured or synthesizable **shared matrix/state datapath** with explicit banking, from which `cycles/step`, Fmax, DSP/LUT/BRAM and local-memory traffic can be obtained. Combined with `T=112` (realistic) and `T=77` (optimistic unattainable oracle bound), that converts the 154--224x arithmetic disadvantage into a concrete area-time/energy bound.

In parallel, algorithmic work has a clear target: reducing useful PC-ALM relaxation below `T=31` would cross the ideal layer-dependency boundary for depth 32. Until then, any claimed advantage over ePC/BP must come from measured fixed-point/locality/throughput effects, not from layer parallelism by itself.
