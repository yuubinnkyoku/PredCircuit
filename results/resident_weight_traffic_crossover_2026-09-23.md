# Resident-weight and dynamic-state traffic crossover (2026-09-23)

## Question

The first-order crossover bound showed that PC-ALM pays about `2T` dense matvec work relative to one reverse BP/ePC credit sweep, while layer-parallel latency only becomes favorable near `T < H`. That model intentionally omitted the memory hierarchy. This note adds a lower-bound traffic model for the hardware candidate now supported by the low-precision experiments.

## Candidate numerical formats

Use the current concrete candidate rather than FP32 words:

- stored state `z`: 15 bits (`fixed15_i3` candidate),
- local state accumulator/datapath: 16 bits with one integer guard bit (`fixed16_i4`),
- dual state `lambda`: 12 bits (current candidate),
- weights: parameter `b_w` bits; examples below use 16 bits.

The 16-bit guard accumulator is local transient storage and is not charged as persistent BRAM state here.

## Dense equal-width model

Let `H` be the number of hidden-state layers, width `N`, and relaxation length `T`.

### Weights

One weight matrix per adjacent layer gives approximately

`B_W = H N^2 b_w` bits.

A transpose matvec does not require a second physical copy if the same matrix can be banked/read in the opposite access order. A design that duplicates weights for simultaneous `W` and `W^T` access instead pays `2 B_W`; this is an implementation choice, not an algorithmic requirement.

For depth 32 (`H=31`), width 64, 16-bit weights:

`B_W = 31 * 64^2 * 16 = 2,031,616 bits = 248 KiB`.

Duplicating for conflict-free simultaneous forward/transpose access would be about 496 KiB.

### Persistent dynamic state

PC-ALM stores both `z` and `lambda`:

`B_dyn,pcalm = H N (15 + 12) = 27 H N` bits.

sPC stores only the state:

`B_dyn,spc = 15 H N` bits.

Thus the lambda state increases persistent dynamic-state capacity by `12/15 = 80%`, while total PC-ALM dynamic state is `27/15 = 1.8x` sPC.

At `H=31, N=64`:

- `z`: 29,760 bits = 3.63 KiB,
- `lambda`: 23,808 bits = 2.91 KiB,
- PC-ALM dynamic total: 53,568 bits = 6.54 KiB.

The persistent dynamic state is therefore only about 2.6% of the single-copy 16-bit weight capacity in this dense width-64 example. Capacity is not the main problem; repeated accesses are.

## On-chip traffic lower bound

Assume weights are resident next to the layer engines, so there is no per-step off-chip weight fetch. Also assume one read and one write of each persistent state element per relaxation step. This is deliberately optimistic: real banking/scheduling may require extra reads.

PC-ALM dynamic-state traffic per step is at least

`Q_dyn,pcalm >= 2 H N (15 + 12) = 54 H N` bits.

For `H=31, N=64` this is

`107,136 bits = 13.08 KiB / step`.

Therefore:

- `T=32`: at least 418.5 KiB of dynamic-state BRAM traffic,
- `T=96`: at least 1.226 MiB,
- `T=128`: at least 1.635 MiB.

For sPC under the same one-read/one-write assumption:

`Q_dyn,spc >= 30 H N` bits/step = 7.27 KiB/step at `H=31,N=64`.

So PC-ALM pays a 1.8x dynamic-state traffic factor per relaxation step before considering whether it reduces the required `T` relative to sPC.

## External-memory crossover

If weights do *not* fit on chip and each dense PC-ALM step must stream a forward and transpose matrix pass, its weight traffic is approximately

`Q_W,pcalm ~= 2 T B_W`.

One BP/ePC reverse credit sweep streams approximately

`Q_W,reverse ~= B_W`

for credit propagation alone. Hence the external weight-traffic ratio is again approximately `2T`. There is no bandwidth crossover for `T >= 1` from the learning rule itself.

Therefore weight residency is not a minor optimization; it is a prerequisite for any plausible memory-traffic advantage of a spatial PC-ALM engine. Once weights are resident, off-chip traffic can be dominated by examples/activations/parameter synchronization rather than rereading `W` and `W^T`, while the cost that remains is local BRAM traffic.

## Capacity scaling and a useful boundary

With 16-bit single-copy weights, dense weight capacity scales as `16 H N^2` bits whereas PC-ALM dynamic state scales only as `27 H N` bits. Their ratio is

`B_dyn,pcalm / B_W = 27 / (16 N)`.

So for `N=64` it is only 2.64%; for `N=8` it is 21.1%. Lambda capacity matters proportionally more in narrow networks, while dense weights dominate capacity in wider networks.

This creates a hardware tension with the algorithmic results: narrow/deep networks are precisely where PC-ALM may be most interesting versus sPC, but that is also where lambda is least negligible as a fraction of the resident storage.

## Crossover conditions after adding memory

The earlier idealized latency condition remains approximately `T_pcalm < H` against one ordered reverse sweep when one engine is available per layer. The memory model adds two necessary conditions rather than relaxing it:

1. weights should be resident/banked on chip (or otherwise reused enough that the `2T` weight-stream penalty is avoided);
2. any reduction in relaxation steps relative to sPC must compensate for PC-ALM's 1.8x persistent-state traffic per step.

For a pure dynamic-state-traffic comparison against sPC, the necessary condition is

`1.8 T_pcalm < T_spc`,

or

`T_pcalm / T_spc < 0.556`.

Thus PC-ALM must cut the sPC relaxation count by more than about 44% before its extra lambda traffic is repaid on this optimistic state-memory metric. This is a concrete threshold that can now be tested against the existing and future depth/width sweeps.

## Interpretation

This model weakens any generic claim that lambda is "cheap" merely because its arithmetic is cheap. Lambda arithmetic is small compared with `N^2` matvecs, but lambda state creates an 80% increase over sPC's persistent state capacity/traffic at the current 15-bit state and 12-bit dual formats.

Conversely, at width 64 the absolute lambda capacity is only 2.91 KiB for 31 hidden layers. The dominant feasibility question is therefore not BRAM capacity but whether banking can sustain the repeated local reads/writes while the matrix engines run.

The strongest FPGA hypothesis is now conditional and falsifiable: **PC-ALM is attractive when weights remain resident, layer-local dynamics are spatially overlapped, and PC-ALM reaches the target gradient/learning quality in less than roughly 0.56x the sPC relaxation steps (for the present 15b/12b formats), ideally also below `H` steps when competing with an ordered BP/ePC reverse wave.**

## Next measurement

Apply this threshold to matched sPC/PC-ALM depth-width sweeps using *minimum T to reach the same BP-cosine/learning-quality target*, not a fixed T. Report `T_pcalm/T_spc`, then overlay the `0.556` state-traffic threshold and the `T_pcalm < H` layer-wave threshold. That directly decides whether the extra dual state is repaid before RTL implementation.