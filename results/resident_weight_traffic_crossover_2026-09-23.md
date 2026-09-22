# Resident-weight and dynamic-state traffic crossover (2026-09-23)

## Question

The first-order crossover bound showed that PC-ALM pays repeated dense matvec work over relaxation while layer-parallel latency only becomes favorable in a sufficiently small-T regime. This note adds a lower-bound memory-traffic model for the current hardware candidate and corrects the dynamic-state accounting to include batch size.

## Candidate numerical formats

Use the current concrete candidate rather than FP32 words:

- stored state `z`: 15 bits (`fixed15_i3` candidate),
- local state accumulator/datapath: 16 bits with one integer guard bit (`fixed16_i4`),
- dual state `lambda`: 12 bits,
- weights: parameter `b_w` bits; examples below use 16 bits.

The 16-bit guard accumulator is local transient storage and is not charged as persistent BRAM state here.

## Dense equal-width model

Let `H` be the number of hidden-state layers, width `N`, batch size `B`, and relaxation length `T`.

### Weights

One weight matrix per adjacent layer gives approximately

`B_W = H N^2 b_w` bits.

A transpose matvec does not require a second physical copy if the same matrix can be banked/read in the opposite access order. A design that duplicates weights for simultaneous `W` and `W^T` access instead pays `2 B_W`; this is an implementation choice, not an algorithmic requirement.

For depth 32 (`H=31`), width 64, 16-bit weights:

`B_W = 31 * 64^2 * 16 = 2,031,616 bits = 248 KiB`.

Duplicating for conflict-free simultaneous forward/transpose access would be about 496 KiB.

### Persistent dynamic state

For batch size `B`, PC-ALM stores both `z` and `lambda` for every sample:

`B_dyn,pcalm = B H N (15 + 12) = 27 B H N` bits.

sPC stores only the state:

`B_dyn,spc = 15 B H N` bits.

Thus the lambda state increases persistent dynamic-state capacity by `12/15 = 80%`, while total PC-ALM dynamic state is `27/15 = 1.8x` sPC. The ratio is independent of batch size.

At the actual main diagnostic shape `B=4, H=31, N=64`:

- `z`: 119,040 bits = 14.53 KiB,
- `lambda`: 95,232 bits = 11.63 KiB,
- PC-ALM dynamic total: 214,272 bits = 26.16 KiB.

The dynamic state is therefore about 10.5% of the single-copy 16-bit weight capacity at batch 4. Capacity is still not dominant at width 64, but it is not the 2.6% implied by the earlier batch-1 arithmetic.

## On-chip traffic lower bound

Assume weights are resident next to the layer engines, so there is no per-step off-chip weight fetch. Also assume one read and one write of each persistent state element per relaxation step. This is deliberately optimistic: real banking/scheduling may require extra reads.

PC-ALM dynamic-state traffic per step is at least

`Q_dyn,pcalm >= 2 B H N (15 + 12) = 54 B H N` bits.

For `B=4,H=31,N=64` this is

`428,544 bits = 52.31 KiB / step`.

Therefore:

- `T=32`: at least 1.635 MiB of dynamic-state BRAM traffic,
- `T=96`: at least 4.904 MiB,
- `T=128`: at least 6.539 MiB.

For sPC under the same one-read/one-write assumption:

`Q_dyn,spc >= 30 B H N` bits/step = 29.06 KiB/step at `B=4,H=31,N=64`.

So PC-ALM still pays exactly a 1.8x dynamic-state traffic factor per relaxation step before considering whether it reduces the required `T` relative to sPC.

## External-memory crossover

If weights do *not* fit on chip and each dense PC-ALM step must stream a forward and transpose matrix pass, its weight traffic is approximately

`Q_W,pcalm ~= 2 T B_W`.

One BP/ePC reverse credit sweep streams approximately

`Q_W,reverse ~= B_W`

for credit propagation alone. Hence the external weight-traffic ratio is again approximately `2T`. There is no bandwidth crossover for `T >= 1` from the learning rule itself.

Therefore weight residency is not a minor optimization; it is a prerequisite for any plausible memory-traffic advantage of a spatial PC-ALM engine. Once weights are resident, off-chip traffic can be dominated by examples/activations/parameter synchronization rather than rereading `W` and `W^T`, while the cost that remains is local BRAM traffic.

## Capacity scaling and batch dependence

With 16-bit single-copy weights, dense weight capacity scales as `16 H N^2` bits whereas PC-ALM dynamic state scales as `27 B H N` bits. Their ratio is

`B_dyn,pcalm / B_W = 27 B / (16 N)`.

For the actual `B=4,N=64` point it is 10.55%; for `B=4,N=8` it is **84.4%**. Thus the earlier batch-1 interpretation understated the narrow-network storage tension by fourfold. Lambda/state capacity becomes a first-order concern precisely in narrow/deep networks, where PC-ALM's algorithmic advantage over sPC has been most interesting.

This also means batch size is a hardware design variable, not merely an experimental detail. Reducing hardware batch from 4 to 1 cuts dynamic-state capacity and traffic fourfold without changing weight capacity, although it may change effective update scaling and therefore must be revalidated algorithmically.

## Crossover conditions after adding memory

The memory model adds two necessary conditions:

1. weights should be resident/banked on chip (or otherwise reused enough that the `2T` weight-stream penalty is avoided);
2. any reduction in relaxation steps relative to sPC must compensate for PC-ALM's 1.8x persistent-state traffic per step.

For a pure dynamic-state-traffic comparison against sPC, the necessary condition is

`1.8 T_pcalm < T_spc`,

or

`T_pcalm / T_spc < 0.556`.

The batch correction changes the absolute BRAM capacity and traffic, but not this ratio because both methods carry the same batch factor.

Crucially, existing measurements already clear this *iteration-ratio* threshold by a wide margin at the main width-64 diagnostic: leaky PC-ALM has a useful point at `T=128`, whereas matched sPC is still 0/5 useful through `T=1024`. Therefore the observed lower bound is

`T_spc,min / T_pcalm > 1024/128 = 8`,

or equivalently `T_pcalm/T_spc,min < 0.125`, with the denominator right-censored because sPC's equal-quality success point has not been observed. This is stronger than the 0.556 traffic break-even, but it is a gradient-geometry result, not yet an end-to-end training or wall-clock result.

At the conservative h15/lambda12 traffic model, comparing the known PC-ALM point with the still-failing sPC point gives

`(128 * 27) / (1024 * 15) = 0.225`.

So the PC-ALM run incurs at most 22.5% of the simple persistent-state traffic of that 1024-step sPC run, despite carrying lambda. Since the sPC endpoint still fails the quality gate, this is a one-sided bound rather than an equal-quality speedup claim.

## Interpretation

The corrected model changes one important conclusion. At width 64, lambda/state capacity is manageable but not negligible at the actual batch size; at width 8 it is nearly as large as the dense weights themselves. The arithmetic overhead of lambda remains small, but narrow-network BRAM pressure is substantially more serious than the earlier batch-1 calculation suggested.

Conversely, the already-measured relaxation-budget separation is large enough that the simple state-traffic break-even is not the current bottleneck. The unresolved question is no longer whether PC-ALM can beat the 0.556 iteration-ratio line in this diagnostic; it already does by a censored margin. The important missing evidence is whether that advantage survives end-to-end learning and a concrete FPGA schedule with banking/port constraints.

The strongest FPGA hypothesis remains conditional and falsifiable: **PC-ALM is attractive when weights remain resident, layer-local dynamics are spatially overlapped, and its large reduction in required relaxation budget survives a real learning task strongly enough to amortize the dual state and banking cost.**

## Next measurement

Do not spend the next run merely extending the sPC T sweep: the existing `>8x` censored separation already clears the 1.8x state-traffic threshold. The higher-value experiment is end-to-end learning on the shared ResidualMLP task with BP, sPC, pure PC-ALM, and leaky PC-ALM, reporting task loss/accuracy together with relaxation T and accumulated MAC/state-traffic estimates. In parallel, any width-8 hardware point must explicitly budget `B`, because at `B=4` the PC-ALM dynamic-state capacity is already about 84% of single-copy 16-bit dense-weight storage.