# P=32 two-tile fused W / W^T cycle schedule

Date: 2026-09-22

## Question

The previous access model left 512 backing-BRAM 72-bit reads per 64x64 layer/relaxation as a target bound. This note checks whether that bound is dependency-feasible with exactly two resident 32x32 q9 tiles (18,432 bit), rather than treating it as a capacity-only estimate.

This remains a cycle/access model, not RTL synthesis or a device-power claim.

## Assumptions inherited from the previous model

- W is 64x64, q9, partitioned as W00/W01/W10/W11, each 32x32;
- P=32 MAC lanes;
- batch B=4;
- a 72-bit backing-store word is used for physical BRAM-event accounting;
- one 32x32 q9 tile is 9,216 bit and needs 128 72-bit bank-read events to load;
- four 72-bit banks can provide 288 bit/cycle, so one tile takes 32 fill cycles;
- the local tile organization supports both row-vector and column-vector access;
- the transpose contribution is linear in the residual, so contributions from disjoint output-row slabs may be accumulated independently.

## Dependency observation

The full 64-element residual does **not** need to exist before any W^T work can start.

Write the residual as two 32-element output slabs r0 and r1. Then

`W^T r = [W00^T r0 + W10^T r1 ; W01^T r0 + W11^T r1]`.

Therefore, once the complete forward prediction for r0 has been formed from both W00 and W01, the pair W00/W01 can immediately be reused for its transpose contribution before either tile is evicted. The same is true for W10/W11 and r1.

This removes the apparent need to retain all four tiles simultaneously.

## Exact two-tile schedule

For output slab 0:

1. load W00 and W01 into the two local tile buffers;
2. for each of B=4 samples, compute the 32 output predictions using both 32-input halves;
3. form/store the 32 residual values r0 for each sample;
4. without evicting W00/W01, traverse the same tiles by columns and accumulate W00^T r0 and W01^T r0 into the two 32-coordinate input-gradient/state-propagation accumulators;
5. evict W00/W01.

Repeat the same sequence for output slab 1 with W10/W11 and r1, adding its transpose contributions into the existing accumulators.

After slab 1, the accumulated vector is exactly W^T r (up to the same arithmetic/quantization rules as a monolithic traversal).

## Backing-BRAM reads

Each of the four tiles is loaded exactly once:

`4 tiles * 128 72-bit events/tile = 512 72-bit bank-read events/layer/relaxation`.

The previous direct-stream baseline was 4,096 events, and the conservative one-tile/reload schedule was 1,024 events. Thus the dependency-valid two-tile schedule reaches

- 8x fewer backing-BRAM read events than direct B=4 streaming;
- 2x fewer than the one-tile schedule;
- one complete q9 matrix payload (4,608 B) per relaxation, with no W/W^T reload.

The earlier 512-event target is therefore achievable at the dependency level.

## MAC cycles

For one 32-row output slab, forward work is

`32 outputs * 2 input chunks * B4 = 256 P=32 MAC cycles`.

Transpose work for that slab is

`64 input coordinates * 1 residual chunk * B4 = 256 P=32 MAC cycles`.

So one slab takes 512 MAC cycles and two slabs take

`C_MAC = 1,024 cycles/layer/relaxation`.

This is the same arithmetic work as the previous schedule; the improvement is exclusively in backing-store locality.

## Fill latency: what two tiles alone can and cannot hide

One tile fills in 32 cycles, so two tiles require 64 fill cycles per slab. With exactly two tile buffers, the current slab's tiles must remain resident through its forward and transpose phases. They therefore cannot simultaneously be overwritten with the next slab.

Without an additional prefetch buffer, the conservative schedule is

`2 slabs * (64 fill + 512 MAC) = 1,152 cycles/layer/relaxation`.

Relative to the ideal 1,024 MAC cycles, this is a 12.5% cycle overhead.

This corrects a possible over-reading of the earlier ping-pong argument: 18.432 kbit is enough to eliminate the W/W^T reload, but not enough by itself to hide all next-slab fills while retaining both current tiles.

To hide fills completely at this granularity, another two-tile bank is the straightforward solution:

- active pair: 18,432 bit;
- prefetch pair: 18,432 bit;
- total: 36,864 bit = 4.5 KiB.

That equals one full 64x64 q9 matrix in capacity, but unlike a passive duplicate it is organized as two transposable slab banks and permits fill/compute overlap. Since 64 fill cycles are far shorter than 512 compute cycles per slab, double-buffering is sufficient in this model.

A smaller asymmetric prefetch design may reduce this capacity, but that requires a more detailed bank-port implementation and is not assumed here.

## Residual and transpose-accumulator context

Fusing by output slab requires retaining only the residual context for the current slab before transpose reuse:

`B * 32 = 128 residual coordinates`.

If provisionally stored at 14 bit, this is 1,792 bit = 224 B. This is not a new persistent PC-ALM state; it is short-lived phase-local context.

The transpose contributions must accumulate across the two output slabs for B*64 = 256 coordinates. At 14 bit this would be 3,584 bit = 448 B if held as a dedicated local accumulator. The exact width should ultimately follow the fixed-point accumulator analysis rather than being assumed equal to h14.

Even with these provisional widths, the phase-local context is much smaller than the two q9 tile buffers.

## h/lambda port conflicts

There is no intrinsic conflict with the *weight* backing BRAM because h/lambda are distinct logical memories. However, a single shared state-memory port could still stall the schedule if h reads, residual writes, transpose accumulation, and lambda updates are serialized through it.

Therefore 512 weight-bank events is now a proven dependency count, but 1,024 no-stall MAC cycles is conditional on providing enough independent state/local-buffer ports. The conservative two-tile/no-prefetch cycle count of 1,152 does not include additional h/lambda port stalls.

The next scheduler refinement should model those ports explicitly rather than revisiting the weight dependency.

## Traffic consequence

With fused-direction reuse, backing weight payload is one q9 matrix per relaxation:

`64*64*9 = 36,864 bit = 4,608 B`.

The existing conservative persistent h+lambda traffic is 1,664 B/layer/relaxation. Therefore weight payload is still

`4,608 / 1,664 = 2.77x`

larger than persistent-state payload for K=1.

If the same resident matrix/slab organization can survive across K relaxation steps, the break-even is again about K>2.77. Thus the cycle-valid W/W^T fusion removes one uncertainty but does not remove the need for cross-relaxation reuse if the goal is to make state traffic dominant.

## Result

The two-tile 18.432-kbit organization is sufficient to make one-fetch-per-weight-per-relaxation dependency-valid. The key algebraic fact is slabwise additivity of W^T r: each output slab's residual can be completed and consumed before moving to the next slab.

The price is now explicit: exactly two tiles imply 128 total fill cycles and a conservative 1,152-cycle layer/relaxation schedule, 12.5% above the 1,024-cycle MAC floor. A 4.5-KiB double-buffered organization can hide those fills in principle and recover the MAC floor, subject to state-memory ports.

This narrows the architecture question. Weight dependency is no longer the main unknown; state-memory banking and whether h/lambda/residual accesses can sustain the 1,024-cycle compute schedule are the next discriminating constraints.