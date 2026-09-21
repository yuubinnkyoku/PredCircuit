# P=32 transposable tile schedule for a 64x64 PC-ALM layer

Date: 2026-09-21

## Purpose

The preceding residency sweep established that model-boundary weight reuse changes the bottleneck, but it did not specify a realizable W / W^T schedule. This note makes the memory boundary concrete for one 64x64 hidden layer at batch B=4 and P=32 MAC lanes.

This is an access/cycle model, not RTL synthesis and not a device power claim. Counts below are explicit about port-width assumptions so that a later FPGA mapping can replace them without changing the schedule logic.

## Assumptions

- one dense hidden matrix W is 64x64;
- stored weight precision is q9;
- P=32 multipliers consume 32 weights per MAC cycle;
- batch B=4 is processed sequentially through the same P=32 lanes, while a fetched local weight vector can be reused across the four batch items;
- the layer is partitioned into four 32x32 tiles W00, W01, W10, W11;
- one tile is 32*32*9 = 9,216 bit = 1.125 KiB;
- the immediate local tile supports row and column vector access (a transposable register/LUTRAM organization, or an equivalent banked structure);
- the layer-local backing store is BRAM-like. For a conservative physical-port count, assume a 72-bit maximum useful read word, so a 288-bit P=32 vector corresponds to four 72-bit bank reads when streamed directly;
- h14 and lambda12 use the existing compact PC-ALM point;
- residuals/intermediate accumulators are streamed or locally buffered and are not counted as persistent h/lambda storage here.

A crucial dependency is retained: the W^T residual-propagation phase follows forward prediction/residual formation. Therefore a single 32x32 local tile cannot in general be loaded once and used for both directions unless enough residual/state context is retained to fuse the phases. The conservative schedule reloads each tile for the transpose phase.

## Baseline: direct BRAM streaming

For one matrix-vector multiply, each of 64 outputs needs two P=32 chunks, so one batch item takes

`64 * 2 = 128 MAC cycles`.

For B=4:

`C_dir = 4 * 128 = 512 cycles/direction`.

Forward W and transpose W^T therefore require

`C_fwd+T = 1024 MAC cycles/layer/relaxation`.

If every P=32 vector is supplied directly from a 72-bit-wide BRAM banking fabric, each MAC cycle needs four physical 72-bit bank reads. Thus the direct-stream baseline is

`R_BRAM,direct = 1024 * 4 = 4,096 72-bit bank-read events/layer/relaxation`.

This count deliberately gives no credit for batch reuse at the BRAM boundary.

## Tiled schedule

For one 32x32 tile, there are 32 row vectors of 32 q9 weights. Loading the tile once from a 72-bit backing store requires

`9,216 / 72 = 128 72-bit bank-read events`.

There are four tiles. A complete forward phase therefore loads

`4 * 128 = 512 BRAM bank-read events`.

Because the conservative dependency-respecting schedule reloads the tiles for W^T, the complete forward+transpose phase uses

`R_BRAM,tile = 2 * 512 = 1,024 72-bit bank-read events/layer/relaxation`.

The MAC work has not disappeared. The local tile still supplies the same 1,024 P=32 MAC cycles across forward and transpose. The difference is where the repeated B=4 accesses occur:

- direct streaming: 4,096 backing-BRAM bank reads;
- transposable tile: 1,024 backing-BRAM bank reads + 1,024 local 288-bit vector reads.

Thus, under these assumptions, tiling reduces backing-BRAM read events by exactly

`4,096 / 1,024 = 4x`,

or 75%, while adding only 9,216 bit of immediate tile capacity per active layer engine.

The factor of four is not accidental: it is the batch reuse B=4. The local tile lets one backing-store load serve all four batch items. If the batch were one, this particular benefit would vanish unless the tile were also reused across relaxation steps or fused phases.

## Can the same tile serve W and W^T without reload?

Capacity alone says yes: one 32x32 tile is only 9.216 kbit. Dependency says not automatically.

Forward prediction must first produce the residual used by transpose-direction propagation. If only one tile is resident, processing W00 forward and immediately using W00^T is generally too early because the relevant residual vector depends on the complete forward sum, including the other input-half tile.

There are two ways to remove the second tile load:

1. keep two tiles for one 32-row output slab (18,432 bit) so the full forward residual for that slab can be formed, then retain/reuse the tiles for transpose contributions; or
2. retain all four tiles for the whole 64x64 matrix (36,864 bit = 4.5 KiB), effectively making the layer-local matrix itself transposable local storage.

The first option is the more interesting next synthesis point. It doubles the immediate tile capacity from 9.216 to 18.432 kbit but may reduce backing-BRAM traffic from 1,024 to approximately 512 72-bit events per relaxation if the dependency schedule can be arranged without spilling residual context. That would be an 8x reduction versus direct B=4 streaming.

This 512-event figure is a target bound, not yet a proven cycle schedule; the exact ordering must be validated with residual availability and state-buffer port conflicts.

## h/lambda accesses for the same layer

For one hidden layer and B=4, there are

`B*N = 256`

hidden-state coordinates and 256 dual coordinates.

Using the existing conservative one-read + one-write per persistent state per relaxation:

- h14: `2 * 256 * 14 = 7,168 bit = 896 B`;
- lambda12: `2 * 256 * 12 = 6,144 bit = 768 B`;
- combined: `13,312 bit = 1,664 B/layer/relaxation`.

The weight backing-store payload for the conservative one-tile schedule is two full q9 matrix reads:

`2 * 64 * 64 * 9 = 73,728 bit = 9,216 B/layer/relaxation`.

So at the *bit-traffic* level, this concrete one-tile schedule is still weight-bound for a single layer:

`9,216 / 1,664 = 5.54x`.

This is an important correction to interpreting the earlier model-boundary K sweep. K>=4 made weight traffic smaller than global h+lambda traffic only when a full weight load was amortized across multiple relaxation steps. The present schedule amortizes across batch, not yet across relaxation steps, so weight traffic remains larger.

If a two-tile/fused-direction schedule reaches one full q9 matrix load per relaxation, weight payload becomes 4,608 B, still 2.77x the h+lambda payload. To make state traffic dominate at the layer-local BRAM boundary, weights must additionally survive across roughly three or more relaxation steps, consistent with the earlier continuous break-even K>2.70.

## Cycle implications

With one P=32 MAC lane group and sequential B=4, arithmetic occupancy is 1,024 MAC cycles per 64x64 layer per relaxation regardless of whether weights come directly from BRAM or a local tile.

A naive non-overlapped tile fill would add backing-store load latency. However, four 72-bit banks can deliver 288 bits/cycle, so a 32x32 tile can be filled in 32 cycles. Double-buffering two 9.216-kbit tile buffers can hide much of this fill behind computation on the other buffer because each tile participates in multiple batch MAC cycles.

Minimum extra local capacity for ping-pong one-tile buffering is

`2 * 9,216 = 18,432 bit = 2.25 KiB`.

This is notably smaller than duplicating the full 64x64 layer (36,864 bit) and far smaller than duplicating the full network's 1.113 Mbit q9 weights.

The exact no-stall condition depends on the chosen row/column traversal and state-buffer ports, so it should be verified in a cycle-accurate scheduler before claiming zero load overhead.

## Consequence for the architecture claim

The useful hardware mechanism is now narrower and more testable:

- q9 stochastic storage keeps the model compact without a FP32 master copy;
- a 32x32 transposable local tile converts B=4 reuse into a 4x reduction in backing-BRAM read events for W/W^T processing;
- this alone does **not** make h/lambda the dominant traffic: the conservative tiled schedule still moves 5.54x more weight bits than persistent-state bits per layer/relaxation;
- cross-relaxation residency of about K>=3 remains necessary before the earlier state-dominated regime appears;
- dual arithmetic remains negligible relative to the 64x64 matrix work, so the design pressure is banking, locality, and state ports rather than lambda ALU throughput.

This avoids the unsupported claim that a small register tile somehow removes all physical weight reads. It instead identifies three independent reuse axes: batch reuse B, forward/transpose reuse, and cross-relaxation reuse K.

## Next discriminating step

The highest-value next hardware experiment is a cycle-accurate scheduler for the two-tile (18.432-kbit) organization. It should test whether forward residual completion and transpose propagation can be ordered so that each q9 matrix weight is fetched from backing BRAM only once per relaxation, while reporting h/lambda port conflicts and stalls.

In parallel, the algorithmic gate still needs end-to-end learning and a fair ePC/BP digital baseline; this access model is a minimal-core planning artifact, not evidence of system-level FPGA superiority.