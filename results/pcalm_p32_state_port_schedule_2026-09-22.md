# P=32 PC-ALM state-port schedule bound

Date: 2026-09-22

## Question

The dependency-valid two-tile schedule reduced a 64x64 q9 layer to 512 backing-weight-BRAM 72-bit read events per relaxation, with a 1,024-cycle MAC floor if tile fills can be hidden. The remaining uncertainty was whether persistent h/lambda traffic and phase-local residual/transpose state force stalls on that 1,024-cycle schedule.

This note gives a conservative access bound. It is still an architecture model, not RTL synthesis or a device-power measurement.

## Inherited configuration

- hidden width N=64;
- batch B=4;
- P=32 MAC lanes;
- h: 14 bit;
- lambda: 12 bit;
- q9 weights;
- 32-row output slabs;
- two active 32x32 q9 tiles, with a second pair available for weight prefetch when targeting the 1,024-cycle floor;
- one relaxation of one 64x64 layer performs 512 forward MAC cycles and 512 transpose MAC cycles.

## Persistent state working set

For one layer and B=4:

- h: `4*64*14 = 3,584 bit = 448 B`;
- lambda: `4*64*12 = 3,072 bit = 384 B`;
- combined persistent working set: `6,656 bit = 832 B`.

If the layer state is loaded once before the relaxation and stored once after it, the backing-state payload is therefore 1,664 B/relaxation, matching the previous traffic model.

With the same 72-bit event unit used for weight accounting, a conservative separately packed count is:

- h load: `ceil(3584/72) = 50` events;
- lambda load: `ceil(3072/72) = 43` events;
- h store: 50 events;
- lambda store: 43 events;
- total: **186 backing-state 72-bit events/layer/relaxation**.

Spread over the 1,024 MAC cycles, this is only `186/1024 = 0.182` backing-state events/cycle on average. Thus backing-state bandwidth itself is not large enough to require a second full-rate 72-bit port if state transfers can be staged outside the inner MAC access path.

## Why the inner loop does not need a 448-bit BRAM state read every MAC cycle

A naive implementation could read one P=32 h chunk per MAC cycle: `32*14 = 448 bit/cycle`. That would turn h memory into a severe port problem. But this repeats the same source activation chunk across many output rows.

For each sample, the complete 64-coordinate source h vector is only 896 bit. Across B=4 it is the 448-B h working set above. Loading that layer-local h once into a register/LUTRAM vector bank lets the P=32 datapath reuse each 32-coordinate chunk across the 32 outputs of a slab instead of returning to backing BRAM.

This is the state analogue of weight residency, but much cheaper in capacity: the entire B=4,N=64 h working set is 448 B.

## Residual production rate

One 32-row slab performs 256 forward MAC cycles and produces `B*32 = 128` completed residual coordinates. Therefore residual completion is only

`128 / 256 = 0.5 residual coordinates/cycle`

on average, even though the MAC datapath consumes 32 multiplications per cycle.

A 14-bit provisional residual buffer for the current slab is 1,792 bit = 224 B. A single scalar write/cycle local memory is already above the average production rate; a small register/LUTRAM bank can absorb bursts at output-completion boundaries without touching the backing h/lambda memory.

## Transpose accumulator

The transpose phase consumes 256 MAC cycles per slab and accumulates into `B*64 = 256` input-side coordinates across the two slabs. The previous note provisionally used 14 bit, which would be 448 B, but this width is not yet justified by fixed-point range analysis.

The important port fact is independent of the exact width: this accumulator is phase-local and should be physically separate from persistent h/lambda storage. If implemented as a local bank/register array with enough banking for the P=32 reduction result, its read-modify-write traffic does not contend with backing-state BRAM.

The exact accumulator width remains an open numerical-format question and must not be silently equated with h14 in RTL.

## Lambda update bandwidth

There are only `B*N = 256` lambda coordinates per layer. Even if every coordinate is read and written once per relaxation, that is 512 scalar lambda accesses over 1,024 MAC cycles, or 0.5 scalar accesses/cycle average. Because the dual update uses the already formed local residual, it can be scheduled at residual-completion boundaries or drained through a narrow local pipeline rather than demanding P=32-wide lambda bandwidth.

Thus lambda's important hardware cost is capacity/persistence and arithmetic, not an intrinsic 32-lane memory port requirement in this schedule.

## Capacity for a no-inner-loop-backing-state design

A minimal layer-local state context, excluding the transpose accumulator width uncertainty, is:

- h working set: 3,584 bit;
- lambda working set: 3,072 bit;
- current-slab residual: 1,792 bit;
- subtotal: **8,448 bit = 1,056 B**.

Adding the provisional 14-bit transpose accumulator gives another 3,584 bit, for **12,032 bit = 1,504 B** total local state/context.

This is smaller than one active q9 two-tile pair (18,432 bit = 2.25 KiB), and far smaller than the 4.5-KiB weight ping-pong capacity used to hide tile fills.

If h/lambda backing transfers also need to be hidden across layer transitions, duplicating only the persistent 832-B state bank is sufficient in capacity terms; it does not require duplicating the phase-local residual/accumulator context.

## Cycle consequence

If state is not prefetched, the conservative backing-state transfer count is 93 72-bit events to load h+lambda and 93 to store them. Serialized around the 1,024 MAC cycles, this would give an upper-bound schedule of

`1024 + 186 = 1,210 cycles`,

an 18.2% overhead.

But this is deliberately pessimistic. Weight double-buffering leaves 512 compute cycles per slab while the next 64-cycle weight fill occurs. The state stream requires only 186 events over the whole relaxation. Separate logical weight/state BRAMs or a small state ping-pong bank therefore provide ample scheduling slack to overlap most or all backing-state transfers.

The stronger result is not that 1,024 cycles is already proven at RTL level. It is that **state backing bandwidth is no longer a plausible fundamental blocker**: the required average is 0.182 72-bit event/cycle, while the expensive P=32-wide accesses can be confined to small local h/accumulator banks.

## Revised memory hierarchy

A concrete minimal hierarchy for the width-64 P=32 core is now:

1. backing q9 weight BRAM;
2. 4.5-KiB transposable weight ping-pong buffer for zero-fill-stall operation (or 2.25 KiB with the known 12.5% fill overhead);
3. 832-B persistent h/lambda layer-local bank;
4. 224-B current-slab residual buffer;
5. transpose accumulator bank, width to be determined by fixed-point range analysis;
6. narrow lambda-update pipeline fed from completed residuals.

This separates the high-bandwidth MAC-near storage from persistent backing state. It also means the PC-ALM-specific lambda state adds only 384 B per B=4,N=64 layer to the local persistent bank in the current 12-bit format.

## Comparison implication

For sPC, removing lambda saves 384 B of local persistent state and its narrow update pipeline. For PC-ALM, that extra state is worthwhile only if the observed reduction in useful relaxation count T survives fair matched-network comparisons. The architecture model therefore preserves the intended scientific comparison: lambda is not free, but its local memory-port cost is small enough that T reduction can dominate total traffic.

This result should not be used to claim an FPGA advantage over ePC or BP by itself. It only removes one implementation objection to the proposed PC-ALM spatial core.

## Result

The h/lambda port question does not force a retreat from the 1,024-cycle MAC target at the bandwidth level. For B=4,N=64,h14,lambda12, the complete persistent state is only 832 B and requires 186 conservative 72-bit backing events for one load+store per relaxation. The inner P=32 bandwidth can be served by roughly 1.5 KiB of layer-local state/context (using a provisional 14-bit transpose accumulator), while lambda itself needs only 0.5 scalar accesses/cycle on average.

The remaining blocker to declaring a concrete no-stall core is therefore narrower: determine the transpose-accumulator numeric width and then assign actual RAM/register banks and ports. The next highest-value experiment is fixed-point range instrumentation of the W^T residual accumulator and lambda update, recording peak/percentile dynamic range across depth and relaxation time. That measurement decides whether the provisional local-memory capacities above survive realistic accumulator widths.