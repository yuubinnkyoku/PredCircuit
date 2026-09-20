# Width-64 PC-ALM dual FIFO schedule bound (2026-09-20)

## Question

The previous overlap note showed that an 8-lane dual engine has ample *average* throughput, but left a real uncertainty: can repeated 64-scalar residual bursts accumulate faster than a 64-entry FIFO can drain?

This note closes that question for the first RTL architecture under consideration: one shared matrix-MAC pool with at most 64 MAC/cycle, width 64, and residual vectors released only after their corresponding dense 64x64 matrix operation is complete. It does **not** claim the same bound for a future architecture with multiple independent matrix engines producing residual vectors concurrently.

## Reused facts

No scientific experiment was rerun.

- Hidden width: `N = 64`.
- Persistent hidden residual/lambda vectors: `batch * hidden_layers = 4 * 31 = 124` vectors, each 64 scalars, hence 7936 scalars total.
- Selected lambda engine: 8 lanes, initiation interval 1, so service rate `C = 8 scalar/cycle` after its synchronous-read pipeline is filled.
- Candidate matrix pool: `P_MAC <= 64 MAC/cycle`.
- A dense width-64 matrix-vector operation requires `64 * 64 = 4096 MAC`.
- A complete 64-scalar residual vector cannot be released before the matrix result on which it depends is available.

## Inter-burst lower bound from matrix work

With one shared matrix pool, two independent 64-wide residual vectors cannot both complete their required dense 64x64 matrix work arbitrarily close together. Even at the widest candidate point,

`4096 MAC / 64 MAC/cycle = 64 cycles`.

Therefore consecutive full 64-scalar residual-vector releases attributable to this shared pool are separated by at least 64 issue cycles. At `P_MAC=32`, the lower bound is 128 cycles and is even easier.

This is stronger than the average-rate argument: it bounds the local burst spacing using the causal matrix work required to create each residual vector.

## Exact FIFO recurrence

For an 8-scalar/cycle consumer,

`q[t+1] = max(0, q[t] + a[t] - 8)`.

Take the worst allowed producer event under this architecture: `a[t] = 64` in one cycle, followed by no new full-vector release for at least 63 cycles.

Immediately after the burst,

`q_peak = 64 - 8 = 56 scalars`.

The remaining 56 scalars drain in seven more service cycles. Thus the queue is empty after 8 total service cycles, while the next full residual vector cannot arrive for at least 64 cycles. There are at least 56 empty cycles of margin before the next burst at `P_MAC=64`.

Hence repeated full-width bursts do not accumulate. The exact capacity required by this idealized release model is 56 scalar slots; a 64-scalar FIFO provides 8 slots of margin:

- minimum bound: `56 * 12 = 672 bit`;
- proposed FIFO: `64 * 12 = 768 bit`.

The proposed FIFO is therefore sufficient for the shared-single-MAC-pool architecture without producer back-pressure from the dual path.

## Relation to the whole-step work model

There are 124 hidden residual vectors. Even if every vector were emitted as a single 64-scalar burst, the dual engine performs exactly

`124 * (64 / 8) = 992 service cycles`

per relaxation step.

The matrix lower bound from just one 64x64 dense operation per residual vector is

`124 * 4096 / 64 = 7936 cycles`

at `P_MAC=64`, already 8x longer than the dual service demand. The repository's fuller matrix-work model is 987136 MAC, or 15424 cycles at 64 MAC/cycle, giving still more scheduling slack. The FIFO proof therefore does not rely on the favorable whole-step average; the per-vector causal spacing alone is enough.

## What this proves, and what it does not

For the first target architecture (`P_MAC <= 64`, one shared matrix pool, no concurrent independent residual producers), the 64-entry residual FIFO is no longer merely a heuristic. Its capacity follows from the matrix-production rate: a full 64-wide burst drains in 8 cycles, while producing the next full vector needs at least 64 cycles.

This does **not** prove a universal 64-entry bound. It must be revisited if any of the following changes:

1. multiple matrix engines can finish independent residual vectors in the same or nearby cycles;
2. a producer bypasses the dense 64x64 matrix dependency and can emit residual vectors faster than one per 64 cycles;
3. the dual engine is stalled by lambda-memory conflicts or downstream credit back-pressure;
4. the integrated schedule requires a different old-lambda/new-lambda ordering than the standalone engine.

The standalone dual RTL currently uses one synchronous lambda read and a one-stage read/compute/write pipeline with unique addresses during a sweep, so item 3 is not implied by an internal same-sweep lambda RAW hazard. It remains an integration property to assert.

## Hardware consequence

For the first integrated scheduler prototype, keep:

- `P_MAC = 64` maximum;
- `P_lambda = 8`;
- residual FIFO capacity = 64 scalars (768 bit at 12 bit/scalar).

There is no throughput reason to widen the dual engine or enlarge the FIFO before integration. The remaining useful RTL experiment is now narrower: verify the causal assumptions and ordering, rather than rediscover queue capacity by brute force.

The integrated testbench should assert:

- no two full residual-vector releases occur less than 64 cycles apart when they share the single 64-MAC pool;
- FIFO occupancy never exceeds 56 under full-width atomic releases (or record the actual smaller peak if residuals stream gradually);
- exactly 7936 lambda scalars update once per relaxation step;
- no dual-path back-pressure stalls the matrix producer;
- credit uses the intended old lambda before the write-back of the new lambda.

If those assertions pass, the PC-ALM-specific arithmetic latency is hidden at this design point; the irreducible hardware premium is then dominated by the 95232-bit lambda state store rather than dual compute or residual buffering.
