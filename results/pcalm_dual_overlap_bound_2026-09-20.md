# Width-64 PC-ALM dual-overlap bound (2026-09-20)

## Question

Can the selected 8-lane lambda engine be hidden behind the matrix/state datapath at the depth-32, width-64, batch-4 design point, or does lambda traffic force a serial 992-cycle tail every relaxation step?

This note does not claim a measured integrated-scheduler result. It derives the conditions an integrated scheduler must satisfy and separates throughput from burst buffering.

## Reused facts

No scientific experiment was rerun.

- Persistent hidden-state / residual / lambda scalars: `S = 4 * 31 * 64 = 7936`.
- Selected dual engine: 8 lanes, 992 words, initiation interval 1 after the synchronous RAM read pipeline.
- One complete lambda sweep therefore issues in 992 cycles (plus pipeline drain).
- Existing matrix-work model: 987,136 MAC per relaxation step.
- Candidate matrix parallelism: 32 or 64 MAC/cycle, giving ideal matrix windows of 30,848 and 15,424 cycles respectively.
- Current software gate: multiplier-free PC-ALM is robust at T=104; the existing sPC comparison remains below criterion at T=1024.

## Throughput condition

Let `R` be the long-run residual production rate in scalar residuals/cycle. An 8-lane dual engine can consume 8 scalar residuals/cycle, so stable overlap requires

`R <= 8`.

Even if every one of the 7936 residual scalars is produced during only the matrix window, the average rates are

- P_MAC=32: `7936 / 30848 = 0.2573` scalar/cycle;
- P_MAC=64: `7936 / 15424 = 0.5145` scalar/cycle.

The corresponding service headroom of the 8-lane engine is therefore about 31.1x and 15.5x. On average throughput alone, lambda update is not a bottleneck.

Equivalently, if lambda work were fully serialized, its 992 issue cycles are 3.22% of the P_MAC=32 matrix window and 6.43% of the P_MAC=64 matrix window. Full overlap can remove that serial term; failure to overlap it is a scheduling/memory-dependency problem, not an arithmetic-throughput problem.

## Burst condition and FIFO bound

Average throughput is insufficient: a 64-wide matrix datapath may expose residuals in bursts wider than the 8-lane consumer.

For a burst of `B` residual scalars arriving in one cycle followed by enough idle cycles, the immediate queue growth is

`Q_peak >= max(0, B - 8)` scalars.

Thus a single 64-scalar burst needs at least 56 scalar slots (672 bits at 12 bit/scalar) if it must be accepted without back-pressure. A 32-scalar burst needs 24 slots (288 bits). These are tiny compared with the 95,232-bit lambda store.

More generally, for any concrete residual-valid trace with arrivals `a[t]`, the exact minimum FIFO capacity for an 8-scalar/cycle consumer is the maximum backlog of the recurrence

`q[t+1] = max(0, q[t] + a[t] - 8)`.

This recurrence should be instrumented in the integrated scheduler testbench. It avoids guessing from the average rate.

## Read-after-write dependency

The current dual engine performs a synchronous read of lambda word `addr`, computes credit from the old lambda and current residual, and writes the updated lambda back one pipeline stage later. Within one sweep, each address is issued once. Therefore there is no same-sweep lambda read-after-write hazard if addresses are unique.

The next relaxation step revisits an address only after the rest of the sweep / matrix schedule, far beyond the one-stage RAM pipeline. The dangerous dependency is instead external: a residual must not be consumed before the corresponding current-state prediction/error is valid, and downstream logic must use the intended old-lambda credit versus newly written lambda according to the software update ordering.

This is precisely what the integrated scheduler must verify bit-exactly.

## Hardware consequence

The first scheduler prototype should keep `P_lambda=8`. Widening the dual engine is not justified by throughput: at P_MAC=64 it already has about 15.5x average service headroom, while previous XC7 synthesis showed that 16 lanes roughly double logic and 32 lanes both quadruple logic and worsen BRAM packing.

The useful new hardware object is therefore not a wider dual engine but a shallow residual FIFO plus ordering assertions around the existing 8-lane lambda RAM.

Recommended initial FIFO capacity: 64 scalars (8 words x 8 lanes = 768 bits). This covers one full 64-scalar instantaneous burst with margin and is less than 1% of the lambda-state bit count. It is a hypothesis to test, not yet a proven sufficient bound for the real matrix schedule.

## Pass/fail criterion for the scheduler prototype

For one complete relaxation step at the width-64 point:

1. no residual is dropped or duplicated;
2. all 7936 lambda scalars are updated exactly once;
3. bit-exact credit and lambda values match the software/standalone RTL ordering;
4. FIFO overflow count is zero;
5. matrix stall cycles attributable to the dual path are zero (preferred) or are reported explicitly;
6. maximum FIFO occupancy is recorded.

If a 64-scalar FIFO passes under the actual residual-valid trace, the previous claim that the 992-cycle dual sweep can be hidden gains an RTL-level basis. If it overflows, the measured occupancy trace—not a wider dual engine by default—should determine the next banking/FIFO change.
