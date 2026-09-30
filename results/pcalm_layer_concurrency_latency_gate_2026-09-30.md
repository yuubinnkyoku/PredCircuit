# Layer-concurrency latency gate (2026-09-30)

## Reused evidence

For depth=32, width=64, batch=4, the existing held-out fixed-point sweep reports:
- T=96: 14/15 useful, mean BP cosine 0.92046.
- T=128: 15/15 useful, mean BP cosine 0.95018.
- T=160: 15/15 useful, mean BP cosine 0.96968.
- T=192: 15/15 useful, mean BP cosine 0.97779.

No experiment is rerun here.

## Best-case layer-concurrency bound

Let H=31 equal-width hidden transitions. Compare deliberately in PC-ALM's favor: every layer has a local core and all H PC-ALM layers can execute a matrix phase concurrently, while a rigid layer-local BP pipeline must respect forward/backward layer dependencies and therefore advances one layer per matrix phase.

Using one layer matrix phase as the latency unit:
- ideal fully layer-parallel PC-ALM update: initial forward H + relaxation 2T + local weight-gradient phase 1. If the initial forward is also dependency-serial, latency is H + 2T + 1.
- rigid layer-local BP update: forward H + activation backward H + weight-gradient phase. Weight gradients can be produced locally during/after backward; a conservative upper bound is 3H, while a fused implementation is closer to 2H.

To give PC-ALM the easier gate, compare against the slower 3H BP bound and omit PC-ALM's serial initial forward, yielding the optimistic necessary condition

    2T + 2 < 3H.

For H=31:

    T < (3*31 - 2)/2 = 45.5.

Thus T<=45 is required even under an unrealistically favorable assumption that all relaxation layers run concurrently and PC-ALM pays no depth-serial forward latency.

Observed T values are well above this:
- T=96: optimistic latency ratio (2T+2)/(3H) = 2.086x
- T=128: 2.774x
- T=160: 3.462x
- T=192: 4.151x

At the first universally useful tested point T=128, ideal layer concurrency is still insufficient by about 2.77x. At the balanced T=192 point it is insufficient by about 4.15x.

Including the actual initial forward makes the bound stricter:

    (H + 2T + 1)/(3H)

which is 3.108x at T=128 and 4.495x at T=192.

## Interpretation

Layer concurrency can recover at most an O(H) factor from PC-ALM's O(T) relaxation work. For the measured depth-32 fixed-point regime, T/H is too large for this mechanism alone to beat even a deliberately handicapped rigid layer-local BP accelerator. A fair reconfigurable BP engine that assigns all MACs to the currently active layer would be stronger still.

Therefore increasing FPGA area to instantiate one PC core per layer does not by itself create a latency crossover for the current algorithm. A meaningful latency advantage requires either substantially reducing T (below roughly 45 in this depth-32 case), exploiting finer-grained asynchronous overlap that changes the effective T, or targeting a workload where BP's dependency/control costs are materially larger than this dense-chain model.
