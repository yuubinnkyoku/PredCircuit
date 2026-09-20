# PC-ALM MAC / element-lane balance at the width-64 design point (2026-09-20)

## Question

The minimal-core plan proposed sweeping both matrix-MAC parallelism `P={8,16,32,64}` and element-update lanes `V={8,16,32,64}`. Before paying for a 4x4 synthesis sweep, ask whether `V` can matter at all in this range.

Use only quantities already established in the repository for B=4, L=32, N=64:

- matrix work per relaxation step: `M = 987,136 MAC`;
- hidden/dual scalar count: `S = 7,936`;
- integrated 8-lane dual engine is II=1 and the streamed frontend accepts the width-64 residual stream without a producer stall;
- matrix and element work are conservatively treated as serial here. Any legal overlap only strengthens the conclusion.

Define

`C_matrix(P) = ceil(M/P)`

and

`C_elem(V) = ceil(S/V)`.

## Exact cycle ratios

| P MAC lanes | C_matrix | V=8 C_elem | elem/matrix | V=16 | V=32 | V=64 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 123,392 | 992 | 0.804% | 496 | 248 | 124 |
| 16 | 61,696 | 992 | 1.608% | 496 | 248 | 124 |
| 32 | 30,848 | 992 | 3.216% | 496 | 248 | 124 |
| 64 | 15,424 | 992 | 6.432% | 496 | 248 | 124 |

For the intended first sweep, even the smallest `V=8` bank contributes at most 6.432% of the matrix cycles under the deliberately pessimistic no-overlap model. Increasing `V` from 8 to 64 can save at most `992-124=868` cycles at `P=64`, i.e. only 5.63% of the `P=64,V=8` serial step time `(15424+992)` before memory effects.

The arithmetic crossover where an 8-lane element bank becomes as slow as the matrix engine is approximately

`P_cross = M / (S/8) = 987136 / 992 = 995.10 MAC lanes`.

Thus the planned `P<=64` region is more than 15x below the arithmetic crossover. State-memory banking can still create a separate bandwidth bottleneck, but widening the element arithmetic alone cannot cure that; it must be measured as `C_mem`.

## Consequence for the first synthesis sweep

Do **not** spend the first synthesis budget on the full 4x4 `(P,V)` Cartesian product. Freeze `V=8` for the primary `P={8,16,32,64}` sweep. This matches the already verified 8-lane dual frontend and keeps the PC-ALM-only logic fixed while the shared matrix datapath scales.

Only synthesize `V>8` if one of these falsifiers occurs:

1. measured state/dual update stalls exceed the analytical `C_elem(8)` budget because banking or writeback serialization couples them to the element lanes;
2. the P=64 implementation shows element/update activity on the critical path or a material Fmax penalty attributable to the 8-lane schedule;
3. a later matrix design moves toward roughly `P~1000`, where arithmetic balance changes qualitatively.

This reduces the immediate synthesis design space from 16 points to 4 without assuming away the memory bottleneck. The primary hardware question becomes cleaner: how does shared MAC parallelism change area, Fmax and cycle count while the already-sufficient PC-ALM element frontend is held constant?

## Interpretation

This strengthens the earlier conclusion that PC-ALM's near-term hardware cost is dominated by lambda storage rather than element arithmetic. At P=64, an entirely non-overlapped 8-lane element pass is only 992 cycles versus 15,424 matrix cycles. The existing streamed frontend has already demonstrated that dual work can consume residuals faster than they are produced, so a larger V is not justified by producer/consumer throughput either.

This is a scheduling result, not an energy result. BRAM accesses, state writes, weight traffic and routing can dominate even when arithmetic cycles do not. Those effects belong in the first measured P sweep rather than being hidden by prematurely widening V.