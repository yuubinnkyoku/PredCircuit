# Short-budget PC-ALM causal propagation front (2026-09-24)

## Setup

Source: successful GitHub Actions run `PC-ALM short-budget probe` #1 on commit `d74819c8`, seeds 980--983. Network depth 32, width 8. Budgets: T={1,2,4,8,12,15,16,24,32}. Methods are the existing state PC baseline (`pc`) and PC-ALM (`pcalm`) using the same model/batch per seed.

The hardware model established earlier gives an optimistic fully layer-parallel latency ratio against one BP/ePC reverse sweep of approximately `2T/H`, H=31, when both W and W^T matvecs sustain the P=8 datapath. Hence latency break-even requires approximately T<=15.

## Result: propagation front

Define the observed credit front as the largest `distance_from_output` whose layer gradient norm exceeds 1e-12. Across four seeds:

| T | sPC/PC front distance | PC-ALM front distance |
|---:|---:|---:|
| 8 | 8,8,8,8 | 8,8,8,8 |
| 12 | 12,12,12,12 | 12,12,12,12 |
| 15 | 14,14,14,13 | 15,15,15,15 |
| 16 | 14,15,14,14 | 16,16,16,16 |
| 24 | 19,19,18,18 | 24,24,24,24 |
| 32 | 21,21,21,21 | 30,30,31,30 |

PC-ALM therefore propagates the nonzero credit front at essentially one layer per primal step in this implementation. It is substantially better than the sPC baseline after the latter starts attenuating/underflowing, but it does not propagate faster than one layer per step.

At T=15 the first layer (distance 31) has exactly zero gradient in all four PC-ALM seeds; its cosine to BP is therefore undefined and its gradient-norm/BP-norm ratio is 0. At T=32 only one of four seeds has a first-layer gradient above the 1e-12 reporting threshold; its first-layer cosine is about 0.181 and norm ratio about 7.8e-9. Thus reaching the layer is not the same as obtaining a useful BP-like update.

All traces report finite values; this is not a divergence/artifact failure.

## Interpretation: a structural latency incompatibility

For a strictly nearest-neighbor, synchronous local relaxation rule initialized from the feed-forward state, information injected at the output cannot affect a layer at graph distance d in fewer than d local update rounds. The experiment directly exhibits this light-cone-like bound: PC-ALM reaches distance T almost exactly.

For depth 32, useful first-layer credit therefore has a structural lower bound T>=31 before considering convergence quality. But the optimistic FPGA latency break-even derived from two matvecs per relaxation step is T<=15. These conditions do not overlap:

`first-layer causal reach: T >= 31`

`optimistic BP/ePC latency break-even: T <= 15`

Consequently, coefficient tuning of alpha/rho/eta_h cannot by itself make the current synchronous nearest-neighbor PC-ALM update beat a one-sweep BP/ePC baseline in end-to-end latency while also updating the first layer. Hyperparameters can change amplitudes, stability and convergence after the causal front arrives, but cannot make information cross more graph edges per synchronous local round.

This does **not** invalidate PC-ALM as a learning rule or its advantage over sPC. It specifically rules out one speedup route for this architecture/model: obtaining full-depth credit in <=15 ordinary local relaxation rounds merely by tuning coefficients or early stopping.

## Hardware implication

The next algorithmic/hardware question is no longer "can alpha/rho make T fall from 112 to 15?". To cross the latency boundary, the update schedule or communication radius must change. Candidate directions should be evaluated by their added hardware cost, e.g. pipelined/wavefront scheduling within a nominal iteration, multi-hop/skip credit paths, block-local solves, or a prospective/error-based mechanism. Any such method must be charged for extra matvecs, wiring/state, and critical-path depth; relabeling multiple local substeps as one iteration does not evade the bound.

The existing PC-ALM coefficient sweeps remain relevant for convergence/precision/energy once the credit front has arrived, and sPC remains the required baseline.
