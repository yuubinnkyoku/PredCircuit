# Minimal PC / leaky-PC-ALM core plan

Date: 2026-09-20
Status: pre-RTL architecture gate; analytical only

## Why this artifact exists

The width-64/depth-32 evidence now clears the project's pre-RTL gate strongly enough to plan a minimal switchable core, but not a full accelerator. The core must preserve the experimental controls: sPC remains a first-class mode, PC-ALM adds a dual state rather than silently changing the base datapath, and stochastic rounding can be enabled independently on the state-gradient and residual/dual paths.

The target is not “PC on FPGA”. The target is a synthesis-ready experiment vehicle that can measure when a spatially parallel local-learning datapath amortizes PC-ALM's extra dual state.

## Frozen algorithmic modes

One parameterized datapath exposes these modes:

| mode | alpha | gamma | dual state | state-grad rounding | residual rounding |
|---|---:|---:|---|---|---|
| sPC | 0 | 0 | disabled/clock-gated | nearest or SR | n/a |
| pure PC-ALM | >0 | 0 | enabled | nearest or SR | nearest or SR |
| leaky PC-ALM | >0 | >0 | enabled | nearest or SR | nearest or SR |

The nominal compact point is h14 / update14 / lambda12. The alternative compact point h15 / update13 / lambda12 must remain configurable because the software results show that bit width is a lattice co-design problem, not a monotone quality knob.

The dual update is

`lambda_next = Q_lambda((1-gamma) * lambda + alpha * Q_r(r))`.

The hidden-state update uses the quantized local state gradient and the configured effective state step. `alpha=0` must make the dual contribution exactly zero so the same matrix/residual datapath can serve as the sPC control.

## Datapath partition

The minimal core has five logical blocks:

1. prediction/residual MAC engine: computes local `W h` predictions and residuals;
2. transpose-credit MAC engine: computes the local `W^T r` contribution needed by hidden-state relaxation;
3. state-update lane bank: quantizes the local state gradient, applies the state-step coefficient, and writes h;
4. dual-update lane bank: quantizes residuals and applies `alpha` plus optional `(1-gamma)` feedback;
5. state/dual SRAM or BRAM banks with explicit read/write counters.

The first RTL should allow the two MAC directions to share one physical MAC array. A compile-time option may duplicate them later. This keeps the first synthesis comparison focused on the incremental dual/state machinery instead of confounding it with aggressive matrix parallelism.

## Rounding architecture

The recent width-64 experiment showed that tensor-shared stochastic rounding preserved useful BP credit on 28/28 trajectories, while role ablation showed that both state-gradient and residual/dual rounding contribute. Therefore the first hardware model does not assume one RNG per lane.

Use one pseudo-random word per quantizer invocation and broadcast it to all active lanes. Each lane still has its own fractional remainder comparator. Expose independent enables for state-gradient SR and residual SR so the four software ablation conditions remain synthesizable.

This is a hypothesis to test in RTL, not a claim that one shared RNG is universally sufficient. The software evidence currently covers the width-64 diagnostic and should not be generalized beyond it without holdout.

## Width-64 storage budget

For B=4, L=32, N=64, hidden scalars `S = B*(L-1)*N = 7,936`.

At h14/lambda12:

- sPC hidden state: `14*S = 111,104 bit = 13.5625 KiB`;
- PC-ALM hidden+dual: `26*S = 206,336 bit = 25.1875 KiB`;
- incremental dual capacity: `95,232 bit = 11.625 KiB`.

With Xilinx-style 36-Kib BRAM blocks, pure capacity lower bounds are 4 BRAM36 for h14 sPC and 6 BRAM36 for h14+lambda12 PC-ALM. These are only capacity bounds: banking, port width, ping-pong buffering, and placement can increase the realized count. The important new implementation question is whether the extra two-or-more BRAMs buy enough reduction in T to reduce total traffic and energy.

## Per-step operation budget

For the current residual MLP shape the existing analytical matrix count is `987,136 MAC/step`. The dual has `S=7,936` element updates per step, about 0.804% of the matrix-MAC count before charging leak separately.

The observed software budget remains the gate:

- useful leaky PC-ALM point: T=128;
- sPC: still 0/5 useful through T=1024.

Thus `T_sPC/T_PCALM > 8`, while the h14/lambda12 persistent-state traffic break-even is only `26/14 = 1.857`. Even with dual traffic, the simple persistent-state traffic ratio at 128 versus 1024 steps is `128*26/(1024*14) = 0.2321`. This is a lower-bound comparison because the 1024-step sPC point still fails the quality gate.

## Cycle model for the first synthesis sweep

Let `P` be the number of MAC lanes. Ignoring pipeline fill and bank conflicts, define

`C_matrix(P) = ceil(987136 / P)` cycles/relaxation step.

Let `V` be the number of element-update lanes. State and dual element work is O(S/V); with separate lane banks,

`C_elem(V) ~= ceil(7936 / V)`.

The first-order step time is

`C_step ~= C_matrix(P) + C_elem(V) + C_mem`,

where `C_mem` is measured from the actual banking schedule rather than assumed zero. Since `C_matrix` dominates for modest P, dual arithmetic should initially be overlapped with matrix streaming where dependencies permit. At high P the design can cross into a state-bandwidth-limited regime; that crossover is exactly what synthesis should locate.

The first synthesis sweep freezes `V=8` and varies only `P in {8,16,32,64}`. For this design point `C_elem(8)=992` cycles, while `C_matrix(P)` is 123,392 / 61,696 / 30,848 / 15,424 cycles respectively. Even at P=64, the non-overlapped element pass is only 6.432% of matrix time, and the arithmetic crossover is near P=995. Widening V before measuring memory stalls would therefore confound the experiment while buying at most 868 cycles per step. Report cycles/step, BRAM36, DSP, LUT, FF, Fmax, and estimated on-chip state traffic. Re-open V>8 only if measured banking/writeback stalls exceed the V=8 budget, the update path materially limits Fmax, or later matrix parallelism approaches the crossover. Do not convert analytical MAC counts into claimed speedups.

## Coefficient implementation choices

Expose exact software coefficients first, then a hardware-friendly coefficient sweep:

- `alpha`: fixed-point constant multiply;
- `1-gamma`: fixed-point constant multiply, with gamma near the tested 0.01--0.02 range;
- effective state step: fixed-point constant multiply.

For each coefficient, compare DSP multiplication against binary/CSD shift-add approximations. A coefficient approximation is accepted only if the existing credit-geometry gate remains satisfied; arithmetic resource reduction alone is not sufficient.

## Required counters and traces

The core must count, per run and preferably per layer:

- relaxation steps;
- state-gradient quantizer zero outputs;
- residual quantizer zero outputs;
- state writes that do not change h;
- state and dual saturation events;
- BRAM/SRAM reads and writes;
- MAC-active and element-update-active cycles.

These counters connect synthesis results to the software mechanism evidence. In particular, the state-lattice-lock hypothesis is not testable in hardware if zero requested updates and zero realized state moves are conflated.

## RTL entry decision

The pre-RTL gate is now considered met for a **minimal experimental core only**:

1. PC-ALM has a meaningful iteration/credit advantage over sPC at the width-64/depth-32 point: sPC is still unusable at T=1024 while PC-ALM has useful T=128 points;
2. compact low precision is viable on held-out seeds (h14/update14/lambda12 and h15/update13/lambda12);
3. the analytical dual arithmetic overhead is sub-1% of matrix MAC work and the observed T separation exceeds the state-traffic break-even;
4. recent stochastic-rounding diagnostics provide a hardware-plausible shared-RNG path rather than requiring independent random generators per neuron lane.

This does **not** authorize a full accelerator claim. End-to-end training, fair ePC/BP runtime comparison, coefficient quantization, and actual shared-MAC synthesis/resource data remain open.

## First falsifiable RTL milestone

Implement only one hidden-layer relaxation tile with switchable sPC / pure PC-ALM / leaky PC-ALM modes, parameterized h/update/lambda widths, shared-RNG stochastic rounding, and the counters above. Verify bit-exactly against a software fixed-point step model before synthesis.

Proceed to multi-layer composition only if the tile satisfies all of:

- bit-exact state and dual updates on deterministic test vectors;
- no unexplained saturation relative to software;
- synthesis produces a resource report for at least P={8,32,64}, initially with V=8 fixed;
- measured cycle/state-traffic model preserves the analytical PC-ALM-vs-sPC break-even margin.

If shared-RNG timing or comparator fanout dominates LUT/Fmax, the next comparison is grouped RNG (one random word per lane group), not immediately one RNG per lane.