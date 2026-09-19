# PC-ALM tensor-shared stochastic rounding (2026-09-19)

## Question

Does fixed14_i1 stochastic rounding still preserve first-layer BP credit when one random variate is broadcast over every element of each update-precision tensor, instead of drawing independent variates per element?

## Reused experiment

No trajectories were rerun for this summary. It aggregates the completed 28-trajectory GitHub Actions experiment (7 model/data seeds that failed under nearest rounding × 4 independent rounding seeds). Conditions match the earlier hard case: depth=32, width=64, T=256, state=fixed15_i3, update=fixed14_i1, dual=fixed12_i1, state_lr=0.234285, dual leak=0.02.

## Result

All 28 trajectories were finite and all 28 satisfied the existing useful-first-layer-credit criterion.

| metric | tensor-shared stochastic rounding |
|---|---:|
| useful | 28/28 |
| finite | 28/28 |
| BP cosine mean | 0.961436 |
| BP cosine std | 0.004995 |
| BP cosine min | 0.952325 |
| BP cosine max | 0.969706 |
| gradient norm ratio mean | 0.871800 |
| relative error mean | 0.290265 |
| realized/requested state movement mean | 0.527678 |
| update zero fraction mean | 0.499016 |
| state zero-step fraction mean | 0.966341 |

Per-seed minima remained above 0.95 except none: the global minimum was 0.952325 (seed 858). Thus broadcasting one random variate across each quantizer invocation did not measurably destroy credit geometry relative to the earlier element-independent result (56/56 useful, mean cosine 0.96095).

## Important confound discovered during review

The current diagnostic monkeypatches `alignment.quantize` whenever `precision == fixed14_i1`. In `run_alignment`, the same update precision is used both for state-gradient quantization (`q_grad`) and for constraint-residual quantization (`q_residual`) before the dual update. Therefore the existing stochastic-rounding experiments do **not** isolate stochastic rounding to `Q(grad_h)` alone: they stochastic-round both state gradients and residuals.

This does not invalidate the empirical result that stochastic rounding of the fixed14_i1 path is robust, but it weakens the stronger mechanistic claim that rescuing lattice lock has already been causally localized to state-gradient rounding alone.

## Interpretation

Strong spatial correlation in the random variates is tolerated in this setting. The dominant requirement may be temporal unbiasedness rather than elementwise independence. However, because gradient and residual quantizers were coupled in the diagnostic, the next causal experiment must separate those two roles before moving further toward an RTL random-number-sharing design.

## Next discriminating experiment

Hold all precisions and schedules fixed and compare, on the same seven hard seeds:

1. stochastic rounding only for state gradients, residual quantization nearest;
2. nearest state gradients, stochastic rounding only for residuals;
3. stochastic rounding for both (existing condition);
4. nearest for both (existing baseline).

If state-gradient-only reproduces the rescue while residual-only does not, the `Q(grad_h)` lattice-lock mechanism is substantially strengthened. If residual-only also rescues, the dual/residual path is part of the mechanism and the RTL cost model must include stochastic rounding there as well.