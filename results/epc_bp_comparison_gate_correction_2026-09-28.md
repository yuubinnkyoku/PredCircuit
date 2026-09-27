# ePC/BP comparison gate correction (2026-09-28)

## Why this correction is needed

The repository already shows that ePC at T=1 has first-layer credit exactly collinear with BP, with norm ratio equal to error_lr. That is an initialization/reparameterization property, not evidence that ePC has reached its predictive-coding equilibrium. Therefore T=1 must not be used as the quality-matched ePC operating point merely because BP cosine is 1.

This also means the existing width-8 hardware lower bound should be read as a digital-compute lower bound, not as a measured quality-matched ePC speedup.

## Required matched-quality gate

For every ePC point report both:

1. BP geometry: global/layerwise cosine, norm ratio, relative error.
2. Equilibrium quality: energy decrease and a stationarity measure (for example max or RMS error-state update / gradient norm), relative to a long-run ePC reference under the same model state.

The minimum ePC T is the first budget satisfying the equilibrium gate; BP geometry is diagnostic and must not be used alone to tune error_lr.

For BP, report the ordinary forward/backward cost as the reference baseline. For sPC and PC-ALM, retain the existing useful-credit geometry gate and additionally report their own residual/stationarity traces so that all PC variants expose convergence rather than only gradient resemblance.

## Immediate experiment

Run a paired depth=32 sweep at width=8 first, because existing ePC discovery seeds 940--944 can be reused without rerunning their T=1 geometry:

- ePC: error_lr in {0.03, 0.1, 0.3}, T in {1,2,4,8,16,32,64,96,128}; add energy and stationarity traces.
- PC-ALM: reuse existing T=112 high-quality point as the realistic narrow reference.
- BP: one forward/backward reference.
- sPC: retain as the state-PC baseline.

Only after an equilibrium-qualified ePC T is known should the same protocol be moved to width=32/64.

## Hardware accounting consequence

Keep two separate comparisons:

- arithmetic work: total dense MACs and state/weight accesses;
- ideal latency: explicit resource-matched engine counts and dependencies.

Do not infer a 2x dual-engine speedup unless the prediction/residual/dual/transpose dependency is implemented and validated at cycle level. The existing dual-state RTL validates lambda read-modify-write throughput, not simultaneous Wz and W^Tq execution.

This protocol prevents two optimistic shortcuts from entering the main claim: treating ePC T=1 as converged, and treating unvalidated dual-matrix overlap as achieved hardware throughput.
