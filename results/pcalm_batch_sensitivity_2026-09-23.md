# PC-ALM batch sensitivity (2026-09-23)

## Setup

- depth=32, width=8
- PC-ALM: alpha=0.925, rho=1.0, eta_h=0.30
- seeds: 970, 972
- 16 mini-batches per seed (32 cases total)
- relaxation budgets: T={32,56,80,112}
- Metrics: global BP-gradient cosine, first-layer BP-gradient cosine/norm ratio, residual and dual summaries.

The two GitHub Actions jobs completed successfully and all 128 measurements were finite.

## Aggregate geometry

| T | global cosine mean | global min | first-layer cosine mean | first-layer min | first-layer norm/BP mean |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.6980 | 0.4513 | 0.7487 | 0.4586 | 5.03e-7 |
| 56 | 0.9064 | 0.8241 | 0.9132 | 0.8327 | 0.3710 |
| 80 | 0.9369 | 0.8760 | 0.9132 | 0.8290 | 1.1623 |
| 112 | 0.9586 | 0.8817 | 0.9636 | 0.8552 | 1.3792 |

T=56 is therefore not uniformly sufficient: the mean direction is good, but some batches remain substantially worse. Increasing T to 112 improves the mean global cosine by 0.0522, with per-batch changes ranging from -0.0443 to +0.1447.

## Required T distribution

Using the deliberately modest criterion global cosine >= 0.90 and first-layer cosine >= 0.90, the earliest measured budget was:

- T=56: 16/32 cases
- T=80: 6/32
- T=112: 9/32
- not reached by T=112: 1/32

Among cases that reached the criterion, mean required T was 76.9. Thus an oracle adaptive schedule over {56,80,112} would reduce relaxation work by about 31% versus always using T=112, but this is only useful if a hardware-visible stopping signal can predict the difficult cases.

A stricter criterion (global >= 0.95, first-layer >= 0.90) required T=56/80/112 in 1/10/16 cases respectively, with 5/32 still below threshold at T=112. The mean over successful cases was 98.1, leaving little adaptive headroom.

## Can residual/dual summaries predict extra settling need?

For each batch define the target as the global-cosine gain from T=56 to T=112. Spearman correlations between this gain and signals visible at T=56 were:

| T=56 signal | Spearman rho with cosine gain |
|---|---:|
| residual L2 sum | +0.005 |
| residual L2 max | -0.043 |
| dual L2 sum | -0.009 |
| max abs dual | -0.041 |

These four simple summaries carry essentially no monotonic information about how much the batch benefits from another 56 relaxation steps. By contrast, the unavailable-in-hardware T=56 global BP cosine correlates strongly with future gain (Spearman rho=-0.861), confirming that difficult batches are real but are not exposed by the current aggregate residual/dual statistics.

First-layer norm/BP ratio has only moderate association (Spearman rho=-0.369) and also requires BP, so it is not a stopping signal.

## Interpretation

The batch-sensitivity hypothesis is supported: required relaxation varies materially by input. However, the simplest proposed adaptive-stopping mechanism is not supported. Aggregate residual magnitude and dual magnitude at T=56 do not identify batches that need more relaxation.

This matters for the FPGA case. An oracle can lower mean T from 112 to about 77 under a lenient geometry criterion, but a realizable controller cannot claim that saving from the present signals. Fixed T=56 is also not defensible as a uniformly high-quality setting.

The next useful test should therefore avoid another coefficient sweep. It should test richer *temporal* convergence signals that remain hardware-local, e.g. residual decay ratios over the last 4-8 steps, dual increment norm ||Delta lambda||, state-update norm ||Delta h||, and sign/oscillation counts. If those also fail to predict required T, adaptive stopping should be deprioritized and the hardware cost model should use the measured fixed-budget cost (likely near T=112 for high gradient fidelity).
