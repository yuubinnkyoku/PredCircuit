# PC-ALM propagation-front diagnosis (2026-09-23)

## Setup

- ResidualMLP: depth 32, width 8, batch 4
- seeds: 970, 971, 972
- PC-ALM: alpha=0.925, rho=1, eta_h=0.25
- budgets: T={1,2,4,8,16,32,56,80,112}
- Metrics: per-layer residual/dual norms, per-layer BP gradient cosine and norm ratio, global BP gradient cosine.

## Main result

The earlier hypothesis that the large training gain between T=56 and T=112 is caused by a late residual/dual front reaching the first hidden layer is not supported.

At T=32, residual and dual activity already occupy 29/31 hidden constraints for every seed (layers 2..30). By T=56, activity reaches all 31/31 constraints, including layer 0. Thus the first layer is reached no later than the interval T=32..56, well before the high-quality T=112 solution.

The first-layer BP cosine nevertheless continues to improve substantially after the front has arrived:

| T | seed 970 | seed 971 | seed 972 | mean |
|---:|---:|---:|---:|---:|
| 32 | 0 | 0 | 0 | 0 |
| 56 | 0.9717 | 0.9522 | 0.9009 | 0.9416 |
| 80 | 0.9764 | 0.9529 | 0.9091 | 0.9461 |
| 112 | 0.9773 | 0.9534 | 0.9114 | 0.9474 |

The global all-layer BP cosine shows the same distinction between front arrival and continued settling:

| T | seed 970 | seed 971 | seed 972 | mean |
|---:|---:|---:|---:|---:|
| 32 | 0.8308 | 0.8578 | 0.8496 | 0.8461 |
| 56 | 0.9703 | 0.9582 | 0.9503 | 0.9596 |
| 80 | 0.9813 | 0.9678 | 0.9632 | 0.9708 |
| 112 | 0.9838 | 0.9699 | 0.9664 | 0.9734 |

The first-layer gradient norm is still very small at front arrival. Its BP norm ratio is only about 0.073--0.077 at T=56, grows to about 0.140--0.149 at T=80, and reaches about 0.225--0.240 at T=112. Therefore the late benefit is not primarily directional alignment: direction becomes good near T=56, while credit magnitude keeps accumulating for many more dual updates.

This separates two time scales:

1. **transport/front time**: roughly 32--56 outer steps for depth 32;
2. **credit-amplitude/settling time**: continues well beyond T=56 and is still materially changing by T=112.

## Interpretation

A coefficient sweep aimed only at making the dual/residual front move faster is unlikely to recover the T=112 training quality at T~32. The front is already present by T=56, yet the first-layer credit magnitude is only about one third of its T=112 magnitude. The more informative target is therefore to accelerate amplitude build-up after arrival while preserving the already-good gradient direction and finite dynamics.

This changes the next parameter-search objective: optimize a short-budget score at T=32/56 that combines first-layer cosine with BP norm ratio (and finite/stability constraints), rather than merely minimizing front-arrival time. Candidate knobs remain alpha, rho, and eta_h, but they should be judged by early credit magnitude, not just propagation depth.

## CI note

The diagnostic workflow itself completed successfully for all three seeds. The contemporaneous ordinary CI failure is non-scientific: pytest, type checking, and ruff lint pass; only `ruff format --check` rejects formatting in `scripts/diagnose_pcalm_propagation_front.py`. This should be fixed independently and must not be interpreted as an experimental failure.
