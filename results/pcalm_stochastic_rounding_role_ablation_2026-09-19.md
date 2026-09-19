# PC-ALM stochastic-rounding role ablation (2026-09-19)

## Question

The earlier fixed14_i1 stochastic-rounding diagnostic affected both state-gradient quantization `Q(g_h)` and constraint-residual quantization `Q(r)` before the dual update. Which path actually contributes to rescuing the seven hard seeds that fail under nearest rounding?

## Reused experiment

No trajectories were rerun. This note aggregates the completed 14-trajectory role-ablation run `35425699738` and its successful aggregate run `35428404468` on the same hard seeds 845, 848, 849, 851, 852, 856, 858. Conditions remain depth=32, width=64, T=256, state=fixed15_i3, update=fixed14_i1, dual=fixed12_i1, state_lr=0.234285, dual leak=0.02. Stochastic rounding uses tensor-shared randomness.

- `gradient`: stochastic `Q(g_h)`, nearest `Q(r)`
- `residual`: nearest `Q(g_h)`, stochastic `Q(r)`

All 14 trajectories were finite and had zero state/update/dual saturation.

## Aggregate result

| role | useful | BP cosine mean | std | min | grad-norm ratio mean | relative error mean | realized/requested mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| gradient-only | 7/7 | 0.934671 | 0.008865 | 0.919812 | 0.791079 | 0.384629 | 0.418965 |
| residual-only | 3/7 | 0.899550 | 0.013647 | 0.877707 | 0.685831 | 0.486693 | 0.296779 |

For comparison, the same seven seeds under nearest rounding had mean BP cosine about 0.8397. Thus gradient-only improves mean cosine by +0.0950 and residual-only by +0.0599. Both interventions improve cosine over nearest on all 7/7 seeds, even though residual-only crosses the existing useful-credit criterion on only 3/7.

## Paired cosine result

| seed | nearest | gradient-only | residual-only | gradient - residual |
|---:|---:|---:|---:|---:|
| 845 | 0.851390 | 0.943512 | 0.910093 | +0.033418 |
| 848 | 0.840982 | 0.940873 | 0.899545 | +0.041329 |
| 849 | 0.779202 | 0.931446 | 0.887783 | +0.043663 |
| 851 | 0.848682 | 0.935567 | 0.902820 | +0.032748 |
| 852 | 0.854936 | 0.925762 | 0.895850 | +0.029912 |
| 856 | 0.871942 | 0.945725 | 0.923055 | +0.022670 |
| 858 | 0.830540 | 0.919812 | 0.877707 | +0.042105 |

Gradient-only beats residual-only on every seed, by 0.0227 to 0.0437 cosine. But residual-only also improves every seed over nearest, so the residual/dual path is causally relevant rather than a spectator.

The previously measured tensor-shared `both` condition was 28/28 useful with mean cosine 0.961436 and minimum 0.952325 across four rounding RNG trajectories per model/data seed. Because those `both` trajectories do not use exactly the same single RNG trajectory as this role ablation, they should be treated as a robustness reference rather than a strict per-seed additive interaction test.

## Interpretation

The simple hypothesis "nearest rounding destroys credit only because sub-LSB state-gradient updates become zero" is insufficient. The data support a coupled low-precision mechanism:

1. quantizing `g_h` directly removes credit-bearing state updates and is the larger effect;
2. quantizing residual `r` changes the dual trajectory through `lambda <- lambda + alpha r` and thereby changes later state gradients;
3. stochastic rounding on either path partially restores BP-like geometry, while stochastic rounding on both paths gives the strongest previously observed result.

This is consistent with a coupled state/dual lattice-lock picture. It also means an eventual PC-ALM RTL cost model should not assume that stochastic rounding is needed only on the state-update datapath.

## Hardware implication

The result narrows the design question. A minimal low-precision PC-ALM core should budget stochastic rounding at both fixed14_i1 quantization sites unless a later experiment proves that the residual path can be widened or rescaled more cheaply. Tensor-shared randomness remains attractive: earlier 28/28 results show that one random variate broadcast per quantizer invocation preserves useful credit when both paths are stochastic.

## Highest-value next experiment

Measure the temporal mechanism directly rather than adding more RNG seeds. On the same hard seeds, log per relaxation step and layer: zero/nonzero events for `Q(g_h)` and `Q(r)`, residual norm, dual norm/increment, and first-layer BP-cosine proxy or local credit direction. Compare nearest, gradient-only, residual-only, and both. Lagged correlations should distinguish direct state-grid rescue from delayed dual-mediated rescue and show whether the residual path contributes with a characteristic time lag. This is the most informative next step before further RNG-sharing reduction or RTL work.
