# Magnitude-window v2: two-stage residual control + Sakana state_lr holdout

Date: 2026-09-18
Runs: 35248858665 (E1 v2, n=20), 35247923364 (Sakana state_lr, n=20).

## E1 stage-2 control (`spc_residual_gain_unit_first`)

Spec control: residual-norm gain, then cosine-preserving **unit first-layer L2**.

| T | control | useful | mean cosine | mean norm ratio | mean rel error |
|---:|---|---:|---:|---:|---:|
| 128 | spc_residual_norm_gain (stage-1) | 0/20 | 0.884 | 0.001 | 0.999 |
| 128 | spc_residual_gain_unit_first (stage-2) | **0/20** | 0.884 | 3.71 | 2.89 |
| 128 | spc_oracle_norm_match | 9/20 (45%) | 0.884 | 1.000 | 0.466 |
| 128 | pcalm_dual_leak | **20/20** | 0.948 | 1.009 | 0.330 |
| 256 | spc_residual_gain_unit_first | **0/20** | 0.830 | 3.71 | 2.97 |
| 256 | spc_oracle_norm_match | 4/20 (20%) | 0.830 | 1.000 | 0.565 |
| 256 | pcalm_dual_leak | **20/20** | 0.956 | 1.021 | 0.317 |

**Reading:** unit first-layer norm is the wrong hardware target for BP-geometry credit. The useful predicate wants first-layer norm **≈ BP scale** (ratio ∈ [0.5,2]), not L2=1. Stage-2 overshoots (~3.7× BP) and fails every seed. Combined with stage-1 (undershoot) and oracle (partial), the sPC rescale family cannot match dual-leak without knowing the BP magnitude.

## Sakana official state_lr holdout (seeds 980–999)

Official depth-32 `state_lr=0.234285` vs project `0.25`. Same ResidualMLP diagnostic.

| state_lr | leak | useful @T=128 | useful @T=256 | cos @128 | norm @128 | rel @128 |
|---|---:|---:|---:|---:|---:|---:|
| sakana 0.234285 | 0 | 45% | 5% | 0.964 | 1.477 | 0.600 |
| sakana 0.234285 | 0.005 | 85% | 85% | 0.960 | 1.227 | 0.414 |
| sakana 0.234285 | **0.01** | **90%** | **100%** | 0.956 | 1.044 | 0.334 |
| sakana 0.234285 | 0.02 | 85% | **100%** | 0.945 | 0.797 | 0.355 |
| project 0.25 | 0 | 55% | 5% | 0.968 | 1.493 | 0.600 |
| project 0.25 | 0.01 | 90% | 100% | 0.960 | 1.065 | 0.328 |
| project 0.25 | 0.02 | 90% | 100% | 0.949 | 0.818 | 0.339 |

**Reading (falsifier 4):** dual-leak useful rates are **stable under official Sakana state_lr**. Pure PC-ALM is lr-sensitive at T=128 (45–55%) and **collapses at T=256 (5%)** — long-horizon overshoot is the real problem, not under-stepping. The magnitude-window / dual-leak conclusion is therefore not an artifact of using project `state_lr=0.25`.

## Frozen cross-line conclusion (holdout-supported)

> Deep local credit fails primarily when **magnitude windows desynchronize** (sPC collapse; pure PC-ALM long-T overshoot; FlyVis residual/gate magnitude stories). Restoring scale without BP knowledge either fails (local residual scales) or only helps a subset of seeds (oracle). **Dual-leak PC-ALM is the only tested local method that reliably opens a fixed-budget, BP-scale, high-cosine credit window**, including under official Sakana coefficients. Exact gradients win by coupling global magnitude control to direction — not because local rules lack information.

## Hardware implication (strengthened)

Bank FPGA entry on **dual-leak PC-ALM + Q3.9/14-14-12**, not on sPC iteration count or post-hoc rescale. Report `state_lr` explicitly; pure PC-ALM is not a robust fixed-T deployment candidate.

## Next experiments

1. Leak sweep under official state_lr at width-64 mixed precision (link to 14/14/12).
2. FlyVis local PC with dual-leak-style magnitude control on the type-pair task (bridge the two lines experimentally).
3. End-to-end training (not only credit geometry) for leaky PC-ALM vs sPC on the shared ResidualMLP task.
