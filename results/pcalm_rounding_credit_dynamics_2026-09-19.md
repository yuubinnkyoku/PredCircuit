# PC-ALM rounding credit dynamics: seed-858 replication

Date: 2026-09-19

## Question

Does the fixed14_i1 stochastic-rounding rescue observed on seed 849 replicate on a second hard width-64/depth-32 seed, and is the improvement concentrated in deep credit rather than the output-adjacent layer?

The four conditions use the same PC-ALM setup and differ only in where stochastic rounding is applied: neither path (`nearest`), the residual/dual path only (`residual`), the state-gradient path only (`gradient`), or both (`both`). The rounding-dynamics workflow records the parameter-gradient cosine to the BP baseline at every relaxation step.

## Seed 858: final step (T=256)

| rounding role | global BP cosine | first/deep weight BP cosine | last/output-side BP cosine |
|---|---:|---:|---:|
| nearest | 0.889716 | 0.830540 | 0.999867 |
| residual only | 0.921557 | 0.866630 | 0.999853 |
| gradient only | 0.952619 | 0.919847 | 0.999820 |
| both | **0.974752** | **0.956387** | 0.999812 |

Relative to nearest, residual-only improves global cosine by **+0.031841** and the first/deep weight cosine by **+0.036089**, while the output-side cosine changes by only -1.38e-5. Gradient-only improves global/deep cosine by +0.062902/+0.089307. Applying stochastic rounding on both paths improves them by **+0.085036/+0.125847**.

This reproduces the qualitative seed-849 ordering:

`nearest < residual-only < gradient-only < both`

and again localizes the useful gain to the deep side rather than the already-saturated output side.

## Time evolution

Global BP cosine for seed 858:

| T | nearest | residual only | gradient only | both |
|---:|---:|---:|---:|---:|
| 32 | 0.682787 | 0.682900 | 0.683475 | 0.683557 |
| 64 | 0.766872 | 0.770060 | 0.771902 | 0.771369 |
| 96 | 0.872060 | 0.900242 | 0.912359 | 0.921177 |
| 128 | 0.885219 | 0.911385 | 0.927554 | 0.941939 |
| 192 | 0.889721 | 0.918181 | 0.944489 | 0.965136 |
| 256 | 0.889716 | 0.921557 | 0.952619 | 0.974752 |

Nearest reaches its maximum global cosine (0.889724) at T=189 and then plateaus. Residual-only continues to improve through the end (maximum 0.921628 at T=255). Its first/deep-layer advantage is not immediate: it is exactly zero through the early locked regime and first becomes nonzero around T=59; by T=64 the advantage is +0.030696 and reaches +0.036089 at T=256. The maximum deep-layer residual-only advantage is +0.059756 at T=81.

This timing is consistent with a dual-mediated accumulation mechanism rather than an instantaneous output-side correction: residual stochastic rounding perturbs the dual update first, and useful deep credit separates only after many relaxation steps.

## Interpretation and limits

The seed-858 replication strengthens three claims:

1. State-gradient quantization is the larger source of fixed-point credit loss, because gradient-only consistently rescues more alignment than residual-only.
2. The residual/dual path nevertheless carries useful information: residual-only improves deep BP alignment while leaving the already-near-perfect output-side alignment unchanged.
3. The two paths are complementary in these hard cases: `both` substantially exceeds either single-path ablation.

This is not yet evidence that PC-ALM is cheaper than sPC in hardware. It establishes a mechanism and a reproducible low-precision benefit inside PC-ALM. The next discriminating experiment should put sPC on the same depth-32/width-64 fixed-point ladder and ask what bit width and relaxation count it needs to match the PC-ALM `both` credit quality. That comparison can then price the extra lambda state against any saved datapath bits and/or iterations.

## Provenance

Source workflow: `PC-ALM rounding dynamics trace`, run 35442035570, commit `9b7b32440de0ec45cfbc066fd4acf493222a6859`. Seed-858 artifacts: nearest, residual, gradient, and both. All four trajectories were finite through T=256.