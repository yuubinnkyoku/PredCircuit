# PC-ALM multi-update training budget sweep (2026-09-23)

## Setup

- ResidualMLP: depth 32, width 8, ReLU
- batch size 4; 24 weight updates
- seeds 970, 971, 972
- PC-ALM: alpha=0.925, rho=1, state_lr=0.25
- budgets T = 8, 16, 32, 56, 112
- corrected ePC baseline: T=1, error_lr=0.3; hidden-layer weight LR compensated by 1/error_lr while output-layer LR is unchanged
- BP baseline uses the same initialization and minibatch sequence.

## Final relative evaluation-loss reduction

| method | seed 970 | seed 971 | seed 972 | mean |
|---|---:|---:|---:|---:|
| BP | 5.707% | 1.841% | 1.023% | 2.857% |
| ePC T=1 | 1.453% | 0.982% | 0.977% | 1.138% |
| PC-ALM T=8 | 2.177% | 0.516% | 0.396% | 1.030% |
| PC-ALM T=16 | 2.476% | 0.586% | 0.421% | 1.161% |
| PC-ALM T=32 | 2.935% | 0.675% | 0.455% | 1.355% |
| PC-ALM T=56 | 3.391% | 0.859% | 0.554% | 1.601% |
| PC-ALM T=112 | 5.367% | 1.732% | 1.121% | 2.740% |

## Main observation

The hoped-for saturation near T ~= depth does **not** occur in this deep/narrow setting. Mean PC-ALM improvement at T=32 is only 49.5% of the T=112 improvement; T=56 reaches 58.4%. Neither reaches 90% of the T=112 improvement (2.466 percentage points) or 90% of the BP improvement (2.572 points). Among the tested budgets, only T=112 crosses both 90% criteria.

The shape is also consistent across all three seeds: relative to each seed's T=112 PC-ALM improvement, T=32 achieves 54.7%, 39.0%, and 40.6%; T=56 achieves 63.2%, 49.6%, and 49.4%. Thus the absence of a T=32 plateau is not driven by one seed.

A second crossover appears at the low-budget end. Mean PC-ALM T=8 (1.030%) is slightly below corrected ePC T=1 (1.138%), T=16 is approximately tied/slightly above (1.161%), and T>=32 is progressively better in loss reduction. This is a quality-vs-iteration tradeoff, not a compute-efficiency win: ePC still uses only one error step per weight update.

## Hardware interpretation

For depth 32, the attractive hypothesis that `T <= L` is enough to retain near-BP multi-update learning is rejected by this small experiment. The T=112 result remains strong against sPC, but its quality advantage over ePC requires substantially more temporal relaxation than a depth-sized budget. Therefore this result does **not** justify moving to RTL yet.

The next high-value question is mechanistic: why does the useful PC-ALM credit emerge mostly between T=56 and T=112? Measure layerwise residual/prediction error and dual lambda trajectories at T checkpoints (e.g. 8,16,32,56,80,112), especially the first-layer BP cosine/norm and the depth/time propagation front. If a late dual propagation event explains the training jump, tune alpha/rho/state_lr or stopping criteria to move that event earlier; if not, the PC-ALM hardware case versus ePC weakens substantially.
