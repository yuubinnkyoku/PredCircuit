# Linear PC-ALM exact-dual credit scaling

Date: 2026-09-15

This is an analytic scalar-chain diagnostic, not a training-accuracy result. The target dual is the exact KKT multiplier, with `-lambda_l` equal to the BP adjoint at hidden layer `l`.

## Baseline

Parameters: `x=0.7`, `y=-0.4`, all weights `w=0.9`, `rho=1`, tolerance `||lambda-lambda*||/||lambda*|| <= 1e-3`, simultaneous/Jacobi-like hidden-state update followed by dual update.

With `state_lr=0.05`, `alpha=0.2`, max 5000 steps:

| depth | steps to 1e-3 | final relative dual error @5000 | final constraint norm @5000 |
|---:|---:|---:|---:|
| 2 | 91 | 6.38e-16 | 2.22e-16 |
| 4 | 368 | 2.20e-15 | 3.85e-16 |
| 8 | 894 | 2.15e-14 | 3.20e-15 |
| 16 | 2325 | 9.66e-6 | 2.27e-7 |
| 32 | >5000 | 8.93e-3 | 9.28e-5 |

So the default diagnostic parameters are too conservative for depth 32; this is a convergence-rate result, not divergence.

## Small coefficient sweep

The sweep below used max 10000 steps and the same `1e-3` relative-dual criterion. It is deliberately small and is only meant to establish whether the depth trend is an artifact of the default coefficients.

Representative rows (`depths = 2,4,8,16,32`):

| alpha | state_lr | steps to tolerance |
|---:|---:|---|
| 0.05 | 0.05 | 132, 382, 1021, 2715, 6692 |
| 0.05 | 0.10 | 215, 324, 514, 1567, 3658 |
| 0.05 | 0.20 | 236, 376, 511, 791, 1863 |
| 0.10 | 0.20 | 110, 164, 256, 783, 1828 |
| 0.20 | 0.20 | 42, 90, 253, 677, 1672 |
| 0.40 | 0.20 | 24, 87, 232, 619, 1540 |
| 0.80 | 0.20 | 21, 89, 221, 579, 1405 |
| 1.00 | 0.20 | **18, 80, 223, 561, 1370** |

Among the tested stable points, `alpha=1.0, state_lr=0.2` is best at depth 32. A log-log fit to its five points gives an empirical exponent about `T ~ L^1.53` over this tiny range. Do not interpret that exponent as asymptotic yet: coefficients were not independently optimized per depth, the network is scalar and linear, and spectral properties change with depth.

## Interpretation

1. PC-ALM does converge to the exact BP/KKT credit in this controlled case.
2. Credit formation is not depth-independent under the current simultaneous local dynamics. Even after coefficient tuning, going from depth 2 to 32 increases the required outer iterations from 18 to 1370.
3. Therefore the hardware case cannot rely on a claim that PC-ALM needs O(1) relaxation steps with depth. Any FPGA advantage must come from cheap/local simultaneous steps, low precision, reduced global memory traffic, and spatial parallelism enough to offset the increased T.
4. This result is intentionally separated from the earlier depth-32 ReLU gradient-geometry result (PC-ALM+dual-leak 19/20 strict successes, median about 96 steps). They use different targets and dynamics and must not be numerically conflated.

## Next experiment

Derive/measure the per-step operation and state-access cost for the same chain and compare total work `T * cost_per_step` against a sequential BP/ePC reverse sweep. Then repeat the dual-target experiment with vector residual blocks and the project's Sakana-style scaling/skips, because the scalar chain lacks the residual architecture that may materially change the spectral scaling.
