# Reduced PC-ALM credit-transfer rank probe (2026-09-24)

## Question

Can the local PC-ALM state `(h, lambda)` be eliminated so that the resulting width-by-width credit transfer

`T = D + C (I - A)^-1 B`

is sufficiently low-rank for a logarithmic-depth parallel prefix credit network?

## Setup

- Current `ResidualMLP` parameterization, ReLU
- width = 8
- state_lr = 0.05, rho = 1.0, alpha = 0.2
- float64 automatic differentiation
- hidden residual blocks only; boundary layers excluded
- exact local Schur elimination with `torch.linalg.solve`
- ranks 1/2/4 are obtained by SVD truncation of each local `T`; full rank 8 is the control
- final metric is after composing every retained hidden-layer transfer, not a per-layer SVD metric

The probe source is `scripts/run_reduced_credit_transfer_probe.py`.

## Depth 8, seeds 980--983

`max cond(I-A) = 3.8609997224` for every run, so the local elimination is not ill-conditioned in this setting.

| seed | rank | final cosine vs full T | relative error | norm ratio |
|---:|---:|---:|---:|---:|
| 980 | 1 | 0.2854 | 0.9967 | 0.01178 |
| 980 | 2 | -0.2766 | 1.0314 | 0.09802 |
| 980 | 4 | -0.0186 | 1.0091 | 0.11763 |
| 981 | 1 | -0.4378 | 1.0000 | 8.28e-6 |
| 981 | 2 | 0.4407 | 0.9827 | 0.04081 |
| 981 | 4 | 0.00135 | 1.0043 | 0.09417 |
| 982 | 1 | -0.0359 | 1.0000 | 9.86e-4 |
| 982 | 2 | 0.6400 | 0.9908 | 0.01445 |
| 982 | 4 | -0.0504 | 1.0452 | 0.25767 |
| 983 | 1 | 0.1935 | 1.0000 | 1.65e-4 |
| 983 | 2 | 0.1308 | 0.9978 | 0.01761 |
| 983 | 4 | 0.6452 | 0.7964 | 0.42046 |

Full rank 8 gives cosine 1.0 and relative error below 5e-15 for all four seeds.

The failure is not primarily a direction-only effect: low-rank composition also strongly suppresses the credit norm. Even rank 4 has norm ratio only 0.094--0.420 across these seeds.

## Depth 32 spot check

For seeds 980 and 981, low-rank composition becomes much more severe:

| seed | rank | final cosine vs full T | relative error | norm ratio |
|---:|---:|---:|---:|---:|
| 980 | 1 | -0.5822 | 1.0000 | 7.32e-17 |
| 980 | 2 | 0.0208 | 1.0000 | 5.74e-11 |
| 980 | 4 | -0.1229 | 1.0000008 | 6.11e-6 |
| 981 | 1 | -0.4242 | 1.0000 | 5.82e-16 |
| 981 | 2 | 0.5574 | 0.9999999 | 1.46e-7 |
| 981 | 4 | -0.0398 | 1.0000051 | 1.27e-4 |

Full rank 8 again reproduces the exact composed transfer to numerical precision.

## Interpretation

This is a negative result for the *naive low-rank reduced-transfer prefix* hypothesis. The earlier low-rank success for the forward/BP residual Jacobian does not survive the PC-ALM local-state elimination used here. The Schur solve itself is well-conditioned (`cond ~= 3.86`), so the failure cannot be attributed to an unstable elimination. Instead, truncating each reduced transfer removes directions that become essential after repeated composition; the resulting credit magnitude collapses exponentially with depth.

This result does **not** show that every possible structured credit operator is impossible. In particular, the current truncation approximates `T` directly. A structure-preserving representation (for example an analytically derived factorization tied to the residual/dual equations) would need separate evidence. However, a generic rank-2/rank-4 SVD prefix network should no longer be treated as a promising route to beat BP/ePC latency.

## Consequence for PredCircuit

The strongest remaining claim for PC-ALM is therefore not low-latency credit transport. Existing experiments still support reduced attenuation relative to sPC, but synchronous nearest-neighbor PC-ALM needs depth-proportional relaxation, Gauss--Seidel trades that support delay for sequential dependence/amplitude decay, and the generic low-rank reduced-transfer escape hatch now fails as well.

The next high-value test should shift from trying to rescue generic prefix transport to quantifying whether PC-ALM's *attenuation advantage over sPC* can justify its dual-state hardware cost under realistic low precision. A compact depth/width sweep at 12/16-bit fixed point, with matched MAC/state-memory accounting and ePC/BP baselines, can answer that without prematurely starting RTL.
