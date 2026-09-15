# ePC deep-credit discovery (seeds 940–944)

Date: 2026-09-15

Shared setting: `ResidualMLP(depth=32, width=8, activation=relu)`, batch size 4.  ePC uses the error-coordinate energy and local post-inference weight credit.  The discovery grid was `error_lr ∈ {0.03, 0.1, 0.3}` and `T ∈ {1,2,4,8,16,32,64,96,128}`.

The legacy BP-like first-layer criterion was cosine >= 0.9, gradient norm ratio in [0.5, 2], and BP-relative error <= 0.6.

## Result

All 135 configurations (5 seeds × 3 learning rates × 9 budgets) remained finite, but none passed the legacy BP-like criterion.  The failure is dominated by gradient magnitude, not absence of immediate deep credit.

At T=1, for every seed and every tested learning rate, first-layer ePC credit is numerically collinear with BP and its norm ratio is the learning rate (within float32 error):

- error_lr=0.03: cosine ≈ 1, norm/BP ≈ 0.03, relative error ≈ 0.97
- error_lr=0.1: cosine ≈ 1, norm/BP ≈ 0.1, relative error ≈ 0.9
- error_lr=0.3: cosine ≈ 1, norm/BP ≈ 0.3, relative error ≈ 0.7

This is expected from the ePC reparameterization: starting from zero errors, reverse-mode differentiation sends the output signal to every error coordinate before the scalar error learning rate is applied.  Thus ePC does not exhibit the sPC-style delayed first-layer signal in this experiment.

After further relaxation, the ePC local weight gradient generally moves away from the BP gradient because ePC is converging toward the exact PC equilibrium gradient, which need not equal the one-shot BP gradient.  Therefore the existing BP-like threshold is useful as a gradient-geometry diagnostic, but it is not a valid standalone convergence criterion for ePC.

## Consequence for the comparison protocol

Do not tune ePC merely to cross the legacy BP norm threshold (for example by choosing error_lr >= 0.5 at T=1).  That would reward a trivial scaling of the first ePC step rather than equilibrium quality.  The fair ePC comparison should add an equilibrium-oriented criterion: energy decrease / stationarity and, where feasible, agreement with a long-run sPC or analytical PC equilibrium.  BP cosine/norm/relative error should remain reported as geometry metrics alongside that criterion.

This discovery therefore changes the next experiment: validate ePC equilibrium behavior first, then compare compute (reverse-mode MACs and memory traffic per ePC step) against sPC and PC-ALM at matched solution quality.
