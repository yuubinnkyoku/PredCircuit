# PC-ALM independent T103 failure budget diagnostic (2026-09-22)

Configuration: depth=32, width=64, batch=4, `state_lr=0.25` (effective state step 1), `update=fixed14_i1`, `state=fixed15_i3`, `dual=fixed12_i1`, dual leak 0.02. These runs reuse the independent-holdout failures seed 860 and 862 and vary only the relaxation budget.

| seed | T | finite | residual | realized/requested | state zero-step | state saturation | cosine to BP | grad norm / BP | relative error | useful |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 860 | 104 | yes | 0.071567 | 0.42034 | 0.75717 | 1.5145e-4 | 0.520845 | 1.46915 | 1.27594 | no |
| 860 | 108 | yes | 0.072210 | 0.41418 | 0.76155 | 1.5051e-4 | 0.528137 | 1.48999 | 1.28306 | no |
| 860 | 128 | yes | 0.071668 | 0.34118 | 0.78475 | 1.4668e-4 | 0.564590 | 1.50747 | 1.25310 | no |
| 862 | 104 | yes | 0.069841 | 1.03497 | 0.73754 | 0 | 0.893234 | 0.79631 | 0.45992 | no |
| 862 | 108 | yes | 0.069757 | 1.03408 | 0.74116 | 0 | 0.900619 | 0.81022 | 0.44391 | yes |
| 862 | 128 | yes | 0.067421 | 1.02145 | 0.76748 | 0 | 0.929268 | 0.86069 | 0.37572 | yes |

## Interpretation

Seed 862 is the expected finite-budget failure: adding relaxation steps monotonically improves the first-layer gradient direction enough to cross the useful-credit threshold between T=104 and T=108.

Seed 860 is qualitatively different. It remains far from BP even at T=128. Increasing T from 104 to 128 improves cosine only 0.5208 -> 0.5646 while the realized/requested late state-step ratio *falls* 0.420 -> 0.341 and the state zero-step fraction rises 0.757 -> 0.785. A small state saturation rate (~1.5e-4) persists, while update and dual saturation remain zero. Therefore seed 860 is not explained by the T=103 budget alone. The current fixed-point state dynamics enter a quantization/saturation-limited trajectory for this seed.

This invalidates using T=103 or T=108 as a globally robust hardware budget until the seed-860 failure mechanism is isolated. The next discriminating experiment is a state-precision ablation at seed 860, T=128 while holding update precision, dual precision, state step, and dual leak fixed. In particular compare fixed15_i3 against fixed16_i3 and FP32 state; recovery with extra state fractional precision would identify the state lattice as the dominant cause, while failure even in FP32 state would redirect attention to update/dual quantization or the underlying PC-ALM dynamics.
