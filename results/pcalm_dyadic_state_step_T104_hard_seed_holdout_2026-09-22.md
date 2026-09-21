# PC-ALM width-64 multiplier-free boundary: recovered T=104 hard-seed holdout

Date: 2026-09-22

Configuration: depth=32, width=64, batch=4, `state_lr=0.25` (effective state step = 1.0), state `fixed15_i3`, update `fixed14_i1`, dual `fixed12_i1`, dual leak 0.02. Existing useful-credit criterion: finite, first-layer cosine >= 0.9, gradient-norm ratio in [0.5, 2.0], relative error <= 0.6.

The previous T=96 sweep over seeds 845--859 gave 10/15 useful. The only failures were seeds 847, 849, 856, 857, and 858, and all five missed only the cosine threshold. A targeted boundary workflow had already run those five seeds at T=104/112/120 (Actions run 35460162003), but its artifacts had not been folded into the current result chain. This note recovers the most informative first boundary point, T=104, without rerunning trajectories.

## T=104 hard-seed results

| seed | cosine to BP | grad-norm ratio | relative error | finite | useful |
|---:|---:|---:|---:|:---:|:---:|
| 847 | 0.901673 | 0.717610 | 0.469962 | yes | yes |
| 849 | 0.907769 | 0.776449 | 0.439546 | yes | yes |
| 856 | 0.904483 | 0.764294 | 0.448959 | yes | yes |
| 857 | 0.903907 | 0.740278 | 0.457960 | yes | yes |
| 858 | 0.903158 | 0.817969 | 0.437679 | yes | yes |

All five formerly failing seeds become useful at T=104. All state/update/dual saturation rates are zero in these five runs. The minimum cosine margin above the 0.9 gate is small: seed 847 is only +0.001673, so T=104 is a boundary point rather than a comfortable margin.

Mean over the five hard seeds at T=104: cosine 0.904198, gradient-norm ratio 0.763320, relative error 0.450821. At T=96 these same seeds had cosines 0.890494--0.896787 and failed only that gate.

## Interpretation

This is stronger than the earlier 96 < T <= 128 localization: for the exact five hard seeds that blocked T=96, the boundary is now 96 < T <= 104. Relative to the previously safe T=128 operating point, T=104 removes 24 of 128 relaxation iterations, i.e. 18.75% of relaxation MAC/state traffic, with no bit-width increase and no stochastic rounding.

This is a targeted hard-seed holdout, not a fresh 15-seed T=104 sweep. The ten seeds that already passed at T=96 were intentionally not rerun, so 15/15 at T=104 is not directly measured and should not be claimed as an experimental count. Given the small +0.001673 worst-case cosine margin, the next high-value test is not T=112/120 (already dominated by T=104 on these hard seeds) but a narrow T=100/102 boundary test on the same five seeds, followed by an independent holdout near the selected budget if the margin remains small.
