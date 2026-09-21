# Width-64 dyadic PC-ALM: T=64 / T=96 artifact recovery

Date: 2026-09-22

This note closes the unaggregated T=64 and T=96 portion of Actions run `35453887542`. No trajectories were rerun. The 30 still-valid per-seed artifacts for seeds 845--859 were downloaded and aggregated.

Configuration matches the previously recorded T=128 result: depth=32, width=64, batch=4, `state_lr=0.25`, so the effective state step is exactly 1.0; state `fixed15_i3`, update `fixed14_i1`, dual `fixed12_i1`, dual leak 0.02.

Useful first-layer credit requires finite values, cosine to BP >= 0.9, gradient-norm ratio in [0.5, 2.0], and relative error <= 0.6.

| budget | useful / 15 | cosine mean | norm-ratio mean | relative-error mean | finite |
|---:|---:|---:|---:|---:|---:|
| 64 | 0/15 | 0.888623 | 0.335981 | 0.717724 | 15/15 |
| 96 | 10/15 | 0.908275 | 0.776529 | 0.438628 | 15/15 |
| 128 | 15/15 | 0.947421 | 0.876736 | 0.328952 | 15/15 |

The T=128 row is the already-recorded result in `pcalm_dyadic_state_step_T128_2026-09-20.md`; it is included only to show the transition and was not recomputed here.

At T=96, the five failures are seeds 847, 849, 856, 857, and 858. Importantly, every one of those failures already satisfies the norm-ratio and relative-error criteria; they miss only the cosine threshold, with cosines 0.891653, 0.896787, 0.891355, 0.890494, and 0.894328 respectively. Thus T=96 is not a catastrophic under-relaxation point: it is a near-threshold operating point whose remaining error is angular alignment.

All T=64/T=96 trajectories are finite. The maximum observed state, update, and dual saturation rates are all zero. Mean late state zero-step fractions are 0.7466 at T=64 and 0.7703 at T=96, so the known fixed-point lattice effect remains present even without saturation.

## Interpretation

The robust width-64 design point remains T=128 if the requirement is 15/15 useful seeds. T=96 cuts relaxation work by 25% but drops strict useful-credit yield to 10/15. Since all five T=96 misses are narrowly below the cosine threshold rather than failing magnitude/range criteria, the highest-value way to reduce T below 128 is not to enlarge numeric range; it is to improve the late angular convergence (for example via rounding/update scheduling) without adding multipliers.

T=64 is too early for this configuration: the mean first-layer norm ratio is only 0.336 and no seed is useful. This gives a concrete lower bound on practical iteration count for the current width-64 low-precision point.

## Hardware implication

For the current multiplier-free state step, T=128 is the conservative architecture target. A T=96 mode is plausible as an approximate/throughput mode, saving 25% of relaxation cycles and state traffic, but it should not replace T=128 as the robust baseline until the five near-threshold angular misses are recovered.
