# PC-ALM width-64 T=103 independent holdout (2026-09-22)

Configuration follows the width-64 dyadic low-precision candidate used for the T-boundary experiments. This holdout uses previously unused seeds 860--874 and T=103. The 15 Actions artifacts from run 35666235122 were recovered rather than rerun.

## Result

- useful first-layer credit: **13/15**
- all 15 trajectories finite
- failures: seeds **860** and **862**
- mean first-layer cosine to BP: 0.889247 (strongly depressed by seed 860)
- mean gradient norm ratio to BP: 0.851156
- mean relative error to BP: 0.475653

### Failure details

| seed | cosine to BP | grad norm ratio | relative error | residual total | late realized/requested | late state zero-step fraction | state saturation |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 860 | 0.518684 | 1.466824 | 1.276690 | 0.071837 | 0.421078 | 0.756304 | 0.0001517 |
| 862 | 0.891911 | 0.792920 | 0.462920 | 0.069943 | 1.035191 | 0.736868 | 0 |

Seed 862 resembles the earlier near-threshold failures: finite, no saturation, acceptable norm ratio and relative error, but cosine below 0.9. Seed 860 is qualitatively different. Its cosine collapses to 0.519, gradient norm overshoots BP by 1.47x, relative error exceeds 1.27, late realized/requested state-step ratio falls to 0.421, and a small nonzero state saturation rate appears. Therefore T=103 is **not** a robust hardware budget on an independent seed set, and seed 860 should not be interpreted as merely a one-step credit-propagation miss.

## Follow-up

A focused diagnostic was launched for seeds 860 and 862 at T=104, 108, and 128. This separates three hypotheses without rerunning successful seeds: (1) T=103 is simply too short, (2) a modest safety margin fixes the failures, or (3) seed 860 enters a distinct low-precision dynamical failure that persists even at the old T=128 baseline.
