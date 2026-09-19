# PC-ALM stochastic update rounding: rounding-seed sensitivity

Date: 2026-09-19

## Question

Does the fixed-point PC-ALM rescue from stochastic update rounding depend on a lucky rounding RNG trajectory?

## Reused experiment

No trajectories were rerun. This note aggregates the already completed 56-trajectory holdout from workflow run `35404473994`: 7 model/data seeds that failed under nearest update rounding (845, 848, 849, 851, 852, 856, 858), each evaluated with 8 independent rounding RNG seeds.

The numerical setting is unchanged from the earlier lattice-lock experiment: depth 32, width 64, T=256, state/update/dual formats `fixed15_i3 / fixed14_i1 / fixed12_i1`, state_lr=0.234285, lattice ratio R=0.93714, and stochastic rounding only on the update quantizer.

## Result

All 56/56 trajectories satisfy the existing `useful_first_layer_credit` criterion.

Overall across 56 trajectories:

- BP-gradient cosine: mean 0.960951, std 0.005739, min 0.947131, max 0.969897.
- Gradient norm ratio to BP: mean 0.871294.
- Relative error to BP: mean 0.292103.
- Late realized/requested state-step ratio: mean 0.531910.
- Late update zero fraction: mean 0.497372.
- Late state zero-step fraction: mean 0.965493.
- State/update/dual saturation remained zero in the underlying trajectories.

Per model/data seed (8 RNG trajectories each):

| seed | useful | cosine mean | cosine std | cosine min | realized/requested mean |
|---:|---:|---:|---:|---:|---:|
| 845 | 8/8 | 0.967476 | 0.001786 | 0.965054 | 0.470947 |
| 848 | 8/8 | 0.966940 | 0.001811 | 0.964107 | 0.505320 |
| 849 | 8/8 | 0.956435 | 0.000824 | 0.955330 | 0.451093 |
| 851 | 8/8 | 0.957744 | 0.001610 | 0.956066 | 0.510604 |
| 852 | 8/8 | 0.957220 | 0.002239 | 0.954046 | 0.651826 |
| 856 | 8/8 | 0.966421 | 0.001779 | 0.964195 | 0.612112 |
| 858 | 8/8 | 0.954422 | 0.004162 | 0.947131 | 0.521465 |

The worst observed trajectory still has cosine 0.9471, comfortably above the 0.9 useful-credit threshold. The largest within-seed cosine standard deviation is only 0.00416 (seed 858).

For the observed 56/56 successes, the exact two-sided 95% Clopper-Pearson lower confidence bound for an exchangeable Bernoulli success probability is about 0.936. This is only a compact robustness summary; trajectories are nested within seven model/data seeds, so it must not be interpreted as evidence over arbitrary unseen networks.

## Interpretation

The earlier 15/15 result is not explained by one lucky stochastic-rounding sequence. On every previously failing model/data seed tested here, eight independent rounding sequences preserve useful first-layer credit.

The dynamics remain highly sparse: about half of quantized updates are still zero and about 96.5% of state steps are zero, yet gradient geometry remains close to BP. This supports the mechanism that stochastic update rounding transmits otherwise sub-LSB, credit-critical updates over the relaxation time axis rather than requiring dense state motion.

This closes the immediate RNG-seed sensitivity question for this holdout. The next hardware-relevant uncertainty is correlation, not marginal RNG quality: how much stochastic-rounding randomness can be shared across elements/layers/steps before the rescue disappears?

## Decision

Do not widen the datapath or start full RTL yet. Preserve sPC as the alpha=0 comparison and PC-ALM as the primary method. The highest-information next experiment is a randomness-sharing sweep at the same hard fixed-point setting, comparing element-wise independent randomness against progressively shared randomness (for example per-layer/per-step and global-per-step). If strongly shared randomness retains credit, the FPGA implementation can use far fewer PRNG resources; if it fails, RNG distribution becomes an explicit area/routing cost in the hardware model.
