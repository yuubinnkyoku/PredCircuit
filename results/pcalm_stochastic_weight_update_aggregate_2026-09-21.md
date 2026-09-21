# PC-ALM 9-bit stochastic weight update aggregate (2026-09-21)

This note records the completed 20-seed aggregate for stateless stochastic updates of `fixed9_i3` stored weights. The PC-ALM setting is depth=32, width=8, state_lr=0.25, rho=1.0, alpha=0.925, dual_leak=0.01, budget=112, with `fixed12_i2` update state. Seeds are 800--819.

For a 9-bit stored-weight LSB of 1/32, an ideal stochastic update emits one LSB with probability `p=min(|lr*g|/LSB,1)`. A k-bit probability comparator is modeled conservatively as `p_eff=floor(p*2^k)/2^k`. Metrics below are analytic expectations over the measured PC-ALM gradient distributions; they are not long-horizon training accuracy results.

| weight lr | RNG bits | expected move rate | relative L1 bias | nonzero probability lost |
|---:|---:|---:|---:|---:|
| 0.001 | 2 | 0.000000 | 1.000000 | 1.000000 |
| 0.001 | 4 | 0.000000 | 1.000000 | 1.000000 |
| 0.001 | 6 | 0.00000194 | 0.995391 | 0.999867 |
| 0.001 | 8 | 0.00001424 | 0.951771 | 0.997177 |
| 0.001 | 12 | 0.00012034 | 0.430565 | 0.811550 |
| 0.01 | 2 | 0.000000 | 1.000000 | 1.000000 |
| 0.01 | 4 | 0.00008991 | 0.974467 | 0.998801 |
| 0.01 | 6 | 0.00040961 | 0.829529 | 0.981356 |
| 0.01 | 8 | 0.00101396 | 0.535441 | 0.879348 |
| 0.01 | 12 | 0.00170611 | 0.076677 | 0.352893 |
| 0.1 | 2 | 0.00237475 | 0.902246 | 0.992598 |
| 0.1 | 4 | 0.00797061 | 0.641317 | 0.926292 |
| 0.1 | 6 | 0.01342153 | 0.330917 | 0.731081 |
| 0.1 | 8 | 0.01638968 | 0.115161 | 0.440508 |
| **0.1** | **12** | **0.01774535** | **0.008797** | **0.079773** |

All 20 seeds were finite for every tested point.

## Interpretation

The 20-seed aggregate confirms that 8-bit RNG is not enough at the nominal weight learning rate 0.01: it loses 53.5% of the ideal expected update magnitude and maps 87.9% of nonzero ideal probabilities to zero. Even 12-bit RNG at lr=0.01 retains 7.67% downward bias and loses 35.3% of nonzero probabilities.

The only tested point reaching <1% expected-update L1 bias is lr=0.1 with a 12-bit comparator (0.8797% mean bias). Its lost-probability fraction is still 7.98%, but those lost events carry little total update mass; this is why lost-count fraction and update-magnitude bias must not be conflated.

This changes the hardware question. Stateless stochastic rounding can avoid dense per-weight residual storage, but it is not sufficient to say that stochastic rounding removes the dead zone: finite probability resolution reintroduces one. Under the measured gradient scale, a 12-bit probability path plus a roughly 10x larger weight-update scale is the first tested configuration that preserves expected update magnitude closely.

This does **not** establish stable learning at weight lr=0.1. The next decisive experiment should therefore be a short matched training trajectory comparing FP32 stored weights, deterministic 9-bit weights, and 9-bit stochastic stored weights at the candidate `(weight_lr=0.1, RNG=12)` point, with identical data/order/initialization. It should record loss, accuracy or task error, BP-gradient cosine, update-event rate, saturation, and divergence. If lr=0.1 is unstable, a denser sweep between 0.01 and 0.1 (with 10/12-bit RNG) is justified; if it is stable, the stateless 9-bit/32-RAMB36 design remains viable without a dense residual memory.
