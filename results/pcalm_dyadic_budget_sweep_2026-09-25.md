# PC-ALM dyadic fixed-point budget sweep (2026-09-25)

Configuration: depth 32, width 64, state/update/dual = 16/14/12 bit, dyadic coefficients, seeds 845–859 (15 held-out seeds). Source: GitHub Actions run 36010493888 aggregate artifact.

| T | finite | useful | BP cosine mean | BP cosine min | norm ratio mean | BP rel. error mean | late state-zero mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 15/15 | 0/15 | 0.89125 | 0.86329 | 0.31161 | 0.73565 | 0.62283 |
| 96 | 15/15 | 14/15 | 0.92046 | 0.89607 | 0.76667 | 0.42081 | 0.68176 |
| 128 | 15/15 | 15/15 | 0.95018 | 0.93952 | 0.86565 | 0.32394 | 0.76691 |
| 160 | 15/15 | 15/15 | 0.96968 | 0.95666 | 0.94001 | 0.24782 | 0.84083 |
| 192 | 15/15 | 15/15 | 0.97779 | 0.96759 | 0.97265 | 0.21135 | 0.89952 |
| 224 | 15/15 | 15/15 | 0.98155 | 0.97027 | 0.98022 | 0.19125 | 0.94311 |
| 256 | 15/15 | 15/15 | 0.98320 | 0.97153 | 0.98215 | 0.18141 | 0.96930 |

All tested budgets were finite. The useful threshold first becomes universal at T=128. However, T=128 is not quality-equivalent to T=256: mean cosine is lower by 0.0330 and mean BP relative error is 0.1425 higher. T=160 removes 37.5% of iterations while retaining mean cosine 0.96968 (minimum 0.95666), but its mean norm ratio remains 0.940. T=192 removes 25% of iterations and is the first tested point with both mean cosine >0.975 and mean norm ratio within 3% of BP (0.97265); minimum cosine is 0.96759. T=224 gives only a further 12.5% iteration reduction versus T=256 but reaches mean cosine 0.98155.

The late state-zero fraction rises monotonically from 0.623 at T=64 to 0.969 at T=256, yet gradient geometry continues to improve. Therefore state-zero fraction alone is not a valid early-stop signal; sparse residual motion remains useful after most state coordinates quantize to no movement.

Working interpretation: use T=192 as the current balanced fixed-budget hardware candidate, T=160 as an aggressive throughput candidate, and T=256 as the quality reference. Do not claim T=128 is equivalent to T=256.
