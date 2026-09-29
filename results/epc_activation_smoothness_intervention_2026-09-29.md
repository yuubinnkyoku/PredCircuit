# ePC activation smoothness intervention — 2026-09-29

Configuration: depth=32, width=8, batch=4, error_lr=0.1, T=128, seeds 960–979, same ResidualMLP scaling/data seeding as `diagnose_epc_relu_mask_switching.py`.

We measured normalized error-coordinate stationarity
`S_T = ||grad_e E_T||_2 / ||grad_e E_0||_2`.

| activation | parameter | median S128 | Q1 | Q3 | max |
|---|---:|---:|---:|---:|---:|
| ReLU | — | 1.3633547e-1 | 9.0117412e-2 | 2.5900340e-1 | 3.2711192e-1 |
| Leaky ReLU | negative slope 0.1 | 7.4170334e-2 | 4.8814401e-2 | 1.0747545e-1 | 2.5688499e-1 |
| Softplus | beta 1 | 6.2370959e-6 | 1.7957988e-6 | 2.2651892e-5 | 1.3571257e-4 |
| Softplus | beta 5 | 1.3920698e-5 | 1.0986596e-6 | 8.0829867e-5 | 6.6854692e-2 |

Interpretation: giving the negative half-line a nonzero slope (Leaky ReLU 0.1) improves the median only about 1.84x and does not remove the stationarity floor. Making the kink smooth (Softplus) changes the result by roughly four orders of magnitude and usually converges to near-stationarity. This supports the narrower hypothesis that the nonsmooth kink / active-set switching, rather than merely dead ReLU units or zero negative slope, is the dominant cause of the ReLU ePC floor under this setup.

Caveat: Softplus beta=5 has one strong outlier (max S128 ~= 0.0669), so sharp smooth approximations can still be seed-sensitive. This is a mechanism result, not yet a learning-performance result. PC-ALM and sPC should be run under the same activation sweep before claiming robustness differences.
