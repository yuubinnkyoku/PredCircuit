# ePC activation smoothness check (2026-09-29)

The previously reported depth-32 ePC stationarity floor was reproduced with the repository default activation, ReLU, not tanh.

For seeds 960-979, depth=32, width=8, batch=4, error_lr=0.1:
- ReLU median ||grad E_T||/||grad E_0|| at T=8,16,32,64,128: 0.152109, 0.107695, 0.107618, 0.125610, 0.136335.
- tanh under the same setup: 0.046851, 0.013971, 0.004102, 0.000351, 0.00000266.

Thus the ~0.1 floor is activation-specific and exactly matches the ReLU configuration used by the current diagnostic default. It should not be attributed to generic nonlinear curvature or to ePC coordinate transformation.

Interpretation: ReLU's nonsmooth active-set changes are the leading hypothesis. Before extending T, compare ReLU against tanh/linear and inspect activation-mask flips / zero preactivations during ePC relaxation.
