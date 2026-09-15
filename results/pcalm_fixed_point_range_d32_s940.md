# PC-ALM fixed-point requirements from an actual depth-32 trajectory

Configuration: ResidualMLP, depth 32, width 8, batch 4, ReLU, seed 940, budget 128, rho=1, alpha=1, Sakana reference activity step `eta_h=0.234285`.

The dedicated Actions run completed successfully and produced a finite FP32 trajectory.

## Dynamic range

- max |lambda| = 1.3512649536
- max |r| = max |Delta lambda| = 0.1598480642
- exact-zero fraction of FP32 dual updates = 13.0576%
- among nonzero elementwise updates:
  - p50 = 1.01327896e-3
  - p10 = 9.60826874e-5
  - p01 = 4.76837158e-7

The old step-by-layer maxima hid this long tail. The elementwise distribution shows that many genuine, nonzero credit updates are substantially smaller than 1e-3.

## Global fixed-point format fitted to observed lambda range

Using the minimum signed integer width that covers the observed FP32 lambda range:

| total bits | format implied by range | LSB | fraction of nonzero FP32 updates below LSB/2 |
|---:|---:|---:|---:|
| 16 | Q2.14 | 6.1035e-5 | 5.21% |
| 12 | Q2.10 | 9.7656e-4 | 30.55% |
| 8 | Q2.6 | 1.5625e-2 | 95.01% |

These are open-loop trajectory diagnostics, not closed-loop quantized PC-ALM results. A lost update here means that a round-to-nearest accumulator at the stated scale would map that isolated FP32 increment to zero; repeated sub-LSB increments can behave differently if an error-feedback accumulator is used.

## Per-layer scaling

Per-layer scaling helps where max |lambda_l| < 1, allowing one extra fractional bit. It does not rescue 8-bit arithmetic: per-layer 8-bit lost-update fractions range roughly from 75% near the input to above 94% in several deeper layers.

For 12 bits, layers with one integer bit have lost-update fractions around 15-20% in many cases, while Q2.10 layers are mostly around 25-34%. The final hidden layer is a difficult case even with Q1.11, at about 38.35%.

Thus per-layer scaling improves the open-loop resolution picture but does not make naive 12-bit rounding obviously safe.

## Interpretation

The measured lambda range is compact enough that overflow is not the main obstacle for this configuration. Resolution of the dual integrator is the limiting issue. In particular, the nonzero median update (~1.0e-3) sits close to the Q2.10 LSB, and the lower tail extends to FP32-scale increments.

This changes the low-precision hypothesis: 12-bit PC-ALM remains plausible, but a plain round-to-nearest dual register should not be assumed adequate. The next decisive experiment is closed-loop quantization, comparing at least:

1. ordinary fixed-point rounding/saturation,
2. per-layer scaling,
3. an error-feedback (fractional residual) accumulator that preserves sub-LSB dual increments.

The relevant outcomes are BP-gradient geometry, residual convergence, oscillation/finite behavior, saturation count, and required relaxation budget T. Eight-bit plain fixed point is a weak candidate given the ~95% open-loop lost-update rate.
