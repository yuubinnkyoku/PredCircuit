# Fixed-point dual-integrator diagnostic

This is an isolated numerical diagnostic for the PC-ALM recurrence
`lambda <- lambda + alpha * r`, not a network-level accuracy result.

Configuration: alpha=1, r_0=0.25, r_t=0.97^t r_0, 256 updates. The FP64
reference converges to lambda=8.3299105239.

| format | range (approx.) | LSB | rel. error | saturations | lost updates | first lost step |
|---|---:|---:|---:|---:|---:|---:|
| Q4.12 / 16b | [-8, 8) | 2^-12 | 3.963% | 146 | 150 | 107 |
| Q6.10 / 16b | [-32, 32) | 2^-10 | 0.0801% | 0 | 51 | 206 |
| Q4.8 / 12b | [-8, 8) | 2^-8 | 4.007% | 56 | 151 | 106 |
| Q6.6 / 12b | [-32, 32) | 2^-6 | 2.272% | 0 | 142 | 115 |
| Q4.4 / 8b | [-8, 8) | 2^-4 | 7.712% | 0 | 187 | 70 |
| Q6.2 / 8b | [-32, 32) | 2^-2 | 30.972% | 0 | 233 | 24 |

The key result is that total bit width alone is not a useful specification.
Q4.12 has excellent fractional resolution but fails because the dual needs a
little more than 8 in this trajectory. Giving two bits back to range (Q6.10)
removes saturation and is much more accurate despite the coarser LSB.

Conversely, 8-bit formats face a hard range/precision tradeoff: Q4.4 retains
precision but has little range headroom, while Q6.2 has range but loses most
late residual updates. This does not yet rule out 8-bit network operation;
per-layer scaling or block floating point could change the result.

## RTL implication

Do not choose a fixed format from activation statistics alone. Measure the
per-layer peak |lambda| and the late-stage distribution of |alpha*r| first.
A candidate format needs both:

1. representable max comfortably above max |lambda|, including transient
   overshoot; and
2. LSB below the residual corrections that still matter for stopping.

The next network-level quantization experiment should therefore log, per layer,
peak |lambda|, peak |r|, the percentile distribution of |alpha*r|, saturation
counts, and zeroed dual-update counts before deciding between 16/12/8 bits.
