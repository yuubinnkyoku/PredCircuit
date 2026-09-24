# Multiplierless PC-ALM coefficient candidate (2026-09-24)

## Motivation

The current width-64 hardware candidate uses official Sakana depth-32 `state_lr=0.234285`, batch 4 (effective hidden-state step `eta_eff=0.93714`), `alpha=0.925`, and leaky-dual stabilization `gamma=0.02`. Existing fixed-point evidence already supports 12-bit dual state and compact 13--15-bit state/update paths, but the hardware cost model still leaves coefficient quantization open.

This note asks a narrower question before RTL: are the three important scalar coefficients already close enough to low-complexity binary fractions that their multipliers can plausibly be implemented with shifts/adds?

## Dyadic/CSD candidates

### Hidden-state step

The implementation's batch-scaled effective step is

`eta_eff = 4 * 0.234285 = 0.93714`.

A particularly simple dyadic approximation is

`eta_eff_q = 15/16 = 0.9375 = 1 - 2^-4`.

Absolute error: `3.60e-4`.
Relative error: `0.0384%`.

Equivalently, at the unscaled parameter level,

`state_lr_q = 15/64 = 0.234375`,

with the same relative error. This needs only a subtract-shift implementation for the effective update.

### Dual injection coefficient

For `alpha=0.925`, use

`alpha_q = 59/64 = 0.921875 = 1 - 2^-4 - 2^-6`.

Absolute error: `3.125e-3`.
Relative error: `0.3378%`.

Thus `alpha * r` can be implemented as `r - (r >> 4) - (r >> 6)` up to the chosen signed fixed-point rounding convention.

### Dual leak

For `gamma=0.02`, use

`gamma_q = 5/256 = 0.01953125 = 2^-6 + 2^-8`.

The actual recurrence coefficient is therefore

`1 - gamma_q = 251/256 = 0.98046875 = 1 - 2^-6 - 2^-8`.

Absolute gamma error: `4.6875e-4`.
Relative gamma error: `2.34375%`.

The dual recurrence becomes

`lambda <- lambda - (lambda >> 6) - (lambda >> 8) + r - (r >> 4) - (r >> 6)`

before final fixed-point rounding/saturation.

## Why this is unusually favorable

All three current research coefficients land near sparse binary expansions at once:

| coefficient | research value | candidate | relative error | nontrivial shifts |
| --- | ---: | ---: | ---: | ---: |
| `eta_eff` | 0.93714 | 15/16 | 0.0384% | 1 |
| `alpha` | 0.925 | 59/64 | 0.3378% | 2 |
| `gamma` | 0.02 | 5/256 | 2.3438% | 2 |
| `1-gamma` | 0.98 | 251/256 | 0.0478% | 2 |

The recurrence uses `1-gamma`, so the relevant multiplicative error in the retained dual term is only about `0.0478%`, not 2.34%.

This is an analytical hardware opportunity, not yet an algorithmic validation. The current low-precision results were obtained with the original floating coefficients before state/update/dual quantization. Iterative PC-ALM can be sensitive to lattice alignment, so a small coefficient perturbation cannot be assumed harmless merely from its relative error.

## Interaction with the known lattice-lock mechanism

The strongest candidate is `eta_eff=15/16`, because it is both hardware-cheap and extremely close to the measured value. However, the existing state15/update14 failure showed that *relative grid spacing*, not nominal coefficient accuracy alone, determines whether small updates survive state rounding. Replacing `0.93714` by exactly `15/16` changes the requested update quantum slightly and can move borderline coordinates across a half-LSB threshold.

Therefore the correct validation is not a floating-point coefficient sensitivity sweep alone. It must rerun the known held-out mixed-precision points with the dyadic coefficients applied before quantization and record useful-credit count, BP cosine/error, zero-update fraction, saturation, and dual dynamic range.

## Hardware implication if validated

If the dyadic candidate preserves the existing credit window, the scalar PC-ALM-specific coefficient multipliers do not require general DSP multipliers:

- hidden-state scaling: one shifted subtract;
- residual injection into the dual: two shifted subtracts from `r`;
- dual retention/leak: two shifted subtracts from `lambda`;
- `rho=1` is already free.

The extra PC-ALM arithmetic would then be dominated by add/subtract and rounding/saturation logic plus the dual state memory, reinforcing the existing cost model's conclusion that BRAM/state traffic, rather than scalar multiplication, is the important PC-ALM overhead.

## Next gate

Run the held-out width-64 seeds 845--859 at `T=256`, leak enabled, for at least the established compact/high-fidelity points:

- state14 / update14 / dual12,
- state15 / update13 / dual12,
- state16 / update14 / dual12,

comparing original coefficients against `(state_lr, alpha, gamma) = (15/64, 59/64, 5/256)`.

A useful gate is to require no loss in useful-seed count and no material degradation in mean BP cosine/error, while checking that the known lattice-lock diagnostics do not worsen. Only after that should these constants be frozen into a minimal RTL core.
