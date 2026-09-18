# Fixed-point lattice LR threshold holdout (2026-09-18)

## Question

For the depth-32 / width-64 PC-ALM fixed-point configuration with `state=fixed15_i3`, `update=fixed14_i1`, and `dual=fixed12_i1`, does the state-update lattice transition occur at the analytically predicted boundary

\[
\eta_{\mathrm{eff}}\Delta_u \ge \Delta_h/2,
\]

rather than being a seed-specific accident?

With batch size 4, `effective_lr = 4 * state_lr`. In this format pair, one update quantum times the effective LR reaches half a state LSB at `effective_lr=1`, i.e. `state_lr=0.25`. The diagnostic therefore predicts two update quanta are needed below 0.25 and one at/above 0.25.

## Source

GitHub Actions run `35289901965` at main commit `422e842ecd785a96ca6ab336d90d5d4b53b0e0a7` produced 15 non-expired artifacts, seeds 845--859. All 90 rows (15 seeds x 6 learning rates) were finite. This report aggregates those artifacts; no experiment was rerun.

## Aggregate results

| state_lr | effective_lr | min update quanta | useful credit | BP cosine mean ± sd | BP relative error mean ± sd | late state zero-step mean ± sd | realized/requested mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.220000 | 0.88000 | 2 | 7/15 | 0.882836 ± 0.045672 | 0.509458 ± 0.112461 | 0.999530 ± 0.000457 | 0.246453 |
| 0.234285 | 0.93714 | 2 | 8/15 | 0.882994 ± 0.046982 | 0.508913 ± 0.113847 | 0.999365 ± 0.000615 | 0.258406 |
| 0.245000 | 0.98000 | 2 | 7/15 | 0.884068 ± 0.043464 | 0.506440 ± 0.108380 | 0.999127 ± 0.000948 | 0.255146 |
| **0.250000** | **1.00000** | **1** | **15/15** | **0.988561 ± 0.004285** | **0.151970 ± 0.027181** | **0.907277 ± 0.012858** | **0.854333** |
| 0.255000 | 1.02000 | 1 | 15/15 | 0.993370 ± 0.003123 | 0.118217 ± 0.023481 | 0.312800 ± 0.013998 | 1.245998 |
| 0.270000 | 1.08000 | 1 | 15/15 | 0.993486 ± 0.003001 | 0.116812 ± 0.022458 | 0.311721 ± 0.016551 | 1.178087 |

Crossing only 0.245 -> 0.250 gives:

- BP cosine: mean change **+0.104492**; improvement in **15/15** seeds; per-seed change range `+0.057601` to `+0.184168`.
- BP relative error: mean change **-0.354470**; improvement in **15/15** seeds; per-seed change range `-0.514579` to `-0.230840`.
- Late state zero-step fraction: mean change **-0.091850**; decrease in **15/15** seeds; per-seed change range `-0.109422` to `-0.064512`.
- Useful first-layer credit changes from **7/15** to **15/15**.

No state, update, or dual saturation was observed in the inspected artifact schema, so the discontinuity is not explained by clipping/overflow.

## Interpretation

The seed-853 observation generalizes across the full 15-seed holdout. The abrupt recovery aligns exactly with the diagnostic's discrete change from two required update quanta to one at `state_lr=0.25`. This strongly supports a quantization-lattice mechanism rather than a smooth learning-rate effect.

The candidate dimensionless hardware design variable is

\[
R = \frac{\eta_{\mathrm{eff}}\Delta_u}{\Delta_h/2}.
\]

For this format pair, `R=1` is a reproducible transition boundary for gradient geometry and state motion. This is evidence for a *candidate* fixed-point design rule, not yet a general rule: the current sweep changes learning rate while keeping the LSB ratio fixed.

An important nuance is that `state_lr=0.25` restores BP geometry before it eliminates most zero state steps (mean zero-step fraction remains 0.907). At 0.255 it falls to 0.313. Therefore the strict `R>=1` boundary appears sufficient to recover useful credit propagation, while additional margin above one is needed for dense realized state motion.

## Next falsification test

Change the state/update LSB ratio while keeping the network, dual format, seeds, and other PC-ALM coefficients fixed. Predict the new transition *before* running it using

\[
\eta_{\mathrm{eff}}^* = \frac{\Delta_h}{2\Delta_u}.
\]

A shifted transition that follows this prediction would distinguish the lattice law from an accidental optimum at `state_lr≈0.25`. At least two LSB ratios on opposite sides of the current ratio should be tested, with points immediately below, at, and above each predicted boundary. Preserve sPC and floating-point PC-ALM as controls; do not reinterpret this numerical-format result as evidence against either baseline.
