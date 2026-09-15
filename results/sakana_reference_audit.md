# Sakana PC-ALM reference audit

Date: 2026-09-15
Reference repository main: `SakanaAI/pc-alm` at tree `660747f61a8a7e547c0ecd2c48c8883380a7d1f6`.

## What was checked

PredCircuit's `ResidualMLP` and PC-ALM inference were compared directly against the released reference equations/code rather than only the paper prose.

The following structural choices match the reference:

- hidden constraints only; the label/output residual is supervised but is not an AL constraint;
- scales `[1/sqrt(input_dim), 1/sqrt(width*depth), ..., 1/width]`;
- skip mask `[False, True, ..., True, False]`;
- first block is linear in its input; later hidden/output blocks apply the selected activation before the matrix multiply;
- free states initialize at the ordinary forward pass;
- AL uses `0.5*rho*||r + lambda/rho||^2` plus supervised loss;
- the activity update compensates for batch-mean energy with an effective step `state_lr * batch_size`;
- each outer PC-ALM iteration performs the primal inner solve first and then `lambda <- lambda + alpha*r`;
- the final weight credit can use either the dual before or after the last dual update, matching the reference's `pre_dual_energy` / `post_dual_energy` switch.

A new regression test independently re-expresses the reference block parameterization and one-step primal-then-dual update and checks PredCircuit against it in float64. This avoids importing Sakana's JAX package into the project while still pinning the released formulation.

## Important parameter correction for future sweeps

The official repository publishes depth-dependent activity steps:

| depth | official state_lr |
|---:|---:|
| 8 | 0.209541 |
| 16 | 0.221921 |
| 32 | 0.234285 |
| 64 | 0.242954 |
| 128 | 0.247725 |

Our scalar-chain exploratory sweep found `state_lr=0.2` best among the coarse values tested at depth 32. That is reassuring but should not be treated as a tuned Sakana reproduction. For residual-MLP comparisons, use the official depth-specific values as the primary reference condition and put independently tuned values in a separate ablation.

## Consequence for the hardware study

The current PredCircuit PC-ALM implementation is structurally aligned with the released Sakana implementation. The next uncertainty is therefore no longer the basic update order. It is quantitative: how many outer iterations are needed for the released residual parameterization to form useful dual credit, and how that iteration count changes under low precision.

This also tightens the FPGA gate. A low-precision result should be compared against the official depth-specific `state_lr`, not against an accidentally under-stepped floating-point baseline.
