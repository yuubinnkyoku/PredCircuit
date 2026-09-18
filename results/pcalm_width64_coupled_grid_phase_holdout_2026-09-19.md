# PC-ALM width-64 coupled grid phase holdout (2026-09-19)

## Setup

- depth 32, width 64, T=256, seeds 845--859
- dual leak = 0.02, state_lr = 0.234285 (R = 0.93714)
- state `fixed15_i3`, update `fixed14_i1`, dual `fixed12_i1`
- interventions: baseline; update quantizer shifted by 0.5 update LSB; dual quantizer shifted by 0.5 dual LSB
- all 15 artifacts from Actions run 35383957127 were present and parsed (45 rows total)

## Aggregate result

| phase target | useful / 15 | mean BP cosine | mean norm ratio | mean relative error | mean late zero-step | mean realized/requested |
|---|---:|---:|---:|---:|---:|---:|
| none | 8/15 | 0.882994 | 0.675067 | 0.508913 | 0.999365 | 0.258406 |
| update +0.5 LSB | **15/15** | **0.959580** | **0.915713** | **0.281614** | **0.994222** | **0.441340** |
| dual +0.5 LSB | 8/15 | 0.885879 | 0.676634 | 0.504797 | 0.999453 | 0.251969 |

Baseline failures were seeds 845, 848, 849, 851, 852, 856, 858. The update-grid intervention rescued **all 7/7** baseline failures. The dual-grid intervention rescued **0/7**.

Failure-seed first-layer BP cosine:

| seed | baseline | update +0.5 LSB | dual +0.5 LSB |
|---:|---:|---:|---:|
| 845 | 0.851390 | 0.949245 | 0.855835 |
| 848 | 0.840982 | 0.946240 | 0.828482 |
| 849 | 0.779202 | 0.934491 | 0.809888 |
| 851 | 0.848682 | 0.945643 | 0.845208 |
| 852 | 0.854936 | 0.941742 | 0.857748 |
| 856 | 0.871942 | 0.964643 | 0.880429 |
| 858 | 0.830540 | 0.917830 | 0.842106 |

## Interpretation

This sharply localizes the observed deterministic fixed-point lattice failure to the update quantizer rather than the state-grid origin or dual-grid origin. A half-LSB update-grid phase shift is sufficient to restore useful credit for every baseline-failing holdout seed, while the corresponding dual-grid shift has essentially no aggregate effect.

This does **not** imply that a fixed half-LSB bias is a suitable hardware remedy: a phase-shifted nearest-neighbor quantizer is biased and changes the discrete vector field. The result should instead be treated as a mechanism probe showing that sub-LSB update information is being destroyed at `Q(delta h)`.

The next discriminating experiment is unbiased stochastic rounding applied only to the update quantizer. If it preserves 15/15 useful credit at the same 14-bit update format and official state_lr, it would test whether preserving the update in expectation can remove lattice lock without increasing datapath width.
