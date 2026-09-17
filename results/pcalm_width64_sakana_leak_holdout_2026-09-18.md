# Width-64 Sakana state_lr × dual-leak holdout (2026-09-18)

Run: 35249816337. Depth 32, width 64, seeds 845–859 (15 independent seeds). The run completed successfully and all 15 seed artifacts were recovered. `state_lr=0.234285` is the official Sakana depth-32 value used by this probe; `0.25` is the project default. Useful-credit is the existing first-layer BP-geometry gate.

## Aggregated results

| state_lr | dual leak | T | useful | mean cosine | mean norm/BP | mean rel. error | mean max |lambda| |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sakana 0.234285 | 0 | 128 | 2/15 | 0.9718 | 1.597 | 0.669 | 0.207 |
| Sakana 0.234285 | 0 | 256 | 0/15 | 0.8264 | 1.429 | 0.828 | 0.249 |
| Sakana 0.234285 | 0.005 | 128 | 15/15 | 0.9693 | 1.340 | 0.448 | 0.174 |
| Sakana 0.234285 | 0.005 | 256 | 15/15 | 0.9368 | 1.227 | 0.458 | 0.201 |
| Sakana 0.234285 | 0.010 | 128 | 15/15 | 0.9658 | 1.150 | 0.322 | 0.150 |
| Sakana 0.234285 | 0.010 | 256 | 15/15 | 0.9780 | 1.137 | 0.263 | 0.170 |
| Sakana 0.234285 | 0.020 | 128 | 15/15 | 0.9576 | 0.889 | 0.296 | 0.118 |
| Sakana 0.234285 | 0.020 | 256 | 15/15 | **0.9943** | **1.020** | **0.108** | 0.134 |
| project 0.25 | 0 | 128 | 1/15 | 0.9773 | 1.620 | 0.677 | 0.210 |
| project 0.25 | 0 | 256 | 0/15 | 0.8195 | 1.360 | 0.792 | 0.248 |
| project 0.25 | 0.005 | 128 | 15/15 | 0.9748 | 1.364 | 0.451 | 0.177 |
| project 0.25 | 0.005 | 256 | 14/15 | 0.9351 | 1.192 | 0.441 | 0.202 |
| project 0.25 | 0.010 | 128 | 15/15 | 0.9714 | 1.174 | 0.316 | 0.153 |
| project 0.25 | 0.010 | 256 | 15/15 | 0.9779 | 1.121 | 0.255 | 0.171 |
| project 0.25 | 0.020 | 128 | 15/15 | 0.9630 | 0.914 | 0.275 | 0.121 |
| project 0.25 | 0.020 | 256 | 15/15 | **0.9943** | **1.024** | **0.109** | 0.136 |

All rows were finite in the recovered artifacts.

## Interpretation

The width-64 holdout closes the main uncertainty left by the width-8 Sakana-state-lr pilot. Pure PC-ALM (`dual_leak=0`) is not robust at this depth/width under either state learning rate: it is only 2/15 useful at T=128 and 0/15 at T=256 with the Sakana rate. This is not a small tuning mismatch; increasing the budget makes the BP geometry worse.

A small dual leak changes the regime. Every Sakana-rate leak condition tested is 15/15 useful at both T=128 and T=256. In particular, leak=0.02 at T=256 gives cosine 0.9943, norm ratio 1.020 and relative error 0.108. The project state_lr=0.25 reaches essentially the same endpoint, so the robust regime is not an artifact of the small state_lr discrepancy.

The leak also reduces dual dynamic range. At the Sakana rate and T=256, mean max |lambda| falls from 0.249 without leak to 0.134 at leak=0.02. That is directly favorable for the planned low-precision dual storage, although fixed-point leak behavior still needs a dedicated quantized holdout before this can be treated as an RTL guarantee.

## Consequence for the research direction

For deep/wide fixed-budget credit, `dual_leak` should no longer be treated as an optional side experiment. The current evidence says the robust PC-ALM candidate is a leaky-dual variant, while pure PC-ALM is strongly budget-sensitive. The next high-value experiment is therefore to combine the already-supported mixed precision (h/update 14 bit, lambda 12 bit) with the robust leak=0.02 regime on holdout seeds, measuring saturation/overflow and BP geometry. That directly tests whether the algorithmic stabilization survives the intended hardware numerical format.
