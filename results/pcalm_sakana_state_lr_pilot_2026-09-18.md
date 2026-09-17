# PC-ALM under official Sakana state_lr (pilot + holdout in flight)

Date: 2026-09-18
Script: `scripts/diagnose_pcalm_sakana_state_lr.py`
Official depth-32 `state_lr = 0.234285` (SakanaAI/pc-alm audit); project default `0.25`.
Local pilot seeds 980–983, T=128; CI holdout seeds 980–999 run 35247923364.

## Local pilot (T=128)

| state_lr | dual_leak | useful rate | mean cosine | mean norm ratio | mean rel error |
|---|---:|---:|---:|---:|---:|
| sakana 0.234285 | 0 | **1/4 (25%)** | 0.950 | 1.402 | 0.617 |
| sakana 0.234285 | 0.01 | **3/4 (75%)** | 0.938 | 0.990 | 0.399 |
| project 0.25 | 0 | 3/4 (75%) | 0.956 | 1.417 | 0.609 |
| project 0.25 | 0.01 | 3/4 (75%) | 0.944 | 1.011 | 0.390 |

## Pilot reading (to be frozen by holdout)

1. **Dual-leak γ=0.01 keeps useful credit at both state_lr values** on the pilot.
2. **Pure PC-ALM is more sensitive to state_lr at fixed T=128**: official 0.234285 had more relative-error failures (norm ratio stays high ~1.4 but rel error just above 0.6) than project 0.25.
3. If the holdout confirms this, the magnitude-window claim is **not an artifact of an under-stepped official baseline** — leak still helps under official coefficients. Conversely, pure PC-ALM hardware claims should quote the state_lr used.
4. Sample seed 980: official pure rel error 0.620 (fail) vs project pure 0.596 (pass) vs leak 0.312/0.295 (pass both).

## Status

- Holdout CI still running at note time.
- Replaces earlier assumption that project 0.25 ≈ official for fixed-T credit quality.
