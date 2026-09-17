# Local-only sPC magnitude controllers (holdout)

Date: 2026-09-18
Scripts: `scripts/diagnose_spc_local_magnitude.py`
CI holdout: Actions 35247010038 (20/20 seeds 960–979), T∈{128,256}, depth32/width8.

## Controls

| control | BP at inference? | rule |
|---|---|---|
| `spc_raw` | no | none |
| `spc_oracle_norm_match` | yes (eval only) | first-layer credit → BP norm |
| `spc_local_layer_rms` | no | layerwise credit / residual RMS (layers >0) |
| `spc_local_residual_match_t{g}` | no | rescale all credit so first-layer norm = `g * mean residual norm` |
| `pcalm_dual_leak` | no | γ=0.01 dual dynamics |

## Holdout aggregate (n=20)

| T | control | useful | mean cosine | mean norm ratio | mean rel error |
|---:|---|---:|---:|---:|---:|
| 128 | spc_raw | 0/20 | 0.884 | 0.000082 | 0.9999 |
| 128 | spc_oracle_norm_match | **9/20 (45%)** | 0.884 | 1.000 | 0.466 |
| 128 | spc_local_residual_match_t2 | **5/20 (25%)** | 0.884 | 0.606 | 0.558 |
| 128 | spc_local_residual_match_t1 | 0/20 | 0.884 | 0.303 | 0.747 |
| 128 | spc_local_layer_rms | 0/20 | 0.884 | 0.000082 | 0.9999 |
| 128 | pcalm_dual_leak | **20/20** | 0.948 | 1.009 | 0.330 |
| 256 | spc_oracle_norm_match | **4/20 (20%)** | 0.830 | 1.000 | 0.565 |
| 256 | spc_local_residual_match_t2 | **4/20 (20%)** | 0.830 | 0.728 | 0.584 |
| 256 | spc_local_residual_match_t1 | 1/20 | 0.830 | 0.364 | 0.727 |
| 256 | pcalm_dual_leak | **20/20** | 0.956 | 1.021 | 0.317 |

## Conclusion

1. **BP-free residual-derived rescale does not beat oracle**, and never approaches dual-leak useful rates.
2. At T=256, local t2 coincides with oracle (20%) but still fails the majority of seeds because direction residual grows as cosine falls (0.884→0.830).
3. Layerwise residual-RMS scaling leaves first-layer deep credit collapsed.
4. **Hardware implication:** post-hoc local rescale of sPC is not a substitute for dual dynamics that *form* usable in-window credit.

Related: stage-2 control `spc_residual_gain_unit_first` (gain then unit first-layer norm) is in `diagnose_spc_magnitude_control.py` and under re-holdout run 35247918652.
