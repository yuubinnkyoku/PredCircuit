# Local-only sPC magnitude controllers (overnight follow-up)

Date: 2026-09-18
Scripts: `scripts/diagnose_spc_local_magnitude.py`
Local pilot: seeds 960–965, T=128, depth32/width8.
CI holdout: Actions 35247010038 / 35246907019 (`spc local magnitude holdout`).

## Question

After the magnitude-window holdout showed oracle first-layer norm restore only partially rescues sPC, can a **BP-free local scale** recover the same useful credit?

Controllers under test (all local except oracle/PC-ALM references):

| control | uses BP? | rule |
|---|---|---|
| `spc_raw` | no | none |
| `spc_oracle_norm_match` | yes (eval only) | rescale first-layer credit to BP norm |
| `spc_local_layer_rms` | no | layerwise credit / residual RMS (layers >0) |
| `spc_local_residual_match_t{g}` | no | rescale **all** credit so first-layer norm = `g * mean(hidden residual norm)` |
| `pcalm_dual_leak` | no | γ=0.01 dual dynamics |

## Local pilot (n=6 seeds, T=128)

| control | useful rate | mean cosine | mean norm ratio |
|---|---:|---:|---:|
| spc_raw | 0/6 | 0.866 | ~0 |
| spc_oracle_norm_match | 2/6 | 0.866 | 1.00 |
| spc_local_residual_match_t2 | 2/6 | 0.866 | 0.72 |
| spc_local_layer_rms | 0/6 | 0.866 | ~0 |
| pcalm_dual_leak | **6/6** | 0.940 | 1.00 |

Per-seed: local `t2` is useful **exactly on the same seeds as the oracle** (960, 965), which are the seeds with high sPC cosine (≥0.96). On low-cosine seeds, even near-unit norm ratio (e.g. 961 ratio 0.80) fails the useful predicate because cosine < 0.9.

## Provisional interpretation

1. A fixed residual-derived local scale can approximate oracle scale **when direction residual is already small**. It does not invent direction.
2. Residual RMS layerwise scaling does not lift first-layer deep credit.
3. Dual-leak PC-ALM remains the only controller that both (a) opens unit-scale credit and (b) raises cosine on every pilot seed — i.e. it changes the credit *content*, not only its magnitude.
4. Therefore the overnight conclusion strengthens: **hardware should not bank on post-hoc sPC rescale; it should bank on dual dynamics that form usable credit in-window.**

## Status

- Pilot recorded; 20-seed CI holdout running.
- Non-deployable caveat: residual-match `g` is not seed-calibrated to BP scale in general; even “success” is conditional on pre-existing cosine.
