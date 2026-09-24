# PC-ALM dyadic coefficient holdout (2026-09-24)

This records the corrected width-64 held-out comparison from GitHub Actions run 35991312147 (seeds 845--859, 15 seeds). The run and CI both completed successfully on commit `7a32acb6`.

The comparison isolates coefficient replacement while keeping the fixed-point configuration and seeds matched. The dyadic candidate uses the hardware-friendly coefficients developed for the current PC-ALM candidate: effective state step `15/16`, dual gain `59/64`, and leak `5/256` (`1-leak = 251/256`).

| precision | coefficients | useful | cosine mean | norm ratio mean | BP relative error mean | gradient error vs original mean / max | late update zero | late state-zero step | saturation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| state14/update14/dual12 | original | 15/15 | 0.968459 | 0.902833 | 0.256994 | 0 / 0 | 0.264852 | 0.949793 | 0 |
| state14/update14/dual12 | dyadic | 15/15 | 0.964226 | 0.908963 | 0.269784 | 0.123514 / 0.192385 | 0.266973 | 0.954451 | 0 |
| state15/update13/dual12 | original | 15/15 | 0.966733 | 0.934623 | 0.254702 | 0 / 0 | 0.995065 | 0.995065 | 0 |
| state15/update13/dual12 | dyadic | 15/15 | 0.964655 | 0.931859 | 0.261573 | 0.057352 / 0.077474 | 0.995154 | 0.995154 | 0 |
| state16/update14/dual12 | original | 15/15 | 0.983727 | 0.975926 | 0.179098 | 0 / 0 | 0.971453 | 0.971453 | 0 |
| state16/update14/dual12 | dyadic | 15/15 | 0.983196 | 0.982150 | 0.181412 | 0.039476 / 0.055273 | 0.969303 | 0.969303 | 0 |

## Interpretation

All six groups are finite and useful on all 15 held-out seeds, and no state/update/dual saturation was observed. Replacing the tuned coefficients by dyadic values therefore does **not** destroy the useful-credit regime in any of the three fixed-point candidates.

The accuracy cost is small at the endpoint metrics: mean first-layer BP cosine drops by 0.00423, 0.00208, and 0.00053 for the 14/14/12, 15/13/12, and 16/14/12 formats respectively. The 16/14/12 point is particularly robust: the dyadic candidate keeps mean cosine at 0.98320 and mean BP relative error rises only from 0.17910 to 0.18141.

The more sensitive metric is the all-gradient deviation from the original-coefficient run. This is largest for 14/14/12 (mean 12.35%, max 19.24%), smaller for 15/13/12 (5.74% / 7.75%), and smallest for 16/14/12 (3.95% / 5.53%). Thus the dyadic replacement is viable, but 14-bit state/update should not be treated as numerically interchangeable with the tuned coefficients.

The pre-existing grid-lock diagnosis remains visible: 15/13/12 has ~99.5% late zero updates/state moves for both coefficient choices. The dyadic coefficients do not cause that pathology and do not cure it. Likewise 16/14/12 remains ~97% late-zero, despite having the best gradient geometry. These high zero fractions should be interpreted as quantized late-stage convergence/grid locking, not saturation (all saturation rates are zero).

## Hardware consequence

For the present candidate, generic scalar multipliers for the PC-ALM-specific coefficients are not required: the dyadic coefficients can be implemented with shifts and adds. The strongest currently tested numerical point is `state16/update14/dual12`; `state14/update14/dual12` remains attractive for storage/traffic but incurs a materially larger trajectory perturbation under dyadic replacement.

This result supports moving the coefficient implementation toward shift-add, but it does **not** by itself satisfy the RTL go criterion. The next high-value comparison is to combine these measured formats with the existing sPC/ePC/BP cost model and report equal-credit-quality cost (T, MACs, state bits, and state traffic), rather than optimizing coefficient arithmetic in isolation.
