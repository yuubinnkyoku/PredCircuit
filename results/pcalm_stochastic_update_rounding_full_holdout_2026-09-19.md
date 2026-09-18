# PC-ALM stochastic update rounding: full 15-seed holdout

Date: 2026-09-19

## Setting

- depth 32, width 64, T=256
- state `fixed15_i3`, update `fixed14_i1`, dual `fixed12_i1`
- state learning rate 0.234285, lattice ratio R=0.93714
- dual leak 0.02
- seeds 845--859

The existing successful GitHub Actions run produced one artifact for every seed. This aggregation reuses those artifacts; it does not rerun the holdout.

## Result

Unbiased stochastic rounding at the update quantizer produced useful first-layer credit on **15/15** holdout seeds. All conditions were finite and the inspected artifacts report zero state/update/dual saturation.

Across all 15 stochastic-rounding trajectories:

- mean BP cosine: **0.9691** (minimum **0.9559**, maximum **0.9842**)
- mean first-layer gradient norm ratio to BP: **0.9093**
- mean first-layer relative error to BP: **0.2543**
- mean late realized/requested state-step norm ratio: **0.5842**

The earlier deterministic nearest-rounding holdout at the same widths and hyperparameters was useful on only 8/15 seeds. Thus the full paired success count changes from **8/15 nearest** to **15/15 stochastic** without increasing nominal precision or changing state learning rate, dual leak, or relaxation budget.

## Interpretation

This strengthens the update-quantizer mechanism: preserving sub-quantum updates in expectation is sufficient to remove every observed useful-credit failure in this 15-seed holdout. It also shows that the seven-seed rescue result was not obtained by degrading the eight seeds that were already useful under nearest rounding.

This still does not establish robustness to the stochastic-rounding RNG trajectory. A dedicated follow-up now varies only the rounding RNG seed on the seven difficult model/data seeds while holding the model, data, numerical formats, and hyperparameters fixed.
