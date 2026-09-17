# Width-64 update/state lattice alignment holdout

Source: GitHub Actions run `35278147978`, commit `8465f515`, seeds 845--859. All 15 jobs completed successfully and all artifacts were recovered without rerunning the experiment.

Configuration: ResidualMLP depth 32, width 64, batch 4, ReLU, `T=256`, `state_lr=0.234285`, `rho=1`, `alpha=0.925`, `dual_leak=0.02`. Hidden state is fixed at `fixed15_i3` and the dual at `fixed12_i1`; only update/residual precision changes.

The experiment was designed after the state-lattice trace showed that `fixed15_i3 + fixed14_i1` nearly freezes the state trajectory. For each update grid we predicted the smallest raw gradient magnitude capable of surviving both update quantization and the subsequent state round-to-nearest step.

## 15-seed aggregate

| update precision | predicted deadzone | min update quanta to move state | useful | cosine | norm/BP | BP-relerr | late realized/requested | late state zero frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed12_i1` | 4.883e-4 | 1 | **15/15** | 0.9655 | 1.0490 | 0.2720 | 1.065 | 0.8506 |
| `fixed13_i1` | **2.441e-4** | **1** | **15/15** | **0.9667** | 0.9346 | **0.2547** | 1.039 | 0.9951 |
| `fixed14_i1` | **3.662e-4** | **2** | **8/15** | **0.8830** | **0.6751** | **0.5089** | **0.258** | **0.9994** |
| `fixed15_i1` | 3.052e-4 | 3 | 12/15 | 0.9217 | 0.7868 | 0.4087 | 0.310 | 0.9987 |
| `fixed16_i1` | 2.747e-4 | 5 | 13/15 | 0.9408 | 0.8467 | 0.3503 | 0.353 | 0.9981 |
| FP32 update | 2.605e-4 | -- | **15/15** | 0.9490 | 0.8681 | 0.3234 | 0.380 | 0.9977 |

All configurations are finite and have zero state/update/dual saturation.

## The predicted lattice deadzone explains the fine-grid ordering

Excluding the very coarse `fixed12_i1` regime, the predicted deadzone orders the observed BP-relative error exactly:

`fixed13_i1 < FP32 < fixed16_i1 < fixed15_i1 < fixed14_i1`.

Across these five grids the Spearman rank correlation between predicted deadzone and mean BP-relative error is 1.0 (Pearson r ~= 0.985 on the five aggregate points). This is a mechanistic falsifier passed by the data: the previously anomalous 15-bit state becomes healthy when the update lattice is changed so a one-quantum update can cross the state half-LSB.

The key numerical relation is:

- state `fixed15_i3` LSB = `2^-11`, half-LSB = `2^-12`;
- update `fixed14_i1` LSB = `2^-12`;
- effective state LR = `batch * state_lr = 0.93714`;
- therefore one update quantum produces only `0.93714 * 2^-12`, just below the state half-LSB and is rounded away.

`fixed13_i1` doubles the update quantum. A single nonzero update then crosses the state threshold, removing the pathological deadzone even though the update datapath is **one bit narrower**.

## Coarse-kick regime

`fixed12_i1` is an intentional counterexample to a one-variable deadzone story. Its predicted raw-gradient deadzone is large, yet it still gives 15/15 useful credit. The late trajectory has much larger coarse state kicks (realized/requested ratio slightly above one) and only ~85% zero state coordinates per step instead of >99%. Thus there are at least two stable regimes:

1. sufficiently fine/matched grids (`fixed13_i1`, FP32, increasingly `fixed16_i1`);
2. a coarse-kick/dither-like regime (`fixed12_i1`) that avoids lattice locking by moving farther when it moves.

The pathological `fixed14_i1` setting lies between them: updates are fine enough to become small, but its one-quantum move is just too small to survive state rounding.

## Hardware implication

Precision must be co-designed as a **pair of lattices**, not selected independently. In this regime, making the update path 14 bits instead of 13 bits makes learning worse. A blanket "more bits is safer" rule is false for the iterative local dynamics.

For state `fixed15_i3`, the best compact tested point is now:

`state=fixed15_i3, update/residual=fixed13_i1, dual=fixed12_i1`.

It is 15/15 useful and has nearly the same mean cosine/BP-relative error as the established `state=fixed14_i3, update=fixed14_i1, dual=fixed12_i1` compact point, but trades one extra stored-state bit for one fewer update-datapath bit.

This does not yet establish which point is cheaper on a concrete FPGA; state storage and update arithmetic have different resource costs.

## Next generalization test

The lattice-locking model predicts a diagonal failure condition whenever update fractional precision is exactly one bit finer than state fractional precision while the effective LR is just below one. New, non-duplicating falsifiers are therefore:

- `state=fixed14_i3` with `update=fixed13_i1` (predicted lock) versus the already-known healthy `state14/update14` control;
- `state=fixed16_i3` with `update=fixed15_i1` (predicted lock) versus the already-known healthy `state16/update14` control.

If both new diagonal points degrade on the same held-out seeds, the result generalizes from a one-off 15-bit anomaly to a reusable fixed-point design rule.
