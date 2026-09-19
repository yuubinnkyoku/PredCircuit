# Multiplier-free PC-ALM state step at T=128

Date: 2026-09-20

Configuration: depth=32, width=64, batch=4, `state_lr=0.25`, hence effective state step `batch * state_lr = 1.0`; state `fixed15_i3`, update `fixed14_i1`, dual `fixed12_i1`, dual leak 0.02. Seeds 845--859. Source artifacts: Actions run 35453887542. No experiment was rerun for this summary.

## Result

All 15/15 seeds satisfy the existing first-layer useful-credit criterion (finite, cosine >= 0.9, gradient-norm ratio in [0.5, 2.0], relative error <= 0.6).

- first-layer BP cosine: mean **0.947421**, sample SD **0.009956**, range **0.934999--0.967850**
- first-layer gradient-norm ratio to BP: mean **0.876736**, SD **0.054253**, range **0.802639--0.952155**
- first-layer relative error to BP: mean **0.328952**, SD **0.036485**, range **0.252459--0.378293**
- late realized/requested state-step norm ratio: mean **0.990709**, SD **0.028096**
- late update zero fraction: mean **0.456552**
- late state zero-step fraction: mean **0.803078**
- state/update/dual saturation: zero in the inspected T=128 artifacts

## Interpretation

The multiplier-free point is not merely a high-T=256 curiosity. At T=128 it remains useful on all 15 independent seeds. Therefore the state update can use effective coefficient exactly one,

`h_next = Q_h(h - Q(g_h))`,

without paying extra relaxation steps relative to the existing T=128 PC-ALM design point. This removes the state-step coefficient multiplier from the minimal RTL datapath while preserving robust deep/first-layer credit alignment.

The high late state zero-step fraction (~80.3%) remains important: multiplier removal does not by itself remove the fixed-point lattice-lock issue. Stochastic rounding / quantizer design remains a separate datapath concern.

The T=64 and T=96 artifacts from the same run should be aggregated next to determine whether T can be reduced below 128 without losing the 15/15 criterion.
