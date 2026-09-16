# PC-ALM dual-leak dynamics: 20-seed fresh sweep (2026-09-17)

Source: GitHub Actions run 35106196956, commit `86528e4362697be21e1738398455080df7f46baf`, seeds 920--939. All 20 jobs succeeded. Each seed sweeps `dual_leak in {0, 0.0025, 0.005, 0.01, 0.02, 0.05}` and every integer budget T=48..160.

The `useful_first_layer_credit` predicate is the same gradient-quality criterion used by the diagnostic. Values below are computed from the recovered CSV artifacts; no experiment was rerun.

## Main result

| leak gamma | oracle seeds with any useful T | best common T (all 20) | useful at that T | median useful steps/seed | median running max |lambda| at T=160 |
|---:|---:|---:|---:|---:|---:|
| 0 | 16/20 | 131 | 8/20 | 20.0 | 1.361 |
| 0.0025 | 18/20 | 125 | 16/20 | 49.0 | 1.208 |
| 0.005 | 19/20 | 126 | 18/20 | 68.5 | 1.120 |
| 0.01 | 19/20 | 129 | **19/20** | **70.5** | 0.968 |
| 0.02 | 19/20 | 148 | **19/20** | 67.5 | 0.753 |
| 0.05 | 12/20 | 138 | 12/20 | 28.0 | **0.536** |

Leak therefore does not merely select a different stopping time. From gamma=0 to 0.01, the median useful-window width grows from 20 to 70.5 measured steps while median running max |lambda| falls from 1.361 to 0.968 (~29% reduction). At gamma=0.02 the dual range shrinks further (~45% vs gamma=0) while retaining 19/20 oracle and 19/20 best-common-T success, but the best common stopping time shifts later to T=148. gamma=0.05 over-damps the dynamics: despite the smallest dual range, oracle success collapses to 12/20.

At the hardware-relevant fixed T=128:

| leak gamma | useful | mean cosine to BP | mean grad norm ratio | mean relative error | median running max |lambda| |
|---:|---:|---:|---:|---:|---:|
| 0 | 7/20 | 0.9628 | 1.5035 | 0.6214 | 1.290 |
| 0.0025 | 16/20 | 0.9621 | 1.3697 | 0.5180 | 1.171 |
| 0.005 | **18/20** | 0.9608 | 1.2568 | 0.4429 | 1.076 |
| 0.01 | **18/20** | 0.9576 | 1.0735 | **0.3562** | 0.917 |
| 0.02 | 16/20 | 0.9462 | 0.8267 | 0.3671 | 0.726 |
| 0.05 | 9/20 | 0.9122 | 0.4783 | 0.6015 | 0.532 |

This exposes a useful distinction. gamma=0.005 is the smallest tested leak reaching 18/20 at T=128 and preserves slightly higher cosine; gamma=0.01 gives essentially unit gradient scale and the lowest mean relative error while reducing dual range more strongly. gamma=0.02 is attractive for range but begins to under-scale the gradient at T=128 and needs a later common T.

## Held-out stopping-time check

Choosing the best fixed T separately for each gamma using only seeds 920--929, then evaluating seeds 930--939 without retuning, gives:

| gamma | train-selected T | train success | holdout success |
|---:|---:|---:|---:|
| 0 | 63 | 3/10 | 3/10 |
| 0.0025 | 108 | 8/10 | 5/10 |
| 0.005 | 108 | 8/10 | 6/10 |
| 0.01 | 109 | **9/10** | **8/10** |
| 0.02 | 111 | **9/10** | 7/10 |
| 0.05 | 138 | 7/10 | 5/10 |

Thus gamma=0.01 is currently the strongest compromise for a fixed-time implementation, while gamma=0.005 is the minimum tested leak that captures most of the benefit. The improvement is consistent with widening/stabilizing the useful temporal window rather than merely moving its center.

## Interpretation for PredCircuit

Dual leak now has evidence for a hardware-algorithm co-design role: moderate leak simultaneously makes a common stopping budget far more robust and reduces lambda dynamic range. The effect is non-monotone, so "more leak" is not generally better. The useful region is approximately gamma=0.005--0.02 for this network, with gamma=0.01 the current nominal point.

This does **not** make leaky PC-ALM identical to the Sakana reference algorithm; pure PC-ALM (gamma=0) remains the reference and must stay reported separately. Leak should be described as an engineering stabilization variant.

## Highest-value next experiment

Quantize the same trajectories/update to candidate fixed-point formats around gamma={0, 0.005, 0.01, 0.02}, starting with 16/12/8-bit saturating fixed point, and measure saturation count, max |lambda|, useful-window width, fixed-T success and gradient geometry. The key question is whether the ~29% dual-range reduction at gamma=0.01 translates into a real fractional-bit/integer-bit saving without losing the 18/20--19/20 stability advantage.