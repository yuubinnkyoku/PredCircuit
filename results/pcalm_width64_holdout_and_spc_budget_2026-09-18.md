# Width-64 mixed-precision holdout and extended sPC budget

Date: 2026-09-18

Shape and evaluation follow the existing width-64 diagnostics: depth 32, width 64, batch 4, and the same BP first-layer credit metrics/useful criterion.

## PC-ALM mixed-precision holdout

Independent seeds 845--859 (15 seeds), T=128. Dual precision is fixed at 12 bit for every configuration below.

| state / update | useful | finite | mean cosine to BP | mean grad-norm ratio | mean relative error to BP | mean residual total |
|---|---:|---:|---:|---:|---:|---:|
| 12 / 14 | 13/15 | 15/15 | 0.932074 | 1.183041 | 0.440507 | 1.110580 |
| 14 / 12 | 15/15 | 15/15 | 0.932705 | 1.225680 | 0.469507 | 0.559998 |
| **14 / 14** | **15/15** | **15/15** | **0.960401** | **1.157310** | **0.343882** | **0.424704** |
| 16 / 12 | 14/15 | 15/15 | 0.934743 | 1.239811 | 0.473654 | 0.534807 |
| 12 / 16 | 15/15 | 15/15 | 0.938323 | 1.187805 | 0.428610 | 1.107720 |

The 14/14/12 design point therefore survives the independent holdout at 15/15 useful and is the strongest tested mixed-precision point by mean BP cosine and relative error. Raising only one side does not reproduce the 14/14 quality; this supports the closed-loop quantization interpretation rather than a single-state precision bottleneck.

## sPC relaxation-budget extension

Seeds 840--844 (5 seeds). This extends the previous T<=256 sweep without recomputing those budgets.

| T | useful | mean cosine to BP | mean grad-norm ratio | mean relative error to BP |
|---:|---:|---:|---:|---:|
| 320 | 0/5 | 0.887207 | 0.016789 | 0.985136 |
| 384 | 0/5 | 0.884331 | 0.029968 | 0.973601 |
| 512 | 0/5 | 0.881214 | 0.062832 | 0.945100 |
| 768 | 0/5 | 0.880286 | 0.135151 | 0.883362 |
| 1024 | 0/5 | 0.882104 | 0.201052 | 0.828101 |

sPC still does not meet the useful first-layer-credit criterion at T=1024. Its gradient norm grows with T, but at T=1024 it is only about 20.1% of BP on average, while direction cosine stays around 0.88. Thus the current evidence gives only a lower bound, T_sPC,min > 1024 for these five seeds, rather than an equal-quality sPC operating point.

Relative to PC-ALM T=128, this implies T_sPC/T_ALM > 8 under the present useful-credit criterion. This is substantially above the 26/14 = 1.857 state-traffic break-even ratio in the current hardware cost model. It should not be presented as an exact speedup because an equal-quality sPC success point has not yet been observed.

## Interpretation

Two FPGA-entry prerequisites are now substantially stronger:

1. Mixed precision: h=14 bit, update/residual=14 bit, lambda=12 bit is stable on 15 independent holdout seeds at width 64/depth 32/T=128.
2. Iteration advantage over sPC: sPC remains unsuccessful through T=1024 on the original five seeds, so the observed iteration-ratio lower bound is >8x.

The next highest-value work is not another dense sPC T sweep. The gap is already large enough for the simple traffic model. Priority should shift to (a) an ePC/BP digital-compute comparison at the same network/credit-quality target, and (b) mapping the 14/14/12 state/update/dual design point onto a concrete FPGA resource/bandwidth model. A sparse larger-T sPC probe is useful only if an equal-quality convergence point is needed for mechanism analysis.