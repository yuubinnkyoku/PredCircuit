# PC-ALM stochastic update rounding: failed-seed rescue

Date: 2026-09-19

## Question

Does unbiased stochastic rounding at the update quantizer rescue the seven seeds that fail under deterministic nearest rounding in the known difficult width-64 fixed-point setting?

## Setting

- depth: 32
- width: 64
- relaxation steps: 256
- state precision: `fixed15_i3`
- update precision: `fixed14_i1`
- dual precision: `fixed12_i1`
- state learning rate: `0.234285`
- lattice ratio R: `0.93714`
- dual leak: `0.02`
- failed nearest-rounding seeds from the prior 15-seed holdout: 845, 848, 849, 851, 852, 856, 858
- stochastic-rounding seeds: 90845, 90848, 90849, 90851, 90852, 90856, 90858

The GitHub Actions stochastic-rounding holdout completed successfully and produced artifacts for all 15 seeds. This note deliberately reuses those artifacts rather than rerunning the experiment.

## Failed-seed paired results

| seed | nearest cosine | stochastic cosine | nearest norm ratio | stochastic norm ratio | nearest relative error | stochastic relative error | nearest realized/requested | stochastic realized/requested | rescue? |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 845 | 0.851390 | 0.963861 | 0.612021 | 0.954198 | 0.576569 | 0.266580 | 0.000000 | 0.450526 | yes |
| 848 | 0.840982 | 0.972145 | 0.438778 | 0.859694 | 0.674179 | 0.259961 | 0.114071 | 0.517619 | yes |
| 849 | 0.779202 | 0.957617 | 0.530470 | 0.839405 | 0.674323 | 0.311357 | 0.023347 | 0.467777 | yes |
| 851 | 0.848682 | 0.955928 | 0.497943 | 0.857713 | 0.634631 | 0.309593 | 0.179902 | 0.519519 | yes |
| 852 | 0.854936 | 0.958917 | 0.606078 | 0.905165 | 0.575338 | 0.288735 | 0.438018 | 0.655287 | yes |
| 856 | 0.871942 | 0.967151 | 0.616457 | 0.851734 | 0.552259 | 0.279177 | 0.379069 | 0.625102 | yes |
| 858 | 0.830540 | 0.958713 | 0.511808 | 0.853077 | 0.641711 | 0.303360 | 0.232023 | 0.524180 | yes |

All 14 paired conditions above were finite and had zero state/update/dual saturation.

## Main result

Stochastic update rounding rescued **7/7** seeds that failed under nearest rounding. Every rescued seed exceeded the existing useful-credit threshold, while preserving the same bit widths, state learning rate, dual leak, and number of relaxation steps.

Across these seven previously failing seeds, the mean first-layer BP cosine changed from approximately **0.8397** under nearest rounding to approximately **0.9620** under stochastic rounding. The mean realized/requested late state-step norm ratio changed from approximately **0.1952** to approximately **0.5371**.

This is stronger than the earlier +0.5-LSB update-grid phase intervention as a hardware-relevant result: stochastic rounding is unbiased in expectation rather than introducing a fixed phase bias.

## Interpretation

The result strongly supports the hypothesis that the principal failure mechanism in this setting is loss of small but credit-critical components at the **update quantizer**. It is inconsistent with a state-grid-only explanation and complements the prior result that dual-grid phase shifts rescued 0/7 failed seeds.

The mechanism is not simply "make every state update nonzero": even with stochastic rounding, the late state zero-step fraction remains high. Instead, preserving a sparse subset of otherwise-lost update events appears sufficient to restore first-layer BP gradient geometry.

This changes the fixed-point design question. Increasing update/state precision is no longer the only demonstrated route out of the R<1 lattice-lock regime; an unbiased rounding rule can recover useful credit at the same nominal widths.

## What this does not establish

- It does not yet show that stochastic rounding is 15/15 useful across all seeds; only the seven previously failing seeds are paired and summarized here, although artifacts exist for all 15.
- It does not yet establish multi-update training accuracy or long-horizon stability.
- It does not yet show that an FPGA stochastic-rounding implementation is cheaper than widening the datapath by one or two bits.

## Highest-value next step

Aggregate all 15 stochastic-rounding artifacts, then test **rounding randomness sensitivity** on the seven difficult model/data seeds using multiple independent rounding seeds. If rescue survives rounding-seed variation, stochastic rounding becomes a credible numerical-format design choice rather than a lucky noise realization. After that, compare its hardware cost (shared LFSR/PRNG, comparator/additional logic, routing) against one-bit and two-bit datapath widening.