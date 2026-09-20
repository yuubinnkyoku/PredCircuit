# PC-ALM area-time break-even from measured width-64 RTL (2026-09-20)

## Question

Does the measured PC-ALM frontend overhead repay itself in resource-time once the observed relaxation-count separation from sPC is included?

This note deliberately uses a conservative accounting rule: charge the **entire** integrated PC-ALM stream frontend to PC-ALM, even though the residual packer/control can in principle be shared with an sPC implementation. This avoids manufacturing an advantage by optimistic resource attribution.

## Inputs already established in the repository

Width-64 / depth-32 / batch-4 design point:

- useful leaky PC-ALM point: `T_PCALM = 128`;
- sPC is still below the useful first-layer-credit criterion at `T_sPC = 1024`, hence `T_sPC,min > 1024`;
- integrated streamed PC-ALM frontend synthesis: `+1739 LC`, `+3 RAMB36E1`, `+0 DSP48` under the conservative all-incremental accounting;
- lambda state: 95,232 bit;
- PC-ALM compact persistent-state traffic ratio versus h14 sPC: `26/14 = 1.857x` per relaxation step.

The observed iteration-ratio lower bound is therefore

`R_T = T_sPC / T_PCALM > 1024 / 128 = 8`.

This is not an equal-quality speedup measurement: the sPC success point has not been reached. It is a conservative lower bound on the relaxation budget needed by sPC relative to a known useful PC-ALM point.

## Generic resource-time break-even

For any additive resource `X`, let `X_base` be the resource count required by the shared sPC datapath and `Delta_X` the additional PC-ALM resource. Resource-time favors PC-ALM when

`(X_base + Delta_X) T_PCALM < X_base T_sPC`.

Equivalently,

`Delta_X / X_base < R_T - 1`.

Using only the observed lower bound `R_T > 8`, a sufficient condition is

`X_base > Delta_X / 7`.

This is useful because it does not require inventing a scalar conversion between LUT/LC, BRAM and DSP.

## LC-time threshold

Charge the full measured integrated frontend as incremental:

`Delta_LC = 1739`.

Then the break-even shared baseline is

`LC_base > 1739 / 7 = 248.43 LC`.

So any matched sPC implementation whose shared matrix/state datapath exceeds **249 LC** already makes the PC-ALM design lower in LC-time at the observed `T=128` versus the still-unsuccessful `T=1024` sPC budget.

This threshold is intentionally conservative because the 312-LC residual packer and some control are not intrinsically PC-ALM-only. Charging only the measured 1,415-LC dual-engine hierarchy would lower the threshold to `1415/7 = 202.14 LC`, but that value is not used as the main claim.

## BRAM-time threshold

The measured PC-ALM increment is three RAMB36E1 blocks:

`Delta_BRAM36 = 3`.

Break-even requires

`BRAM36_base > 3/7 = 0.4286`.

Since physical BRAM counts are integral, **one shared RAMB36E1 or more** is sufficient for PC-ALM to win BRAM-time under the same conservative T ratio, even after charging all three lambda BRAMs only to PC-ALM.

This does not say PC-ALM uses fewer BRAMs instantaneously; it says the extra three blocks are held active for sufficiently fewer relaxation steps that `BRAM * relaxation-step` is lower.

## DSP-time threshold

The integrated PC-ALM frontend uses no DSP48 primitives, so

`Delta_DSP = 0`.

For any positive shared DSP count, PC-ALM's DSP-time is strictly lower whenever `T_PCALM < T_sPC`. There is no DSP resource-time break-even penalty from the measured dual frontend.

## Persistent-state traffic check

The independent memory-traffic model gives a per-step PC-ALM/sPC persistent-state ratio of

`26/14 = 1.857`.

Across the observed budgets,

`traffic_PCALM / traffic_sPC < (128 * 26) / (1024 * 14) = 0.23214`.

Thus the compact PC-ALM point consumes **less than 23.3%** of the modeled persistent-state traffic of the 1024-step sPC run, despite carrying lambda. This is consistent with the resource-time result: the >8x iteration separation is much larger than the 1.857x per-step state-traffic penalty.

## Margin form

It is useful to invert the calculation. For a particular shared baseline, the required relaxation ratio is

`R_required = 1 + Delta_X / X_base`.

Examples for the conservative `Delta_LC=1739`:

| shared baseline LC | required `T_sPC/T_PCALM` |
| ---: | ---: |
| 256 | 7.793 |
| 512 | 4.396 |
| 1,024 | 2.698 |
| 2,048 | 1.849 |
| 4,096 | 1.425 |

The measured lower bound `>8` clears even the deliberately tiny 256-LC shared-baseline case. A realistic dense width-64 matrix datapath is expected to be substantially larger than 249 LC, but that expectation should be replaced by synthesis before making a system-level area-efficiency claim.

## Interpretation and limits

This closes the first-order **resource-time gate**, not the accelerator-efficiency question.

The result is stronger than simply saying that dual arithmetic is only ~0.8% of matrix MACs. Even when the whole measured `1739 LC + 3 RAMB36E1` frontend is pessimistically treated as PC-ALM-only, the observed relaxation separation gives generous break-even thresholds: `LC_base >= 249` and `BRAM36_base >= 1`.

However, `LC * step` and `BRAM * step` are proxies, not joules. They omit clock frequency, switching activity, BRAM dynamic energy, weight traffic, and any schedule-dependent stalls. They also compare a known-useful PC-ALM point against an sPC run that is still below criterion at 1024, because an equal-quality sPC point has not yet been observed. The correct wording is therefore that PC-ALM clears a conservative resource-time gate under the current diagnostic, not that an FPGA implementation is already proven more energy-efficient than sPC, ePC, GPU/NPU, or BP.

## Consequence for the RTL gate

The original hardware-entry condition required (1) a meaningful PC-ALM advantage over sPC, (2) realistic low precision, and (3) a resource model in which lambda cost can plausibly be repaid.

At the current width-64 diagnostic, all three now have quantitative support:

1. useful PC-ALM at T=128 while sPC remains below criterion at T=1024 (`R_T > 8`);
2. held-out useful fixed-point points with 12-bit lambda already exist;
3. measured all-incremental frontend overhead breaks even in LC-time for any shared baseline above 249 LC and in BRAM-time for any shared baseline using at least one RAMB36E1.

This is enough to justify a **minimal switchable sPC / leaky-PC-ALM core** as the next RTL artifact, but not a full accelerator. The highest-value missing hardware number is now the synthesized shared matrix/state datapath area and Fmax, because it converts the generous analytical margin into an actual device-specific area-time figure. In parallel, end-to-end learning and the ePC/BP digital baselines remain necessary before making the central scientific efficiency claim.