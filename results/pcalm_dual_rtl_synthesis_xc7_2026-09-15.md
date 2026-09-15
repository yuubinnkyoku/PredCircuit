# sPC vs PC-ALM local RTL synthesis comparison

Date: 2026-09-15

This is a **local credit/dual-update block comparison**, not a full predictive-coding layer and not an FPGA-board measurement. Yosys 0.33 was used with the XC7 mapping flow as an early architecture sanity check.

## Modules

- sPC baseline: `rtl/spc_credit.sv`
- PC-ALM candidate: `rtl/pcalm_dual_update.sv`
- data width: 12 bit
- rho = 1

## Initial arbitrary-coefficient implementation

The first PC-ALM RTL used Q*.12 constants for alpha ~= 0.925 and retain ~= 0.99.

### Generic synthesis

| resource | sPC credit | PC-ALM dual/credit |
|---|---:|---:|
| generic cells | 12 | 38 |
| multipliers | 0 | 2 |
| sequential cells | 0 | 2 `$adffe` |

### XC7 technology mapping

| resource | sPC credit | PC-ALM dual/credit | delta |
|---|---:|---:|---:|
| estimated LCs | 46 | 141 | +95 |
| DSP48E1 | 0 | 2 | +2 |
| FDCE | 0 | 13 | +13 |
| CARRY4 | 28 | 73 | +45 |
| LUT2 | 16 | 46 | +30 |
| LUT3 | 22 | 72 | +50 |
| LUT4 | 1 | 1 | 0 |
| LUT5 | 2 | 17 | +15 |
| LUT6 | 21 | 51 | +30 |

The two DSPs came from the alpha and retain products and are an implementation choice, not an algorithmic requirement.

## Holdout-validated dyadic implementation

A fresh 20-seed software holdout validated the hardware-friendly coefficients

- `alpha = 237/256 = 0.92578125`
- `retain = 253/256 = 0.98828125`, equivalent to `dual_leak = 3/256`
- `rho = 1`

before the RTL was changed. The dual update is now exactly implemented as

```
253 * lambda = (lambda << 8) - (lambda << 1) - lambda
237 * r      = (r << 8) - (r << 4) - (r << 1) - r
lambda_next  = round((253 * lambda + 237 * r) / 256)
credit       = r + lambda_pre_dual
```

There are no general multipliers in this datapath.

### XC7 technology mapping after shift-add replacement

| resource | sPC credit | PC-ALM dyadic dual/credit | delta vs sPC | change vs arbitrary PC-ALM |
|---|---:|---:|---:|---:|
| estimated LCs | 46 | **203** | +157 | +62 |
| DSP48E1 | 0 | **0** | 0 | **-2** |
| FDCE | 0 | 13 | +13 | 0 |
| CARRY4 | 28 | **31** | +3 | -42 |
| LUT6 | 21 | 141 | +120 | +90 |

Both Icarus simulation and Yosys `check` pass after the replacement. RTL-smoke run `34925112689` completed successfully on commit `b5ff3e0c84818945c415a75403bd9e21fee355d4`.

## Interpretation

The co-design trade-off is now explicit rather than hypothetical:

- the dual lane is **DSP-free**;
- removing 2 DSP48E1 per scalar lane costs about **+62 estimated LCs per PC-ALM lane** relative to the first multiplier-based PC-ALM block;
- compared with the minimal sPC credit block, one complete dyadic PC-ALM dual/credit lane costs about **+157 estimated LCs** plus 13 flip-flops for lambda/state flags.

For the current architectural example with 32 dual lanes, direct replication projects to roughly:

- arbitrary-coefficient PC-ALM: 64 DSP48E1 plus about 3040 LC over the sPC credit lanes;
- dyadic PC-ALM: **0 DSP48E1** plus about **5024 LC** over the sPC credit lanes.

These are linear lane-replication projections, not place-and-route measurements. They nevertheless show why the dyadic form is useful: DSP pressure disappears, but LUT/LC pressure becomes the next design constraint. Time-sharing or pipelining the dual update is therefore a meaningful design-space variable rather than blindly instantiating one lane per state scalar.

The software-side matched-credit experiment should be combined with this lane cost when choosing dual parallelism. PC-ALM reaches useful deep credit in far fewer relaxation steps than sPC at depth 32, so fewer dual lanes may still hide dual work under the main MAC schedule.
