# First sPC vs PC-ALM local RTL synthesis comparison

Date: 2026-09-15

This is a **local credit/dual-update block comparison**, not a full predictive-coding layer and not an FPGA-board measurement. Yosys 0.33 was used with the XC7 mapping flow as an early architecture sanity check.

## Modules

- sPC baseline: `rtl/spc_credit.sv`
- PC-ALM candidate: `rtl/pcalm_dual_update.sv`
- data width: 12 bit
- PC-ALM coefficient representation: Q*.12 constants for alpha ~= 0.925, retain ~= 0.99, rho = 1

## Generic synthesis

| resource | sPC credit | PC-ALM dual/credit |
|---|---:|---:|
| generic cells | 12 | 38 |
| multipliers | 0 | 2 |
| sequential cells | 0 | 2 `$adffe` |

The PC-ALM block has the expected persistent lambda register plus its saturation flag. The two generic multipliers come from the alpha and retain coefficient products; rho=1 is optimized away.

## XC7 technology mapping

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

Both modules pass Yosys `check` with zero reported problems. The PC-ALM testbench also passes under Icarus Verilog.

## Interpretation

The LUT/register overhead is small enough to keep investigating, but **2 DSPs per scalar dual-update lane is not an acceptable final architecture** if many dual lanes are spatially replicated. This is an implementation issue rather than an algorithmic requirement: alpha and retain are compile-time constants.

The next co-design candidate therefore replaces arbitrary constant multipliers with dyadic coefficients:

- `alpha = 237/256 = 0.92578125`
- `retain = 253/256 = 0.98828125`, equivalent to `dual_leak = 3/256`
- `rho = 1`

These can be implemented using shifts and additions/subtractions:

```
alpha * r  = r - (r >> 4) - (r >> 7) - (r >> 8)
retain * l = l - (l >> 7) - (l >> 8)
credit     = r + l
```

A 20-seed algorithmic holdout is being used to decide whether this coefficient approximation is acceptable before replacing the reference RTL datapath. If it preserves the tuned operating point, the next RTL target is a DSP-free dual update and a new XC7 synthesis comparison.
