# Width-8 transpose-friendly weight-source synthesis (2026-09-24)

Commit under test: `0c9dfdc07cc2924d4b1bd9cff3718dba3c656c8c`.
Workflow: `Weight source synthesis`, run 35907070868 (success).
Target flow: Yosys `synth_xilinx` / XC7 mapping.

## Question

For an 8x8, 16-bit weight matrix, can PC-ALM feed both `W h` and `W^T c` at 8 weights/cycle without paying a prohibitive transpose-access cost? We compared:

1. `weight_source_single_copy`: one logical copy of W, with row/column selection from the same stored matrix.
2. `weight_source_dual_orientation`: explicit W and W^T copies, both updated on a weight write.

Both interfaces deliver 8 weights in parallel, so either can preserve the arithmetic lower bound of 8 active cycles per 8x8 matvec (16 active cycles for the two matvecs in one PC-ALM layer/relaxation, excluding control/state overhead).

## Synthesis result

| source | estimated LCs | RAM64M | notable LUT cells | total cells |
|---|---:|---:|---|---:|
| single-copy | 70 | 42 | LUT2=12, LUT3=70 | 263 |
| dual-orientation | 112 | 84 | LUT3=112 | 335 |

The dual-orientation implementation therefore costs about 1.60x estimated LCs and exactly 2x RAM64M in this synthesis. The absolute increment is 42 estimated LCs and 42 RAM64M per width-8 layer weight source.

Yosys maps the single-copy `w` storage to Xilinx LUTRAM (`$__XILINX_LUTRAM_QP_`). The dual-orientation design maps both `w` and `wt` to the same LUTRAM family. No DSP cost is introduced by either source itself.

## Interpretation

The earlier bank-conflict result does **not** imply that transpose access must serialize to 64 cycles. A single logical W copy synthesized as multi-port LUTRAM plus selection logic can already expose the required 8-wide row/column interface. For width 8, explicit W/W^T duplication is therefore not required to preserve the 16-active-cycle arithmetic bound.

The trade-off is now concrete: dual orientation spends 2x distributed RAM and +60% estimated LCs at the weight-source level, while single-copy spends selection/multi-port LUTRAM logic but is smaller in this synthesis. This favors the single-copy source as the next integration candidate.

This is still a synthesis/resource result, not a placed-and-routed timing result. It does not establish that the single-copy transpose network meets the same Fmax after connection to the P=8 MAC adder tree, nor does it include state/dual memory delivery. Those are the remaining hardware uncertainties.

## Consequence for the PC-ALM vs ePC/BP bound

Because both sources can structurally supply 8 weights/cycle, the optimistic width-8 arithmetic latency model remains `2T/H` relative to one sequential reverse sweep under full layer replication, rather than the conflict-serialized `9T/H` bound. With H=31, pure layer parallelism still requires roughly `T <= 15` to match the reverse-sweep latency. The weight-source result removes one reason that this bound might have been unattainable, but it does not make current T=77--112 competitive.

## Next experiment

Integrate `weight_source_single_copy` with `shared_mac_array(P=8)` as one synthesis top, retaining the dual-orientation version as an A/B reference. Measure combined LUT/FF/RAM/DSP and, where the available flow permits, timing/Fmax. This directly tests whether transpose selection or the MAC reduction tree becomes the critical implementation cost.
