# PC-ALM hardware break-even note (2026-09-15)

This note turns the current depth-32 / width-8 experiments and analytical cost model into explicit RTL go/no-go conditions. It deliberately does **not** claim measured FPGA performance.

## Current candidate

For depth=32, width=8, input_dim=8, output_dim=4, batch=4:

- major MACs per relaxation step: 16,128 for both sPC and PC-ALM
- free hidden state: 992 scalars
- PC-ALM dual state: +992 scalars
- PC-ALM dual update: 992 scalar updates/step
- engineering candidate: alpha=0.925, rho=1, eta_h=0.25, dual leak=0.01
- robust common relaxation budget from the fresh holdout: T=128

The dual leak is an engineering extension, not the unmodified Sakana PC-ALM baseline. The pure PC-ALM baseline remains in all comparisons.

## Matched-credit holdout: the iteration advantage is now measured

A fresh 20-seed depth-32 / width-8 holdout compared sPC, pure PC-ALM and leaky PC-ALM using the same strict first-layer credit criterion:

- cosine to BP >= 0.9
- gradient norm ratio to BP in [0.5, 2]
- relative error to BP <= 0.6
- all values finite

Results:

- sPC: **0/20** seeds reached the criterion by T=256
- pure PC-ALM: **17/20** reached by T=160; median first successful T=96
- leaky PC-ALM: **19/20** reached by T=160; median first successful T=96, IQR 80--112
- at one fixed budget T=128, leaky PC-ALM was useful on **19/20 = 95%** of seeds

At the common T=128 point, leaky PC-ALM averaged:

- first-layer cosine to BP: **0.9491**
- first-layer gradient-norm ratio to BP: **1.0135**
- first-layer relative error to BP: **0.3412**

At sPC T=256, sPC averaged:

- first-layer cosine to BP: 0.8055
- first-layer gradient-norm ratio to BP: **0.00439**
- first-layer relative error to BP: 0.9964

The critical failure is therefore not merely poor angle: the deep sPC credit magnitude is still only about **0.44% of BP** after 256 relaxation steps.

The fixed-budget curve is stored in `pcalm_matched_credit_common_budget_900_919.csv`; the per-method minimum-budget summary is stored in `pcalm_matched_credit_summary_900_919.csv`.

## Cycle break-even

With `P_mac` MAC lanes and `P_dual` dual-update lanes,

```
C_spc  = ceil(16128 / P_mac)
C_pcalm_overlap = max(ceil(16128 / P_mac), ceil(992 / P_dual))
C_pcalm_serial  = ceil(16128 / P_mac) + ceil(992 / P_dual)
```

For the exploratory point P_mac=128, P_dual=32:

- sPC: 126 cycles/step
- PC-ALM with overlapped dual update: 126 cycles/step
- PC-ALM with serialized dual update: 157 cycles/step

Thus PC-ALM beats sPC when:

- overlap: `T_pcalm < T_spc`
- serialized: `T_pcalm/T_spc < 126/157 = 0.80255`

For the 19 seeds where leaky PC-ALM succeeds at T=128 while sPC still fails at T=256, the useful sPC budget is known only to satisfy `T_spc > 256`. That censoring is already enough to prove:

- overlapped cycle ratio `< 128/256 = 0.5`
- serialized cycle ratio `< (157*128)/(126*256) = 0.623`

So, on those 19/20 seeds, the current analytical model predicts **more than 2x** relaxation-cycle advantage if the dual update is hidden under the MAC path, and **more than about 1.60x** even if the dual update is serialized. These are model-based lower bounds, not measured FPGA speedups.

## State storage and pessimistic state-traffic break-even

At equal 12-bit storage precision PC-ALM carries twice as much persistent relaxation state because lambda has the same shape as the hidden state:

- sPC hidden state: 11,904 bit
- PC-ALM hidden + dual state: 23,808 bit

For this small experiment the total is only about 2.91 KiB, but the 2x relative cost matters when scaling.

A deliberately pessimistic proxy assumes every persistent scalar is read/written once per relaxation step:

```
traffic_spc   ~ T_spc * S
traffic_pcalm ~ T_pcalm * 2S
```

PC-ALM therefore wins this proxy only when `T_pcalm/T_spc < 0.5`.

The T=128 holdout now reaches that boundary in a useful way: for 19/20 seeds, PC-ALM succeeds at 128 while sPC fails through 256, hence the unknown useful sPC budget is strictly greater than 256 and

```
2 * 128 < T_spc
```

for those seeds. In other words, **even the pessimistic 2x persistent-state traffic proxy is favorable on 19/20 seeds in this deep/narrow regime**. This still does not substitute for an SRAM/register-file energy model.

## Low-precision status

The low-precision gate has now been passed for the complete arithmetic skeleton at the current width-8 operating point.

Already supported:

- hidden-state storage: 12-bit fixed point
- dual/lambda storage: 12-bit fixed point
- state/dual update signals: 12-bit fixed point
- MAC input operands (weights and activations): 12-bit fixed point
- sequential MAC accumulator: **12-bit fixed point at width=8**, with the caveat that its required integer range must still be checked as width grows

A fresh 20-seed MAC-operand holdout using `fixed12_i3` operands preserved the same 0.8 useful-rate seen with FP32 operands at T=112. Relative gradient error versus the FP32-operand implementation averaged about **0.064**, with negligible saturation. The lower useful rate is therefore a property of the fixed T=112 operating point, not a 12-bit operand failure.

The separate accumulator holdout used fixed12_i3 operands and quantized the running dot-product accumulator after every MAC. Results over 20 fresh seeds were:

- FP32 accumulator: useful rate 0.90, cosine 0.9305
- fixed16_i6: useful rate 0.90, FP32-accumulator gradient error 0.0547, no saturation
- **fixed12_i5: useful rate 0.90, cosine 0.9314, FP32-accumulator gradient error 0.0625, no saturation**
- fixed12_i6: useful rate 0.80, gradient error 0.1167

The largest pre-quantization accumulator magnitude observed was about 23.7, so the signed `fixed12_i5` range (approximately -32 to +31.98) was sufficient at width=8. `fixed12_i6` spent one extra bit on integer range and lost one fractional bit, which was worse despite the larger range. This is a useful warning: accumulator format should follow measured dynamic range, not simply maximize integer bits.

The summary is stored in `pcalm_accumulator_precision_summary_820_839.csv`.

## RTL gate decision

The previous RTL gate required all three of the following:

1. PC-ALM shows a meaningful advantage over sPC in depth, iteration count, width, or BP-gradient agreement.
2. Low precision is realistic.
3. A hardware cost model suggests the lambda overhead can be paid back.

For the depth-32 / width-8 deep-narrow regime, all three are now satisfied strongly enough to justify a **minimal RTL prototype**:

- sPC loses its first-layer credit magnitude even at T=256, while leaky PC-ALM is robust at T=128 on 19/20 seeds;
- the stored states, update values, MAC operands and width-8 accumulator all tolerate 12-bit fixed point;
- the measured iteration gap exceeds both the current cycle break-even and, on 19/20 seeds, the pessimistic 2x state-traffic break-even.

This is **not** yet a go-ahead for a large full-network FPGA accelerator. Width/depth scaling and target-device synthesis remain necessary.

## First RTL prototype status

A minimal PC/PC-ALM-switchable block now exists as `rtl/pcalm_dual_update.sv`. It implements the local difference between the two algorithms:

```
lambda_next = (1 - leak) * lambda + alpha * residual
credit      = rho * residual + lambda
```

with saturating fixed-point arithmetic, a persistent lambda register, explicit minibatch dual reset and immediate sPC-mode bypass of stale lambda.

`rtl/tb_pcalm_dual_update.sv` checks reset, sPC bypass, two nominal PC-ALM updates, update disable, dual clear, positive/negative saturation and mode switching. The GitHub `RTL smoke` workflow now passes both:

- Icarus Verilog simulation: `PASS tb_pcalm_dual_update`
- Yosys generic synthesis/check: **0 problems**

The generic Yosys netlist currently contains 38 cells, including 2 multipliers and 2 asynchronously reset/enabled registers. This count is only a synthesizability sanity check; it is **not** a LUT/DSP/BRAM or Fmax result for a specific FPGA family.

## Next hardware questions

The next steps are deliberately narrower than a full accelerator:

1. check accumulator range as width grows beyond 8;
2. compare the synthesized extra PC-ALM dual block against the corresponding sPC credit-only baseline;
3. add randomized bit-accurate Python-vs-RTL vectors, not just hand-written cases;
4. then synthesize for a concrete FPGA family and measure LUT/DSP/register/BRAM/Fmax deltas.

Only after those pass should the MAC/state-update datapath be integrated into a complete layer engine.
