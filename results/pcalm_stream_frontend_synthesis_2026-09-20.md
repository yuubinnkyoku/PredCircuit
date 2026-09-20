# Integrated PC-ALM stream frontend synthesis (2026-09-20)

## Scope

This records the first width-64 synthesis point that integrates the continuous 1-scalar/cycle residual packer with the 8-lane PC-ALM dual engine and the full lambda state used by the width-64 experiment.

Source run: GitHub Actions run 35496077451, head `e9bdcd8e9dc2507f2e8c27da5706f018ae2452ac`.

Configuration:

- residual input: 1 scalar/cycle, 12 bit
- packer: 1 -> 8 lanes
- dual lanes: 8
- dual depth: 992 words
- lambda scalars: 992 * 8 = 7936
- lambda information bits: 7936 * 12 = 95,232 bit
- target: Yosys `synth_xilinx -family xc7`

The workflow simulation and synthesis both completed successfully.

## Synthesis result

| Item | Result |
|---|---:|
| Estimated logic cells | **1,739 LC** |
| FDRE | **393** |
| RAMB36E1 | **3** |
| DSP48 | **0** |
| Total cells | 3,933 |

Hierarchy-local estimates in the same integrated synthesis were:

| Block | Estimated LC | FDRE | RAMB36E1 |
|---|---:|---:|---:|
| residual packer | 312 | 184 | 0 |
| 8-lane dual engine | 1,415 | 135 | 3 |
| integrated frontend | 1,739 | 393 | 3 |

The integrated LC estimate is only 12 LC above the sum of the two hierarchy-local estimates (1,727 LC), so the wrapper/control itself is negligible at this design point. The integrated FF count is not additive in the same way because the top-level/interface/control registers are reported separately after synthesis.

No DSP48 primitive is present. The full 95,232-bit lambda state maps to three RAMB36E1 blocks, matching the previously observed width-64 8-lane memory packing.

## Interpretation

The streamed frontend removes the earlier 64-wide serializer while preserving the full scientific lambda state. The resulting PC-ALM-specific frontend is therefore dominated by the dual engine plus lambda BRAM, not by residual width conversion.

This closes an important hardware-side uncertainty: the 1-scalar/cycle schedule does not require a large residual buffer, and the integrated implementation remains DSP-free with only three 36-kbit BRAMs for lambda.

The next high-value comparison is no longer another standalone frontend micro-optimization. The useful question is whether the reduction in relaxation steps observed for PC-ALM versus sPC repays 1,739 LC + 3 RAMB36E1 in area-time / energy terms once the shared matrix datapath is included. That model should explicitly keep sPC as the alpha=0 / no-lambda-state baseline and avoid charging shared MAC/W/W^T resources only to PC-ALM.
