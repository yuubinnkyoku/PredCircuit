# 8-lane PC-ALM dual-engine state-transition validation (2026-09-16)

## Status

Commit `4d32f8d2131b4192d3f21153a7b1144840340977` completed both CI and RTL smoke successfully. The RTL smoke job explicitly ran the new `tb_dual_engine_8lane.sv` state-transition test before synthesis.

Observed test output:

```text
PASS: dual_engine_8lane state transition 0 -> 237 -> 234 verified across all 8 lanes
```

The same job then completed the 8-lane and 16-lane XC7 synthesis stages and uploaded the synthesis logs.

## What this validates

The test exercises the integrated synchronous-RAM datapath rather than only the scalar arithmetic primitive:

1. clear/zero-fill of the dual-state memory;
2. synchronous read of a packed 8-lane word;
3. one-cycle alignment of address, residual, and valid state;
4. computation of credit from the pre-update dual state;
5. write-back of the updated dual state to the same address;
6. persistence of that state across a bubble and a later read.

For residual `r = 256` and initial dual state `lambda_0 = 0`, the implemented dyadic update gives

`lambda_1 = round((253*0 + 237*256)/256) = 237`.

The credit is formed from the pre-update state, so the first credit is

`credit_0 = lambda_0 + r = 256`.

After a bubble, revisiting the same address with `r = 0` recovers the stored value and gives

`lambda_2 = round(253*237/256) = 234`.

All eight lanes pass this `0 -> 237 -> 234` trajectory.

## Consequence for the cycle model

This removes an important ambiguity in the earlier 8-lane hardware model: the one-stage synchronous-RAM pipeline and its address/data alignment are functionally correct for a write-back/re-read sequence. Therefore the existing steady-state model of one issued word per cycle plus one pipeline drain remains consistent with the implemented RTL.

For 992 dual values at 8 lanes, there are 124 packed words, so the modeled sweep remains approximately `124 + 1 = 125 cycles` rather than requiring a two-cycle cost per word.

This is still a functional/cycle-level result, not a post-place-and-route timing result. Whether the 125-cycle dual sweep is actually hidden by the 126-cycle MAC budget still depends on the achieved clock-frequency ratio.

## Research interpretation

The previously measured 8-lane XC7 cost (1408 estimated LCs, 3 RAMB18E1, 0 DSP48) can now be associated with an engine whose integrated state-memory transition has been checked, rather than only with syntactically valid/synthesizable RTL. This strengthens the hardware-cost evidence but does not by itself satisfy the FPGA-go condition: the software-side PC-ALM advantage and low-precision stability criteria remain separate requirements.

## Next discriminator

The highest-value hardware discriminator is now post-place-and-route timing (or an equivalent implementation-level timing estimate) for the 8-lane dual engine and the matched MAC datapath. The 8-lane design hides under a 126-cycle MAC budget only if approximately

`f_dual / f_MAC >= 125/126 = 0.99206`.

If that ratio is met, increasing dual lanes offers little throughput benefit for this operating point. If it is not met, the next search should target the smallest parallel/banked design that restores timing slack while preserving block-RAM mapping where possible.
