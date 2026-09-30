# Equal-quality whole-update cost gate (2026-09-30)

## Reused evidence

For depth=32, width=64, batch=4, the existing held-out sweep on `research/pcalm-width64-equal-quality-cost` reports:
- T=160: 15/15 useful seeds, mean BP cosine 0.96968.
- T=192: 15/15 useful seeds, mean BP cosine 0.97779.
- T=256: mean BP cosine 0.98320.

No experiment is rerun here.

## Whole-update dense-MAC lower bound

Let one dense matrix pass across all H=31 equal-width hidden transitions and the batch be one unit. Ignoring edge layers and applying the same omission to both methods:

- BP training update: forward + activation backward + weight-gradient outer product = 3 units.
- PC-ALM training update: initial forward + T relaxation steps with two matrix phases each + one final local weight-gradient phase = 2T+2 units.

Therefore

    R_MAC(T) = (2T + 2) / 3.

This gives:
- T=160: 107.33x BP
- T=192: 128.67x BP
- T=256: 171.33x BP

The earlier 2T=320/384/512x figures compare PC-ALM relaxation work only with one BP/ePC reverse-credit sweep. They are valid for credit-path comparison, but should not be presented as the whole-training-update ratio.

## Energy crossover gate

If all non-MAC costs are temporarily ignored, a PC-ALM accelerator can beat a BP implementation in energy/update only if its average energy per useful dense MAC is lower by more than R_MAC:

    e_MAC,PC / e_MAC,BP < 1 / R_MAC.

Required reductions:
- T=160: >107.3x
- T=192: >128.7x
- T=256: >171.3x

This is a necessary, not sufficient, condition under the dense-MAC model. Repeated state movement and control only make the PC-ALM requirement harder; avoided external-memory traffic can make the BP reference more expensive and must therefore be measured explicitly rather than guessed.

## Consequence

The current width-64 equal-quality result is a numerical-implementability result, not yet a compute/energy advantage. At the balanced T=192 point, low precision and locality must compensate roughly 129x more whole-update dense MAC work before PC-ALM can beat a three-pass BP update on energy in this simplified model.

The next decisive measurement is therefore energy and bytes moved per matrix pass for matched-precision BP/ePC and PC-ALM dataflows. Coefficient sweeps cannot resolve this gate.
