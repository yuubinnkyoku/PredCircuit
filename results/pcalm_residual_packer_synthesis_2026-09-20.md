# PC-ALM streamed residual packer synthesis (2026-09-20)

## Status

Commit `0a8392ae` completed both the dedicated **Residual packer RTL** workflow and the normal CI successfully. The synthesis artifact `residual-packer-synthesis` was recovered rather than re-running synthesis.

## XC7 synthesis result

Target: `rtl/residual_packer_1to8.sv`, 12-bit residuals, 8 lanes.

| metric | streamed 1→8 packer | previous atomic 64→8 serializer |
|---|---:|---:|
| estimated logic cells | **322** | 836 |
| FDRE | **184** | 772 |
| DSP48 | **0** | 0 |
| BRAM | **0** | 0 |

The streamed packer therefore reduces estimated logic cells by **514 LC (61.5%)** and flip-flops by **588 (76.2%)** relative to the previous 64-wide serializer design point.

The packer RTL accepts a scalar stream and only backpressures when an arriving scalar would complete the next 8-lane word while the previous completed word is still blocked. Therefore a consumer that accepts at least one 8-lane word per 8 producer cycles permits continuous 1-scalar/cycle input.

## Interpretation

The measured 322 LC is still not “free”, but it is small enough that the earlier 836-LC serializer is no longer a credible first implementation. More importantly, the packer cost is not intrinsic to PC-ALM arithmetic; it is an interface-width adaptation between a 1-scalar/cycle residual schedule and the already-selected 8-lane dual engine.

For the current shared 64-MAC schedule, the residual producer is expected to emit at most about 1 scalar/cycle while the dual engine consumes 8 scalars in a word. The next high-value check is therefore end-to-end timing rather than another standalone packer optimization: connect the packer to the 8-lane dual engine and verify 64 consecutive residuals with zero producer stalls, correct address order, pre-update-λ credit, and correct λ write-back.

## Design consequence

Keep the first hardware point at:

- shared matrix MAC pool (up to 64 MAC/cycle),
- scalar residual stream,
- 1→8 residual packer,
- 8-lane shared dual engine,
- 12-bit λ storage.

Do not restore the atomic 64-wide serializer unless a later matrix schedule actually requires burst output.
