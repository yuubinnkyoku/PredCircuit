# PC-ALM weight-only precision, unclipped i3 formats (2026-09-21)

Configuration: depth 32, width 8, ReLU residual MLP, seeds 800--819, PC-ALM alpha=0.925, rho=1, state_lr=0.25, dual_leak=0.01, budget=112. Only stored weights are quantized; activations and MAC arithmetic remain FP32. Update-state precision remains fixed12_i2. Fixed-point weight formats use three integer bits, giving range [-8, 8), so the observed pre-quantization maximum |W|=4.3186 does not clip.

| weight precision | finite | useful | mean first-layer cosine to BP | mean all-gradient relative error vs FP32-weight PC-ALM | mean saturation |
|---|---:|---:|---:|---:|---:|
| FP32 | 20/20 | 16/20 | 0.936019 | 0 | 0 |
| fixed12_i3 | 20/20 | 15/20 | 0.935846 | 0.057806 | 0 |
| fixed10_i3 | 20/20 | 16/20 | 0.937698 | 0.069637 | 0 |
| fixed9_i3 | 20/20 | 16/20 | 0.937384 | 0.080939 | 0 |
| fixed8_i3 | 20/20 | 17/20 | 0.935174 | 0.098316 | 0 |

The FPGA-relevant 10-to-9-bit boundary shows no degradation in finite rate or the current useful-credit criterion: both are 20/20 finite and 16/20 useful. Mean first-layer cosine to BP changes only from 0.937698 to 0.937384 (-0.000314), while the all-gradient relative error against the FP32-weight PC-ALM reference rises from 0.069637 to 0.080939 (+0.011302 absolute). This is a measurable but modest perturbation, not a convergence failure.

For the current 30 x 64 x 64 depth-packed weight-store model on RAMB36E1, this makes 9-bit weights a meaningful hardware candidate: at P=32 the legal 4096x9 mode reduces the predicted weight memory from 64 to 32 RAMB36, whereas 10-bit weights remain in the 2048x18 mode and require 64. P=64 still requires 64 RAMB36 because the bank count itself is the lower bound.

This result is a diagnostic on gradient geometry, not yet a full learning-accuracy result. Before treating 9-bit storage as the default RTL format, validate it on an actual training trajectory (weight updates quantized after each update) and compare final loss/accuracy and convergence against 10-bit/FP32. The present straight-through diagnostic quantizes weights during prediction but does not establish long-horizon accumulation behavior of quantized stored weights.
