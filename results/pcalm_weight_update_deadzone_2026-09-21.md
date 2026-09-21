# PC-ALM stored-weight update dead-zone diagnostic (2026-09-21)

Configuration: depth=32, width=8, PC-ALM alpha=0.925, rho=1, state_lr=0.25, dual_leak=0.01, budget=112, 20 seeds (800--819). Weights alone are quantized; activations/MAC arithmetic remain FP32. Fixed-point formats use integer_bits=3, so saturation is zero in this run.

The diagnostic models hardware with no hidden high-precision master copy:

`W_before = Q(W)`

`W_after = Q(W_before - lr * g)`

and reports the fraction with `W_after == W_before`.

| stored weight | dead @ lr=0.001 | dead @ lr=0.01 | dead @ lr=0.1 |
|---|---:|---:|---:|
| FP32 | 13.18% | 10.20% | 9.80% |
| 12-bit i3 | 100.00% | 99.893% | 93.321% |
| 10-bit i3 | 100.00% | 100.00% | 99.330% |
| 9-bit i3 | 100.00% | 100.00% | 99.821% |
| 8-bit i3 | 100.00% | 100.00% | 99.965% |

The FP32 equality rate is not a quantization dead zone; it includes exact zero gradients / numerically unchanged FP32 values and is retained only as a reference. The fixed-point result is decisive: under a direct read-modify-requantize update, the present PC-ALM gradient scale is far below one stored-weight LSB for nearly all weights. In particular, the 9-bit i3 candidate (LSB=1/32) has 100% dead updates at lr=0.01 and still 99.821% at lr=0.1.

This does **not** invalidate the earlier 9-bit forward/credit result (mean first-layer BP cosine 0.937384, 20/20 finite, 16/20 useful). It shows that good gradient geometry and writable low-precision state are separate requirements.

Hardware consequence: a bare 9-bit stored-weight design is not a viable online-learning state representation at these update scales. A residual/error-feedback accumulator, stochastic rounding, or a substantially different weight-update scaling is required if 9-bit BRAM storage is to retain its 32-RAMB36 P=32 advantage. A full FP32 master copy would defeat much of that memory argument, so the next comparison should quantify the smallest residual state that restores update activity.
