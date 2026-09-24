# Reverse Gauss-Seidel PC-ALM: causality versus attenuation

Date: 2026-09-24

## Question

The short-budget experiment established that the current synchronous/Jacobi-style local dynamics advances useful credit by at most about one layer per relaxation step.  For depth 32 this gives a causal requirement near 31 steps, while the optimistic fully-layer-parallel hardware break-even against one BP/ePC reverse sweep is about T <= 15.

A natural escape is a reverse Gauss-Seidel schedule: update deeper hidden states first and immediately expose the changed residual/dual to the next shallower layer in the same sweep.  This can remove the strict one-layer-per-outer-step support bound, but it serializes the layer updates.  The important question is whether the credit that crosses many layers in one sweep has usable amplitude.

## Linear-chain bound

Use the existing scalar-chain model from `scripts/diagnose_linear_pcalm_credit.py`, with the repository defaults

- weight `w = 0.9`
- hidden-state step `eta_h = 0.05`
- dual step `alpha = 0.2`
- penalty `rho = 1.0`.

Consider the most favorable fused reverse schedule: after changing a deeper state, recompute its local residual and immediately update its dual before processing the next shallower layer.  Around the forward-feasible initialization, a newly introduced downstream credit contributes to the next shallower state gradient through the `-w (lambda + rho r)` term.  The state change contributes a factor `eta_h * w`; immediate residual/dual exposure contributes at most the local factor `(rho + alpha)` in this first-order path.  Therefore the per-layer same-sweep transmission factor is approximately

```
g = eta_h * |w| * (rho + alpha)
  = 0.05 * 0.9 * 1.2
  = 0.054.
```

After `d` same-sweep layer crossings the amplitude is bounded at first order by roughly `g^d`:

| crossings d | g^d | log10(g^d) |
|---:|---:|---:|
| 8 | 7.23e-11 | -10.14 |
| 15 | 9.68e-20 | -19.01 |
| 30 | 9.37e-39 | -38.03 |

Thus reverse Gauss-Seidel changes the problem from a strict support/causality barrier into an extreme attenuation barrier.  In exact real arithmetic the first layer can become nonzero within one reverse sweep, but with these stable baseline coefficients the signal crossing the full depth-32 chain is around 1e-38 of the injected scale at first order.  This is far below FP32 useful dynamic range for gradient geometry and, more importantly for PredCircuit, far below any proposed 12--16 bit fixed-point resolution.

A direct floating-point diagnostic of the fused reverse schedule agrees qualitatively: the first sweep becomes numerically indistinguishable from zero after only about 9--10 layer crossings at a 1e-15 support threshold.  This is not a proof that every nonlinear/network setting has exactly factor 0.054; it is a mechanism diagnostic showing that merely changing Jacobi to Gauss-Seidel does not make a 31-layer same-sweep credit path useful under the current stable coefficient scale.

## Hardware interpretation

Reverse Gauss-Seidel also removes the main layer-parallel advantage.  A reverse sweep has a true layer-to-layer dependency, so 31 layer updates must be ordered even if each layer has a dedicated MAC engine.  Calling that ordered chain one `T=1` iteration would therefore hide the actual latency rather than improve it.

Consequently there are now two distinct cases:

1. **Jacobi / synchronous PC-ALM:** preserves layer spatial parallelism, but measured credit support advances approximately one layer per relaxation step; depth 32 needs about 31 steps before the first layer can even receive credit.
2. **Reverse Gauss-Seidel / fused PC-ALM:** can transmit support through multiple layers inside one sweep, but serializes those layers and attenuates the same-sweep path approximately as `(eta_h |w| (rho+alpha))^d` in the scalar diagnostic.

Neither schedule by itself closes the latency gap to ePC/BP.  A useful new architecture must therefore do more than reorder the same nearest-neighbor equations: it needs either a multi-layer credit operator whose gain does not decay exponentially with distance, a hierarchical/skip credit path, or a block solve whose extra arithmetic and wiring are explicitly charged to the hardware model.

## Consequence for the next experiment

Do not spend a large sweep on `alpha`, `rho`, or `eta_h` solely to make reverse Gauss-Seidel cross 31 layers in one sweep.  To make the scalar same-sweep gain non-decaying would require `eta_h |w| (rho+alpha)` near one; with `eta_h=0.05` and `|w|=0.9`, that means `rho+alpha` near 22.2, far outside the current stable baseline scale and likely to change the dynamics qualitatively.

The higher-value next test is a **block/hierarchical credit path**: compare block sizes 1, 2, 4, 8 under an equal MAC budget, and measure both effective propagation distance and BP-gradient cosine.  This directly asks how much extra local connectivity is required to beat the one-hop causal wall without simply renaming a serial reverse sweep as one iteration.
