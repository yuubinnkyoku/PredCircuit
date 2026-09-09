# Cancellation, conditioning, and the alignment-learning gap

This note records the current mechanism-level interpretation of the FlyVis classification experiments. The main result is no longer adequately described as "high gradient alignment but no learning." The local rule is operating in a regime where task gradients from different motion directions nearly cancel, the local signal is extremely small coordinate-wise, and the optimizer changes the geometry of that signal substantially.

## 1. Cancellation amplifies small per-direction errors

For four direction-specific exact descent vectors `g_c`, define the cancellation ratio

```text
kappa = ||sum_c g_c|| / sum_c ||g_c||
```

At initialization on the biological `extent=2` crop, this ratio is extremely small (roughly `1.6e-3` in the direction-balance diagnostic). Therefore a local approximation can be very accurate for every individual direction and still have a poor aggregate direction after the four large vectors cancel.

A degree-preserving rewire gives a substantially larger surviving aggregate signal. In the local branched-Adam control at learning rate `0.01`, the degree-rewired graph reaches mean cross-entropy `1.3480`, versus `1.3644` for the biological graph. The paired difference is about `-0.01645` with a 95% interval of approximately `[-0.01847, -0.01444]`.

This is not because the degree-rewired task is intrinsically easier. With the exact CE gradient at the same learning rate, the biological graph reaches roughly `0.601` CE while the degree-rewired graph is much worse at roughly `0.884`. The rewire therefore helps the current local rule specifically, consistent with topology changing the conditioning of locally available credit.

## 2. Exact-gradient noise control: amplification is real, but not sufficient

To isolate cancellation from all local-rule details, `scripts/run_flyvis_noisy_per_direction_ce_oracle.py` starts from the exact CE gradient for each direction and adds an orthogonal perturbation with a controlled per-direction cosine. Gradient norms are preserved before aggregation.

| Per-direction cosine | Mean aggregate cosine | Noise / exact aggregate norm | Mean CE after training | Mean accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1.0000 | 1.000 | 0.000 | 0.6020 | 100.00% |
| 0.9999 | 0.858 | 0.697 | 0.9622 | 100.00% |
| 0.9990 | 0.546 | 3.095 | 1.2556 | 100.00% |
| 0.9950 | 0.230 | 9.209 | 1.3354 | 98.75% |
| 0.9900 | 0.132 | 15.003 | 1.3598 | 95.63% |
| 0.9800 | 0.071 | 24.499 | 1.3750 | 84.38% |

This is direct causal evidence that near-cancellation can turn a tiny individual angular error into a much larger aggregate error. In particular, a per-direction cosine of `0.995` is not "almost exact" in the aggregate geometry of this task.

However, the same experiment also falsifies a stronger version of the cancellation hypothesis. Random orthogonal errors large enough to reduce aggregate cosine to about `0.23` still permit nearly perfect classification accuracy. Therefore cancellation alone does **not** explain why the real local rule can remain near chance.

The remaining difference is likely the **structure** of the local residual: which coordinates it occupies, whether its signs are persistent across epochs, whether errors are correlated between task directions, and how those errors interact with recurrent-state changes and optimizer moments.

## 3. Nudge depth exposes a signal-strength / direction-quality tradeoff

At `beta=0.03`, increasing the number of matched free/nudged branch steps increases the component of local credit parallel to the exact CE descent, but eventually degrades angular accuracy.

The 20-seed initialization diagnostic and 10-seed, 100-epoch Adam training give:

| Nudge steps | Raw cosine | Projection coefficient `alpha` | Adam(local) vs Adam(exact) cosine | Coordinates with `|local| > 10 eps` | Mean CE after | Mean accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.780 | 0.0477 | 0.397 | 4.86% | 1.3833 | 41.25% |
| 2 | 0.776 | 0.0934 | 0.525 | 7.94% | 1.3644 | 41.56% |
| 4 | 0.755 | 0.1793 | 0.617 | 13.72% | 1.3540 | 28.13% |
| 8 | 0.688 | 0.3319 | 0.586 | 26.90% | 1.3242 | 47.50% |
| 16 | 0.542 | 0.5774 | 0.426 | 48.86% | **1.3193** | **73.75%** |
| 32 | 0.363 | 0.9181 | 0.216 | 73.51% | 1.4001 | 28.75% |

There is a genuine optimum near 16 branch steps under the current optimizer. Short branches have cleaner direction but extremely weak coordinate-wise credit. Very long branches produce a much larger signal but accumulate nonlinear/state-dependent direction error and fail badly.

The optimum is therefore not a maximum of raw cosine, gradient norm, or first-step Adam alignment alone.

## 4. Adam operates inside the local-credit scale

The coordinate diagnostic `scripts/diagnose_flyvis_coordinate_snr.py` shows that for the default `nudge_steps=2`, `beta=0.03`, and `Adam epsilon=1e-8`:

- raw local/exact cosine is about `0.776`;
- `Adam(local)` versus `Adam(exact)` cosine is only about `0.525`;
- about `76.9%` of local-credit coordinates have magnitude at or below `epsilon`;
- about `92.1%` are at or below `10 * epsilon`;
- the median absolute local-credit coordinate is only about `1.25e-9`.

Thus Adam is not merely removing a harmless global scale factor. Its epsilon and per-coordinate normalization act directly on the bulk of the local-credit distribution.

At larger nudge depth, more coordinates escape the epsilon-dominated regime, but optimizer-space direction quality eventually collapses. This supplies a concrete conditioning mechanism for the observed intermediate optimum.

## 5. Current causal tests

Two tests now have the highest value.

### Adam-epsilon shift

If the nudge-depth optimum is partly set by the scale of local credit relative to Adam epsilon, then changing epsilon should move the optimum:

- lowering epsilon should allow shallower nudges to become useful earlier;
- raising epsilon should favor stronger/deeper nudges until nonlinear direction error dominates.

The workflow `.github/workflows/flyvis-branched-adam-epsilon.yml` tests `epsilon = 1e-9` and `1e-7` across nudge depths `2, 4, 8, 16, 32`, with the existing `1e-8` sweep as the center condition.

### Matched residual-structure control

The noisy exact-gradient experiment uses random orthogonal error and still learns surprisingly well. The next diagnostic should compare the actual local residual with synthetic residuals that have the same norm and aggregate cosine but different coordinate structure. A particularly useful comparison is

```text
L = alpha G + R_local
L_random = alpha G + R_random
```

with `R_random` orthogonal to `G` and norm-matched to `R_local`.

If `L_random` learns substantially better than `L` under the same optimizer, then the missing mechanism is not residual magnitude alone; it is structured, persistent local-credit error.

## Current interpretation

The strongest statement supported by the present experiments is:

> Biological recurrent topology changes both cancellation and conditioning of locally available credit. Near-cancellation strongly amplifies local approximation error, but the failure of the current local rule additionally depends on signal scale, optimizer coordinate conditioning, nonlinear nudge dynamics, and likely structured residual error across coordinates and training time.

This is stronger and more specific than either "biological topology is better" or "high gradient cosine should imply learnability." It also defines concrete falsification experiments rather than treating gradient alignment as an endpoint.
