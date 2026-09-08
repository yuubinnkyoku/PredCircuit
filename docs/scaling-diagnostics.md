# Local-credit scaling diagnostics

This note isolates why the online two-phase predictive-coding rule that learns on the extent-1 FlyVis crop does not automatically scale to extent 2. It is a mechanistic pilot, not a biological claim.

## Question

The extent-1 crop has 443 nodes and 7,698 directed edges. Extent 2 has 1,203 nodes and 26,636 edges. Exact gradients through the same predictive-coding dynamics solve both sizes, but the online local rule with a fixed scalar learning rate learns at extent 1 and remains near chance at extent 2.

Three simple explanations were separated:

1. the local direction becomes inaccurate as the circuit grows;
2. the direction remains useful but its per-edge magnitude shrinks;
3. coordinate-wise adaptive normalization restores magnitude but rotates an approximate local direction away from the exact descent direction.

The diagnostic tracks a fixed set of individual motion examples at epochs 0, 1, 5, 10, 25, 50, and 100 during the actual online training trajectory. `raw_cosine` compares the local edge direction with the exact same-PC descent direction. `effective_cosine` compares the optimizer's actual proposed edge step with that exact direction.

## 1. Raw local credit remains well aligned at extent 2

With scalar SGD-style local updates (`learning_rate=10`), the optimizer cannot change the direction, so raw and effective cosine are identical.

### Biological wiring

| Epoch | Extent-1 cosine | Extent-2 cosine | Extent-1 mean |local edge| | Extent-2 mean |local edge| |
|---:|---:|---:|---:|---:|
| 0 | 0.906 | 0.912 | 3e-6 | 0.92e-6 |
| 10 | 0.975 | 0.981 | 5e-6 | 1.55e-6 |
| 25 | 0.932 | 0.943 | 5e-6 | 1.43e-6 |
| 50 | 0.844 | 0.904 | 7e-6 | 1.42e-6 |
| 100 | 0.773 | 0.883 | 10e-6 | 1.25e-6 |

The key negative result is that **extent-2 raw local credit does not collapse directionally**. It remains highly aligned with the exact gradient throughout training and is, if anything, more aligned than extent 1 late in this diagnostic.

The corresponding exact-gradient signal is also smaller at extent 2. At epoch 100, the mean absolute exact edge gradient is about `1.40e-6` at extent 2 versus about `1.0e-5` at extent 1. Thus the larger circuit changes the natural per-edge scale of both the true and approximate credit signals.

The raw local/exact L2 norm ratio remains near one at both scales: for biological wiring it is 1.27 at initialization and 0.93 at epoch 100 for extent 2. The approximation is therefore not simply vanishing relative to the oracle; rather, **the whole per-edge credit scale becomes smaller as the circuit expands**.

Despite the high cosine, extent-2 scalar-LR-10 training remains at 25% biological accuracy and MSE 0.08341 at epoch 100 in this three-seed trajectory diagnostic. High instantaneous alignment is therefore necessary but not sufficient for useful accumulated learning at a fixed global step scale.

## 2. Per-synapse Adam is the wrong normalization for this approximate direction

Per-coordinate Adam was tested because it can increase tiny local updates without requiring one very large global learning rate. It does increase applied per-edge magnitude, but it badly rotates the update away from the exact descent direction.

For biological wiring at extent 1 with local Adam (`lr=0.001`):

| Epoch | Raw cosine | Effective Adam cosine | Mean |effective edge step| |
|---:|---:|---:|---:|
| 0 | 0.906 | 0.132 | 7.90e-4 |
| 10 | 0.926 | 0.066 | 1.31e-4 |
| 25 | 0.936 | 0.073 | 1.25e-4 |
| 50 | 0.947 | 0.079 | 1.32e-4 |
| 100 | 0.954 | 0.076 | 1.60e-4 |

At extent 2 the same effect is even clearer:

| Epoch | Raw cosine | Effective Adam cosine | Mean |effective edge step| |
|---:|---:|---:|---:|
| 0 | 0.912 | 0.097 | 7.02e-4 |
| 10 | 0.923 | 0.049 | 1.20e-4 |
| 25 | 0.928 | 0.053 | 1.20e-4 |
| 50 | 0.932 | 0.064 | 1.28e-4 |
| 100 | 0.941 | 0.068 | 1.57e-4 |

So Adam succeeds at the narrow engineering goal of producing edge updates around `1e-4`, but those effective updates have cosine only about 0.04-0.13 with the exact descent direction. The raw local signal still has cosine around 0.9.

This explains why the extent-2 Adam sweep does not rescue learning despite restoring a seemingly favorable update magnitude. Coordinate-wise normalization treats the residual error in the approximate local gradient as if every coordinate were an independently trustworthy gradient estimate. The resulting diagonal preconditioner can strongly amplify low-magnitude/noisy coordinates and rotate the full vector.

This is also visible at extent 1: raw local direction remains highly aligned under the Adam training trajectory, yet Adam training does not reproduce the useful extent-1 scalar-SGD behavior. The failure is therefore not evidence against magnitude adaptation in general; it is evidence against **coordinate-wise** magnitude adaptation for this approximate local credit signal.

## 3. Revised scaling hypothesis

The earlier hypothesis “extent 2 fails because the local gradient points in the wrong direction” is rejected by this diagnostic. The hypothesis “just use per-synapse Adam to restore magnitude” is also rejected.

The remaining clean test is a **single scalar normalization** of the complete local update. One scalar can restore a chosen mean edge-step magnitude while preserving the local vector direction exactly, including the relative pattern across synapses. This distinguishes magnitude dilution from coordinate-wise preconditioning.

A direction-preserving scalar-normalized experiment is implemented in `scripts/run_flyvis_temporal_scalar_normalized.py`. It chooses a scalar from the current mean absolute edge direction and applies that same scalar to the edge and bias directions. The extent-2 sweep tests target mean absolute edge updates of `3e-5`, `1e-4`, and `3e-4`.

If this rescues extent 2, the scaling bottleneck is primarily a scalar credit-scale problem and suggests size/degree-normalized local step rules. If it fails while raw per-example cosine remains high, the next mechanism to test is **interference across sequential examples**: individually good local directions may accumulate differently from exact gradients over a complete four-direction update cycle.
