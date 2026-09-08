# Local credit geometry in the FlyVis crop

This note tracks the current mechanism-level evidence about the local predictive-coding credit used by PredCircuit. The original working hypothesis was that the main failure appeared only when scaling from the smaller FlyVis crop to `extent=2`. The 10-seed residual experiment falsified that narrow explanation: a harmful non-oracle component is already present at `extent=1`.

## Question

For one cycle of the four motion directions, let the cycle-summed local predictive-coding credit be `L` and the exact BPTT descent direction for the same samples be `G`. Decompose

```text
L = L_parallel + L_perp
L_parallel = dot(L, G) / dot(G, G) * G
L_perp = L - L_parallel
```

The projection uses exact credit and is therefore a diagnostic oracle, not a candidate biological or hardware-local learning rule. The scientific question is whether a practical local rule can suppress the harmful part without access to `G`.

## Oracle projection pilot

On the `extent=2` crop, 5 seeds, 100 epochs, `beta=0.03`, learning rate 10:

| Training direction | Mean accuracy | Mean MSE | Mean margin |
| --- | ---: | ---: | ---: |
| local `L` | 30.000% | 0.079773 | 0.005461 |
| parallel `L_parallel` | 56.875% | 0.083066 | -0.000026 |
| parallel, rescaled to `||L||` | 48.750% | 0.082921 | -0.000747 |
| residual only `L_perp` | 25.625% | 0.085098 | -0.034094 |
| exact/oracle `G` | 58.750% | 0.083260 | 0.000001 |

Removing the orthogonal residual gives a large accuracy recovery. Rescaling the parallel component to the original local-credit norm still beats the original local update, so update magnitude alone cannot explain this pilot.

## Residual attenuation: 10-seed replication

Train with

```text
C(r) = L_parallel + r * L_perp
```

where `r=0` removes the residual and `r=1` recovers the original local credit. Both crop sizes use 10 paired seeds, 100 epochs and the same temporal task.

| Residual scale `r` | extent=1 accuracy | extent=2 accuracy |
| ---: | ---: | ---: |
| 0.00 | **51.875%** | **50.000%** |
| 0.25 | 43.438% | 49.375% |
| 0.50 | 40.000% | 49.063% |
| 0.75 | 40.000% | 41.563% |
| 1.00 | **31.250%** | **33.125%** |

The `r=0` versus `r=1` improvement is +20.625 percentage points at `extent=1` and +16.875 points at `extent=2`. Therefore the harmful residual is not specifically an `extent=2` scaling pathology. The stronger current statement is:

> The present local temporal PC rule contains a generally harmful credit component relative to the exact task descent direction.

The remaining scale question is whether the *composition* or source of that residual changes with circuit size, rather than whether it first appears at large extent.

## Hierarchical topology nulls

To locate which part of biological wiring changes credit geometry, the diagnostic now compares four graphs:

- `biological`: the published FlyVis retinotopic crop;
- `type_pair_rewire`: exact in/out degrees and source/target cell-type pairs are preserved while fine partner identity is rewired;
- `pair_rotation`: each cell-type pair keeps its internal spatial offset pattern, but different type pairs are independently rotated by multiples of 60 degrees, breaking cross-type orientation alignment;
- `degree_rewire`: only global directed node degrees are preserved, destroying cell-type and retinotopic organization. This is a deliberately coarse null and changes the dynamical scale substantially.

Twenty seeds with five diagnostic cycles per seed give:

| extent | topology | Mean local/oracle cosine | Mean residual fraction |
| ---: | --- | ---: | ---: |
| 1 | biological | 0.405 | 0.877 |
| 1 | type-pair rewire | 0.348 | 0.908 |
| 1 | pair rotation | 0.267 | 0.917 |
| 1 | global degree rewire | 0.532 | 0.829 |
| 2 | biological | 0.409 | 0.869 |
| 2 | type-pair rewire | 0.431 | 0.882 |
| 2 | pair rotation | 0.219 | 0.941 |
| 2 | global degree rewire | 0.617 | 0.774 |

The cleanest positive result is the pair-rotation control. Biological minus pair-rotation cosine is +0.137 at `extent=1` and +0.191 at `extent=2`. A paired post-hoc sign-flip analysis over the 20 seed means gives approximately `p=0.022` and `p=0.003`, respectively; bootstrap 95% intervals for the mean differences are approximately `[0.035, 0.256]` and `[0.085, 0.299]`. The corresponding biological-minus-null residual-fraction differences are -0.040 and -0.072.

Fine partner rewiring is much weaker and inconsistent across extents. This points away from individual synapse partner identity as the only explanation and toward **alignment of spatial motifs across cell-type pathways** as one structural ingredient that helps local credit geometry.

The global degree-rewired graph has even higher cosine than the biological graph, but it also changes local and oracle credit norms by roughly an order of magnitude. It is therefore not evidence that random wiring is biologically superior. It is evidence that global degree preservation is too weak a null for a clean credit-geometry comparison unless dynamical scale is also controlled.

## Oracle-free coherence gate

A candidate hardware-local filter accumulates, for each synapse `e`, its signed and absolute local credits over the four task samples:

```text
q_e = |sum_k l_k,e| / (sum_k |l_k,e| + eps)
delta_w_e proportional to q_e^p * sum_k l_k,e
```

No exact gradient or backward credit transport is used. The state needed at an edge is only its running signed credit and absolute-credit accumulator.

The 10-seed `extent=2` sweep shows that the first 5-seed improvement was not universal:

| LR | p=0 | p=0.25 | p=0.5 | p=0.75 | p=1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | **36.250%** | 31.563% | 33.750% | 31.563% | 20.000% |
| 30 | 29.375% | 31.875% | 34.375% | **35.625%** | 32.813% |
| 40 | 27.188% | 27.813% | 31.875% | 35.000% | **35.313%** |

At low LR the gate hurts, while at higher LR stronger gating helps. This creates an important confound: the gate may simply reduce effective update magnitude rather than selectively remove bad credit.

The next control therefore rescales each gated cycle back to the ungated mean-absolute credit magnitude before applying the global learning rate. If the gate still helps, its per-synapse reweighting contains information beyond a smaller effective step size. That norm-matched experiment is implemented in `scripts/run_flyvis_temporal_coherence_norm_control.py`.

## Objective mismatch

The exact oracle used in the current geometry diagnostics minimizes MSE to a hand-set four-output target vector, while the main behavioral metric is four-way direction classification. MSE, cross-entropy, margin and accuracy can disagree strongly.

A second new experiment therefore replaces the squared output nudge with the exact cross-entropy force on the four T4 output activities:

```text
dL_class / dz = softmax(z) - one_hot(y)
```

Only the output population receives this supervised force. Hidden-state teaching still propagates through the recurrent predictive-coding dynamics, and synaptic learning remains free-vs-nudged and local. The implementation is `scripts/run_flyvis_temporal_classification_nudge.py`.

## Current falsification targets

The highest-value tests are now:

1. **Norm-matched coherence:** does coherence filtering beat an ungated update when total credit magnitude is held fixed?
2. **Classification-aligned nudge:** does matching the output teaching force to the actual classification objective improve accuracy, cross-entropy, or local/exact credit alignment?
3. **Controlled coarse topology null:** can the apparent advantage of the global degree-rewire be removed by matching dynamical scale, cell-type block statistics, or both?
4. **Mechanistic localization:** which cell-type pairs or cross-type spatial alignments contribute most to the biological-versus-pair-rotation credit difference?

These tests distinguish a generic optimizer/step-size effect from a genuinely local credit-assignment mechanism tied to connectome structure.
