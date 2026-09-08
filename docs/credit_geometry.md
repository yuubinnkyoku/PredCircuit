# Local credit geometry in the FlyVis crop

This note records the current diagnostic evidence for why the local predictive-coding update stops scaling cleanly from the smaller FlyVis crop to `extent=2`.

## Question

For a cycle of the four motion directions, let the cycle-summed local predictive-coding credit be `L` and the exact descent direction for the same samples be `G`. The main diagnostic question is whether the failure is mostly an update-magnitude problem or a direction problem.

We decompose the local credit into the component parallel to exact credit and an orthogonal residual:

```text
L = L_parallel + L_perp
L_parallel = dot(L, G) / dot(G, G) * G
L_perp = L - L_parallel
```

The projection uses exact credit and is therefore a diagnostic oracle, not a candidate biological or hardware-local learning rule.

## Projection experiment

Common settings:

- FlyVis retinotopic crop: `extent=2`
- 4 motion directions
- 5 seeds
- 100 epochs
- `beta=0.03`
- learning rate `10`
- 7 frames, 2 inference steps per frame
- 8 test repeats per direction

| Training direction | Mean accuracy | Mean MSE | Mean margin |
| --- | ---: | ---: | ---: |
| local `L` | 30.000% | 0.079773 | 0.005461 |
| parallel `L_parallel` | 56.875% | 0.083066 | -0.000026 |
| parallel, rescaled to `||L||` | 48.750% | 0.082921 | -0.000747 |
| residual only `L_perp` | 25.625% | 0.085098 | -0.034094 |
| exact/oracle `G` | 58.750% | 0.083260 | 0.000001 |

Removing the orthogonal residual nearly doubles the gain above 25% chance accuracy. Rescaling the parallel component to the original local-credit norm still leaves a large advantage over the original local update, so the effect cannot be explained only by total update magnitude.

The residual by itself is approximately chance-level and produces a strongly negative mean classification margin.

## Residual attenuation

The diagnostic direction

```text
C(r) = L_parallel + r * L_perp
```

was trained with the same settings. `r=0` removes the residual and `r=1` recovers the original local credit.

| Residual scale `r` | Mean accuracy |
| ---: | ---: |
| 0.00 | 56.875% |
| 0.25 | 53.125% |
| 0.50 | 51.250% |
| 0.75 | 44.375% |
| 1.00 | 30.000% |

Accuracy falls almost monotonically as the orthogonal residual is restored. The current strongest mechanism hypothesis is therefore:

> The extent-2 bottleneck is dominated by a harmful component of cycle-summed local credit that is orthogonal to the exact descent direction.

This is stronger than the earlier explanations based only on finite nudge size, small updates, optimizer choice, or sequential processing of the four directions.

## Exact-credit interpolation

A separate diagnostic interpolates between local and exact credit:

```text
C(alpha) = (1 - alpha) * L + alpha * G
```

With 10 seeds and the same `extent=2`, `beta=0.03`, learning-rate-10 regime:

| Exact fraction `alpha` | Mean accuracy |
| ---: | ---: |
| 0.00 | 33.125% |
| 0.10 | 35.938% |
| 0.25 | 43.750% |
| 0.50 | 47.188% |
| 1.00 | 60.000% |

The monotonic accuracy trend is consistent with credit direction quality, rather than update size alone, being a limiting factor.

## Metric warning

MSE and directional classification accuracy do not agree in these diagnostics. For example, the raw local update can have lower MSE while having much worse direction accuracy than the projected or oracle update. The four T4 outputs are being used as direction scores, so MSE to the current hand-set target vector is not sufficient as the sole task metric.

New scaling experiments therefore also record softmax cross-entropy over the four T4 readout scores, alongside accuracy, margin, and MSE.

## Next falsification tests

Two experiments are now the priority:

1. Repeat residual attenuation with 10 paired seeds at `extent=1` and `extent=2`. If harmful orthogonal residual is specifically a scaling problem, restoring it should hurt `extent=2` substantially more than the smaller crop.
2. Replace oracle projection with a genuinely local filter. The first prototype is a per-synapse credit-coherence gate. For local credits `l_k,e` observed at one synapse over the four motion samples:

```text
q_e = |sum_k l_k,e| / (sum_k |l_k,e| + eps)
delta_w_e proportional to q_e^p * sum_k l_k,e
```

This requires only the synapse's own accumulated credit and absolute-credit accumulator. It does not use exact gradients or backward credit transport. `p=0` is the original cycle-summed local update.

The coherence gate is only promising if it improves paired-seed task performance without an oracle and remains useful when crop extent increases or topology is perturbed.
