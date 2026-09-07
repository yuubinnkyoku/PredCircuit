# Pilot results

This file records early results that changed the experimental design. These are pilot results, not biological conclusions.

## 1. Synthetic topology-control pilot

Task: a two-input linear mapping on a sparse directed graph with layers `[2, 8, 1]`. Each topology was trained for 200 epochs with 10 paired seeds under predictive coding (PC) and a same-topology BPTT baseline.

### Final MSE

| Learning rule | Topology | Mean final MSE |
|---|---|---:|
| PC | biological/base layered graph | 0.156195 |
| PC | degree-preserving rewire | 0.024413 |
| PC | rank-pair-preserving rewire | 0.153307 |
| PC | ER matched | 0.099621 |
| PC | no feedback | 0.168751 |
| PC | no reciprocal edges | 0.183692 |
| BPTT | biological/base layered graph | 1.91e-10 |
| BPTT | degree-preserving rewire | 2.28e-10 |
| BPTT | rank-pair-preserving rewire | 1.61e-10 |
| BPTT | ER matched | 0.056250 |
| BPTT | no feedback | 2.67e-10 |
| BPTT | no reciprocal edges | 1.38e-10 |

The large apparent PC advantage of the ordinary degree-preserving rewire initially looked interesting. It was also suspicious: the rank-pair-preserving rewire, which keeps the layer-to-layer edge counts, was almost identical to the base graph.

Paired PC comparison across the 10 seeds:

- base vs ordinary degree-preserving rewire: ordinary rewire had lower MSE in 10/10 seeds; Wilcoxon p = 0.00195;
- base vs rank-pair-preserving rewire: mean difference = 0.00289; Wilcoxon p = 0.08398;
- base vs no-feedback: no clear pilot-scale difference.

The correct question became: did the ordinary rewire accidentally make the supervised task easier?

## 2. The ordinary degree-preserving null creates shortcut paths

We measured directed input-to-output path structure over 1,000 independently generated synthetic graphs.

| Topology | Reachable input-output pairs | Direct-edge fraction | Shortest path |
|---|---:|---:|---:|
| base | 1.000 | 0.000 | 2.0 always |
| degree-preserving rewire | 1.000 | **0.975 mean** | **1.0 always** |
| rank-pair-preserving rewire | 1.000 | 0.000 | 2.0 always |
| no feedback | 1.000 | 0.000 | 2.0 always |
| no reciprocal edges | 1.000 | 0.000 | 2.0 always |
| ER matched | 0.966 mean | 0.348 mean | variable / sometimes unreachable |

For the ordinary degree-preserving rewire, the median direct-edge fraction was 1.0 and the mean input-output shortest path was 1.025. In other words, a null model that exactly preserved every node's in/out degree nevertheless created direct task shortcuts in essentially every graph.

### Design decision

An unconstrained degree-preserving rewire is **not a valid primary null** for a task whose input/output semantics are tied to circuit position. It remains useful as a deliberately destructive control.

The primary topology null must preserve the relevant coarse circuit organization. In the current code this is `rank_pair_preserving_rewire`, which preserves:

- every node's in-degree and out-degree;
- every `(source rank, target rank)` edge count;
- therefore feedforward/lateral/feedback edge counts;
- the weight distribution inside each rank pair when measured strengths are present.

Future experiments must also report input-output reachability and shortest-path distributions so topology randomization cannot silently change task difficulty.

## 3. FlyVis cell-type-level real-connectome pilot

To move beyond synthetic graphs without immediately requiring the full MaleCNS dataset, we used the published FlyVis connectome scaffold pinned at commit `92b3845cc426dd309a1a0e1b3890156c42e14021`.

The FlyVis retinotopic filters were collapsed to a cell-type-level graph by summing synapse counts over spatial offsets. The parsed graph contains:

- 65 cell-type nodes;
- 605 directed type-level edges;
- 8 input types;
- 34 output types.

This collapse intentionally discards retinotopy. It can test graph/inference conditioning, but it cannot support claims about motion vision or other retinotopic computations.

We clamped random sensory inputs and ran 100 predictive-coding inference steps over 20 seeds. The main quantity here is `energy_ratio = final_energy / initial_energy`; smaller means that the inference procedure descended farther in that model's own energy landscape. It is **not** task accuracy and energies from different weight configurations are not a common objective.

### Step-size sweep

Mean energy ratio over 20 seeds:

| Step size | Biological topology, random weights | Degree rewire, random weights | Rank-pair rewire, random weights | Biological measured strengths | Weight-shuffled strengths | Rank-pair rewire + strengths |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 0.9141 | 0.9099 | 0.9101 | 0.8764 | 0.7919 | 0.8157 |
| 0.02 | 0.9016 | 0.8949 | 0.8971 | 0.8563 | 0.7640 | 0.7881 |
| 0.03 | 0.8980 | 0.8897 | 0.8932 | 0.8424 | 0.7540 | 0.7769 |
| 0.05 | 0.8958 | 0.8860 | 0.8908 | 0.8221 | 0.7455 | 0.7664 |
| 0.08 | 0.8950 | 0.8847 | 0.8899 | 0.8015 | 0.7401 | 0.7591 |

All configurations had monotonically non-increasing recorded energy at every tested step size.

For topology-only comparisons, the biological graph was not clearly separated from either rewired control at any tested step size. At step size 0.03, for example, paired Wilcoxon p values were approximately 0.177 versus the ordinary degree rewire and 0.154 versus the rank-pair rewire.

When measured signed synapse strengths were used, the exact biological assignment descended its own energy more slowly than shuffled/rewired controls. At step size 0.03:

- biological measured strengths: mean ratio 0.8424;
- global weight shuffle: 0.7540, Wilcoxon p = 8.2e-05 versus biological;
- rank-pair rewire: 0.7769, lower than biological in 20/20 seeds, Wilcoxon p = 1.91e-06.

The direction persisted across all five tested step sizes. This is a robust **conditioning observation**, but it is not evidence that biological weights are worse: the weight manipulations change the energy landscape itself, and no behavioral/prediction target is being evaluated here.

## 4. What these pilots changed

The synthetic pilot produced a useful negative result: a seemingly strong topology effect disappeared once task-path shortcuts were controlled. This strengthens, rather than weakens, the experimental design.

The FlyVis pilot shows that the implementation can ingest a published biological scaffold and perform stable PC inference, but static cell-type-level energy minimization is not a sufficient biological benchmark.

The next scientifically meaningful target was therefore to preserve retinotopy and introduce a temporal motion task. Sections below report the first experiments on that target.

## 5. Retinotopic FlyVis motion task

The retinotopic pilot uses an extent-1 FlyVis circuit with 443 nodes and 7,698 directed edges. Moving bars are presented as short temporal sequences. The supervised readout is the central T4a/T4b/T4c/T4d group, evaluated as a four-way direction discrimination task.

The primary destructive-but-degree-matched null is `type_pair_preserving_rewire`. It preserves the cell-type pair of every edge, edge count, node degree, and the edge-slot strength distribution, but destroys fine retinotopic partner identity.

A second, more constrained null is `pair_rotation`: the spatial offset pattern for each cell-type pair is independently rotated by multiples of 60 degrees. This preserves each pair's local receptive-field geometry and edge count while scrambling the alignment of those geometries between type pairs.

Unless explicitly stated otherwise, the results below use random initial trainable weights. They therefore test biological **wiring topology**, not the measured FlyVis synapse-strength assignment.

## 6. Same predictive-coding dynamics are learnable with exact gradients

A useful control is to keep the predictive-coding state dynamics and sparse FlyVis graph fixed while differentiating the final supervised loss exactly through those dynamics.

With Adam at learning rate 0.01 for 100 epochs over 10 paired seeds:

| Topology | Mean final MSE | Mean accuracy | Mean margin |
|---|---:|---:|---:|
| biological | **0.023788** | **1.000** | **0.295149** |
| type-pair rewire | 0.038318 | **1.000** | 0.184927 |

All 10 seeds reached 100% direction accuracy. Thus failure of the early local rules was not caused by an unlearnable task or by insufficient representational capacity of the predictive-coding circuit. The biological topology also converged to a lower supervised error under the same exact-gradient optimizer in this pilot.

Plain exact-gradient SGD at moderate learning rates did not move appreciably because the exact gradients are small. This became important when interpreting local-learning step sizes.

## 7. Local-gradient alignment depends strongly on batching

For a single motion example, the temporal two-phase local update is surprisingly well aligned with the exact same-PC gradient. The edge-direction cosine similarity is approximately 0.885 on the biological topology and 0.929 on the type-pair rewire.

However, after averaging a balanced batch containing opposing motion directions, combined local-vs-exact alignment falls to roughly 0.25-0.36 through training. The main failure is in edge plasticity, while bias-gradient alignment remains much higher.

The diagnostic indicates that exact supervised edge gradients from opposing directions cancel much more strongly than the residual errors of the local approximation. Therefore, a balanced batch can amplify approximation error relative to the true batch direction even when the per-example local rule is well aligned.

This changed the main local-learning protocol from balanced-batch training to online/stochastic updates.

## 8. Online temporal local PC learns when the update scale is large enough

Earlier online experiments used learning rates that were far too small for the observed local-statistic magnitude. A sweep over 0.1, 0.3, 1, 3, and 10 showed the first clear learning at 3 and a much stronger effect at 10.

A 20-seed paired replication gave:

| Local-PC learning rate | Topology | Mean final MSE | Mean accuracy | Mean MSE improvement |
|---:|---|---:|---:|---:|
| 3 | biological | **0.082976** | **0.3313** | **0.000358** |
| 3 | type-pair rewire | 0.083301 | 0.2547 | 0.000032 |
| 10 | biological | **0.077390** | **0.4297** | **0.005945** |
| 10 | type-pair rewire | 0.082608 | 0.2766 | 0.000725 |

For learning rate 10, the biological circuit had lower MSE in 17/20 paired seeds. The paired mean difference `biological - rewire` was -0.005218, with a bootstrap 95% interval of approximately [-0.00717, -0.00310]. A two-sided Wilcoxon signed-rank test gave p = 0.000261. Accuracy was higher by 0.1531 on average (15.3 percentage points), with a bootstrap 95% interval of approximately [0.0672, 0.2422] and two-sided Wilcoxon p = 0.00742.

At learning rate 3 the effect is smaller but points in the same direction: the biological circuit had lower MSE in 17/20 seeds, with two-sided Wilcoxon p = 0.000134; the paired accuracy difference was +0.0766 with p = 0.00321.

The accuracy result is not identical to a positive class margin: at learning rate 10 the mean margin remained slightly negative for both topologies and was highly variable. The robust result at this stage is therefore lower MSE and higher finite-sample direction accuracy, not solved classification.

## 9. The topology effect survives a stronger retinotopic null

At learning rate 10, 10 paired seeds were compared across biological wiring, `pair_rotation`, and `type_pair_preserving_rewire`.

| Topology | Mean final MSE | Mean accuracy | Mean MSE improvement |
|---|---:|---:|---:|
| biological | **0.076989** | **0.4500** | **0.006342** |
| pair rotation | 0.083482 | 0.2531 | -0.000148 |
| type-pair rewire | 0.083307 | 0.2438 | 0.000026 |

The biological circuit had lower MSE than `pair_rotation` in 10/10 seeds. The paired mean MSE difference was -0.006493, bootstrap 95% interval approximately [-0.00887, -0.00402], with two-sided Wilcoxon p = 0.00195. Accuracy was higher by 0.1969 on average, p = 0.00781.

Against the type-pair rewire, biological wiring had lower MSE in 9/10 seeds; two-sided Wilcoxon p = 0.00586. Mean accuracy was higher by 0.2063, p = 0.0156.

Because `pair_rotation` keeps edge count exactly matched and preserves each cell-type pair's local offset geometry while scrambling alignment across pairs, this result argues against the effect being explained only by degree, type-pair counts, or the existence of local receptive fields. The remaining hypothesis is that **cross-type alignment of retinotopic wiring** matters for effective online local credit assignment. This is still a small-circuit pilot and requires scale/generalization controls.

## 10. Edge and bias plasticity are not interchangeable

At learning rate 10, a 10-seed ablation separated trainable edge weights and node biases.

| Topology | Plasticity | Mean final MSE | Mean accuracy |
|---|---|---:|---:|
| biological | full edge+bias | **0.076989** | **0.4500** |
| biological | edges only | 0.082192 | 0.3688 |
| biological | bias only | 0.083763 | 0.2500 |
| type-pair rewire | full edge+bias | 0.083307 | 0.2438 |
| type-pair rewire | edges only | 0.083619 | 0.3563 |
| type-pair rewire | bias only | 0.083761 | 0.2500 |

On the biological topology, edge plasticity alone is enough to produce measurable learning relative to bias-only training: MSE 0.082192 versus 0.083763 (two-sided Wilcoxon p = 0.00195), and accuracy 0.3688 versus 0.25 (p = 0.0156). Adding bias plasticity to edge plasticity further reduces MSE to 0.076989 (full vs edges-only p = 0.00977), although the corresponding accuracy difference is not significant in this 10-seed pilot (p = 0.219).

The rewired topology behaves differently: edges-only reaches higher mean finite-sample accuracy than the full update, while neither condition shows the same MSE learning seen in the biological graph. This interaction is exploratory, but it suggests that the topology effect is not simply a generic benefit from updating more parameters.

## 11. Current interpretation

The strongest result so far is narrower than “biological wiring is better.” It is:

> In this small retinotopic FlyVis motion task, a two-phase online predictive-coding update learns substantially more on the biological wiring than on controls that preserve cell types, degree, edge count, and even within-type-pair receptive-field geometry.

The evidence now separates several possibilities that were previously confounded:

- the task and PC dynamics are learnable, because exact gradients reach 100%;
- the local rule is not random, because per-example alignment with the exact gradient is high;
- balanced batching is harmful because gradient cancellation exposes local-approximation residuals;
- sufficient online update scale allows the local rule to learn;
- the advantage persists under a rotated-retinotopy null, implicating finer cross-type spatial organization;
- edge plasticity is necessary for the biological topology's effect, while bias plasticity appears synergistic for MSE.

Important remaining controls are measured connectome-strength initialization, larger retinotopic extents, multiple temporal tasks/stimulus families, training-length/learning-rate robustness, and a direct hardware-locality cost model. The current results should remain labeled as pilot evidence until those controls are complete.
