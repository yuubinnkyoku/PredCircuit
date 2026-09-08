# FlyVis temporal predictive-coding findings

This note collects the current retinotopic/temporal results in one place. They are pilot results and should not be read as a biological-learning claim without the controls listed below.

## Experimental system

The task uses a finite retinotopic crop of the published FlyVis circuit. Moving Gaussian bars are presented as short sequences. The central T4a/T4b/T4c/T4d cells are evaluated as a four-way direction readout.

The main extent-1 crop contains 443 nodes and 7,698 directed edges. Unless stated otherwise, trainable edge weights are randomly initialized and the graph supplies wiring structure rather than physiological synaptic efficacy.

The local learning protocol is a two-phase online predictive-coding update. A free trajectory and a softly nudged trajectory are run through the same recurrent predictive-coding dynamics; every edge is updated from the difference of endpoint-local prediction-error statistics. No autograd graph or BPTT is used for that update.

Two topology controls are important:

- `type_pair_preserving_rewire` preserves source/target cell types, every node's in/out degree, edge count, and edge-slot strengths while destroying fine retinotopic partner identity;
- `pair_rotation` preserves the spatial offset pattern within each cell-type pair but independently rotates those patterns by multiples of 60 degrees, destroying cross-type retinotopic alignment while retaining local receptive-field geometry.

## 1. Online local PC shows an extent-1 topology effect

A 20-seed paired replication gave:

| Local-PC learning rate | Topology | Mean final MSE | Mean accuracy | Mean MSE improvement |
|---:|---|---:|---:|---:|
| 3 | biological | **0.082976** | **0.3313** | **0.000358** |
| 3 | type-pair rewire | 0.083301 | 0.2547 | 0.000032 |
| 10 | biological | **0.077390** | **0.4297** | **0.005945** |
| 10 | type-pair rewire | 0.082608 | 0.2766 | 0.000725 |

At learning rate 10, biological wiring had lower MSE in 17/20 paired seeds. The paired mean difference `biological - rewire` was -0.005218, with bootstrap 95% interval approximately [-0.00717, -0.00310] and two-sided Wilcoxon p = 0.000261. Mean accuracy was 15.3 percentage points higher, with p = 0.00742.

At learning rate 3 the effect was smaller but pointed in the same direction: biological wiring had lower MSE in 17/20 seeds, p = 0.000134, and mean accuracy was 7.66 percentage points higher, p = 0.00321.

## 2. The effect survives a stronger retinotopic null

At learning rate 10 over 10 paired seeds:

| Topology | Mean final MSE | Mean accuracy | Mean MSE improvement |
|---|---:|---:|---:|
| biological | **0.076989** | **0.4500** | **0.006342** |
| pair rotation | 0.083482 | 0.2531 | -0.000148 |
| type-pair rewire | 0.083307 | 0.2438 | 0.000026 |

Biological wiring beat `pair_rotation` in MSE in 10/10 seeds. The paired MSE difference was -0.006493, bootstrap 95% interval approximately [-0.00887, -0.00402], Wilcoxon p = 0.00195. Accuracy was higher by 19.69 percentage points on average, p = 0.00781.

This narrows the structural hypothesis: the result cannot be explained only by cell-type adjacency, edge count, node degree, or the existence of local receptive fields. The remaining candidate is the *alignment of those receptive-field geometries across cell types*.

## 3. The local-PC advantage is strongly target-alignment dependent

The canonical direction-to-T4 class assignment is `(1, 2, 0, 3)` for directions `(0, 90, 180, 270)` degrees. We cyclically shifted the target classes while keeping stimuli, graph, optimizer budget, and local rule fixed.

| Target shift | Biological MSE | Rewire MSE | Biological accuracy | Rewire accuracy |
|---:|---:|---:|---:|---:|
| 0, canonical | **0.076989** | 0.083307 | **0.4500** | 0.2438 |
| 1 | 0.082131 | 0.083898 | 0.2375 | 0.2594 |
| 2 | 0.086132 | **0.085871** | 0.1438 | **0.2563** |
| 3 | 0.082908 | 0.083781 | 0.2031 | 0.2281 |

For canonical targets, the paired MSE advantage `rewire - biological` was +0.006318 and biological wiring won in 9/10 seeds (Wilcoxon p = 0.00391). For shifts 1, 2, and 3, the corresponding paired tests were not significant.

The interaction contrast

`canonical topology advantage - mean(noncanonical topology advantages)`

was +0.005525 in MSE advantage, positive in 9/10 seeds, with Wilcoxon p = 0.00977, exact sign-flip p = 0.00195, and bootstrap 95% interval approximately [0.00212, 0.00979]. The corresponding accuracy interaction was +25.94 percentage points, positive in 10/10 seeds, with p = 0.00195 and bootstrap interval approximately [18.44, 33.28] percentage points.

This is currently one of the most informative controls. It says the extent-1 biological advantage is not a generic preference for any arbitrary four-way target mapping. However, it does **not** yet prove a local-credit-specific effect, because the biological topology may simply encode a task prior that helps any optimizer on the canonical mapping. A matched online exact-gradient/BPTT target-shift experiment is therefore required.

## 4. Exact gradients show that topology is also a generic task prior

Using the same predictive-coding state dynamics but differentiating the supervised loss exactly through the unrolled recurrent inference, Adam at learning rate 0.01 solves the extent-1 task over 10 paired seeds:

| Topology | Mean final MSE | Mean accuracy | Mean margin |
|---|---:|---:|---:|
| biological | **0.023788** | **1.000** | **0.295149** |
| type-pair rewire | 0.038318 | **1.000** | 0.184927 |

Thus the task is learnable and the biological topology helps exact-gradient optimization as well. The scientifically stronger question is no longer “does biological wiring help?” but rather “does fine biological wiring help the *local approximation to credit assignment* more than it helps exact-gradient learning?”

A useful diagnostic supports the plausibility of the local rule: for a single motion example, local-vs-exact edge-gradient cosine similarity is about 0.885 on biological wiring and 0.929 on the rewire. Balanced batches reduce combined alignment to roughly 0.25-0.36 because exact gradients from opposing directions cancel more strongly than the residual error of the local approximation. This motivated online/stochastic local updates.

## 5. Plasticity ablation

At learning rate 10 over 10 seeds:

| Topology | Plasticity | Mean final MSE | Mean accuracy |
|---|---|---:|---:|
| biological | edge + bias | **0.076989** | **0.4500** |
| biological | edges only | 0.082192 | 0.3688 |
| biological | bias only | 0.083763 | 0.2500 |
| type-pair rewire | edge + bias | 0.083307 | 0.2438 |
| type-pair rewire | edges only | 0.083619 | 0.3563 |
| type-pair rewire | bias only | 0.083761 | 0.2500 |

On biological wiring, edge plasticity clearly matters: edge-only versus bias-only MSE gives p = 0.00195, and accuracy gives p = 0.0156. Adding bias plasticity to edge plasticity further improves MSE (full versus edge-only p = 0.00977). The same cooperative effect is not seen on the rewired graph.

## 6. Connectome-strength initialization is not a simple win

The graph contains signed synapse-count-derived edge strengths. We compared random trainable initialization with initialization proportional to those measured structural strengths.

At learning rate 1, neither initialization learns appreciably.

At learning rate 3, structural-strength initialization improves descriptive performance across biological and control topologies. For example:

| Topology / init | Mean final MSE | Mean accuracy |
|---|---:|---:|
| biological / structural strength | **0.082882** | **0.3969** |
| biological / random | 0.083017 | 0.3406 |
| pair rotation / structural strength | 0.082887 | 0.3906 |
| pair rotation / random | 0.083335 | 0.2500 |
| rewire / structural strength | 0.083043 | 0.3625 |
| rewire / random | 0.083302 | 0.2594 |

The benefit is therefore not specific to biological fine wiring.

At learning rate 10, the picture reverses strongly on biological wiring:

| Topology / init | Mean final MSE | Mean accuracy |
|---|---:|---:|
| biological / random | **0.076989** | **0.4500** |
| biological / structural strength | 0.084385 | 0.2625 |
| pair rotation / random | **0.083482** | 0.2531 |
| pair rotation / structural strength | 0.087068 | 0.2219 |
| rewire / random | 0.083307 | 0.2438 |
| rewire / structural strength | **0.082714** | 0.2688 |

This should not be interpreted as evidence that biological synaptic strengths are poor. Synapse count is only a structural proxy for physiological efficacy, and the same local learning-rate scale changes the recurrent state conditioning when initial weight magnitudes change. The result instead shows that initialization scale and state dynamics must be normalized before measured-strength claims are meaningful.

## 7. Fixed hyperparameters do not scale from extent 1 to extent 2

The extent-2 crop contains 1,203 nodes and 26,636 edges. Reusing the extent-1 settings unchanged (`weight_lr=10`, two recurrent inference steps per frame) gave:

| Topology | Mean final MSE | Mean accuracy | Mean absolute edge update |
|---|---:|---:|---:|
| biological | 0.083378 | 0.2500 | 0.000032 |
| type-pair rewire | **0.083046** | 0.1969 | 0.000033 |

The biological circuit remains at chance rather than reproducing the extent-1 learning effect. Its mean edge update is about four times smaller than the extent-1 learning-rate-10 run (~0.000125).

This is an important negative result. The current local-PC advantage does **not** automatically scale with circuit size under fixed hyperparameters. Possible explanations include update dilution with degree/edge count, insufficient recurrent inference depth, changed conditioning, and longer effective credit paths. These possibilities must be separated experimentally rather than hidden by retuning only the successful condition.

## 8. Training-state locality has a temporal-depth crossover

For a streaming two-phase local-PC implementation, a simple temporary-state model stores the current node state plus one free-phase edge statistic and one free-phase node/bias statistic. For BPTT, the comparison uses only recurrent node-state history, deliberately excluding activation/intermediate/autodiff bookkeeping. BPTT numbers are therefore lower bounds.

At batch size 1 and 16-bit values:

| Extent | Nodes | Edges | Local temporary bytes | Crossover recurrent depth |
|---:|---:|---:|---:|---:|
| 1 | 443 | 7,698 | 17,168 | 19 |
| 2 | 1,203 | 26,636 | 58,084 | 24 |
| 3 | 2,345 | 57,663 | 124,706 | 26 |

For the present 7-frame × 2-step task (14 recurrent steps), the BPTT state-history lower bound is still smaller: 13,290 bytes at extent 1, 36,090 at extent 2, and 70,350 at extent 3. At larger temporal depth the linear BPTT history crosses the local edge-statistic storage: for example, at 28 recurrent steps the lower bound is 69,774 bytes at extent 2 and 136,010 at extent 3, both exceeding the local estimate.

Therefore local predictive learning should not be advertised as automatically memory-cheaper. Its hardware advantage is a *streaming/local communication structure whose memory does not grow with unroll depth*; whether this beats BPTT depends on temporal depth, sparsity, statistic representation, and the real autodiff implementation.

## Current interpretation

The current extent-1 evidence supports a narrow hypothesis:

> Fine cross-type retinotopic alignment in the FlyVis wiring is associated with more effective online two-phase predictive-coding learning on the biologically aligned T4 motion task.

Three qualifications are essential.

First, the exact-gradient control shows that biological wiring is also a useful generic task prior, so a topology advantage by itself is not evidence about local credit assignment.

Second, the target-shift interaction is strongly suggestive but must be compared with the *same target shifts under exact-gradient/BPTT learning*. If exact gradients solve all shifts similarly while local learning loses the biological advantage off the canonical mapping, the local-credit interpretation becomes substantially stronger. If exact gradients show the same target-specific interaction, the result is better described as task/topology alignment.

Third, extent 2 currently fails under the extent-1 hyperparameters. A credible scaling claim requires either a principled size-normalized local rule or a prespecified scale sweep showing why inference/update magnitudes change.

The next decisive controls are therefore: matched online BPTT across all four target shifts; extent-2 exact-gradient learnability; and an extent-2 sweep over local update scale and inference depth. Multiple stimulus families and eventually a prediction task that does not bake the T4 class labels directly into the readout are also needed before stronger biological claims.
