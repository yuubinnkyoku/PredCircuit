# FlyVis temporal predictive-coding findings

This note collects the current retinotopic/temporal results in one place. They are pilot results and should not be read as a biological-learning claim without the controls listed below.

## Experimental system

The task uses a finite retinotopic crop of the published FlyVis circuit. Moving Gaussian bars are presented as short sequences. The central T4a/T4b/T4c/T4d cells are evaluated as a four-way direction readout.

The main extent-1 crop contains 443 nodes and 7,698 directed edges. Extent 2 contains 1,203 nodes and 26,636 edges. Unless stated otherwise, trainable edge weights are randomly initialized and the graph supplies wiring structure rather than physiological synaptic efficacy.

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

## 3. The local-PC topology advantage is strongly target-alignment dependent

The canonical direction-to-T4 class assignment is `(1, 2, 0, 3)` for directions `(0, 90, 180, 270)` degrees. We cyclically shifted the target classes while keeping stimuli, graph, training budget, and local rule fixed.

| Target shift | Biological MSE | Rewire MSE | Biological accuracy | Rewire accuracy |
|---:|---:|---:|---:|---:|
| 0, canonical | **0.076989** | 0.083307 | **0.4500** | 0.2438 |
| 1 | 0.083555 | **0.082749** | 0.1750 | **0.2813** |
| 2 | 0.083542 | **0.083440** | 0.1375 | **0.3063** |
| 3 | **0.083476** | 0.084800 | 0.2344 | **0.2938** |

For canonical targets, the paired MSE advantage `rewire - biological` was +0.006317 and biological wiring won in 9/10 seeds (Wilcoxon p = 0.00586). For shifts 1, 2, and 3 the mean MSE advantages were -0.000805, -0.000102, and +0.001324 respectively; none gave a comparable reliable biological advantage.

The within-seed interaction

`canonical topology advantage - mean(noncanonical topology advantages)`

was +0.006178 in MSE, positive in 10/10 seeds, with Wilcoxon p = 0.00195. The analogous accuracy interaction was +31.77 percentage points, positive in 9/10 seeds, p = 0.00586.

Thus the extent-1 local-PC topology effect is strongly tied to the canonical T4 target organization rather than being a generic advantage for arbitrary permutations of the four outputs.

## 4. Matched online exact gradients remove the target-specific interaction

A decisive control uses the **same predictive-coding state dynamics, one-sample online schedule, stimulus jitter schedule, target shifts, and graph pair**, but computes exact gradients through the unrolled recurrent dynamics and updates weights/biases with Adam at learning rate 0.01.

Over 10 paired seeds for every target shift:

| Target shift | Biological MSE | Rewire MSE | Biological accuracy | Rewire accuracy |
|---:|---:|---:|---:|---:|
| 0 | **0.020227** | 0.041275 | 1.000 | 1.000 |
| 1 | **0.020038** | 0.041904 | 1.000 | 1.000 |
| 2 | **0.019510** | 0.040863 | 1.000 | 1.000 |
| 3 | **0.021507** | 0.041553 | 1.000 | 1.000 |

Biological wiring improves exact-gradient MSE for **every** target permutation: mean `rewire - biological` advantage is +0.02105, +0.02187, +0.02135, and +0.02005 for shifts 0 through 3, with biological lower in 10/10 seeds in every condition (Wilcoxon p = 0.00195 for each shift).

Crucially, exact-gradient learning does **not** show the canonical-specific interaction seen under local PC. Its interaction

`shift-0 topology advantage - mean(shifts-1..3 topology advantages)`

is -0.000041, positive in only 6/10 seeds, Wilcoxon p = 0.922. In contrast, the same interaction under local PC is +0.006178. Taking the paired difference between the local-PC and exact-gradient interactions gives +0.006219, Wilcoxon p = 0.00586.

This is the strongest current evidence for a **local-credit-specific topology interaction**. The biological graph is a generic task prior under exact gradients, but only the approximate local predictive-coding update depends strongly on the biological T4 target alignment.

The correct claim is therefore narrower than “biological wiring learns better”:

> Fine FlyVis retinotopic wiring appears to make the local predictive-coding credit signal selectively effective when the supervised objective is aligned with the circuit's biological T4 organization, while exact gradients can exploit the same wiring for arbitrary output permutations.

This remains a model result on a synthetic moving-bar task, not evidence that the fly itself implements this exact learning rule.

## 5. Exact gradients establish learnability and a generic topology prior

A separate balanced-batch exact-gradient control at extent 1 also solves the task. With Adam at learning rate 0.01 for 100 epochs over 10 paired seeds:

| Topology | Mean final MSE | Mean accuracy | Mean margin |
|---|---:|---:|---:|
| biological | **0.023788** | **1.000** | **0.295149** |
| type-pair rewire | 0.038318 | **1.000** | 0.184927 |

A gradient-alignment diagnostic gives another useful clue. For a single motion example at extent 1, the temporal two-phase local edge update is well aligned with the exact same-PC gradient: cosine similarity is about 0.885 on biological wiring and 0.929 on the rewire. Balanced batches reduce combined alignment to roughly 0.25-0.36 because exact gradients from opposing directions cancel more strongly than residual error in the local approximation. This motivated online/stochastic local updates.

## 6. Plasticity ablation

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

## 7. Connectome-strength initialization is not a simple win

The graph contains signed synapse-count-derived edge strengths. We compared random trainable initialization with initialization proportional to those measured structural strengths.

At learning rate 1, neither initialization learns appreciably. At learning rate 3, structural-strength initialization improves descriptive performance across biological and control topologies, so the benefit is not specific to biological fine wiring.

At learning rate 10, structural-strength initialization degrades the biological and pair-rotation conditions relative to random initialization: biological random reaches MSE 0.076989 / accuracy 0.4500, whereas biological structural-strength initialization reaches MSE 0.084385 / accuracy 0.2625.

This should not be interpreted as evidence that biological synaptic strengths are poor. Synapse count is only a structural proxy for physiological efficacy, and changing initial weight magnitudes changes recurrent conditioning and the appropriate update scale.

## 8. Local-PC learning currently fails to scale to extent 2, but exact gradients do not

At extent 2 (1,203 nodes, 26,636 edges), simply reusing the successful extent-1 local-PC settings (`weight_lr=10`, two recurrent inference steps per frame) gives chance-level biological accuracy and essentially no MSE improvement.

An exploratory 5-seed scale sweep tested learning rates 10, 30, and 100 with two or four inference steps per frame:

| Local LR | Steps/frame | Biological MSE | Biological accuracy | Rewire MSE | Rewire accuracy |
|---:|---:|---:|---:|---:|---:|
| 10 | 2 | 0.083390 | 0.2500 | **0.082700** | 0.1250 |
| 10 | 4 | 0.083280 | 0.2500 | **0.082313** | 0.2813 |
| 30 | 2 | 0.083923 | 0.2500 | **0.083352** | 0.2500 |
| 30 | 4 | **0.090098** | 0.2500 | 0.092125 | 0.2500 |
| 100 | 2 | **0.088947** | 0.2500 | 0.090468 | 0.2500 |
| 100 | 4 | 0.177699 | 0.2500 | **0.137062** | 0.2500 |

At learning rate 100 with four steps, some runs became non-finite. Increasing update scale or inference depth therefore does not rescue the extent-2 local rule; large scale eventually destabilizes it.

This is **not** a representational-capacity or task-learnability failure. Exact unrolled gradients at extent 2, with the same PC dynamics and two steps per frame, solve the task in all five seeds:

| Topology | Mean final MSE | Mean accuracy | Mean margin |
|---|---:|---:|---:|
| biological | **0.011321** | **1.000** | **0.421842** |
| type-pair rewire | 0.024163 | **1.000** | 0.289366 |

The extent-2 exact-gradient result is particularly useful: the larger circuit has ample capacity and can propagate useful exact credit, while the present local approximation does not. The next scaling question is therefore about **local-gradient quality / normalization**, not about whether the circuit can solve the task.

## 9. Training-state locality has a temporal-depth crossover

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

The strongest current result is now a **learning-rule × topology × target-alignment interaction** rather than a simple topology effect.

At extent 1, local predictive coding gains a large biological-wiring advantage only for the canonical T4 target assignment. Exact-gradient learning, under a matched online schedule, reaches 100% accuracy for every target permutation and retains almost the same biological-vs-rewire MSE advantage across all four permutations. The canonical-specific topology interaction is present under local PC and absent under exact gradients; the paired difference between those interactions is significant in this 10-seed pilot (p = 0.00586).

That is consistent with the hypothesis that biological fine wiring can make a particular **local credit-assignment signal** more useful when the task matches the circuit's native organization. It does not prove that biological learning uses the implemented two-phase PC rule, and it does not yet generalize beyond this task/crop.

The major negative result is equally important: the current local rule does not scale automatically to extent 2. Neither larger learning rates nor doubling recurrent inference depth rescues it, while exact gradients solve extent 2 easily. The next mechanistic experiment should therefore compare local-vs-exact gradient direction and norm across extents and identify whether signal quality, degree normalization, path length, or recurrent conditioning causes the scaling breakdown.

Before a stronger claim, the result should also survive additional stimulus families and a genuinely predictive objective (for example next-frame or sensory-cancellation prediction) that does not define success directly through the canonical T4 class labels.
