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

The next scientifically meaningful target is therefore:

1. preserve FlyVis retinotopic offsets instead of collapsing them;
2. add a temporal predictive-coding task, such as next-frame / moving-stimulus prediction;
3. compare biological wiring with nulls that preserve cell type, retinotopic displacement statistics, degree, and coarse sensory depth;
4. then compare PC local learning against a same-topology BPTT control.

Until that experiment exists, no result in this file should be phrased as “the biological connectome helps/hurts predictive coding.”
