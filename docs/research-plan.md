# Research plan

## Central question

Can predictive coding exploit structural properties of biological neural circuits that are absent from matched null graphs?

The important word is **structural**. A raw comparison against an Erdős–Rényi graph is too weak: degree distribution, sparsity, reciprocity, modularity, edge signs, and synapse counts can all differ at once. The research therefore proceeds by progressively stronger controls.

## Hypotheses

### H1 — topology matters under a local learning rule

A connectome-constrained predictive-coding network will differ from degree-matched rewired controls in at least one of:

- final task error,
- samples/updates to criterion,
- inference steps to a stable energy,
- robustness to node/edge perturbation.

This is deliberately non-directional at first. Assuming the biological graph must win would bias the study.

### H2 — feedback and reciprocal motifs matter more for predictive coding than for feedforward baselines

Predictive coding uses recurrent error-driven inference. Removing anatomically backward or reciprocal connections should therefore alter inference dynamics disproportionately if those motifs carry useful predictive/error interactions.

### H3 — biological strength/sign information adds information beyond binary adjacency

Compare, in order:

1. binary adjacency,
2. adjacency + synapse-count magnitude,
3. adjacency + neurotransmitter-derived sign where justified,
4. magnitude/sign-shuffled controls.

### H4 — any advantage should survive a fair null model

A claimed connectome effect must survive at least an in/out-degree-preserving directed rewire. Stronger controls should preserve cell-type blocks or anatomical rank where possible.

## Phase A — software validation

- Unit-test local update equations.
- Verify inference energy behavior on small graphs.
- Verify directed rewiring exactly preserves in/out degree.
- Run tiny tasks only as implementation checks.
- Add deterministic seeds and machine-readable CSV output.

No biological conclusions belong in this phase.

## Phase B — connectome ingestion and descriptive analysis

For each chosen circuit:

- number of neurons and edges,
- edge-weight distribution,
- in/out-degree distributions,
- reciprocal edge fraction,
- strongly connected components,
- rank/anatomical feedback fraction when definable,
- type composition,
- neurotransmitter/sign coverage.

The first MaleCNS-wide task is descriptive, not training: identify subgraphs whose biological inputs/outputs and computational role are interpretable enough to support a task.

## Phase C — topology-controlled predictive-coding experiment

For a selected circuit, construct:

- `biological`: original adjacency;
- `degree_rewire`: preserve every node's in/out degree;
- `weight_shuffle`: keep edges, shuffle strengths;
- `no_reciprocal`: remove two-node reciprocal motifs;
- `no_feedback`: remove edges that oppose a justified anatomical/task ordering;
- `er_matched`: same node/edge counts, as a deliberately weak null.

Run multiple seeds. Report the complete seed distribution, not just the best run.

## Phase D — learning-rule control

Use the same topology under:

- predictive coding with local updates;
- BPTT/autograd recurrent baseline;
- optional Hebbian/eligibility-trace baseline later.

This separates “the graph is useful” from “the graph is especially useful for predictive coding.”

## Phase E — robustness and efficiency

Measure:

- random edge deletion curves,
- random node silencing curves,
- weight noise,
- inference iterations to tolerance,
- number of edge-local messages per example,
- number of state values that must be retained for an update.

These are the measurements that can later connect cleanly to FPGA/NPU work.

## Candidate circuits

Do not choose solely because a subgraph is easy to download. Prefer circuits with a clear sensory input and experimentally studied computation. The fly visual motion system is an unusually strong first candidate because an existing connectome-constrained BPTT model provides a comparison point. MaleCNS can then broaden the analysis beyond the visual system.

## Failure conditions worth keeping

Negative outcomes are informative if controls are strong:

- degree-preserving rewiring performs identically;
- predictive coding is more sensitive to biological feedback than expected but does not improve task error;
- biological topology helps BPTT equally, implying no PC-specific interaction;
- synapse-count information matters while exact wiring motifs do not.
