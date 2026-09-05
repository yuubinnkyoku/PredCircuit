# Experiment matrix

| Axis | Levels | What it isolates |
|---|---|---|
| Learning | predictive coding / BPTT | local inference-learning vs backprop-through-time |
| Topology | biological / degree-rewired / ER-matched | exact wiring vs low-order graph statistics |
| Directionality | original / feedback removed | role of backward/recurrent pathways |
| Reciprocity | original / reciprocal removed | two-node recurrent motifs |
| Strength | synapse counts / binary / shuffled | information in measured connection magnitude |
| Sign | biological estimate / shuffled / unsigned | excitatory-inhibitory constraint where justified |
| Damage | 0–50% edge or node perturbation | robustness |
| Scale | small subcircuit → larger circuit | whether effects survive scaling |

## Primary outcomes

1. Held-out task error/accuracy.
2. Updates or examples to a fixed criterion.
3. Inference energy versus inference step.
4. Robustness area under the damage-performance curve.
5. Edge-local operations and retained state per training example.

## Minimum statistical discipline

- Fix the experiment definition before looking at the final test set.
- Use multiple random seeds for initialization and null-graph generation.
- Pair biological/null runs by seed when possible.
- Report all seeds and uncertainty intervals.
- Avoid tuning each topology separately unless that is explicitly a second experiment.
- Keep failed runs and numerical-instability counts visible.

## First concrete milestone

A small real circuit should produce one table with rows = topology variants and columns = primary outcomes, plus graph-statistics verification that the degree-preserving null actually preserves the intended quantities.
