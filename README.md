# PredCircuit

**Predictive coding and local learning in biological and artificial neural circuits.**

PredCircuit is a research codebase for asking a narrow question from several directions:

> How do predictive-coding learning dynamics interact with the actual topology of neural circuits?

The repository deliberately separates **learning rule**, **circuit topology**, and **hardware**. The current phase is software-only: arbitrary-graph predictive coding, topology-controlled ablations, connectome ingestion, and reproducible benchmarks. FPGA/NPU work can be added later without changing the scientific core.

## Current scope

- Predictive-coding inference on arbitrary directed graphs.
- Strictly local synaptic update: postsynaptic prediction error × presynaptic activity.
- Same-topology BPTT baseline for later controlled comparisons.
- Topology controls: degree-preserving rewiring, feedback/lateral-edge removal when anatomical ranks are known, and reciprocal-edge removal.
- Graph metrics and robustness helpers.
- MaleCNS v1.0 bulk-data and neuPrint ingestion paths.
- Small synthetic CPU sanity checks.
- Tests and CI.

This is **not** yet evidence that a biological connectome improves predictive coding. Synthetic experiments only validate the machinery. Biological claims require prespecified MaleCNS/fly visual-system experiments and proper null models.

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e '.[dev]'
pytest
```

Optional neuPrint support:

```bash
pip install -e '.[malecns,dev]'
```

## MaleCNS v1.0

The official MaleCNS dataset is not vendored here. It is large and is licensed separately (CC-BY). Download only what an experiment needs:

```bash
python scripts/download_malecns.py annotations
python scripts/download_malecns.py connectivity  # about 1.1 GB
```

Or query a small neighborhood through neuPrint:

```bash
export NEUPRINT_TOKEN='...'
python scripts/neuprint_query.py --type DNge104
```

Never commit a neuPrint token or raw MaleCNS dumps.

## Research program

The first useful biological experiment is not “run the whole fly brain.” It is a controlled comparison:

1. Extract a defined circuit/subgraph with a biological reason for its input/output interpretation.
2. Train predictive coding with local updates.
3. Construct null graphs that preserve progressively more structure.
4. Compare task performance, inference convergence, robustness, and communication/update cost.
5. Ablate feedback, reciprocal motifs, synapse strengths, and signs separately.
6. Only then ask which structural features matter.

See `docs/research-plan.md` and `docs/experiment-matrix.md`.

## Repository layout

```text
src/predcircuit/       core library
scripts/               reproducible experiment/data entry points
tests/                 unit tests
docs/                  hypotheses, experiment design, references
data/                   documentation only; raw data are gitignored
results/                curated results can live here; generated runs are ignored
.github/workflows/      CI
```

## Design rule

A result is only interesting if topology and learning rule are not confounded. When comparing a biological circuit with a null graph, keep node count, edge count, initialization, task, and training budget fixed; preserve in/out degree whenever that is the intended control.

## Status

Foundation stage. The code implements the first local predictive-coding model and connectome plumbing; real-connectome experiments still need circuit selection and biological input/output semantics.
