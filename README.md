# PredCircuit

**Predictive coding and local learning in biological and artificial neural circuits.**

PredCircuit is a research codebase for asking a narrow question from several directions:

> How do predictive-coding learning dynamics interact with the actual topology of neural circuits?

The repository deliberately separates **learning rule**, **circuit topology**, and **hardware**. The current phase is software-first: arbitrary-graph predictive coding, controlled FlyVis retinotopic experiments, connectome ingestion, exact-gradient controls, and analytical locality-cost models. FPGA/NPU work can be added later without changing the scientific core.

## Current scope

- Predictive-coding inference on arbitrary directed graphs.
- Local synaptic plasticity based on postsynaptic prediction error and presynaptic activity, including two-phase free/nudged updates.
- Exact unrolled-gradient controls through the same predictive-coding state dynamics.
- Retinotopic FlyVis motion experiments with constrained topology nulls.
- Null models that preserve cell-type pairs, exact node degree, or within-pair receptive-field geometry.
- MaleCNS v1.0 bulk-data and neuPrint ingestion paths.
- Analytical temporary-state/locality estimates for local PC versus BPTT lower bounds.
- Reproducible scripts, tests, type checking, linting, and CI.

The strongest current result is still a **pilot**: on a small extent-1 FlyVis crop, online two-phase local PC learns more on biological retinotopic wiring than on fine-wiring nulls, and that advantage is strongest for the canonical T4 direction assignment. Fixed hyperparameters do not yet reproduce the effect at extent 2, and exact-gradient learning also benefits from biological topology. These controls make the current question more specific: whether biological fine wiring disproportionately helps *local credit assignment*, rather than merely supplying a generic task prior.

See `docs/temporal-findings.md` for the current evidence and caveats, and `docs/pilot-results.md` for the experiment history including negative results that changed the design.

## Development stack

PredCircuit uses **uv** for environments/dependencies, **ty** for type checking, **pytest** for tests, and **ruff** for linting/formatting.

```bash
uv sync
uv run pytest
uv run ty check src tests scripts
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
```

To apply formatting locally:

```bash
uv run ruff format src tests scripts
```

Optional neuPrint support:

```bash
uv sync --extra malecns
```

## MaleCNS v1.0

The official MaleCNS dataset is not vendored here. It is large and is licensed separately (CC-BY). Download only what an experiment needs:

```bash
uv run python scripts/download_malecns.py annotations
uv run python scripts/download_malecns.py connectivity  # about 1.1 GB
```

Or query a small neighborhood through neuPrint:

```bash
uv sync --extra malecns
export NEUPRINT_TOKEN='...'
uv run python scripts/neuprint_query.py --type DNge104
```

Never commit a neuPrint token or raw MaleCNS dumps.

## Research program

The useful biological experiment is not “run the whole fly brain.” It is a controlled comparison:

1. Extract a defined circuit/subgraph with a biological reason for its input/output interpretation.
2. Train predictive coding with local updates.
3. Construct null graphs that preserve progressively more structure.
4. Compare against exact-gradient learning through the same state dynamics.
5. Measure task performance, gradient alignment, inference behavior, robustness, and communication/update cost.
6. Ablate retinotopic alignment, feedback, reciprocal motifs, synapse strengths, signs, and plasticity sites separately.
7. Test whether any effect survives larger circuits and multiple tasks before making a biological claim.

See `docs/research-plan.md`, `docs/experiment-matrix.md`, and `docs/temporal-findings.md`.

## Repository layout

```text
src/predcircuit/       core library
scripts/               reproducible experiment/data entry points
tests/                 unit tests
docs/                  hypotheses, experiment design, references, findings
data/                   documentation only; raw data are gitignored
results/                curated results can live here; generated runs are ignored
.github/workflows/      CI and reproducible experiment runs
```

## Design rule

A result is only interesting if topology and learning rule are not confounded. When comparing a biological circuit with a null graph, keep node count, edge count, initialization, task, and training budget fixed; preserve task-relevant coarse organization and in/out degree whenever those are intended controls. A topology null must also be checked for accidental task shortcuts.

## Status

Active pilot stage. The code now supports real FlyVis retinotopy, temporal local-PC learning, exact-gradient controls, constrained topology nulls, target-alignment tests, scaling experiments, and hardware-locality estimates. The next goal is to separate generic topology/task alignment from a genuinely local-credit-specific effect and to determine why the current local rule does not automatically scale from extent 1 to extent 2.
