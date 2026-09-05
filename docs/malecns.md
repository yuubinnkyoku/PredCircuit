# MaleCNS v1.0 notes

The MaleCNS project provides a full adult male *Drosophila* CNS connectome. The official v1.0 dataset is exposed through neuPrint and bulk Google Cloud Storage files.

## Useful official artifacts

- Dataset name for neuPrint: `male-cns:v1.0`
- Curated annotations: about 13 MB Feather
- Per-neuron neurotransmitter predictions: about 42 MB Feather
- Full segment-to-segment connectivity: about 1.1 GB Feather
- Synapse-point and partner tables: much larger and unnecessary for the first graph-level experiments

PredCircuit intentionally starts with the segment-level weighted graph instead of synapse coordinates. Coordinates become relevant only if spatial locality itself becomes a hypothesis.

## Authentication

Bulk files are public. neuPrint programmatic queries require an account/API token. Store it only in the environment variable `NEUPRINT_TOKEN`; `.env` is gitignored.

## Data policy

Raw MaleCNS data are not committed to this repository. The upstream dataset is CC-BY and should be cited directly. Derived tiny tables needed to reproduce a figure may be committed later with provenance and license notes.

## Extraction strategy

Whole-CNS optimization is not the first target. Use one of two routes:

1. **Biology-first:** choose a known circuit/cell-type set, query its 1-hop/2-hop neighborhood, and preserve biological inputs/outputs.
2. **Graph-first exploratory analysis:** threshold the full weighted graph and search for candidate modules, but treat any discovered hypothesis as exploratory until tested on a separately defined circuit.

The `graph_from_connectivity` helper can keep seed neurons plus their strongest neighbors, but this is only a technical extractor, not a scientifically justified circuit definition by itself.
