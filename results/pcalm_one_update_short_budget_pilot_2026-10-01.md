# One-update short-budget gradient pilot (2026-10-01)

This is a local CPU reproduction of the current branch source at `c90d61f8`, using seed 0, the first minibatch, depth 32, width 64, ReLU, batch size 4, and `state_lr=15/64`. It is a diagnostic pilot, not a multi-seed training result.

## Gradient geometry versus same-weight BP

| method | T | global cosine | layer-0 cosine | layer-0 norm/BP | dual zero fraction | max |lambda| |
|---|---:|---:|---:|---:|---:|---:|
| sPC | 80 | 0.5667 | 0.2924 | 4.68e-8 | - | - |
| PC-ALM FP32 official-like | 64 | 0.8992 | 0.867 | 0.560 | 0.2708 | 0.0700 |
| PC-ALM fixed official-like 14/16/12 | 64 | 0.9094 | 0.865 | 0.646 | 0.5266 | 0.0684 |
| PC-ALM fixed stabilized 14/16/12 | 64 | 0.8289 | 0.860 | 0.302 | 0.5659 | 0.0508 |
| PC-ALM fixed stabilized 14/16/12 | 80 | 0.9080 | 0.872 | 0.604 | 0.4859 | 0.0645 |
| PC-ALM fixed stabilized 14/16/12 | 96 | 0.9264 | 0.888 | 0.694 | 0.4276 | 0.0771 |

No tested PC-ALM condition had dual saturation in this pilot.

## Interpretation

The strongest immediate result is not a 12-bit-dual failure. At T=64, fixed official-like PC-ALM is essentially as well aligned with BP as FP32 and is slightly better in global cosine for this seed. The 12-bit dual has many more exact zeros, but this does not stop first-layer credit in the tested minibatch.

PC-ALM clearly beats sPC on credit propagation at comparable budget: sPC T=80 has layer-0 gradient magnitude only 4.68e-8 of BP, whereas FP32 PC-ALM T=64 reaches 0.560 and fixed official-like reaches 0.646. This is direct evidence for a ballistic-vs-diffusive advantage in this PredCircuit setting, although it is only one seed/update.

The stabilized coefficients at T=64 reduce gradient magnitude substantially (layer-0 norm/BP 0.302) and global cosine to 0.829; increasing T to 80-96 recovers both. This suggests the extra T penalty may be driven more by the stabilization coefficients/leak than by fixed-point storage itself.

## Next gate

Run the existing short-budget training experiment for multiple updates/seeds before changing lambda width. Prioritize FP32 official T64 vs fixed official T64 vs fixed stabilized T64/80/96 and sPC T80. If the fixed-official ~= FP32 relation persists, deprioritize lambda-bit sweeps and isolate alpha/leak instead.
