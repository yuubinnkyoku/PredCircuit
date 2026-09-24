# Low-rank residual credit-transport probe (2026-09-24)

## Question

Can a tree/prefix credit path exploit a compressed local operator, rather than paying for a full dense 8x8 Jacobian at every layer?

This is deliberately a **forward-linearized BP transport proxy**, not yet the PC-ALM inference Jacobian. It tests whether the residual network geometry itself immediately rules out low-rank composition.

## Setup

- depth 32, width 8, ReLU residual MLP
- same parameterization as `ResidualMLP`
- seeds 980--999, batch size 4 (80 sample paths)
- exact hidden credit transport uses each local Jacobian `J = I + R`
- approximations preserve the identity skip exactly and truncate only `R = J-I` by SVD
- diagonal-only baseline also measured

The probe was evaluated independently in float64 before committing the reproducible script `scripts/run_credit_jacobian_rank_probe.py`.

## Results

| approximation | median cosine to exact deep adjoint | mean cosine | 10th-percentile cosine | median relative error | median norm ratio |
|---|---:|---:|---:|---:|---:|
| diagonal | 0.8763 | 0.8507 | 0.6804 | 0.4917 | 0.8909 |
| I + rank-1(R) | 0.9356 | 0.9169 | 0.8149 | 0.3954 | 0.9063 |
| I + rank-2(R) | **0.9702** | **0.9564** | **0.9026** | 0.2701 | 1.0000 |
| I + rank-4(R) | **0.99949** | **0.99658** | **0.98992** | **0.04465** | 1.0000 |
| full rank-8 | 1.0000 | 1.0000 | 1.0000 | 0 | 1.0000 |

## Interpretation

The naive question "is the 8x8 Jacobian low-rank?" is misleading for a residual network because the identity skip is full-rank by construction. The useful decomposition is `J = I + R`: identity transport is wiring/addition, while only `R` needs an expensive learned transform.

On this proxy, rank 2 already preserves deep credit direction surprisingly well, and rank 4 is nearly exact across 30 hidden transitions. Therefore the previous worst-case assumption that a composable credit operator necessarily costs a full dense 8x8 transform is too pessimistic for this residual architecture.

This does **not** establish a PC-ALM speedup. A prefix tree must compose operators, and products of `(I + U V^T)` terms can increase effective rank; truncation after composition introduces another error source. Also, the PC-ALM local inference/dual Jacobian is not identical to the forward-network Jacobian measured here.

## Next decisive experiment

Extract the actual one-step PC-ALM credit-state Jacobian around a representative inference trajectory, split its skip/identity component from the correction, and repeat rank-1/2/4 compression. Then explicitly compose compressed operators in a balanced tree with re-truncation at every merge. Measure first-layer BP cosine, MACs, coefficient storage, and tree depth. If rank 2--4 survives composition, a logarithmic-depth FPGA prefix path becomes plausible; if rank grows rapidly under composition, this route should be rejected before RTL work.
