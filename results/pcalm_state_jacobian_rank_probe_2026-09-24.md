# PC-ALM one-step state-Jacobian rank probe (2026-09-24)

## Question

The forward residual-network proxy showed that preserving the identity skip exactly and compressing only `J-I` can retain deep BP credit direction at low rank. Does the same low-rank structure survive in the **actual PC-ALM one-outer-step state map** over hidden states and dual variables?

## Setup

This is a direct float64 autograd probe of the current `predcircuit.pcalm` equations, not a forward-BP proxy.

- residual MLP, ReLU, width 8
- free initialization from the forward pass
- zero initial duals
- one PC-ALM outer step with one hidden-state gradient step
- `eta_h = 0.05`, `rho = 1.0`, `alpha = 0.2`
- state is concatenated `(h_0,...,h_{H-1}, lambda_0,...,lambda_{H-1})`
- map is exactly: hidden AL-gradient step -> residual recomputation -> `lambda <- lambda + alpha r`
- full Jacobian obtained with float64 autograd
- rank energy means cumulative squared singular-value energy of `J_state - I`

The probe was evaluated independently before recording these numbers. It is a structural diagnostic; it does not yet claim a hardware speedup or a trained-network result.

## Results

### Depth 8, width 8, seed 980

State dimension is 112. Cumulative energy of the global correction `J_state - I`:

| rank | energy captured |
|---:|---:|
| 1 | 0.0406 |
| 2 | 0.0783 |
| 4 | 0.1507 |
| 8 | 0.2824 |
| 16 | 0.5146 |
| 32 | 0.8255 |
| 64 | 0.9671 |

Representative 16x16 same-layer `(h_l, lambda_l)` diagonal state blocks are also not rank-1/2 corrections. Rank-2 captures only about 0.248--0.265 of correction energy; rank-4 about 0.482--0.500; rank-8 about 0.922--0.949.

### Depth 32, width 8, seed 980

State dimension is 496. Cumulative energy of `J_state - I`:

| rank | energy captured |
|---:|---:|
| 2 | 0.0158 |
| 4 | 0.0313 |
| 8 | 0.0619 |
| 16 | 0.1212 |
| 32 | 0.2369 |
| 64 | 0.4512 |
| 128 | 0.7710 |
| 256 | 0.9508 |

## Interpretation

The encouraging low-rank result for the forward residual credit operator does **not** transfer directly to the full PC-ALM `(h, lambda)` one-step dynamics. At depth 32, rank 4 captures only about 3.1% of the correction energy, while roughly half the 496-dimensional state rank is required to reach about 95%.

This is an important negative result: a prefix circuit should not be designed by globally approximating the entire PC-ALM one-step state map as `I + U V^T` with rank 2--4. The dual/state coupling makes that object much less compressible than the forward-network credit proxy.

However, this does not yet rule out a compressed **credit-only transfer operator**. The global state Jacobian contains local self-dynamics, primal relaxation, and dual integration that a prefix credit path need not reproduce. The next useful object is therefore the Schur/adjoint transfer from a deeper-layer credit perturbation to the adjacent shallower-layer credit after eliminating local state variables, not the whole state-transition Jacobian.

## Consequence for the research direction

Keep the previous forward `I + low-rank(R)` result as evidence that the residual network geometry itself is compressible, but do not use it as evidence that PC-ALM dynamics are low rank. Any logarithmic-depth prefix proposal now has to demonstrate low-rank structure in a reduced credit-transfer operator after local elimination. If that reduced operator is also near full-rank, the prefix route should be rejected before RTL work.
