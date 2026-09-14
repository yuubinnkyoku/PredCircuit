# PC-ALM reference protocol

This document freezes the first software-comparison target before implementing a hardware-oriented variant.

## Reference

- Official implementation: `SakanaAI/pc-alm`
- Reference main tree inspected: `660747f61a8a7e547c0ecd2c48c8883380a7d1f6`
- Core reference file: `pcalm/inference.py`
- Paper: Seely & Gould, *Augmented Lagrangian Predictive Coding* (arXiv:2605.31022)

Do not claim agreement with PC-ALM until the implementation is checked against this frozen reference.

## Update semantics to preserve

For hidden constraints `r_i = h_i - f_i(h_{i-1}; W_i)`, the reference uses the shifted augmented-Lagrangian energy

`L = supervised_loss + sum_i rho/2 * ||r_i + lambda_i/rho||^2`.

Per minibatch, hidden states start at the feed-forward values and hidden duals start at zero. A PC-ALM outer step is:

1. Run `inner_steps` primal gradient-descent steps on hidden states with duals fixed.
2. Recompute hidden residuals.
3. Apply local dual ascent `lambda_i <- lambda_i + alpha * r_i`.

The final outer iteration runs the primal solve and then computes one more dual update. The official default weight-credit timing is `pre_dual_energy`: the weight gradient uses the duals *before* that final dual update. `post_dual_energy` exists as an explicit alternative and must not be silently substituted.

The reference energy is averaged over the batch. Its hidden-state solver multiplies `state_lr` by the runtime batch size so that `state_lr` corresponds to the paper's per-sample activity step. Missing this factor changes finite-T convergence and is an implementation bug, not a scientific result.

Standard PC is retained as the state-based PC baseline: feed-forward hidden initialization, zero duals, and primal relaxation on the same quadratic penalty energy. BP is the ordinary gradient of the feed-forward supervised loss.

## First comparison gate

Before FPGA RTL work, implement a small PyTorch benchmark that uses one identical residual MLP, initialization, batch and loss for BP / sPC / PC-ALM. ePC is added once its update convention is fixed with an equally explicit reference.

The first diagnostic should be inference-only at fixed weights rather than full training. Sweep depth and relaxation budget and record:

- cosine similarity and relative error of each method's parameter gradient versus BP;
- per-layer `||r_i||` and, for PC-ALM, `||lambda_i||` over inference time;
- finite / oscillatory / divergent behavior;
- number of local state updates, MAC estimate and state bytes.

The most informative initial regime is deep and narrow, because this is where the PC-ALM paper reports diffusive sPC credit decay and faster PC-ALM credit propagation. Full training and low-precision sweeps should follow only after the update-order parity tests pass.

## Required parity tests

1. `alpha = 0` must remove dual accumulation and reduce PC-ALM to the corresponding repeated primal-relaxation dynamics for the same total number of primal steps.
2. A tiny deterministic linear network must approach the BP gradient as PC-ALM inference converges.
3. Batch-size changes must not change the effective per-sample state step when the data are duplicated.
4. `pre_dual_energy` and `post_dual_energy` must be tested separately; the default benchmark uses `pre_dual_energy`.
5. All reported sweeps must record finite status and the maximum absolute dual value so numerical instability is not mistaken for a learning effect.
