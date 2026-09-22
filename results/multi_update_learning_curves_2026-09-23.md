# Multi-update learning curves: BP / sPC / PC-ALM / leaky PC-ALM / ePC

Date: 2026-09-23

## Setup

- ResidualMLP: depth 32, width 8, ReLU
- train size 64, eval size 256, batch 4
- 24 weight updates, same initialization and minibatch sequence per method
- weight LR 0.01
- sPC: T=1024
- PC-ALM: T=112, alpha=0.925, rho=1, eta_h=0.25
- leaky PC-ALM: same, dual leak=0.01
- ePC: T=1, error_lr=0.3; weight LR divided by 0.3 to compensate the known T=1 global credit scale
- seeds: 970, 971, 972

## Final evaluation loss and relative reduction

| seed | BP | sPC | PC-ALM | leaky PC-ALM | ePC |
|---|---:|---:|---:|---:|---:|
| 970 | 1.764691 (-5.71%) | 1.865042 (-0.35%) | 1.771066 (-5.37%) | 1.790604 (-4.32%) | 1.841229 (-1.62%) |
| 971 | 1.066094 (-1.84%) | 1.085082 (-0.09%) | 1.067282 (-1.73%) | 1.071518 (-1.34%) | 1.073591 (-1.15%) |
| 972 | 0.812412 (-1.02%) | 0.819332 (-0.18%) | 0.811611 (-1.12%) | 0.813404 (-0.90%) | 0.804331 (-2.01%) |

Mean relative loss reduction over the three seeds:

- BP: 2.857%
- sPC: 0.206%
- PC-ALM: 2.740%
- leaky PC-ALM: 2.189%
- ePC: 1.592%

Relaxation steps after 24 updates:

- sPC: 24,576
- PC-ALM / leaky PC-ALM: 2,688
- ePC: 24

## Interpretation

PC-ALM's one-shot gradient-geometry advantage over sPC survives repeated weight updates in this small deep/narrow task. Pure PC-ALM nearly tracks BP over all three seeds while using 9.14x fewer relaxation steps than the already-unsuccessful sPC T=1024 baseline. Leaky PC-ALM is consistently weaker than pure PC-ALM here, so dual leak should not be treated as a default improvement.

The ePC result is deliberately not summarized as a win or loss. It is vastly cheaper in relaxation steps and is best on seed 972, but weaker on seeds 970 and 971. With only three seeds and 24 updates, this is enough to reject the claim that one-shot BP-like ePC credit automatically implies BP-like multi-update learning, but not enough to establish a robust PC-ALM advantage over ePC.

The next high-value experiment is therefore an ePC-vs-PC-ALM learning-efficiency boundary rather than further sPC budget extension: increase seeds and updates, and sweep ePC error_lr / weight-step normalization while keeping PC-ALM T fixed. Compare loss reached per cumulative MAC/state-access estimate, not relaxation T alone.
