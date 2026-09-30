# PC-ALM per-iteration dependency lower bound (2026-09-30)

## Question

Can the layer-concurrency latency gate be relaxed from roughly T<=15 by assuming one PC-ALM iteration is faster than one BP layer phase?

## Dependency DAG from the current implementation

For hidden state z_l, the augmented-Lagrangian state gradient contains:
- the local constraint residual r_l = z_l - f_l(z_{l-1}; W_l),
- the next-layer residual r_{l+1}, propagated through W_{l+1}^T (and the activation derivative),
- the dual terms attached to the same residuals,
- at the final free layer, the supervised output term propagated through W_out^T.

All r_l can be evaluated concurrently across layers once the current z snapshot is fixed. However the W_{l+1}^T correction for z_l depends on r_{l+1} from that same snapshot. Therefore an exact synchronous Jacobi-style iteration has at least two dependent dense matrix phases:

1. prediction/residual phase: W_l z_{l-1} for all layers in parallel;
2. correction phase: W_{l+1}^T q_{l+1} for all layers in parallel, where q includes the just-computed residual/dual contribution.

The dual update lambda_l <- lambda_l + alpha r_l is elementwise and can be fused after residual production; it does not remove the residual -> transpose-multiply dependency.

Thus, if one BP layer matrix phase is the latency unit and PC-ALM/BP use comparable per-layer MAC throughput, the algorithmic lower bound is c>=2 for an exact synchronous iteration. c<2 requires either extra within-layer hardware that makes a PC matrix phase faster than the BP matrix phase, or a changed schedule/algorithm that overlaps/stales residuals.

## Corrected latency gate

With H hidden transitions, fully layer-parallel PC-ALM has best-case critical path

    L_ALM >= H + 2T

before small elementwise/final-gradient overheads.

A layer-distributed BP engine with forward and fused backward/weight-gradient has approximately

    L_BP ~= 2H

in the same matrix-phase units.

Therefore a latency crossover requires

    H + 2T < 2H
    T < H/2.

For H=31 this is T<15.5, i.e. about T<=15.

The more general cT<H expression remains algebraically valid, but c is not a free parameter: for the exact synchronous update implemented in src/predcircuit/pcalm.py, c has an algorithmic two-matrix-phase lower bound under equal MAC throughput.

## Consequence for current evidence

Existing held-out width-64 results have T=96/128/160/192, so even ideal layer concurrency remains far outside the latency crossover. At T=96, the relaxation budget alone is 6.2x the T=15 gate; at T=128 it is 8.5x.

This does not rule out PC-ALM hardware research. It changes the highest-value question: either reduce T by a large factor, or test a deliberately altered pipelined/stale-residual schedule and measure how much gradient geometry is lost. Merely assigning a favorable c<1 to the same synchronous equations is not a valid hardware speedup claim.
