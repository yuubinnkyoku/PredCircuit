# Block PC-ALM credit propagation: latency/work lower bound

Date: 2026-09-24

## Motivation

The measured synchronous depth-32 PC-ALM dynamics advances credit support by about one layer per relaxation step, while reverse Gauss-Seidel can move support farther in one sweep only by serializing layer updates and strongly attenuating the same-sweep path. The proposed intermediate design is a block-local solve of B adjacent layers. This note asks whether block sizes B={1,2,4,8} can by themselves beat one explicit BP/ePC reverse sweep when all internal work is charged.

Let H=31 hidden transitions and width N=8. One reverse matrix-vector operation per transition costs N^2 MACs, so a BP/ePC reverse sweep costs H*N^2 = 1,984 MACs. The existing PC-ALM relaxation model needs both forward/residual and transpose/credit matrix-vector operations, i.e. 2*H*N^2 = 3,968 MACs per full-network relaxation.

## Causality bound for non-overlapping B-layer blocks

Suppose a block operator can expose credit across at most B adjacent layer boundaries before another inter-block synchronization is required. Even granting perfect parallel execution of all currently independent blocks, crossing H boundaries requires at least

    waves(B) = ceil(H / B)

inter-block waves.

For H=31:

| B | minimum waves |
|---:|---:|
| 1 | 31 |
| 2 | 16 |
| 4 | 8 |
| 8 | 4 |

This removes the T>=31 outer-step wall only if the B-layer operator really propagates through B layers internally. The internal latency cannot be ignored.

## If the block is only a fused sequence of nearest-neighbor updates

A B-layer Gauss-Seidel/fused block has a true dependency chain of at least B local layer updates. With one local matrix-vector stage taking c cycles, the optimistic end-to-end critical path is bounded by

    ceil(H/B) * B * c >= H*c.

Thus B merely trades 31 global waves for fewer, longer waves; it does not beat the depth-proportional reverse dependency. For B={1,2,4,8}, the normalized lower bounds ceil(31/B)*B are respectively 31, 32, 32, 32 local stages. This is essentially the same serial depth as a BP/ePC reverse sweep before charging PC-ALM's extra residual/dual work.

## If the block uses a direct dense multi-layer credit operator

To genuinely collapse B sequential nearest-neighbor dependencies, the block must implement a composed/multi-layer operator rather than merely reschedule them. A generic dense operator over B width-N layer states has dimension BN and therefore O((BN)^2)=O(B^2*N^2) coefficients/MACs per application. Even under perfect parallelism across blocks, the work across H/B blocks scales as O(H*B*N^2), i.e. approximately B times a single reverse sweep, while coefficient/state storage also grows with B.

For the width-8, H=31 reference, the simple H*B*N^2 work scale is:

| B | approximate composed-credit MACs | ratio to BP/ePC reverse |
|---:|---:|---:|
| 1 | 1,984 | 1x |
| 2 | 3,968 | 2x |
| 4 | 7,936 | 4x |
| 8 | 15,872 | 8x |

This is not an implementation claim for a particular solver; it is a design-space warning. A block method only creates a useful FPGA opportunity if its multi-layer operator has exploitable structure (sparsity, low rank, recurrence/prefix composition, or a reusable factorization) that makes the effective cost materially subquadratic in B while retaining usable BP-gradient geometry.

## Consequence

Plain block size is not a free new axis. There are two limiting cases:

1. **Nearest-neighbor block sweep:** O(B) internal dependency depth, so end-to-end latency remains O(H); this approaches a reverse sweep and sacrifices the layer-parallel advantage.
2. **Direct generic B-layer solve:** can reduce synchronization depth toward O(H/B), but generic arithmetic/storage grows toward O(H*B*N^2), already 4x one reverse sweep at B=4 and 8x at B=8 in the reference network.

Therefore an exhaustive B={1,2,4,8} experiment using only serial Gauss-Seidel inside each block has low information value: it cannot change the latency lower bound. The higher-value target is a *structured associative/hierarchical credit operator* whose compositions can be evaluated as a tree/prefix network. Such an operator could in principle reduce propagation depth from O(H) toward O(log H), but its arithmetic, wiring, coefficient storage, and gradient fidelity must be charged explicitly. This is qualitatively different from merely grouping layers.

## Hardware decision rule

Do not advance a block PC-ALM design toward RTL solely because it reaches the first layer in fewer named outer iterations. Require all three:

- measured BP-gradient cosine/norm at matched total MAC count;
- physical critical-path/schedule depth including all intra-block dependencies;
- block operator cost growing slowly enough with B that it does not simply recreate or exceed an explicit ePC/BP reverse sweep.

The next algorithmic experiment should therefore test a structured multi-layer composition (for example a linearized/prefix credit recurrence) against explicit reverse propagation, rather than a plain serial block Gauss-Seidel schedule.