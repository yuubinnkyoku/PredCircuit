# Measured dual-cost crossover update (2026-09-16)

This note combines the fair scalar XC7 common-credit-core result with the later, more trustworthy vector dual-engine synthesis and behavioral validation. The scalar result remains useful for understanding the recurrence cost, but it is no longer used as the network-level area estimate.

## Scalar recurrence measurement

The fair common-core XC7 synthesis gives, per scalar 12-bit PC-ALM dual element:

- 221 LUT2--LUT6 primitives,
- 31 CARRY4,
- 13 FF-class cells,
- 0 DSP48.

For H=L-1 hidden layers of width N, fully spatial replication would scale approximately as 221 H N LUT primitives, 31 H N CARRY4s and 13 H N FF-class cells. For H=31, N=8 this gives 54,808 LUT primitives, 7,688 CARRY4s and 3,224 FF-class cells before storage/interconnect. This remains an upper-bound-style argument against one updater per lambda state, not a physical network area estimate.

## Verified shared vector engine supersedes scalar extrapolation

The later 8-lane RTL keeps all 992 persistent 12-bit lambda values in a 124 x 96-bit synchronous memory and streams eight lambda/residual pairs per issued word. XC7 synthesis maps it to:

- 1,408 estimated LCs,
- 266 CARRY4,
- 129 FDRE,
- 3 RAMB18E1,
- 0 DSP48.

The shared state path has also passed a behavioral RAM read/update/writeback test across all eight lanes, including the trajectory `0 -> 237 -> 234`. Therefore these vector figures, rather than `8 * scalar cost`, are the current hardware-cost reference.

## Corrected overlap model

There are 992 scalar dual updates per relaxation step for batch=4 and H*N=248. With P_dual lanes, the current synchronous-read pipeline needs

    C_dual(P_dual) = ceil(992 / P_dual) + 1

cycles for a full sweep, where the extra cycle is the pipeline drain. The major MAC-side analytical budget is 126 cycles/step at P_mac=128.

Thus the cycle-only hiding condition is

    ceil(992 / P_dual) + 1 <= 126.

Equivalently,

    ceil(992 / P_dual) <= 125,

so the minimum integer lane count remains

    P_dual,min = 8.

But the old statement that eight lanes take exactly 124 cycles was incomplete: the verified engine needs an estimated 125-cycle sweep including drain. This matters because the apparent two-cycle margin against the 126-cycle MAC path is actually only one cycle.

The real-time condition is stricter:

    C_dual / f_dual <= C_MAC / f_MAC.

At eight lanes,

    f_dual / f_MAC >= 125/126 = 0.99206.

So eight lanes are the minimum cycle-count design, but they only remain throughput-neutral if the routed dual engine reaches at least 99.21% of the MAC-side clock.

## 16-lane comparison

Keeping the same 11,904 state bits while doubling to 16 lanes gives a 62 x 192-bit logical memory and an estimated 63-cycle sweep. XC7 synthesis gives:

- 2,826 estimated LCs,
- 530 CARRY4,
- 432 FDRE,
- 64 RAM64M,
- 0 RAMB18E1,
- 0 DSP48.

Its real-time hiding condition relaxes to

    f_dual / f_MAC >= 63/126 = 0.5.

However, logic almost exactly doubles: 2,826 / 1,408 = 2.007. The LC-cycle products are 176,000 for 8 lanes and 178,038 for 16 lanes, only +1.16% apart. Parallelism is therefore acting mainly as an area-for-latency exchange, while the memory mapping becomes qualitatively worse at 16 lanes under the current XC7/Yosys flow (BRAM -> distributed RAM).

## Width dependence and architectural consequence

The qualitative width argument still holds: dual work is O(B H N), while dense layer MAC work is O(B H N^2), so a shared dual engine becomes easier to amortize as width grows. Very narrow networks are the hardest overlap case because the MAC path shrinks relative to the O(H N) dual sweep.

The current architectural rule is therefore:

1. store lambda in compact memory rather than replicating recurrence logic per state;
2. share a small bank of fixed-point dual-update lanes;
3. choose the minimum lane count satisfying the *real-time* overlap constraint, not merely the cycle-count constraint;
4. include the discrete RAM mapping regime in the resource model;
5. do not increase beyond eight lanes unless post-route timing shows `f_dual/f_MAC < 0.99206`.

This supersedes the earlier scalar-times-lanes area estimate and the earlier 124-cycle statement. The next decisive measurement is post-place-and-route timing for the verified 8-lane dual engine versus the comparable MAC datapath.