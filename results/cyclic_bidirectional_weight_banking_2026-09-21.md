# Conflict-free cyclic banking for bidirectional 64x64 weight access

Date: 2026-09-21

## Question

The previous capacity bound left one key uncertainty: can a *single* physical copy of each 64x64 hidden-layer weight matrix sustain both row-oriented `W h` and column-oriented `W^T c` accesses without bank conflicts?

For the current candidate parallelisms P in {8,16,32,64}, the answer is yes at the logical-bank level. A simple cyclic mapping is sufficient; full transposed duplication is not required.

## Mapping

For weight W[r,c] in a 64x64 hidden layer, with P dividing 64, assign

`bank(r,c) = (r + c) mod P`.

Within each bank, store the remaining coordinates (and layer index) as the local address.

A forward P-wide group fixes r and accesses P consecutive columns `c0 ... c0+P-1`. Its bank indices are

`(r+c0+k) mod P`, k=0..P-1,

which are exactly all P residues once each.

A transpose P-wide group fixes c and accesses P consecutive rows `r0 ... r0+P-1`. Its bank indices are

`(r0+k+c) mod P`, k=0..P-1,

again exactly all P residues once each.

Therefore both access directions are conflict-free for aligned/consecutive P-element groups. This is a constructive counterexample to the concern that `W` versus `W^T` necessarily requires a second transposed weight copy.

## BRAM packing consequence

There are 30 hidden 64x64 matrices. Hidden-layer weight bits are

`30 * 64 * 64 * 14 = 1,720,320 bit`.

Under the cyclic mapping, each of P logical banks stores an equal share:

`bits_per_bank = 1,720,320 / P`.

Using 36,864 bit per RAMB36 as the same capacity model used previously, a straightforward one-logical-bank-per-BRAM-chain realization needs:

| P | bits/logical bank | RAMB36 per bank | hidden-weight RAMB36 | pure hidden capacity floor |
|---:|---:|---:|---:|---:|
| 8 | 215,040 | 6 | 48 | 47 |
| 16 | 107,520 | 3 | 48 | 47 |
| 32 | 53,760 | 2 | 64 | 47 |
| 64 | 26,880 | 1 | 64 | 47 |

The small input/output edge matrices contain only

`(8*64 + 64*4)*14 = 10,752 bit`.

There is enough aggregate slack in every row above to hold these edge weights too (subject to an address layout that preserves the required ports), so they do not force another RAMB36 by capacity alone.

Thus a plausible single-copy bidirectional design has an important discontinuity:

- P=8 or 16: about 48 RAMB36 for weights, essentially the 47-block capacity floor;
- P=32 or 64: about 64 RAMB36 with the simple cyclic realization, because each logical bank must remain independently readable even though its capacity is under-filled.

This is still far below the 94-RAMB36 lower bound of storing complete W and W^T copies, but it shows that the cost of high spatial parallelism can appear as *bank fragmentation* rather than access stalls.

## Relation to PC-ALM overhead

The measured PC-ALM dual state costs 3 RAMB36. Relative to the cyclic single-copy weight design, lambda therefore adds:

- 3/48 = 6.25% at P=8/16;
- 3/64 = 4.69% at P=32/64.

So the PC-ALM-specific lambda storage remains a small BRAM term compared with the shared weight subsystem. More importantly, increasing P from 16 to 32 can cost +16 RAMB36 in this simple memory organization, over five times the entire measured lambda-memory increment, even though the algorithm is unchanged.

## What this does and does not prove

This proves conflict freedom for the regular dense 64x64 hidden-layer traces at the logical-bank level. It does not yet prove a particular RTL will infer the predicted RAMB36 count, meet timing, or avoid extra routing/control cost. It also assumes P divides 64 and P-wide groups are scheduled consecutively/aligned in the natural way.

The result changes the next experiment. A duplicated W/W^T memory is no longer the right default reference. The useful hardware comparison is now:

1. cyclic single-copy banking with P={8,16,32,64};
2. measured RAMB36 and post-place Fmax;
3. forward and transpose stall counters, expected to be zero for the regular hidden-layer trace;
4. comparison of the P=16 -> P=32 cycle halving against the predicted +16-RAMB36 fragmentation step.

If RTL confirms this model, P=16 becomes a particularly interesting area-efficiency point: it preserves the near-capacity-minimum 48-block weight store while halving matrix cycles relative to P=8. P=32 must earn its extra 16 BRAM through latency/energy benefits rather than being assumed superior from DSP count alone.
