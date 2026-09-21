# PC-ALM weight residency / reuse sweep

Date: 2026-09-21

## Question

The preceding write-energy bound showed that q9 stochastic rounding reduces changed-weight writes from 90.79% to 1.0092%, but that this saves <1% of total weight-memory transactions if all weights are reread from BRAM on every one of the 112 relaxation steps. This note asks how much cross-relaxation reuse is required before weight traffic stops dominating, and whether the required local storage is plausible.

This is a traffic/capacity model, not a vendor power estimate. In particular, moving a read from one BRAM to another equally expensive SRAM does not make the read energy disappear.

## Width-64 gate point

Use the main hardware-gate shape from `docs/pcalm-hardware-cost-model.md`:

- batch B = 4
- depth L = 32
- hidden width N = 64
- input I = 8
- output O = 4
- PC-ALM relaxation budget T = 112 for the stochastic-weight trajectory schedule
- compact persistent state = h14 + lambda12
- stored weights = q9

The dense weight count is

`W = I*N + (L-2)*N^2 + N*O = 123,648` scalars.

At 9 bits/weight this is

`W_bits = 1,112,832 bit = 135.844 KiB`.

The persistent PC-ALM state traffic from the existing model is

`S_tx = 412,672 bit/relaxation = 50.375 KiB/relaxation`,

so over T=112:

`S_total = 46,219,264 bit = 5.510 MiB/training-step`.

The q9 stochastic writeback payload, if only changed weights are physically written, is

`0.010092 * W * 9 = 11,231 bit ~= 1.371 KiB/training-step`.

The dense-FP32-like changed-weight fraction would instead correspond to about 123.3 KiB of q9-equivalent write payload. The sparse writeback is therefore small once read traffic is amortized.

## Reuse sweep

Let K be the number of relaxation steps served by one full q9 weight load across the modeled memory boundary. The number of full loads is `ceil(T/K)` and the boundary weight-read traffic is

`W_read(K) = ceil(112/K) * 135.844 KiB`.

| K | full weight loads | weight-read traffic | weight read / persistent-state traffic | q9 sparse writeback / (read+state+write) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 112 | 14.858 MiB | 2.696x | 0.0067% |
| 2 | 56 | 7.429 MiB | 1.348x | 0.0106% |
| 4 | 28 | 3.714 MiB | 0.674x | 0.0149% |
| 8 | 14 | 1.857 MiB | 0.337x | 0.0186% |
| 16 | 7 | 0.929 MiB | 0.169x | 0.0208% |
| 32 | 4 | 0.531 MiB | 0.096x | 0.0227% |
| 112 | 1 | 0.133 MiB | 0.024x | 0.0243% |

Two thresholds follow immediately.

1. Weight-boundary reads fall below persistent h+lambda traffic once K >= 4. The exact continuous break-even is about K > 2.70, so K=4 is the first swept point.
2. Even with complete K=112 reuse, persistent-state traffic remains about 41x larger than the one-time weight load. Once weights are resident, optimizing h/lambda movement matters much more than sparse weight writeback.

Thus cross-relaxation weight residency is useful, but it changes the bottleneck rather than eliminating memory traffic.

## Capacity cost of residency

A full q9 width-64 model needs 1.113 Mbit (135.8 KiB) of resident weight storage. This is not a free register cache. Implementing a duplicate full-model local copy in flip-flops would be implausibly expensive; implementing it in BRAM simply moves the repeated accesses to another BRAM and does not by itself remove physical SRAM read energy.

The useful architecture is therefore not `global BRAM -> duplicate BRAM cache`. It is to make the weight store itself local to the layer/MAC engine and avoid repeatedly crossing a more expensive interconnect or higher memory level. If a tile is held in registers close to P MAC lanes, only the tile's storage is duplicated, not the whole model.

For one 64x64 hidden-layer matrix:

- q9 matrix capacity = `64*64*9 = 36,864 bit = 4.5 KiB`;
- this is exactly one 36-kbit-class block in raw capacity terms;
- a P=32 row/column tile of 32 weights is only `32*9 = 288 bit` of immediate register storage, though a real bidirectional schedule may require multiple such words/banks.

This makes a tiled hierarchy plausible: layer-local BRAM holds the model, while a small register/LUTRAM working set feeds repeated forward / transpose-direction MAC schedules. The energy benefit then depends on how many MAC uses each local fetch serves, not merely on K at the model boundary.

## Important correction to the previous interpretation

`K=112` should not be interpreted as "weights are read only once physically during 112 relaxations." Matrix computation still consumes the weights on every relaxation. K only counts crossings of the chosen memory boundary. If the resident store is BRAM, its internal read ports still toggle every iteration. A genuine dynamic-energy claim requires a hierarchy with different per-access costs, e.g. BRAM -> register/LUTRAM tile, plus measured/synthesized access counts.

Therefore the strongest supported result is architectural:

- K >= 4 is enough to make *boundary* weight traffic smaller than existing PC-ALM h+lambda traffic at the width-64 point;
- after that point, state traffic is the first-order memory target;
- q9 stochastic sparse writeback remains useful for capacity and write-port activity, but contributes little to total traffic once reads/state are counted.

## Consequence for PC-ALM vs sPC

This bottleneck shift does not erase PC-ALM's current algorithmic advantage. The existing hardware model gives per-step persistent-state traffic ratio PC-ALM/sPC = 26/14 = 1.857, while the measured useful-credit iteration separation is >8 (PC-ALM useful by T=128; sPC still unusable at T=1024). Therefore reducing T remains much more important than eliminating the dual state.

But for a minimal core, the memory architecture should now expose two reuse levels explicitly:

1. layer-local weight storage / tile reuse for forward and transpose access;
2. h/lambda state buffering and banking, because state traffic becomes dominant around K>=4.

## Next discriminating step

Do not spend the next experiment on a larger K sweep: the analytical threshold is already clear. The next high-value hardware model is a concrete 64x64 layer tile schedule with P=32 lanes that counts physical accesses at each level (BRAM, LUTRAM/register tile, h buffer, lambda buffer) for both W and W^T directions. Compare at least two organizations:

- BRAM-streamed weights each MAC word;
- layer-local BRAM plus a small transposable register/LUTRAM tile.

The schedule should report cycles, BRAM reads, local-buffer reads, h/lambda reads+writes, and extra storage. Only after those counts exist does substituting device-specific energy/access values become scientifically meaningful.