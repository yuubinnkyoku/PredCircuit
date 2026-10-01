# T64 PC-ALM hardware bound update (2026-10-01)

## Scope

This note connects the one-update depth-32/width-64 fixed-point pilot to the existing hardware cost model. It is an analytical bound, not an FPGA speedup claim. The T64 point is still only one seed/update and must survive multi-update training before becoming the default operating point.

## Measured algorithmic point reused

From `pcalm_one_update_short_budget_pilot_2026-10-01.md`:

- fixed official-like PC-ALM, update/state/dual = 14/16/12 bit, T=64;
- global same-weight BP cosine = 0.9094;
- layer-0 cosine = 0.865;
- layer-0 gradient norm / BP = 0.646;
- dual zero fraction = 0.5266;
- max |lambda| = 0.0684;
- no dual saturation;
- matched sPC T=80 has layer-0 norm/BP = 4.68e-8.

This does not replace the older held-out long-T evidence. It is a new optimistic short-budget point.

## Width-64 persistent-state traffic at T64

For B=4, H=31, N=64 there are

`S = B H N = 7936`

persistent hidden-state coordinates.

Using the actual pilot state16 + lambda12 format, the optimistic one-read/one-write lower bound is

`Q_step = 2 S (16 + 12) = 444,416 bit = 54.25 KiB / step`.

At T=64:

`Q_T64 = 3.391 MiB`

of local persistent-state traffic.

A matched state16 sPC step is

`2 S 16 = 253,952 bit = 31.0 KiB / step`.

Therefore PC-ALM pays a 28/16 = 1.75x state-traffic penalty per relaxation step. The iteration-count crossover against state16 sPC is

`T_sPC / T_pcalm > 1.75`.

Against the pilot sPC T=80, T64 PC-ALM already uses 1.40x as much persistent-state traffic (3.391 MiB versus 2.422 MiB), but the sPC credit is essentially absent at layer 0, so this is not an equal-quality comparison.

Using the older right-censored width-64 result that sPC is still below the useful-credit gate at T=1024, a T64 PC-ALM point would imply an optimistic one-sided traffic ratio

`(64*28)/(1024*16) = 0.109375`.

That is at most 10.94% of the simple persistent-state traffic of the still-failing T1024 sPC endpoint, if the T64 quality survives the held-out/multi-update gate. This is not yet a measured equal-quality speedup.

## Dense arithmetic bound

The existing width-64 model gives 987,136 matrix MACs per relaxation step. Charging one dual coefficient operation per hidden scalar adds 7,936 operations, for 995,072 matrix-plus-dual operations per T64 step.

Thus T64 costs approximately

`64 * 995,072 = 63,684,608`

matrix-plus-dual operations.

Relative to a BP/ePC-style single reverse credit sweep, the simplified whole-update dense-work ratio remains approximately

`(2T+2)/3 = 43.33x`

at T=64. The short-budget pilot therefore improves the earlier T112/T128 arithmetic picture substantially but does not create a compute crossover against BP/ePC.

## Layer-parallel latency bound

With H=31 and two matrix phases per PC-ALM relaxation step, the ideal dependency-phase bound is

`H + 2T = 159`

matrix phases at T64, versus about `2H = 62` for a fused forward/reverse BP schedule.

The ideal ratio is therefore about 2.56x. A pure layer-parallel latency crossover still requires roughly

`T < H/2 = 15.5`.

Hence T64 is valuable, but not enough to justify a claim that spatialization alone beats BP/ePC.

## Hardware interpretation

The new T64 pilot strengthens PC-ALM versus sPC: if it survives multi-update/held-out validation, the extra lambda state can be amortized by the relaxation-budget reduction despite state16+lambda12 carrying 1.75x the persistent-state traffic per step.

It does not clear the stronger PC-ALM-versus-ePC/BP gate. Against digital T=1-style credit propagation, the repeated matrix work remains the dominant obstacle. Weight residency is mandatory; otherwise forward/transpose weight streaming reproduces an approximately 2T traffic penalty.

The hardware claim should therefore remain conditional:

1. PC-ALM versus sPC: promising short-budget credit propagation with a plausible local-memory crossover.
2. PC-ALM versus ePC/BP: no performance/energy crossover established; must be supported by measured locality/energy or a much smaller T.
3. Full RTL accelerator: still premature. A minimal switchable core/resource model remains the appropriate next hardware artifact.

## Next gate

Validate fixed official-like T64 over multiple updates and held-out seeds before replacing the established long-T operating point. In parallel, build a banked weight/state source around the existing shared MAC datapath and measure operand stalls. The software comparison should include ePC, because its digital reparameterization removes the sPC signal-decay bottleneck and is the relevant digital competitor rather than sPC alone.
