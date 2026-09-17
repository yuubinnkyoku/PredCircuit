# PC-ALM mixed-precision hardware cost model

This note turns the current width-64 credit, low-precision, and compute results into a first-order hardware model. It is intentionally analytical: it is a gate before RTL, not a synthesis claim.

## Current algorithmic candidate

The robust candidate is now **leaky-dual PC-ALM**, not pure PC-ALM:

`lambda <- (1 - gamma) lambda + alpha r`, with `gamma` around 0.01--0.02 in the current deep residual-MPL diagnostics.

At depth 32 / width 64, official Sakana depth-32 `state_lr=0.234285`, and held-out seeds 845--859, pure PC-ALM degrades at long T, whereas leak 0.02 at T=256 gives 15/15 useful first-layer credit and near-BP FP32 geometry. The leak is therefore an engineering stabilization variant and must remain clearly separated from pure Sakana PC-ALM in reporting.

The low-precision evidence no longer supports one universal bit width. The currently defensible points are:

- compact: hidden state `fixed14_i3`, update/residual `fixed14_i1`, dual `fixed12_i1`;
- alternative compact grid: hidden state `fixed15_i3`, update/residual `fixed13_i1`, dual `fixed12_i1`;
- higher-fidelity: hidden state `fixed16_i3`, update/residual `fixed14_i1`, dual `fixed12_i1`.

All three use a 12-bit dual. State/update precision must be selected **jointly** because the iterative dynamics can lock to an unfavorable quantization lattice; more bits are not monotonically better.

## Network shape used for the main gate

The matched width-64 diagnostic uses:

- batch `B=4`,
- model depth `L=32` including the output block,
- hidden width `N=64`,
- input dimension `I=8`,
- output dimension `O=4`.

`ResidualMLP` has `L - 1 = 31` hidden states. Define

`S = B (L - 1) N` hidden-state scalars.

For the current shape:

`S = 4 * 31 * 64 = 7,936` scalars.

## Persistent inference-state storage

For an h14 compact implementation, ignoring weights, traces, and optional double buffering:

- sPC with h14: `14 S` bits,
- PC-ALM with h14 + lambda12: `(14 + 12) S = 26 S` bits.

At width 64:

| method | persistent inference state | bytes | KiB |
| --- | ---: | ---: | ---: |
| sPC h14 | 111,104 bit | 13,888 | 13.5625 |
| PC-ALM h14 + lambda12 | 206,336 bit | 25,792 | 25.1875 |
| PC-ALM dual increment | +95,232 bit | +11,904 | +11.625 |

Thus the dual raises persistent inference-state capacity by `12 / 14 = 85.7%` relative to h14 sPC. This is the main PC-ALM storage penalty at the compact design point.

For the state15/update13 alternative, PC-ALM persistent inference state becomes `27 S`, while an h15 sPC baseline uses `15 S`; the corresponding per-step persistent-state ratio is `27/15 = 1.8`. Whether h14/update14 or h15/update13 is cheaper depends on BRAM packing versus arithmetic width and must be resolved on a concrete FPGA.

The extra dual storage is `12 B (L - 1) N` bits: linear in depth and width, not `N^2` like dense weights.

## Per-relaxation-step arithmetic

For a streaming implementation that reuses each residual rather than materializing an autograd graph, a first-order matrix-MAC count for one hidden-state relaxation step is

`M_base = B [ I N + 2 (L - 2) N^2 + 2 N O ]`.

The terms are:

1. forward hidden predictions: `B [I N + (L-2) N^2]`,
2. transpose-direction residual propagation: `B (L-2) N^2`,
3. output prediction and its transpose contribution: approximately `2 B N O`.

For the current shape:

`M_base = 987,136 MAC/step`.

PC-ALM additionally performs an elementwise dual update. Charging the `alpha r` coefficient operation conservatively as one multiply-add-like operation per hidden scalar gives

`M_dual = S = 7,936 operations/step`,

only `0.804%` of the matrix-MAC count. Dual leak adds another constant-coefficient operation `(1-gamma) lambda`; the exact DSP/LUT price depends on coefficient representation and whether the two constant coefficients are implemented by DSPs, CSD/shift-add logic, or folded into an existing datapath. It does not change the asymptotic conclusion: the dual arithmetic is `Theta(B L N)` versus `Theta(B L N^2)` matrix work.

At fixed T, PC-ALM's important overhead is therefore persistent state and state traffic, not matrix arithmetic.

## State-memory traffic

A deliberately conservative model counts one read and one write of every persistent inference state per relaxation step, with residuals streamed rather than stored persistently:

- sPC h14: `2 * 14 S = 222,208 bit/step = 27.125 KiB/step`,
- PC-ALM h14 + lambda12: `2 * 26 S = 412,672 bit/step = 50.375 KiB/step`.

PC-ALM is `26/14 = 1.857x` heavier per step in persistent-state traffic at this compact point.

This is not total memory traffic. Weight traffic can dominate unless weights are kept/reused locally, and ping-pong state buffers change physical BRAM-port requirements. The ratio is useful because it isolates the dual-state penalty.

## Updated relaxation-budget evidence

The earlier 128-versus-256 comparison is obsolete. The current evidence is stronger:

- leaky PC-ALM reaches useful deep credit by T=128 on the width-64 matched exploratory seeds;
- on official Sakana `state_lr`, leak 0.02 at T=256 is 15/15 useful on held-out seeds and nearly BP-aligned in FP32;
- sPC remains 0/5 useful at T=320, 384, 512, 768, and **1024**; at T=1024 its first-layer norm is still only about 0.20 times BP.

Therefore the measured iteration-budget separation satisfies

`T_sPC,min / T_PCALM > 1024 / 128 = 8`

for the known T=128 PC-ALM success point. This is an **iteration-count lower bound**, not an 8x wall-clock speedup claim: the equal-quality sPC success point has not been observed, and per-step implementations differ.

### Arithmetic lower-bound comparison

Even using the conservative dual arithmetic charge and only the observed lower bound,

`128 * (987,136 + 7,936) / (1024 * 987,136) ~= 0.126`.

Thus PC-ALM's matrix-plus-dual operation budget at T=128 is below about 12.6% of a 1024-step sPC run that still fails the credit gate.

### Persistent-state traffic lower-bound comparison

For h14/lambda12:

`128 * 26 / (1024 * 14) ~= 0.232`.

So even after carrying the extra dual, the compact PC-ALM point uses about 23.2% of the simple persistent-state traffic of a 1024-step h14 sPC run. Again, the sPC run still has worse credit, so this is not yet an equal-quality benchmark.

The generic h14/lambda12 state-traffic break-even remains

`T_sPC / T_PCALM > 26/14 = 1.857`.

The measured budget lower bound `>8` clears this analytical threshold with substantial margin.

## Low precision is a dynamical co-design problem

The key numerical result is no longer merely "12--14 bit works". The hidden-state and update grids interact with the effective state learning rate.

With official `state_lr=0.234285` and batch 4, the implementation's effective state step is

`eta_eff = 4 * 0.234285 = 0.93714`.

For state `fixed15_i3` and update `fixed14_i1`:

- update-gradient LSB = `2^-12`,
- one update quantum requests `eta_eff * 2^-12 = 2.2879e-4`,
- state15 half-LSB = `2^-12 = 2.4414e-4`.

A one-quantum update therefore falls just below the state rounding threshold. The held-out trajectory diagnosis shows the consequence: late in relaxation, `fixed15_i3 + fixed14_i1` has about 99.94% zero state-coordinate updates and realizes only about 28% of the requested state-step norm; useful credit plateaus at 8/15.

Changing only the update grid falsifies an intrinsic "15-bit state is bad" interpretation:

| state15 update grid | useful | mean cosine | mean BP-relative error |
| --- | ---: | ---: | ---: |
| fixed12_i1 | 15/15 | 0.9655 | 0.2720 |
| **fixed13_i1** | **15/15** | **0.9667** | **0.2547** |
| fixed14_i1 | **8/15** | 0.8830 | 0.5089 |
| fixed15_i1 | 12/15 | 0.9217 | 0.4087 |
| fixed16_i1 | 13/15 | 0.9408 | 0.3503 |
| FP32 update | 15/15 | 0.9490 | 0.3234 |

Among the fine-grid points excluding the intentionally coarse 12-bit regime, the analytically predicted state-rounding deadzone orders the observed BP-relative error exactly in this sweep. Making the update path one bit **wider** from 13 to 14 bits can make learning much worse.

This implies an RTL rule: do not select h/update bit widths independently. The relative lattice spacing and `eta_eff` must be checked so the common smallest nonzero update magnitudes do not repeatedly fall below the hidden-state half-LSB. A simple width-only monotonicity assumption is unsafe for iterative local learning.

## Scaling implications

For deep width-N residual MLPs:

- matrix work per step is `Theta(B L N^2)`,
- h storage is `Theta(B L N)`,
- PC-ALM dual storage/update is `Theta(B L N)`.

Therefore the arithmetic fraction spent on the dual shrinks roughly as `1/N`; the BRAM/bandwidth fraction does not. Wider layers make the dual computationally cheaper relative to matrix work, but do not remove its state-memory cost.

A highly parallel design with many MACs can become state-bandwidth limited, where the per-step dual traffic matters. A MAC-shared design remains compute dominated, where reducing T is more valuable and dual arithmetic is nearly free.

## RTL gate status

For the current depth-32/width-64 research point, three preconditions are now substantially supported:

1. **algorithmic advantage over sPC:** leaky PC-ALM has a measured useful-credit point while sPC is still unusable at 1024 steps, giving a required-iteration lower-bound ratio greater than 8;
2. **realistic low precision:** multiple held-out fixed-point designs are 15/15 useful with a 12-bit dual; compact h14/update14/dual12 and h15/update13/dual12 points are both viable, while h16/update14/dual12 provides a higher-fidelity option;
3. **dual overhead can be amortized:** the observed T separation is far beyond the 1.857x simple state-traffic break-even and the dual arithmetic overhead is small relative to dense matrix work.

These results justify moving from a vague RTL gate to a **minimal-core planning phase**, but not yet a full accelerator implementation or system-level efficiency claim.

The remaining high-value gaps are:

- generalize the lattice-lock rule beyond the state15/update14 anomaly (a diagonal holdout is in progress);
- perform end-to-end learning, because one-step gradient geometry is not training accuracy;
- complete a fair ePC/BP digital-compute comparison, including wall-clock/runtime characteristics on GPU/NPU rather than only analytical MAC counts;
- map the compact and higher-fidelity fixed-point points onto a specific FPGA's BRAM/DSP/LUT packing and cycle schedule;
- choose hardware-friendly binary/CSD approximations for `alpha`, `gamma`, and `eta_h`, then verify that those coefficient quantizations preserve the credit window.

The next RTL artifact should therefore be a **parameterized minimal PC/sPC/leaky-PC-ALM core and resource model**, not yet a full network accelerator. The core should make `alpha=0` (or equivalent dual disable) a clean sPC mode, expose the dual-leak coefficient, and allow state/update/dual widths to be independently configured so the lattice-alignment rule can be tested in synthesis.