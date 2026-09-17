# PC-ALM mixed-precision hardware cost model

This note turns the current width-64 credit and precision results into a first-order hardware model. It is intentionally analytical: it is a gate before RTL, not a synthesis claim.

## Design point

Current software evidence motivates the provisional mixed-precision point

- hidden state `h`: 14 bit,
- state-update / residual datapath: 14 bit,
- PC-ALM dual `lambda`: 12 bit.

The matched width-64 diagnostic uses `depth=32`, `width=64`, batch size 4, input dimension 8, output dimension 4. `ResidualMLP` has `depth - 1 = 31` hidden states because the final block is the supervised output block.

Define

- `B`: batch size,
- `L`: model depth including the output block,
- `N`: hidden width,
- `I`: input dimension,
- `O`: output dimension,
- `S = B (L - 1) N`: number of hidden-state scalars.

For the current diagnostic,

`S = 4 * 31 * 64 = 7,936` scalars.

## Persistent inference-state storage

Ignoring weights, traces, and optional double buffering:

- sPC with 14-bit hidden state: `14 S` bits,
- PC-ALM at 14/14/12: `(14 + 12) S = 26 S` bits.

At width 64 this is

| method | persistent inference state | bytes | KiB |
| --- | ---: | ---: | ---: |
| sPC h14 | 111,104 bit | 13,888 | 13.5625 |
| PC-ALM h14 + lambda12 | 206,336 bit | 25,792 | 25.1875 |
| PC-ALM dual increment | +95,232 bit | +11,904 | +11.625 |

Thus the dual raises persistent inference-state capacity by `12 / 14 = 85.7%` relative to h14 sPC. This is the main PC-ALM storage penalty at the current design point.

The scaling law is simple: the extra dual storage is `12 B (L - 1) N` bits, linear in depth and width. It does **not** scale as `N^2` like the weights.

## Per-relaxation-step arithmetic

For a streaming implementation that reuses each residual rather than materializing an autograd graph, a first-order matrix-MAC count for one hidden-state relaxation step is

`M_base = B [ I N + 2 (L - 2) N^2 + 2 N O ]`.

The terms are:

1. forward hidden predictions: `B [I N + (L-2) N^2]`,
2. transpose-direction residual propagation: `B (L-2) N^2`,
3. output prediction and its transpose contribution: approximately `2 B N O`.

For `B=4, L=32, N=64, I=8, O=4`, this gives

`M_base = 987,136 MAC/step`.

PC-ALM additionally performs the elementwise dual update

`lambda <- lambda + alpha r`.

If multiplication by `alpha` is conservatively charged as one multiplier operation per hidden scalar, that adds at most

`M_dual = S = 7,936 multiply-add-like operations/step`,

or only `0.804%` of the matrix-MAC count. If constant-coefficient `alpha` is implemented with shift/add or folded into the residual datapath, the DSP penalty can be lower. This must be checked by synthesis later.

So at fixed `T`, PC-ALM's important overhead is state storage/access, not matrix arithmetic.

## State-memory traffic

A deliberately conservative lower-order model counts one read and one write of every persistent inference state per relaxation step, with residuals streamed rather than stored persistently:

- sPC h14: `2 * 14 S = 222,208 bit/step = 27.125 KiB/step`,
- PC-ALM h14 + lambda12: `2 * 26 S = 412,672 bit/step = 50.375 KiB/step`.

Again PC-ALM is `26/14 = 1.857x` heavier per step in state traffic.

This is not total memory traffic. Weight traffic can dominate unless weights are kept/reused locally, and an implementation with ping-pong state buffers changes physical BRAM port requirements. The ratio is nevertheless useful because it isolates the cost introduced by the dual state.

## Does the lower relaxation budget pay for lambda?

The width-64 matched diagnostic found useful first-layer credit for FP32 PC-ALM already at `T=128` (5/5 seeds), while sPC was still 0/5 at `T=256`. Therefore there is no equal-quality sPC break-even point yet: sPC has not reached the criterion by the largest measured budget.

A weaker, conservative comparison is still informative: compare PC-ALM at 128 steps with sPC at 256 steps even though the latter has worse credit.

### Matrix work

Including the conservative dual multiply cost,

`128 * (987,136 + 7,936) / (256 * 987,136) = 0.504`.

So PC-ALM uses about **50.4%** of the matrix-plus-dual arithmetic budget of a 256-step sPC run.

### Persistent-state traffic

Using the state-only traffic model,

`128 * 26 / (256 * 14) = 0.929`.

So despite carrying the extra 12-bit dual, PC-ALM at 128 steps uses about **92.9%** of the persistent-state traffic of sPC at 256 steps.

This gives a useful hardware break-even rule. Let `R = T_spc / T_pcalm`. Ignoring the small dual arithmetic term, PC-ALM wins on total persistent-state traffic when

`R > 26 / 14 = 1.857`.

For arithmetic, because the dual update is only about 0.8% of the matrix work at the current shape, the break-even is approximately

`R > 1.008`.

The measured budget separation is already `R >= 2` in the sense that PC-ALM passes at 128 while sPC still fails at 256. Therefore the current result clears both analytical break-even ratios, but **not yet an equal-quality comparison** because the minimum successful sPC `T` is unknown.

## Scaling implications

For deep, width-N residual MLPs:

- matrix work per step is `Theta(B L N^2)`,
- h storage is `Theta(B L N)`,
- PC-ALM's additional lambda storage/update is also `Theta(B L N)`.

Therefore the *arithmetic* fraction spent on the dual shrinks roughly as `1/N`; the *state-memory* fraction does not. Wider layers make the dual computationally cheaper relative to matrix work, but do not remove its BRAM/bandwidth cost.

This distinction matters for architecture choice. A highly parallel design with enough MACs can become state-bandwidth limited, in which case the 1.857x per-step state traffic is real. A MAC-shared design remains compute dominated, where reducing `T` is much more valuable and the lambda arithmetic is nearly free.

## RTL gate status

The software evidence now satisfies two parts of the proposed RTL gate for at least the width-64/depth-32 diagnostic:

1. **algorithmic advantage over sPC:** PC-ALM passes the first-layer credit criterion at T=128 for 5/5 seeds; sPC is 0/5 at T=256;
2. **plausible low precision:** h14/update14/lambda12 is close to FP32 in the current 5-seed mixed-precision probe.

The analytical cost model also shows that a 2x or larger reduction in required T is enough to amortize the added dual state even under a state-traffic model.

This is promising but is **not yet sufficient to start full RTL**. The missing pieces are:

- find or bound the minimum successful sPC budget so the T advantage is measured at equal credit quality;
- validate h14/update14/lambda12 on held-out seeds rather than only the 5 exploratory seeds;
- turn the analytical model into an FPGA-specific BRAM/DSP/LUT/cycle model, including weight placement, state buffering, memory ports, and the chosen layer/neuron/MAC parallelism;
- compare against ePC and BP on the digital-compute side before claiming a system-level advantage.

The next highest-value experiment is therefore a holdout validation of the 14/14/12 design point together with a targeted sPC budget extension, rather than RTL implementation.