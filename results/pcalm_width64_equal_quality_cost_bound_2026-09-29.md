# Width-64 equal-quality PC-ALM vs BP/ePC cost bound (2026-09-29)

## Purpose

Combine the held-out width-64 dyadic fixed-point quality result with the existing arithmetic/latency model. This is an analytical lower bound, not a measured FPGA speedup.

Matched network: depth 32 (H=31 hidden transitions), width N=64, batch B=4. Current hardware-friendly candidate uses state/update/dual = 16/14/12 bit with dyadic coefficients. The held-out budget sweep (seeds 845--859) gives:

- T=160: 15/15 useful, mean BP cosine 0.96968, mean norm ratio 0.94001, mean BP relative error 0.24782.
- T=192: 15/15 useful, mean BP cosine 0.97779, mean norm ratio 0.97265, mean BP relative error 0.21135.
- T=256: quality reference, mean BP cosine 0.98320, mean norm ratio 0.98215, mean BP relative error 0.18141.

T=192 is therefore the current balanced equal-quality candidate; T=160 is an aggressive throughput candidate.

## Arithmetic lower bound

For dense equal-width layers:

```
M_reverse = H*N^2 = 31*64^2 = 126,976 MACs
M_step    = 2*H*N^2 = 253,952 MACs / PC-ALM relaxation step
```

Parameter-gradient outer products are omitted symmetrically.

| method / budget | credit/dynamics MACs | ratio to one BP/ePC reverse sweep |
| --- | ---: | ---: |
| BP/ePC reverse sweep | 126,976 | 1x |
| PC-ALM T=160 | 40,632,320 | 320x |
| PC-ALM T=192 | 48,758,784 | 384x |
| PC-ALM T=256 | 65,011,712 | 512x |

The ratio is exactly 2T and is independent of width N under this dense-MAC model. Widening from N=8 to N=64 therefore does not create an arithmetic crossover; it only increases absolute work.

## Ideal layer-parallel latency bound

Suppose every layer has a P=N=64 dot-product engine, so one length-64 dot product is consumed per active cycle. This requires H*N = 31*64 = 1,984 MAC lanes/DSP-equivalents before considering duplication for simultaneous forward/transpose phases.

One PC-ALM relaxation step still needs two local matrix phases, while one conventional reverse sweep traverses H layers once. The ideal latency ratio is therefore

```
C_PC / C_reverse = 2T/H.
```

For H=31:

- T=160: 320/31 = 10.32x slower than one reverse sweep.
- T=192: 384/31 = 12.39x slower.
- T=256: 512/31 = 16.52x slower.

The ideal latency crossover remains T < H/2 = 15.5. The current balanced width-64 point T=192 is 12.39x above that boundary in latency ratio. Even the aggressive T=160 point is 10.32x above it.

Thus layer spatialization alone cannot produce a BP/ePC latency advantage at the current algorithmic budgets. Two matrix engines per layer can at most attack the local two-phase factor at roughly double the matrix resources and still leaves a large T/H gap.

## Persistent storage and local traffic

Using state=16 bit and dual=12 bit:

```
dynamic PC-ALM state = B*H*N*(16+12)
                      = 222,208 bit
                      = 27.125 KiB
```

Single-copy 16-bit weights:

```
weights = H*N^2*16
        = 2,031,616 bit
        = 248 KiB
```

Dynamic state is therefore only 10.94% of one resident weight copy at width 64. This is qualitatively different from width 8, where dynamic state was 84.4% of one weight copy. As width grows, weight capacity scales as O(HN^2) while PC-ALM state scales as O(BHN); lambda becomes less important for raw capacity, although it still doubles the relaxation-state component relative to sPC.

A one-read/one-write lower bound for dynamic state traffic is 54.25 KiB per relaxation step:

- T=160: 8.48 MiB
- T=192: 10.17 MiB

This traffic can stay on chip if banked appropriately, but it is repeated local movement that BP/ePC does not pay at the same multiplicity.

## Interpretation

The width-64 low-precision result is strong evidence that PC-ALM remains numerically implementable: 16/14/12-bit dyadic arithmetic reaches 15/15 useful seeds and high BP-gradient agreement. It does **not** establish computational superiority over BP/ePC.

In fact, combining quality and cost makes the current bottleneck sharper: the balanced T=192 point carries 384x the dense credit/dynamics MAC count of one reverse sweep and a 12.39x ideal latency disadvantage even under one fully spatialized P=64 engine per layer. The central research question therefore cannot be answered by "FPGA parallelism" alone.

The FPGA case must come from at least one of:

1. a much smaller useful relaxation budget (for pure latency crossover at depth 32, T<=15);
2. measured energy/op and locality advantages large enough to compensate a roughly 10--12x ideal latency gap and hundreds-fold work count;
3. a schedule/architecture that reuses or fuses the repeated local matrix work in a way not represented by the dense-MAC lower bound.

This is a useful negative result: width scaling improves the relative *capacity* cost of lambda but does not improve the fundamental arithmetic ratio.

## Consequence for the RTL gate

The minimal PC/sPC-switchable RTL remains justified as a mechanism/cost probe because PC-ALM has a measured deep-credit advantage over sPC and low precision is realistic. A claim of FPGA performance/energy superiority over ePC/BP is **not** justified yet.

The highest-value next measurement is therefore not another coefficient or bit-width sweep. It is an end-to-end layer-engine measurement of energy/cycle/operand traffic for one PC-ALM relaxation step versus one reverse-credit sweep, with resident weights and matched precision. That measurement can be combined with the 2T work multiplier to determine the T at which a real FPGA energy crossover could exist.
