# Tang Nano 9K PC-ALM mapping bound (2026-09-25)

This note refines the layer-parallel hardware model for the current width-64, depth-32, T=80 candidate.

## Device bound

Tang Nano 9K / GW1NR-9 provides 468 Kbit BSRAM and 20 18x18 multipliers. A signed 14x14 weight/state MAC fits one 18x18 multiplier.

For a 64x64 hidden layer at 14-bit weights:
- weights/layer = 64*64*14 = 57,344 bit.
- h + lambda for batch=4 = 4*64*(16+12) = 7,168 bit/layer.
- weights + h + lambda = 64,512 bit/layer (ignoring buffers/fragmentation).

Raw BSRAM capacity therefore permits at most floor(468*1024 / 64,512) = 7 such resident layers; practical P_L will be lower after buffering and BSRAM fragmentation.

## DSP-balanced layer parallelism

Let P_L be resident/active layers and P_MAC=floor(20/P_L) multipliers per active layer. Ignoring pipeline fill, local scalar work, and memory-port stalls, the dominant 64-wide matrix-vector term scales as

    F(P_L) = ceil(31/P_L) * ceil(64/P_MAC).

For candidate P_L:
- 1: P_MAC=20, F=124
- 2: P_MAC=10, F=112
- 4: P_MAC=5,  F=104  (best among tested integer partitions)
- 5: P_MAC=4,  F=112
- 6: P_MAC=3,  F=132
- 7: P_MAC=2,  F=160

Thus on this small FPGA, maximizing layer parallelism is not optimal: spreading 20 DSPs too thin increases per-layer dot-product serialization. The simple compute-balanced point is P_L=4, P_MAC=5.

At T=80 this dominant-factor model gives 80*104 = 8,320 normalized MAC slots per relaxation/update, versus 9,920 for P_L=1 and 12,800 for P_L=7. P_L=4 is ~16% below P_L=1 and ~35% below P_L=7 in this simplified metric.

## Interpretation

The earlier fully spatial 31-layer model is an upper bound, not a realizable Tang Nano 9K design. For the available board, the relevant research question is joint allocation of layer parallelism and neuron/MAC parallelism under both BSRAM and DSP constraints.

The dual lambda is not the dominant blocker. At P_L=4 its raw local storage is only 4*4*64*12 = 12,288 bit; weights dominate resident memory. The immediate RTL prototype, if/when task-level T=80 is validated, should therefore target a 4-layer tile with about 5 multipliers per active layer rather than a fully unrolled network.

Caveats: this is a resource-bound model, not synthesis. It omits BSRAM aspect-ratio/port conflicts, residual buffers, weight-update traffic, activation logic, routing, clock rate, and the fact that forward/backward matrix-vector phases may have different reuse opportunities.
