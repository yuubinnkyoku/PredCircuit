# PC-ALM hardware break-even note (2026-09-15)

This note turns the current depth-32 / width-8 experiment and the analytical cost model into explicit RTL go/no-go inequalities. It deliberately does **not** claim measured FPGA performance.

## Current point

For depth=32, width=8, input_dim=8, output_dim=4, batch=4, 12-bit state/weights/dual:

- major MACs per relaxation step: 16,128 for both sPC and PC-ALM
- free hidden state: 992 scalars = 11,904 bit
- PC-ALM dual state: +992 scalars = +11,904 bit
- PC-ALM dual update: 992 scalar updates/step
- current useful PC-ALM operating point: T=112, alpha=0.925, rho=1, eta_h=0.25, dual leak=0.01

The precision experiments already support 12-bit hidden-state storage (range about +/-8), 12-bit lambda storage (range about +/-2), and 12-bit update signals at this operating point. The matrix-product datapath is still FP32 in those experiments, so this is not yet a full 12-bit-core claim.

## Cycle break-even

With `P_mac` MAC lanes and `P_dual` dual-update lanes, the existing analytical model gives

```
C_spc  = ceil(16128 / P_mac)
C_pcalm_overlap = max(ceil(16128 / P_mac), ceil(992 / P_dual))
C_pcalm_serial  = ceil(16128 / P_mac) + ceil(992 / P_dual)
```

For the concrete exploratory point P_mac=128, P_dual=32:

- sPC: 126 cycles/step
- PC-ALM if dual update overlaps the MAC path: 126 cycles/step
- PC-ALM if serialized: 157 cycles/step

Therefore PC-ALM beats sPC in relaxation cycles when

- overlap: `T_pcalm < T_spc`
- serialized: `157*T_pcalm < 126*T_spc`, i.e. `T_pcalm/T_spc < 0.80255`

At the current PC-ALM point T=112, the serialized design needs the comparable-quality sPC point to require at least **140 steps** (strictly, T_spc > 139.56). With overlap, any comparable-quality sPC point above 112 steps loses on this lower-bound cycle model.

This makes overlap of the lambda update a first-class architectural requirement: it changes the required iteration-count advantage from about 20% to merely requiring fewer iterations.

## State-storage and state-traffic break-even are different

At equal 12-bit precision, PC-ALM stores twice as many persistent relaxation-state bits as sPC because lambda is the same shape as the free hidden state:

- sPC: 11,904 bit
- PC-ALM: 23,808 bit

The absolute PC-ALM state+lambda footprint is still only about 2.91 KiB for this small experiment, but the **relative** 2x cost matters when scaling width, depth, or batch size.

A deliberately pessimistic lower-order traffic proxy is to assume every persistent scalar is read/written once per relaxation step. Under that proxy, total state traffic scales as

```
traffic_spc   ~ T_spc * S
traffic_pcalm ~ T_pcalm * 2S
```

so PC-ALM only wins state traffic if `T_pcalm/T_spc < 0.5`. For T_pcalm=112 this would require T_spc >224.

This proxy is not an implementation claim: lambda may be kept in registers/local SRAM and state/dual accesses can be fused with the local datapath. Its value is that it exposes a real risk hidden by the MAC-only model. **A cycle advantage does not automatically imply a memory-energy advantage.**

## Architectural consequence

The next hardware model should separate three regimes instead of reporting one generic FPGA advantage:

1. **MAC-bound, dual-overlapped:** PC-ALM wins as soon as its useful T is lower than sPC's.
2. **dual-update-bound or serialized:** PC-ALM needs roughly a 20% T reduction at the 128/32-lane point.
3. **state-memory-bound with no locality benefit:** PC-ALM may need close to a 2x T reduction because of the extra lambda state.

Thus the most important hardware question is no longer just LUT/DSP cost of the lambda adder. It is whether layer-local state and lambda can remain close enough to the compute lanes that the extra lambda traffic does not reach the shared-memory bottleneck.

## Immediate experimental implication

Do not move to full RTL solely because the 12-bit storage experiments passed. First finish quantizing `Wz` and `W^T r`, then compare sPC and PC-ALM at a matched credit-quality target and report **minimum T**, not a fixed common T. Feed those measured T values into both the cycle inequality and a locality-aware state-traffic model.

A particularly informative target is the current strict input-credit criterion (cosine >=0.9, norm ratio in [0.5,2], relative error <=0.6). If sPC still cannot meet it by T=224 while PC-ALM remains useful around T=112, then even the pessimistic 2x-state traffic proxy becomes favorable to PC-ALM; if sPC reaches it well below 224, compute-cycle and memory-energy conclusions may diverge.
