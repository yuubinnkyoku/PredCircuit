# Measured dual-cost crossover update (2026-09-16)

This note combines the fair XC7 common-credit-core synthesis result with the existing depth-32 / width-8 relaxation/cycle model. It is a resource-model update, not a placed-and-routed FPGA performance claim.

## Measured incremental dual cost

The fair common-core XC7 synthesis gives, per scalar 12-bit PC-ALM dual element:

- 221 LUT2--LUT6 primitives,
- 31 CARRY4,
- 13 FF-class cells,
- 0 DSP48.

The sPC specialization optimizes to a wire for rho=1, so a standalone area ratio is undefined. The useful quantity is this incremental PC-ALM cost.

For H=L-1 hidden layers of width N, a fully spatial dual implementation therefore scales as approximately:

- LUT increment = 221 H N,
- CARRY4 increment = 31 H N,
- FF increment = 13 H N.

For the current H=31, N=8 experiment (248 scalar dual elements), naive full replication would imply about 54,808 LUT primitives, 7,688 CARRY4s and 3,224 FF-class cells. This is deliberately an upper-bound style extrapolation: synthesis across a vector/layer implementation may share or restructure logic, and FPGA slice packing means primitive counts are not physical LUT-site counts.

The key consequence is nevertheless clear: fully replicating the current one-element dual recurrence is unlikely to be the attractive architecture. The measured cost favors sharing/pipelining the dual updater.

## Dual-lane sharing changes the area conclusion

The existing cycle model has 992 scalar dual updates per relaxation step because batch=4 and H*N=248. At P_mac=128, the major MAC path costs 126 cycles/step.

To hide the dual update under that MAC path, only

    ceil(992 / P_dual) <= 126

is required, hence

    P_dual >= ceil(992 / 126) = 8.

This is much smaller than the previously illustrative P_dual=32 point. With eight shared dual lanes, the measured one-element synthesis extrapolates to roughly:

- 1,768 LUT primitives,
- 248 CARRY4s,
- 104 FF-class cells,
- 0 DSP48,

for the arithmetic lanes, plus storage/control/interconnect not captured by this scalar multiplication. The 992 persistent lambda values still require storage, but they do not require 992 copies of the recurrence combinational logic.

At P_dual=8, dual work takes exactly 124 cycles/step, so it can fit under the 126-cycle MAC path in the analytical overlap schedule. Thus the previous >2x relaxation-cycle lower bound for the 19/20 useful seeds is preserved in the model while reducing replicated dual arithmetic by 4x relative to P_dual=32 and 124x relative to one updater per scalar state.

## Width dependence

For equal-width layers, both the number of scalar dual updates and the major MAC count scale with H and batch, but dual work is O(H N) while dense matvec work is O(H N^2). Holding P_mac fixed, the minimum dual lanes needed to hide the recurrence therefore decreases relative to MAC work as N grows. In the idealized no-overhead model,

    P_dual_min ~= ceil((B H N) / ceil(MACs_step / P_mac)).

For the current measured MAC count, P_dual_min=8 at N=8. This reverses one earlier concern: although a fully spatial dual block has a large per-element LUT cost, dense wider networks amortize a shared dual engine better, not worse. Very narrow networks remain the hardest hardware case because the MAC path becomes short enough that more dual lanes are needed to keep the recurrence hidden.

## Updated architectural conclusion

The measured 221-LUT scalar recurrence should not be replicated once per lambda state. A more credible minimal layer/network engine is:

1. keep lambda values in registers/BRAM/distributed RAM;
2. instantiate a small bank of shared, pipelined 12-bit dual-update lanes;
3. stream lambda/residual pairs through those lanes while the MAC datapath computes the next state-gradient/prediction work;
4. choose P_dual just large enough that dual latency does not exceed the MAC path.

At the current depth-32,width-8,batch-4 point, the analytical minimum is eight lanes. This is now the highest-value RTL target: synthesize an 8-lane vector dual engine with realistic lambda storage and compare its LUT/FF/BRAM/Fmax against the 128-lane MAC-side resource budget. Only that vector synthesis can replace the scalar primitive extrapolation with a trustworthy network-level area number.
