# Width-64 PC-ALM dual-engine synthesis (2026-09-20)

## Scope

This note records the first apples-to-apples XC7 synthesis comparison in which every dual-engine point stores the full persistent dual state of the current depth-32, width-64, batch-4 experiment:

- dual scalars: `4 * 31 * 64 = 7936`
- dual format: 12 bit
- payload: `7936 * 12 = 95,232 bit`
- synthesis: Yosys `synth_xilinx -family xc7`
- source commit: `e78759adcc28b4d260553ca7d0de8732de972045`
- RTL-smoke run: `35469489979` (success)
- artifact: `rtl-synthesis-smoke`, id `10592132668`

The earlier width-8 engine figures must not be used as width-64 storage estimates.

## Measured XC7 mapping

| dual lanes | words | update cycles / relaxation step | estimated LCs | mapped BRAM | physical BRAM bits | payload efficiency |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 992 | 992 | 1,403 | 3 x RAMB36E1 | 110,592 | 86.1% |
| 16 | 496 | 496 | 2,806 | 3 x RAMB36E1 | 110,592 | 86.1% |
| 32 | 248 | 248 | 5,608 | 11 x RAMB18E1 | 202,752 | 47.0% |

All three points use zero DSP48E1 cells. Logic scales approximately linearly with lane count. The 32-lane point also crosses a BRAM-width packing boundary and nearly doubles physical BRAM bits for the same 95,232-bit payload.

The multiplier-free state-update unit independently maps to 110 estimated LCs, 0 DSP and 0 BRAM.

## Latency hiding against matrix work

The current analytical relaxation-step matrix work is 987,136 MACs. Under ideal `P_MAC`-way MAC parallelism:

| `P_MAC` | matrix cycles | 8-lane dual cycles / matrix cycles |
|---:|---:|---:|
| 8 | 123,392 | 0.80% |
| 32 | 30,848 | 3.22% |
| 64 | 15,424 | 6.43% |

Thus even the 8-lane dual engine has more than 15x cycle slack relative to a 64-lane matrix datapath, provided dual updates can be scheduled concurrently with matrix work and memory-port dependencies do not serialize the two paths.

## Design decision

For the first width-64 PC-ALM core, use **`P_lambda = 8` as the baseline**.

It is the Pareto point for the current design space:

- same BRAM count and packing efficiency as 16 lanes;
- half the logic of 16 lanes;
- much better BRAM packing and one quarter the logic of 32 lanes;
- its 992 update cycles fit comfortably inside even the modeled 64-lane matrix window of 15,424 cycles.

Increasing `P_lambda` is therefore not justified by arithmetic throughput alone. A wider dual engine should only be introduced if an integrated schedule demonstrates a real memory-port or dependency bottleneck.

## Interpretation

At width 64, the PC-ALM-specific arithmetic path is no longer the leading hardware concern. The important incremental cost is the persistent 95,232-bit lambda state and its banking/ports. The measured 8-lane implementation stores that state with 86.1% raw BRAM-bit utilization while keeping dual arithmetic DSP-free.

Combined with the existing software result that multiplier-free PC-ALM reaches the deep-credit criterion at T=104 whereas sPC is still below criterion at T=1024, the remaining hardware question is narrower: can the 8-lane lambda memory traffic be overlapped with the shared matrix/state datapath without introducing stalls?

The next RTL milestone should therefore be an integrated scheduler/banked-memory prototype, not a wider standalone dual engine.