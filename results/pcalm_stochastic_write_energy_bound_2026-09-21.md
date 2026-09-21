# PC-ALM stochastic-weight write sparsity: transaction-energy bound

Date: 2026-09-21

## Motivation

The 128-step trajectory experiment found that 9-bit stored weights with 12-bit stochastic rounding (`q9_sr12_lr0p1`) retained about 93.2% of the FP32-lr=.1 MSE improvement while moving only 1.0092% of stored weights per training step. FP32-lr=.1 moved 90.79%.

This note asks whether that write sparsity can materially reduce *total weight-memory dynamic energy* in the current PC-ALM schedule. It is deliberately a transaction-count bound, not a vendor power estimate.

## Current trajectory schedule

The stochastic-training diagnostic uses depth 32, width 8, batch 4, and `budget=112` relaxation steps per training step. The weight update occurs once after the 112-step relaxation.

For a depth-32 residual MLP with input 8, hidden width 8, and output 4, the dense weight-scalar count is

`W = 8*8 + 30*8*8 + 8*4 = 2,016`.

The measured logical writes per training step are therefore approximately

- FP32 lr=.1: `0.9079 W = 1,830` changed weight scalars;
- q9 stochastic-12 lr=.1: `0.010092 W = 20.3` changed weight scalars.

So write-enable gating can suppress about 98.89% of the writes that would occur if every changed FP32 weight were written.

## But relaxation reads dominate if weights live in BRAM

Let `R` be the number of full-weight-array BRAM-read equivalents per relaxation step. `R=1` is an optimistic lower bound; a straightforward implementation that separately streams the forward prediction and transpose-direction propagation is closer to `R=2` unless the same fetched weights are retained/reused locally.

Per training step, weight-memory transactions in scalar equivalents are approximately

`N_tx = T R W + f_write W`,

where `T=112` and `f_write` is the logical write fraction.

Replacing FP32-like dense changed-weight writes (`f_write=0.9079`) by the measured stochastic q9 writes (`f_write=0.010092`) therefore reduces total weight-memory transaction count by only

`(0.9079 - 0.010092) / (112 R + 0.9079)`.

This gives:

| assumed BRAM weight reads per relaxation step `R` | total transaction reduction from write sparsity |
| ---: | ---: |
| 1 | 0.795% |
| 2 | 0.399% |
| 4 | 0.200% |

Thus the ~90x reduction in logical changed-weight writes does **not** imply anything close to a ~90x reduction in total weight-memory traffic. Under a BRAM-streamed 112-step relaxation schedule, read traffic overwhelms the once-per-training-step writeback.

## Energy break-even in symbolic form

Transaction counts alone are not energy. Let `E_r` and `E_w` be dynamic energy per scalar-equivalent BRAM read and write. Ignoring RNG/comparator overhead for the moment, the fractional weight-memory energy reduction is

`Delta_E / E_dense = (0.9079 - 0.010092) E_w / (112 R E_r + 0.9079 E_w)`.

To obtain even a 10% reduction in weight-memory energy from write sparsity alone requires

`E_w / E_r > 12.47 R`.

Therefore:

- for `R=1`, writes would need to cost more than about 12.5x reads;
- for `R=2`, more than about 24.9x;
- any 12-bit RNG/comparator energy makes the required ratio slightly worse.

This is a strong warning against presenting the measured 1.0092% write rate as a direct power advantage.

## Where stochastic q9 still helps hardware

The negative result is specific to *dynamic write energy under repeated BRAM weight streaming*. The stochastic q9 candidate still has two independent hardware advantages already supported by the experiments/model:

1. **capacity/packing:** the 9-bit stored-weight design can retain the favorable BRAM packing point without an FP32 master copy or dense residual accumulator;
2. **write-port activity:** if an architecture keeps weights resident in local registers/LUTRAM/cache across many relaxation steps, or otherwise amortizes BRAM reads, sparse writeback can become relevant because `R` falls toward zero at the BRAM boundary.

This shifts the architecture question. The valuable optimization is not merely `write_enable = changed`; it is **weight residency/reuse across relaxation steps**. If the same local weight tile can serve both forward and transpose-direction operations for many iterations before returning to BRAM, then the 1% write rate may translate into a meaningful memory-energy reduction. If weights are streamed from BRAM every relaxation step, it probably will not.

## RNG/comparator overhead

The 12-bit stochastic rounder needs a random threshold and comparison for each candidate weight update, but only once per training step, not once per relaxation step. Its operation count is therefore `Theta(W)` per training step versus `Theta(T W)` weight accesses / matrix work during relaxation. This suggests the RNG/comparator arithmetic is unlikely to dominate at `T=112`, but no LUT/power claim should be made until a concrete LFSR sharing schedule is synthesized or modeled.

## Consequence for the RTL gate

The 128-step numerical result remains a positive low-precision result, but the proposed *dynamic-power* interpretation is narrowed:

- **supported:** q9 stochastic rounding preserves learning trajectory surprisingly well with ~1% logical changed-weight writes and no dense high-precision master state;
- **not supported:** ~1% writes by itself implies a large total training-energy saving;
- **new architectural requirement:** demonstrate cross-relaxation weight residency/reuse, or quantify a memory hierarchy in which writeback energy is a non-negligible fraction of training energy.

## Next discriminating step

Before extending training again, add a weight-traffic model with a reuse parameter `K` = number of relaxation steps served per BRAM weight load. Sweep `K = 1, 2, 4, 8, 16, 32, 112` and include the existing state-traffic model. This will show whether a feasible tile/register budget can move the design from the read-dominated regime into one where q9 sparse writeback has measurable system-level value. A later synthesis can then replace symbolic `E_r/E_w` with device-specific numbers.
