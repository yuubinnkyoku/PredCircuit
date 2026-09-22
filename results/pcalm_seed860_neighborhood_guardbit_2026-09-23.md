# PC-ALM seed-860 neighborhood guard-bit scan (2026-09-23)

## Question

Seed 860 was a known stress case where `fixed15_i3` state/accumulator clipping destroyed first-layer BP-gradient alignment and an `i4` guard range recovered it. This scan asks whether that behavior is locally common in seed space or specific to the seed-860 trajectory.

## Setup

- depth: 32
- width: 64
- relaxation steps: 128
- update: `fixed14_i1`
- dual: `fixed12_i1`
- persistent state: `fixed15_i3`
- conditions compared:
  - `fixed15_i3`, writeback interval 1
  - `fixed16_i4`, writeback interval 32
- seeds: 850--869 excluding 860
- seed 860 is intentionally not rerun; its previously recorded stress-case result remains the reference.

## Result

Across all 19 newly tested neighboring seeds:

- both conditions were finite in 19/19 seeds;
- both conditions passed the current useful-credit criterion in 19/19 seeds;
- accumulator saturation rate was exactly 0 in every run;
- state-writeback saturation rate was exactly 0 in every run;
- `fixed15_i3` and `fixed16_i4` produced numerically identical reported residual, first-layer cosine, norm ratio, and relative error for every seed;
- first-layer BP cosine across the 19 seeds had mean **0.948864** and minimum **0.929268** (seed 862).

Therefore the pathological seed-860 behavior is not a contiguous feature of the nearby seed-number range. The evidence is consistent with a trajectory-specific rare excursion rather than a local region of bad initializations.

## Interpretation

Together with the earlier seed 0--99 scan, the `i4` bit should be treated as a **guard-range bit**, not as a generally required precision bit. In ordinary tested trajectories, `fixed15_i3` never reaches the boundary, so `15_i3` and `16_i4` collapse to the same quantized trajectory. In the known seed-860 stress trajectory, however, a very small number of range violations can redirect the deep recurrent relaxation enough to damage gradient geometry.

This supports a hardware candidate with 15-bit stored state plus one extra integer guard bit in the local accumulator/datapath. It does **not** establish a population frequency for such excursions: seed 860 was previously selected as a stress case, so combining it naively with the ordinary-seed scans would introduce selection bias.

## Decision

Further blind seed scanning now has lower information value than returning to the central hardware question. The low-precision candidate is sufficiently concrete for first-order cost modeling:

- stored state: 15-bit `i3` candidate;
- local accumulator: 16-bit `i4` candidate;
- dual state remains independently quantized (current useful candidate 12-bit);
- seed 860 remains a required regression/stress test.

The next highest-value work is to extend the existing hardware crossover model with explicit weight residency/banking and state/dual traffic, then compare PC-ALM's iteration advantage over sPC against the extra dual-state and repeated-relaxation cost. ePC and BP should remain digital baselines rather than assuming an FPGA advantage a priori.
