# Width-64 state-lattice dynamics: why 15-bit state is worse than 14/16 bit

Source: GitHub Actions run `35277252027`, commit `5ec57134`, seeds 845--859. All 15 jobs completed successfully and all artifacts were recovered without rerunning the experiment.

Configuration: ResidualMLP depth 32, width 64, batch 4, ReLU, `T=256`, official Sakana depth-32 `state_lr=0.234285`, `rho=1`, `alpha=0.925`, `dual_leak=0.02`. Update/residual precision is fixed at `fixed14_i1` and the dual at `fixed12_i1`; only hidden-state precision changes among FP32, `fixed14_i3`, `fixed15_i3`, and `fixed16_i3`.

The diagnostic records every outer step: raw/quantized residual norm, dual norm and dual-step norm, requested and realized hidden-state update norm, state rounding-error norm, zero-update fractions, consecutive state-step cosine, saturation, and BP-credit geometry at checkpoints.

## Checkpoint credit geometry

| state | useful @96 | @128 | @160 | @192 | @224 | @256 | mean cosine @256 | mean norm/BP @256 | mean BP-relerr @256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 13/15 | 15/15 | 15/15 | 15/15 | 15/15 | 15/15 | 0.9820 | 0.9585 | 0.1897 |
| `fixed14_i3` | 1/15 | 10/15 | **15/15** | 15/15 | 15/15 | 15/15 | 0.9685 | 0.9028 | 0.2570 |
| `fixed15_i3` | 1/15 | 5/15 | 7/15 | 7/15 | 8/15 | **8/15** | **0.8830** | **0.6751** | **0.5089** |
| `fixed16_i3` | **14/15** | **15/15** | 15/15 | 15/15 | 15/15 | 15/15 | **0.9837** | **0.9759** | **0.1791** |

`fixed15_i3` does not merely need a slightly later stopping point. Its useful count plateaus at 7--8/15 after T=160, while its cosine/norm/relative-error change only minimally from T=192 to T=256. In the same interval `fixed14_i3`, `fixed16_i3`, and FP32 continue to improve.

First useful checkpoint among the measured checkpoints:

- FP32: 15/15 eventually useful, median T=96, latest T=128.
- `fixed16_i3`: 15/15, median T=96, latest T=128.
- `fixed14_i3`: 15/15, median T=128, latest T=160.
- `fixed15_i3`: only 8/15 ever useful, median T=128 among successes, latest success T=224.

## Late-window dynamics (T=225--256 mean over 15 seeds)

| state | residual | dual norm | dual-step norm | requested state-step norm | realized state-step norm | realized/requested | state zero-step frac | consecutive step cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 0.05272 | 2.0204 | 0.00838 | 0.00657 | 0.00657 | 1.000 | 0.9703 | -0.306 |
| `fixed14_i3` | 0.05705 | 1.9511 | 0.02858 | 0.02789 | 0.02037 | 0.730 | 0.9498 | -0.074 |
| `fixed15_i3` | **0.04861** | **1.6723** | **0.00353** | 0.01695 | **0.00468** | **0.276** | **0.99937** | **-0.569** |
| `fixed16_i3` | 0.05255 | 2.0406 | 0.00830 | 0.00650 | 0.00668 | 1.028 | 0.9715 | -0.318 |

All state/update/dual saturation rates are zero. Therefore this is not clipping.

The 15-bit trajectory is instead almost frozen by the state lattice: in the final 32 steps, 99.94% of state coordinates do not move on a given step and the realized state-step norm is only about 27.6% of the requested quantized-gradient step norm. The remaining moves alternate more strongly (`cos(delta_h_t, delta_h_{t-1}) ~= -0.57`). Its dual dynamics also nearly stop (`dual-step norm ~=0.0035`) and the credit norm stalls well below BP.

The low residual of the 15-bit trajectory is therefore **not** evidence of better credit. It reaches a low-motion quantized attractor with worse BP geometry.

## Why this particular lattice is pathological

The update gradient format is `fixed14_i1`, so its LSB is `2^-12`. State updates use `effective_lr = batch * state_lr = 4 * 0.234285 = 0.93714`. One update-gradient quantum therefore requests

`delta_h_min = 0.93714 * 2^-12 = 2.28794e-4`.

For the three state formats:

| state | frac bits | state LSB | half LSB | minimum update-gradient quanta needed to cross half-LSB |
|---|---:|---:|---:|---:|
| `fixed14_i3` | 10 | 9.7656e-4 | 4.8828e-4 | 3 |
| `fixed15_i3` | 11 | 4.8828e-4 | 2.4414e-4 | **2** |
| `fixed16_i3` | 12 | 2.4414e-4 | 1.2207e-4 | **1** |

The 15-bit case is especially close to a threshold: a one-quantum update (`2.2879e-4`) falls only ~6.3% below the half-LSB state threshold (`2.4414e-4`) and is rounded away. The 16-bit state preserves a one-quantum update. The coarse 14-bit state discards small updates too, but when it moves it makes much larger lattice jumps and does not enter the same low-motion attractor on these seeds.

This supports a **quantization-lattice locking** interpretation. It does not yet prove an exact finite-period limit cycle.

## Hardware implication

Bit-width alone is not a monotone quality knob for iterative local learning. The relative grids of:

1. update/residual precision,
2. effective state learning rate, and
3. hidden-state LSB

must be co-designed.

Current defensible points remain:

- compact: `state=fixed14_i3`, `update=fixed14_i1`, `dual=fixed12_i1` (15/15 useful);
- higher-fidelity: `state=fixed16_i3`, `update=fixed14_i1`, `dual=fixed12_i1` (15/15 useful and essentially FP32 checkpoint geometry).

`fixed15_i3` is specifically disfavored with the current update grid and official state learning rate.

## Next falsifier

Vary the update lattice and/or `state_lr` around the state15 threshold while holding all else fixed. If `fixed15_i3` recovers when a one-quantum update crosses the state half-LSB, that would directly support lattice locking rather than an intrinsic 15-bit-width pathology. A particularly cheap hardware probe is to compare `fixed13_i1`, `fixed14_i1`, `fixed15_i1`, and FP32 update precision with state fixed at `fixed15_i3` and dual at `fixed12_i1`.
