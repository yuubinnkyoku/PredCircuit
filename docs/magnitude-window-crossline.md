# Magnitude-window cross-line conclusion

Status: **holdout-supported (E1 n=20; E2 n=19)**.
Primary tables: `results/magnitude_window_holdout_2026-09-18.md`.
Follow-up local controllers: `results/spc_local_magnitude_pilot_2026-09-18.md` (20-seed holdout: local t2 ≤ oracle, dual-leak 20/20).
Spec: `docs/compose/spec/magnitude-window-crossline.md`.

## Claim under test (post-holdout)

> Local credit methods often already carry useful direction information; they fail when credit magnitude or temporal synchronization leaves a usable window. Magnitude/timing stabilizers can recover performance **only when direction residual is already small**. Dual-leak PC-ALM is the tested method that improves **both** magnitude and direction-quality of the usable window.

E1 holdout weakened the strong form of the claim: oracle first-layer norm restore is **necessary but not sufficient** (45% useful at T=128, 20% at T=256), so the claim cannot be “magnitude alone.”

## Explicit caveats

1. **Gradient geometry ≠ training accuracy.** Useful first-layer credit is a diagnostic against BP, not a trained-task score.
2. **Dual leak ≠ pure Sakana PC-ALM.** γ=0 is the reference; leak is an engineering stabilization variant (`pure_pcalm_fixed_budget_gap`, dual-leak dynamics note).
3. **FlyVis residual attenuation used an oracle projection** (`docs/credit_geometry.md`); it is mechanism evidence, not a deployable local rule.
4. **ePC negative is for this residual-MLP diagnostic**, not a refutation of PC-ALM-paper training results at official settings.
5. **MAC/state numbers are analytical lower-bound models**, not place-and-route measurements.

## Shared mechanism table

| System | Direction signal | Magnitude / sync failure | Stabilizer with evidence |
|---|---|---|---|
| FlyVis local temporal PC | biological vs pair-rotation cosine gap | harmful residual component dominates credit | oracle residual attenuation `r=0` recovers accuracy |
| Type-pair shared credit 4m+r | late shared mean aligns with hard-margin gradient | weight-norm explosion; late gate removes recovering credit | stay ungated late; Goldilocks init band ~0.08 |
| sPC deep residual MLP | holdout cosine ~0.88 at T=128 | norm ratio ~8e-5 of BP; oracle restore only 45%/20% useful | dual-leak PC-ALM 20/20 at T=128 and 256 |
| Pure PC-ALM | cosine often ~0.95+ | non-monotone useful window; seed desync | dual leak γ∈{0.005–0.02} widens window, shrinks \|λ\| |
| PC-ALM + dual leak | holdout cosine ~0.95, norm ≈1 at T=128/256 | leak 0.05 over-damps | γ=0.01 nominal; Q3.9 dual state preserves FP32 |
| ePC this protocol | cosine ≤0.74 to BP | 0/19 useful; high lr diverges energy | not competitive under BP or stationarity gates here |
| Exact gradient / BPTT | solves extent 1–2 | well-scaled by construction | global control; not local |

## E1 holdout (seeds 960–979, 20/20 collected)

Network: ResidualMLP depth=32, width=8, batch=4, ReLU, state_lr=0.25, rho=1.

| T | control | useful | mean cosine | mean norm ratio | mean relative error |
|---:|---|---:|---:|---:|---:|
| 128 | `spc_raw` | 0/20 | 0.884 | 0.000082 | 0.9999 |
| 128 | `spc_oracle_norm_match` | 9/20 (45%) | 0.884 | 1.000 | 0.466 |
| 128 | `spc_residual_norm_gain` (stage-1 only) | 0/20 | 0.884 | 0.001 | 0.999 |
| 128 | `pcalm_dual_leak` γ=0.01 | **20/20** | 0.948 | 1.009 | 0.330 |
| 256 | `spc_raw` | 0/20 | 0.830 | 0.0044 | 0.996 |
| 256 | `spc_oracle_norm_match` | 4/20 (20%) | 0.830 | 1.000 | 0.565 |
| 256 | `pcalm_dual_leak` γ=0.01 | **20/20** | 0.956 | 1.021 | 0.317 |

Review note: stage-1 residual-norm gain alone does **not** implement the spec’d two-stage control. Stage-2 (`residual_gain` then unit first-layer norm) is now implemented as `spc_residual_gain_unit_first` and requires a short re-holdout to freeze its rate.

## E2 holdout (19 seeds) + compute accounting

| error_lr | best stationarity_ok rate | useful BP rate | energy |
|---:|---:|---:|---|
| 0.1 | 0.21 at T=64 | 0/19 all T | mild decrease |
| 0.3 | 0.11 at T=16 | 0/19 all T | noisy |
| 0.8 / 1.0 | 0 | 0/19 all T | often increases / diverges |

### Matched-budget compute model (analytical, depth32/width8/batch4)

| family | MACs/step (lower bound) | total MACs @ T=128 | persistent state bits |
|---|---:|---:|---:|
| sPC | 64,512 | 8.26e6 | 31,744 |
| PC-ALM (+dual MAC) | 68,480 | 8.82e6 | 63,488 |
| ePC (2× matrix accounting) | 129,024 | 1.65e7 | 31,744 (error coords) |
| BP once | 129,024 | 1.29e5 | free-state only |

No ePC operating point in this protocol matches dual-leak PC-ALM’s useful-credit rate, so the digital-compute comparison does not rescue ePC under the current gate.

## Hardware implication

At width-64/depth-32, PC-ALM+dual-leak 14/14/12 clears a state-traffic break-even lower bound (`T_sPC/T_ALM > 8` vs break-even 26/14≈1.857) while sPC has no equal-quality point. Linear-chain evidence still forbids claiming depth-independent T. The architectural story is: **local simultaneous steps + magnitude-stabilized dual dynamics + compact low-precision state**, not “sPC + more T” and not ePC T=1 collinearity.

## Falsifiers (updated)

1. ~~E1 oracle restore fixes sPC~~ — **refuted on holdout** (only 45%/20% useful).
2. E2 ePC competitive at matched quality/compute — **not supported** in this protocol.
3. FlyVis residual attenuation benefits exact gradients equally — still open.
4. Dual-leak benefits vanish on official Sakana `state_lr` or held-out widths — **refuted for state_lr** (`results/magnitude_window_v2_and_sakana_lr_2026-09-18.md`: leak 90–100% useful at T=128/256 under official 0.234285; pure PC-ALM collapses to 5% at T=256).
5. Stage-2 local residual gain matches dual-leak useful rates — **refuted** (0/20 useful; unit first-layer norm is the wrong target vs BP-scale criterion).

## Next experiment

1. **Width-64 mixed precision × official state_lr × dual-leak** — confirm 14/14/12 still holds when `state_lr=0.234285`.
2. **Bridge lines:** apply dual-leak-style magnitude control on the FlyVis type-pair local rule (test whether the same “window” mechanism explains late-gate/Goldilocks).
3. **End-to-end training** of leaky PC-ALM vs sPC on the shared ResidualMLP supervised task (credit geometry alone is not enough for a systems claim).

## Evidence pointers

- `results/magnitude_window_holdout_2026-09-18.md`
- `results/magnitude_window_v2_and_sakana_lr_2026-09-18.md`
- `results/spc_local_magnitude_pilot_2026-09-18.md`
- `results/pcalm_sakana_state_lr_pilot_2026-09-18.md`
- `results/pcalm_width64_holdout_and_spc_budget_2026-09-18.md`
- `results/pcalm_dual_leak_dynamics_20seed_2026-09-17.md`
- `results/pcalm_dual_q39_20seed_2026-09-17.md`
- `results/pure_pcalm_fixed_budget_gap_2026-09-16.md`
- `results/sakana_reference_audit.md`
- `docs/credit_geometry.md`
- `docs/research-log-2026-09-13-type-pair-credit.md`
- arXiv:2605.31022; SakanaAI/pc-alm @ `660747f6`
