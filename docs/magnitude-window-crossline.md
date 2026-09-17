# Magnitude-window cross-line conclusion

Status: designed / in-progress. This note is the cross-line synthesis target for the overnight compose-next feature `magnitude-window-crossline`. It may update as E1/E2 holdouts land.

## Claim under test

> Local credit methods often already carry useful direction information; they fail when credit magnitude or temporal synchronization leaves a usable window. Magnitude/timing stabilizers recover performance without inventing a new global algorithm.

If E1 shows that sPC remains geometrically wrong after oracle first-layer norm matching, the claim must be weakened.

## Shared mechanism table (pre-holdout synthesis from committed results)

| System | Direction signal | Magnitude / sync failure | Stabilizer with evidence |
|---|---|---|---|
| FlyVis local temporal PC | biological vs pair-rotation cosine gap | harmful residual component dominates credit | oracle residual attenuation `r=0` recovers accuracy (credit_geometry.md) |
| Type-pair shared credit 4m+r | late shared mean aligns with hard-margin gradient | weight-norm explosion; late gate removes recovering credit | stay ungated late; Goldilocks init band ~0.08 (research-log 2026-09-13) |
| sPC deep residual MLP | cosine ~0.88 at large T | norm ratio ~2% of BP at T=1024 (width-64) | none yet — E1 tests scale vs direction |
| Pure PC-ALM | cosine often ~0.95+ | non-monotone useful window; seed desync | dual leak γ∈{0.005–0.02} widens window, shrinks \|λ\| |
| PC-ALM + dual leak | cosine ~0.95, norm ≈1 at T=128 | leak 0.05 over-damps | γ=0.01 nominal; Q3.9 dual state preserves FP32 |
| ePC T=1 | collinear with BP | scale equals error_lr | BP-like criterion is wrong axis — E2 uses equilibrium/stationarity |
| Exact gradient / BPTT | solves extent 1–2 | well-scaled by construction | global control; not local |

## Hardware implication (provisional)

At width-64/depth-32, PC-ALM+dual-leak 14/14/12 already clears a state-traffic break-even lower bound (`T_sPC/T_ALM > 8` vs break-even 1.857) while sPC has no equal-quality point. The architectural story is not “local rules need O(1) T with depth” (linear-chain evidence says T grows ~L^1.5 even when tuned). The story is: **local simultaneous steps + magnitude-stabilized dual/residual dynamics + compact low-precision state** can beat global credit when the useful window is synchronized.

## Falsifiers

1. E1: oracle norm-matched sPC still fails useful-credit on most seeds while cosine is low.
2. E2: ePC cannot reach stationarity-quality credit at compute comparable to sPC/PC-ALM, or equilibrium credit disagrees with long-run sPC.
3. A new FlyVis experiment shows residual attenuation benefits exact gradients equally (topology-agnostic magnitude effect only).
4. Dual-leak benefits disappear on held-out widths/depths or under official Sakana `state_lr`.

## Evidence pointers

- `results/pcalm_width64_holdout_and_spc_budget_2026-09-18.md`
- `results/pcalm_dual_leak_dynamics_20seed_2026-09-17.md`
- `results/pcalm_dual_q39_20seed_2026-09-17.md`
- `results/pure_pcalm_fixed_budget_gap_2026-09-16.md`
- `results/pcalm_observable_stopping_holdout_2026-09-16.md`
- `results/epc_credit_discovery_940_944.md`
- `results/linear_pcalm_credit_scaling.md`
- `docs/credit_geometry.md`
- `docs/research-log-2026-09-13-type-pair-credit.md`
- `docs/temporal-findings.md`
- arXiv:2605.31022 (PC-ALM); SakanaAI/pc-alm @ `660747f6`

## E1/E2 outputs

Pending CI holdouts seeds 960–979 via:

- `scripts/diagnose_spc_magnitude_control.py`
- `scripts/diagnose_epc_equilibrium_compute.py`

When artifacts land, replace this section with aggregate tables and a verdict on the claim.
