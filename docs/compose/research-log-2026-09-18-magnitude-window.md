# Research log: magnitude-window cross-line (compose-next overnight)

Session: 2026-09-18 (UTC+9)
Branch: main (worktree add blocked by shared-.git policy)
Base at start: `305cda2` (origin/main after pull)

## Decision (Never-Ask / user 全自分で決めて)

- Research direction: dual-leak window + cross-line magnitude conclusion
- Workspace: continue on main (isolated worktree not available)
- Loop: cron heartbeat `17 * * * *` job `ce47e887` + continuous in-session work

## What was built

- `src/predcircuit/magnitude_control.py` — sPC/PC-ALM credit helpers, oracle norm match, residual gain, ePC stationarity, MAC accounting
- `scripts/diagnose_spc_magnitude_control.py` — E1 diagnostic
- `scripts/diagnose_epc_equilibrium_compute.py` — E2 diagnostic
- `scripts/aggregate_magnitude_window.py`
- `tests/test_magnitude_window_diagnostics.py`
- Workflows: `spc-magnitude-control-holdout.yml`, `epc-equilibrium-compute-holdout.yml`
- Docs: `docs/compose/spec/magnitude-window-crossline.md`, `docs/magnitude-window-crossline.md`

## Verification

| Command | Result |
|---|---|
| `uv run pytest` | PASS (46) |
| `uv run ruff check/format` | PASS |
| `uv run ty check` (with malecns extra) | PASS |
| CI on `df81f35` / `7ea9b2b` / `078b5b7` | PASS (CI green) |

## Local pilot (non-trivial evidence)

T=128, depth32/width8, seeds 960–967:

- sPC raw useful 0/8, cosine 0.871, norm ratio 8e-5
- oracle first-layer norm match useful 3/8
- residual-norm gain useful 0/8
- PC-ALM dual leak γ=0.01 useful 8/8, cosine 0.936, norm ≈1

**Interpretation:** deep sPC failure is primarily magnitude collapse; scale restore only works when direction residual is already small; dual-leak improves both axes.

## GitHub Actions notes

- FlyVis workflows watch `src/predcircuit/**` and re-fired on the new module; cancelled to save capacity.
- Holdouts dispatched: run 35244069612 (sPC mag), 35244061762 (ePC equilibrium).

## Overnight follow-up (local magnitude)

- Implemented `scripts/diagnose_spc_local_magnitude.py` + CI workflow.
- Local pilot n=6: residual-match t2 useful only when cosine already high (same seeds as oracle); dual-leak 6/6.
- Conclusion: post-hoc local rescale cannot replace dual dynamics for credit *content*.
- Results: `results/spc_local_magnitude_pilot_2026-09-18.md`
- Reviewer subagent general-1 failed (UnknownError); general-2 re-spawned.

## Pending overnight (updated)

1. Collect local-magnitude 20-seed holdout artifacts
2. Ingest general-2 review findings; fix any criticals
3. Keep dual-leak/Sakana state_lr follow-up as next scientific experiment after review
4. Cron heartbeat `ce47e887` remains armed
