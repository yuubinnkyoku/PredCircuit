# Research log: type-pair shared credit & local credit geometry

Session date: 2026-09-12/13 (UTC+9)
Branch: main
Starting SHA: `0bffcfc`
Ending SHA: see `git log` head at close (research-log commits through `b5ae482` and later)

## CI / infrastructure

| Item | Status |
|---|---|
| Starting main SHA | `0bffcfc` "Run local power norm-control trajectory 1480-1499" |
| CI at start | **RED** — ruff format on `diagnose_flyvis_type_pair_local_power_norm_control_trajectory.py` |
| Format fix | `a607aa5` (rebased/applied) |
| CI after format fix | **GREEN** (verified on subsequent pushes) |
| Experiment workflows this session | mechanism 1540-1559 success; full-horizon holdout 1500-1519 success; replication 1520-1539 success; 200-epoch 1560-1579 success |
| Implementation bugs found | none in this session's scripts |
| CI failures other than format | none |

Separation of failure modes: the only CI failure was ruff formatting. No scientific workflow failed. No missing artifacts were silently dropped (all expected seed counts verified).

## Collected / executed experiments

### 1. Norm-control trajectory 1480-1499 (collected)

- **Workflow run**: 34701414713 (success, 3m41s)
- **Seeds**: 1480-1499 (20/20 present)
- **Finite**: 800/800 rows True
- **Design**: branch at epoch 80, then 0/1/5/10/20 steps under `threshold` (rho05), `local_power`, `delta_norm`. Both random and biological_strength init. 4 eval reps.
- **Artifact**: `results/artifacts/norm-control-1480-1499/`

**local_power vs delta_norm (update-norm control), biological_strength, branch_step=20:**

| metric | mean Δ | 95% CI | dz | +/− seeds | Wilcoxon p |
|---|---:|---|---:|---|---:|
| cross_entropy | +0.000019 | [+0.000010, +0.000029] | +0.94 | 17/3 | 0.0017 |
| accuracy | +0.000260 | [−0.001902, +0.002423] | +0.06 | 3/3 | 0.463 |
| hard_margin | +0.000012 | [−0.000043, +0.000068] | +0.11 | 9/11 | 1.000 |
| soft_margin | −0.000026 | [−0.000039, −0.000013] | −0.95 | 3/17 | 0.0015 |

**local_power vs threshold (rho05), biological_strength, branch_step=20:**

| metric | mean Δ | 95% CI | dz | +/− seeds | Wilcoxon p |
|---|---:|---|---:|---|---:|
| cross_entropy | +0.000089 | [+0.000069, +0.000109] | +2.08 | 20/0 | 0.0001 |
| accuracy | +0.006510 | [−0.000732, +0.013753] | +0.42 | 10/1 | 0.029 |
| hard_margin | +0.000296 | [+0.000171, +0.000420] | +1.11 | 19/1 | 0.0002 |
| soft_margin | −0.000123 | [−0.000151, −0.000095] | −2.06 | 0/20 | 0.0001 |

**Interpretation:** local_power improves hard_margin over rho05, but is statistically indistinguishable from the update-norm control on hard_margin, and is slightly *worse* on CE/soft_margin. The "selective local power" story is not supported over "just suppress large updates."

Suppression fractions @ step=20, bio: threshold 0.347, local_power 0.366 (added 0.019), delta_norm 0.364 (added 0.018). Random init suppresses only ~3% of groups.

### 2. Full-horizon rule holdout 1500-1519 (collected, parallel agent)

- **Workflow run**: 34704492342 (success, 5m3s)
- **Seeds**: 1500-1519 (20/20)
- **Finite**: 800/800 True
- **Design**: train from **epoch 0**, rules `local` (m+r), `standard` (4m+r), `threshold` (rho05), `local_power`, `delta_norm`. Both inits. Save at 20/40/60/80/100. 4 eval reps. Locked thresholds 0.5 / 0.35 / 5e-8 / 0.002.
- **Artifact**: `results/artifacts/full-horizon-1500-1519/`
- **Replication 1520-1539**: run 34704834673 success (5m43s), 20/20 finite.

### 2b. Combined 40-seed full-horizon (1500-1539)

Pooled holdout + replication. n=40 paired seeds. t_crit≈2.023.

**local_power vs rho05, biological_strength:**

| h | CE Δ (dz) | acc Δ (dz) | hard Δ (dz) | soft Δ (dz) |
|---:|---|---|---|---|
| 20 | −0.000009 (−0.22) | +0.00260 (+0.29) | +0.000008 (+0.18) | +0.000012 (+0.22) |
| 40 | +0.000053* (+1.31) | −0.00117 (−0.18) | +0.000006 (+0.05) | −0.000072* (−1.30) |
| 60 | +0.000252* (+4.18) | +0.00456 (+0.25) | +0.000057 (+0.15) | −0.000339* (−4.18) |
| 80 | +0.000361* (+4.76) | +0.00482 (+0.19) | +0.000242* (+0.51) | −0.000488* (−4.73) |
| 100 | +0.000393* (+3.55) | +0.01146* (+0.64) | +0.000597* (+1.11) | −0.000539* (−3.59) |

h=100 sign counts: CE 40/0, acc 29/8, hard 35/5, soft 0/40. All four metrics significant.

**local_power vs delta_norm, biological_strength:**

| h | CE Δ | acc Δ | hard Δ | soft Δ |
|---:|---|---|---|---|
| 20 | +0.000029* | +0.00104 | **−0.000017*** | −0.000038* |
| 40 | +0.000059* | +0.00065 | **−0.000023*** | −0.000080* |
| 60 | +0.000061* | +0.00378* | **+0.000054*** | −0.000083* |
| 80 | +0.000068* | +0.00404* | **+0.000125*** | −0.000093* |
| 100 | +0.000098* | +0.00521* | **+0.000326*** | −0.000136* |

Hard-margin crossover vs the update-norm control occurs between h=40 and h=60 and then grows. CE/soft are worse at every horizon.

**Absolute means, biological_strength, combined 40 seeds, h=100:**

| rule | CE | acc | hard | soft |
|---|---:|---:|---:|---:|
| local | 1.37256 | **0.61211** | −0.00012 | −1.08017 |
| standard | 1.36540 | 0.52044 | −0.00647 | −1.07030 |
| threshold | 1.36987 | 0.58099 | −0.00033 | −1.07652 |
| local_power | 1.37027 | 0.59245 | **+0.00026** | −1.07706 |
| delta_norm | 1.37017 | 0.58724 | −0.00006 | −1.07693 |

local_power is the only rule with a **positive** mean hard margin at h=100. Plain local still has the best accuracy.

**Random init, h=100:** all local_power−rho05 effects |dz| ≤ 0.24, non-significant.

**Absolute means, biological_strength:**

| horizon | rule | CE | acc | hard | soft |
|---:|---|---:|---:|---:|---:|
| 20 | local | 1.3758 | 0.598 | +0.0059 | −1.0846 |
| 20 | standard | 1.3764 | 0.623 | +0.0060 | −1.0854 |
| 20 | threshold | 1.3768 | 0.626 | +0.0058 | −1.0859 |
| 20 | local_power | 1.3767 | 0.628 | +0.0059 | −1.0858 |
| 20 | delta_norm | 1.3767 | 0.628 | +0.0059 | −1.0858 |
| 40 | local | 1.3727 | 0.597 | +0.0070 | −1.0804 |
| 40 | standard | 1.3730 | 0.708 | +0.0081 | −1.0808 |
| 40 | threshold | 1.3743 | **0.727** | +0.0080 | −1.0826 |
| 40 | local_power | 1.3744 | **0.729** | +0.0080 | −1.0827 |
| 40 | delta_norm | 1.3743 | 0.728 | +0.0080 | −1.0826 |
| 60 | local | 1.3736 | 0.622 | +0.0043 | −1.0816 |
| 60 | threshold | 1.3734 | 0.709 | +0.0067 | −1.0813 |
| 60 | local_power | 1.3736 | 0.713 | +0.0067 | −1.0816 |
| 80 | local | 1.3733 | 0.616 | +0.0033 | −1.0812 |
| 80 | threshold | 1.3719 | 0.643 | +0.0057 | −1.0793 |
| 80 | local_power | 1.3723 | 0.652 | +0.0060 | −1.0798 |
| 100 | local | 1.3727 | **0.591** | −0.0013 | −1.0803 |
| 100 | standard | 1.3663 | 0.491 | −0.0092 | −1.0715 |
| 100 | threshold | 1.3702 | 0.561 | −0.0018 | −1.0770 |
| 100 | local_power | 1.3706 | 0.573 | −0.0010 | −1.0775 |
| 100 | delta_norm | 1.3705 | 0.569 | −0.0014 | −1.0774 |

**Critical trajectory fact:** every shared-credit rule (standard / rho05 / local_power / delta_norm) peaks in accuracy around epoch 40 (~0.73) and then decays. Plain local is flatter and is the **best accuracy rule at epoch 100** (0.591). Standard 4m+r collapses hardest (0.491).

**local_power vs rho05, biological_strength, horizon=100:**

| metric | mean Δ | dz | +/− | Wilcoxon p |
|---|---:|---:|---|---:|
| cross_entropy | +0.00038 | +3.40 | 20/0 | significant |
| accuracy | +0.01276 | +0.68 | 15/4 | significant |
| hard_margin | +0.00077 | +1.63 | 19/1 | significant |
| soft_margin | −0.00052 | −3.51 | 0/20 | significant |

**local_power vs delta_norm, bio, h=100:**

| metric | mean Δ | dz | +/− |
|---|---:|---:|---|
| CE | +0.00011 | +1.52 | 19/1 |
| acc | +0.00417 | +0.44 | 10/4 |
| hard | +0.00043 | +1.02 | 18/2 |
| soft | −0.00015 | −1.57 | 0/20 |

**Pareto summary @ h=100, bio:** local_power beats rho05 on hard_margin and accuracy, pays with CE and soft_margin. Against the update-norm control it is only marginally better on hard_margin and not better on CE. Absolute hard margins of all shared-credit rules are still negative at h=100.

**Random init @ h=100:** local_power ≈ rho05 ≈ delta_norm on every metric (effects ~1e-5 or smaller). The rule does not destroy random init, but also does almost nothing there (suppression fraction ~3%).

### 3. Boundary actual-update geometry 1420-1439 (collected, pre-existing)

- **Workflow run**: 34696053260
- **Seeds**: 1420-1439, finite True, 3270 rows (boundary groups only, 0.35 ≤ ρ < 0.5)
- **Artifact**: `results/artifacts/boundary-geometry-1420-1439/`

@ h=100, among boundary groups:
- `mean_w_c > 5e-8` (local_power gate): harmful rate **80.2%** vs 45.2% when not hit (n=172 hits)
- Strongest linear local correlates of oracle hard-margin effect: `weight_credit_dot` (r=0.364), `weight_mean_times_credit_mean` (r=0.348), `credit_norm` (r=0.389), `suppression_delta_norm` (r=0.399)
- `rho` itself: r=−0.009 (useless as a continuous predictor)

### 4. Local proxy mechanism discovery 1540-1559 (executed this session)

- **Workflow run**: 34704738176 (success, 2m31s)
- **Seeds**: 1540-1559 (20/20)
- **Finite**: 72360/72360 True
- **Design**: all type-pair groups (not just boundary). Oracle `grad(hard_margin) · Δu_g` with Δu_g = clip(low) − clip(high) (gain 3→1). Oracle used **only as diagnostic**, never as a learning rule. Horizons 0/20/40/60/80/100.
- **Artifact**: `results/artifacts/mechanism-1540-1559/`
- **Script**: `scripts/diagnose_flyvis_type_pair_local_proxy_mechanism.py`
- **Workflow**: `.github/workflows/flyvis-type-pair-local-proxy-mechanism.yml`

**Harmful rates (oracle_hard > 0):** ~0.48-0.59 across horizons (near chance overall).

**Proxy ranking @ h=100 (discovery seeds — not confirmation evidence):**

| proxy | AUROC | AUPRC | Pearson | precision@power-rate | direction |
|---|---:|---:|---:|---:|---|
| cov_w_c | **0.593** | 0.595 | 0.038 | 0.654 | as_is |
| credit_norm | 0.577 | 0.575 | 0.088 | 0.611 | as_is |
| credit_std | 0.577 | 0.574 | 0.057 | 0.606 | as_is |
| delta_u_norm | 0.570 | 0.579 | 0.270 | 0.630 | as_is |
| credit_mean_abs | 0.567 | 0.577 | 0.226 | 0.630 | as_is |
| weight_std | 0.557 | 0.559 | 0.045 | 0.558 | as_is |
| mean_w_c (local_power) | 0.523 | 0.568 | 0.188 | 0.646 | as_is |
| rho | 0.511 | 0.515 | 0.039 | 0.492 | inverted |
| mean_w_times_mean_c | 0.507 | 0.537 | 0.424 | 0.629 | inverted |
| weight_delta_dot | 0.504 | 0.536 | −0.475 | 0.580 | as_is |

**local_power operating point @ h=100:** 167/12060 groups (1.4%), harmful rate **80.8%**, oracle mean +3.36e-6. Non-hits: 51.3%.

**Decision:** cov_w_c has the best overall AUROC, but at the local_power suppression rate its precision (0.654) is essentially tied with mean_w_c (0.646). Per the pre-registered rule ("明確に上回りそうなら freeze して confirmation"), this is **not** a clear win. **No new proxy rule was frozen.** Focus stays on understanding local_power.

## Hypothesis status

| Hypothesis | Status | Evidence |
|---|---|---|
| local_power is a selective local rule beyond update-norm control | **Weakly supported on hard_margin/acc, not on CE/soft** | 40-seed h=100: hard +0.000326* (dz=0.75, 33/7), acc +0.0052* vs delta_norm; CE/soft worse at every horizon |
| local_power improves hard_margin over rho05 on biological init | **Supported (with cost)** | 40-seed h=100: Δhard +0.000597, dz=1.11, 35/5; CE and soft_margin significantly worse (40/0 and 0/40) |
| Shared type-pair credit (4m+r or gated) helps from epoch 0 | **Partially supported, horizon-dependent** | Big accuracy win at h=40 (~0.73 vs local 0.60), advantage decays; at h=100 local is best (0.612) |
| local_power destroys random init | **Falsified** | random-init effects |dz|≤0.24, all n.s. |
| rho alone identifies harmful groups | **Falsified** | AUROC 0.511 |
| mean(w*c) is a good harmful-group identifier | **Weakly supported at sparse operating point** | 80.8% precision among 1.4% hits; AUROC only 0.523 overall |
| A simple local proxy clearly dominates local_power | **Not supported** | cov_w_c better AUROC but tied at matched suppression rate |

## Verdict on local_power as "有力候補"

Against the user's six minimum conditions (using the combined 40-seed confirmation):

1. **bio hard_margin > rho05 on independent hold-out** — **YES.** Δhard +0.000597, CI excludes 0, 35/40 positive, growing with horizon.
2. **≥ update-norm control, not "just suppress large updates"** — **MARGINAL YES on hard_margin/acc, NO on CE/soft.** Hard margin and accuracy are significantly above delta_norm from h=60 onward, but CE and soft_margin are worse at every horizon. The gain is not *only* update-norm suppression, but it is not free either.
3. **does not clearly destroy random init** — **YES.**
4. **all seeds finite** — **YES** (1600 + 72360 rows finite).
5. **CE / soft-margin cost quantified** — **YES.** vs rho05 @ h=100: CE +0.000393 (40/0), soft −0.000539 (0/40). vs delta_norm: CE +0.000098, soft −0.000136.
6. **explainable on the time axis** — **YES for the hard-margin story.** vs delta_norm the hard-margin sign flips between h=40 and h=60 and then grows monotonically. The accuracy advantage over rho05 also grows. However the *absolute* accuracy of every shared-credit rule still peaks ~h=40 and decays toward (or below) plain local by h=100.

**Revised conclusion:** local_power meets conditions 1, 3, 4, 6 and marginally 2. It does **not** get a free lunch on CE/soft_margin. It is best described as:

> a sparse (≈1–2% of groups), high-precision (≈81%) harmful-group filter that buys hard-margin and a small accuracy gain over rho05 and over the update-norm control, paid for by a small but highly consistent CE and soft-margin cost. It is the only rule with positive mean hard margin at epoch 100, but plain local still wins on accuracy.

It is **not** a general replacement for local credit. The dominant structural finding remains that *all* shared-credit rules are transient under epoch-0 training.

## Most concise current local learning-rule candidate

**After the 200-epoch reversal, the recommendation changes:**

```
For short-to-medium horizons (≲120 epochs): plain local m+r is
competitive or best; gating shared mean credit (rho05) helps accuracy
in the early peak but the advantage is gone by ~epoch 100.

For long horizons (≥180 epochs): ungated shared type-pair mean credit
(standard 4m+r = +3·mean on every group) is clearly best —
accuracy 0.79, hard margin +0.22, CE 1.13 at epoch 200.

local_power's extra mean(w·c) gate helps only inside the mid-training
valley (h≈80–140) and becomes harmful after the recovery starts.
Do not use it as a long-horizon rule.

Open mechanism question: the ρ≥0.5 gate that defines rho05/local_power
appears to remove the groups whose shared mean credit drives the late
margin explosion. The gate is a brake on the recovery.
```

No simpler or stronger local proxy than `mean(w·c)` was confirmed for the valley window, and none is needed once the 200-epoch result is in hand.

## Implementation / CI notes

- Only CI issue: ruff format of the norm-control trajectory script. Fixed; CI green after `a607aa5`.
- A duplicate 40-seed full-horizon workflow (`flyvis-type-pair-full-horizon-rules`, seeds 1500-1539) was started this session and then **cancelled** after discovering a parallel agent's 20-seed holdout on the same seeds. No science lost; the 20-seed holdout + 20-seed replication covers the 40-seed requirement.
- Mechanism discovery uses held-out hard-margin gradient strictly as a diagnostic oracle. It is never written back into any training direction.

## 200-epoch extension (1560-1579) — MAJOR REVERSAL

- **Workflow run**: 34705564171 (success, 8m19s)
- **Seeds**: 1560-1579 (20/20), finite 1600/1600
- **Design**: same five rules, epoch 0→200, save every 20. Fresh seeds and a **different eval jitter base** (9_100_000 vs 6_900_000), so this is also a mild jitter hold-out.
- **Artifact**: `results/artifacts/full-horizon-200-1560-1579/`

### Accuracy trajectory, biological_strength

| h | local | standard | rho05 | local_power | delta_norm |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.663 | 0.687 | 0.691 | 0.690 | 0.690 |
| 40 | 0.554 | 0.637 | 0.657 | 0.657 | 0.655 |
| 60 | 0.585 | 0.632 | 0.674 | 0.678 | 0.677 |
| 80 | 0.596 | 0.552 | 0.602 | 0.620 | 0.616 |
| 100 | 0.610 | 0.501 | 0.556 | 0.568 | 0.561 |
| 120 | 0.601 | 0.517 | 0.540 | 0.551 | 0.543 |
| 140 | 0.631 | 0.534 | 0.583 | 0.588 | 0.579 |
| 160 | 0.572 | 0.614 | 0.496 | 0.499 | 0.498 |
| 180 | 0.541 | **0.712** | 0.569 | 0.549 | 0.568 |
| 200 | 0.505 | **0.790** | 0.671 | 0.644 | 0.667 |

### Hard margin / CE at h=200, bio

| rule | CE | acc | hard | soft |
|---|---:|---:|---:|---:|
| local | 1.3699 | 0.505 | −0.0120 | −1.0763 |
| **standard** | **1.1309** | **0.790** | **+0.2204** | **−0.7295** |
| threshold | 1.2995 | 0.671 | +0.0453 | −0.9789 |
| local_power | 1.3096 | 0.644 | +0.0350 | −0.9930 |
| delta_norm | 1.3014 | 0.667 | +0.0425 | −0.9816 |

**Standard 4m+r is the clear winner at epoch 200 on every metric.** The h=80–140 "collapse" is a valley, not a permanent failure. CE improves from ~1.37 to 1.13; hard margin goes from ~0 to +0.22; accuracy reaches 79%.

### local_power vs rho05, bio (selected horizons)

| h | CE Δ | acc Δ | hard Δ | soft Δ |
|---:|---|---|---|---|
| 80 | +0.00031* | +0.018* | +0.00056* | −0.00042* |
| 100 | +0.00034* | +0.012 | +0.00073* | −0.00046* |
| 120 | +0.00046* | +0.012* | +0.00060 | −0.00063* |
| 140 | +0.00070* | +0.005 | +0.00022 | −0.00096* |
| 160 | +0.00152* | +0.003 | +0.00021 | −0.00210* |
| 180 | +0.00475* | −0.020 | −0.00439 | −0.00644* |
| 200 | +0.01008* | **−0.027*** | **−0.01038*** | −0.01407* |

The local_power hard-margin and accuracy advantage over rho05 **peaks around h=80–120 and then reverses**. By h=200 local_power is worse than rho05 on all four metrics. The extra `mean(w·c)` suppression is a medium-horizon regularizer that becomes harmful in the long run.

### Revised scientific picture

1. Shared type-pair mean credit (4m+r) has a **U-shaped / multi-phase** accuracy trajectory: early peak (~h=20–60), a deep valley (~h=80–160), then a strong late recovery that by h=200 far exceeds plain local.
2. The h=100 snapshot that made plain local look best was **sampling the valley**. Any claim about "which rule is better" that stops at epoch 100 is unreliable.
3. local_power's selective extra suppression helps in the valley (h=80–140) and hurts after the recovery begins (h≥180). It is not a durable improvement over rho05.
4. Absolute hard margin of standard 4m+r reaches +0.22 at h=200 — the first time any rule produces a clearly positive margin in this series.

### Verdict update on local_power

local_power is **demoted further**. Against the six conditions:

1. bio hard > rho05 — YES only in the h=80–140 window; **NO at h=200** (−0.010, 1/19).
2. ≥ update-norm control — NO at h=200 (worse on all metrics).
3. random init safe — still YES.
4. finite — YES.
5. CE/soft cost — YES, and the cost **grows** with horizon (CE +0.010 at h=200).
6. time-axis story — the story is now "helps in the valley, hurts in the recovery," which is the opposite of a durable learning-rule improvement.

**Final position this session:** the interesting object is no longer local_power. It is the **U-shaped shared-credit trajectory** itself, and in particular why plain 4m+r recovers so strongly after epoch 160 while the gated rules (rho05 / local_power / delta_norm) recover less. Gating the shared mean credit appears to **delay or damp the late recovery**.

## Temporal Jaccard of selected type-pair groups (from mechanism 1540-1559)

Computed offline from the mechanism-discovery artifact (all type-pair groups, horizons 0/20/40/60/80/100).

**local_power band (0.35 ≤ ρ < 0.5 and mean(w·c) > 5e-8):**

| transition | Jaccard | persist | appear | disappear | mean n_prev | mean n_curr |
|---|---:|---:|---:|---:|---:|---:|
| 0→20 | 0.000 | 0.000 | 6.30 | 2.35 | 2.35 | 6.30 |
| 20→40 | 0.097 | 0.165 | 4.90 | 5.20 | 6.30 | 6.00 |
| 40→60 | 0.095 | 0.166 | 3.95 | 4.90 | 6.00 | 5.05 |
| 60→80 | 0.107 | 0.176 | 3.85 | 4.20 | 5.05 | 4.70 |
| 80→100 | 0.117 | 0.285 | 7.15 | 3.50 | 4.70 | 8.35 |

Consecutive-run lengths of power-band membership: n=570 runs, **mean 1.15**, median 1, max 5. 508/570 runs last exactly one horizon.

**rho05 band (ρ ≥ 0.5) for comparison:** Jaccard 0.57–0.66 across the same transitions — much more stable.

**Mechanistic implication:** local_power does **not** lock onto a small set of persistently harmful type-pair groups. Each evaluation horizon suppresses a largely different ~5–8 groups. The hard-margin gain is therefore better described as a **bursty, rotating stochastic suppressor** (dropout-like on the mean-credit channel) than as targeted identification of a stable harmful set. The 80.8% oracle-precision figure is a same-time-point statement and does not imply temporal consistency.

## Shared-credit recovery mechanism (1580-1589)

- **Workflow run**: 34706943323 (success, 4m3s)
- **Seeds**: 1580-1589 (10/10), finite 400/400
- **Design**: train local / standard / threshold / local_power from epoch 0 to 200; at each horizon log weight-norm, ρ distribution, shared-mean fraction, and cosine of the shared-mean component with the held-out hard-margin gradient. Gradient is diagnostic only.
- **Artifact**: `results/artifacts/recovery-1580-1589/`

### Weight-norm growth (the primary driver)

| h | local | standard | threshold | std/lcl | thr/lcl |
|---:|---:|---:|---:|---:|---:|
| 20 | 33.6 | 33.7 | 33.7 | 1.005 | 1.004 |
| 80 | 35.6 | 37.1 | 36.3 | 1.041 | 1.018 |
| 120 | 37.0 | 40.9 | 38.3 | 1.107 | 1.037 |
| 160 | 38.3 | 50.5 | 41.3 | 1.319 | 1.079 |
| 180 | 39.0 | 59.7 | 44.2 | 1.530 | 1.133 |
| 200 | 40.0 | **71.1** | 49.2 | **1.778** | 1.229 |

Standard 4m+r nearly doubles its weight norm relative to local by epoch 200. The gated rules grow far less. The late margin explosion is a **weight-norm phenomenon** enabled by ungated shared mean credit.

### ρ≥0.5 group fraction declines with training

| h | ρ≥0.5 | ρ≥0.35 |
|---:|---:|---:|
| 20 | 0.350 | 0.472 |
| 80 | 0.314 | 0.465 |
| 140 | 0.250 | 0.392 |
| 180 | 0.166 | 0.307 |
| 200 | **0.133** | 0.268 |

As training proceeds the credit geometry becomes less mean-dominated. The rho05 gate therefore fires less often late — but the groups it still suppresses are increasingly the ones whose shared mean is aligned with the margin gradient, so the gate removes helpful credit exactly when it would matter.

### Shared-mean ↔ hard-margin gradient cosine

| h | standard | threshold | local_power |
|---:|---:|---:|---:|
| 20 | −0.002 | −0.004 | −0.004 |
| 80 | +0.009 | +0.010 | +0.003 |
| 140 | +0.038 | −0.056 | −0.062 |
| 180 | +0.056 | +0.025 | +0.042 |
| 200 | **+0.060** | +0.015 | +0.030 |

Standard's shared-mean component becomes positively aligned with the hard-margin gradient in the late phase. Threshold and local_power show weaker or negative alignment in the valley (h=120–160) because they have been suppressing the high-ρ groups that carry this alignment.

### Mechanism summary

```
Ungated 4m+r late recovery
  = (1) continued weight-norm growth (1.78× local at h=200)
  + (2) shared-mean component rotating into alignment
        with the held-out hard-margin gradient (cos +0.06)
  + (3) the ρ≥0.5 population shrinking (0.35 → 0.13),
        so the credit geometry itself becomes more residual-like
        and the extra mean gain acts on a different structure
        than it did in the valley.

The rho05 / local_power gates block (1) and (2) by suppressing
exactly the groups whose shared mean is most aligned with the
margin gradient in the late phase. The gate that regularizes
the valley becomes a brake on the recovery.
```

## Late-gate causal intervention (1600-1619, parallel agent)

- **Workflow run**: 34707716130 (success, 2m56s)
- **Seeds**: 1600-1619 (20/20), finite 400/400
- **Design:** pretrain as ungated standard 4m+r for **180 epochs**, then branch into `standard` (continue), `rho05`, `local_power`, or `local` for 0/1/5/10/20 further steps. Jitter base 10_300_000.
- **Artifact:** `results/artifacts/late-gate-intervention-1600-1619/`

### Absolute means after 180+20 epochs

| branch | CE | acc | hard | soft |
|---|---:|---:|---:|---:|
| standard (continue) | **1.099** | **0.839** | **+0.249** | **−0.679** |
| rho05 gate | 1.116 | 0.827 | +0.231 | −0.704 |
| local_power gate | 1.114 | 0.831 | +0.234 | −0.702 |
| switch to local | 1.131 | 0.791 | +0.213 | −0.730 |

### Paired Δ vs continuing standard, branch_step=20

| branch | CE Δ | acc Δ | hard Δ | soft Δ |
|---|---|---|---|---|
| rho05 | +0.0163* | −0.0120 | **−0.0177*** | −0.0244* |
| local_power | +0.0150* | −0.0073 | **−0.0154*** | −0.0228* |
| local | +0.0317* | −0.0471* | **−0.0365*** | −0.0506* |

Effects grow monotonically with branch_step (0 → 1 → 5 → 10 → 20), a clean dose–response.

### Causal statement

**Applying the rho05 or local_power gate after 180 epochs of ungated 4m+r significantly reduces the hard-margin gain and worsens CE and soft margin.** Switching to plain local is even worse. Continuing ungated shared type-pair mean credit is the best late-phase action.

This is the causal counterpart of the correlational gradient-alignment result: the gate removes shared-mean credit that, in the late phase, is aligned with the held-out hard-margin gradient. The gate that regularizes the mid-training valley is a brake on the late recovery.

## Next single experiment

**Biological-strength scale perturbation.** Train standard 4m+r and local to epoch 200 under `use_biological_strength` multipliers of roughly {0.5, 1.0, 1.5} (or an equivalent lighter/heavier init), and check whether the late recovery is specific to one init scale or is a generic property of shared mean credit. Also still open: a far-jitter hold-out that re-evaluates saved weights under a disjoint jitter base.

## Artifact index (local)

```
results/artifacts/norm-control-1480-1499/
results/artifacts/full-horizon-1500-1519/
results/artifacts/full-horizon-1520-1539/
results/artifacts/full-horizon-200-1560-1579/     # U-turn discovery
results/artifacts/recovery-1580-1589/             # recovery mechanism
results/artifacts/late-gate-intervention-1600-1619/ # causal late-gate brake
results/artifacts/boundary-geometry-1420-1439/
results/artifacts/mechanism-1540-1559/
results/artifacts/branched-1460-1479/
results/artifacts/local-power-holdout-1440-1459/
```
