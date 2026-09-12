# Research log: type-pair shared credit & local credit geometry

Session date: 2026-09-12/13 (UTC+9)
Branch: main
Starting SHA: `0bffcfc`
Ending SHA: (updated at close)

## CI / infrastructure

| Item | Status |
|---|---|
| Starting main SHA | `0bffcfc` "Run local power norm-control trajectory 1480-1499" |
| CI at start | **RED** — ruff format on `diagnose_flyvis_type_pair_local_power_norm_control_trajectory.py` |
| Format fix | `a607aa5` (rebased/applied) |
| CI after format fix | **GREEN** |
| Experiment workflows this session | local-proxy mechanism (success); full-horizon holdout 1500-1519 (success, parallel agent); full-horizon replication 1520-1539 (in progress at close) |
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

```
local (m+r) is the safest default from epoch 0.
If shared type-pair mean credit is used at all, gate it:
  suppress (extra gain 3→1) when ρ = ||m||/||r|| ≥ 0.5
  and additionally when 0.35 ≤ ρ < 0.5 and mean(w·c) > 5e-8
but expect an early accuracy peak (~epoch 40) and later decay,
and expect a CE / soft-margin cost relative to rho05 alone.
```

No simpler or stronger local proxy than `mean(w·c)` was confirmed.

## Implementation / CI notes

- Only CI issue: ruff format of the norm-control trajectory script. Fixed; CI green after `a607aa5`.
- A duplicate 40-seed full-horizon workflow (`flyvis-type-pair-full-horizon-rules`, seeds 1500-1539) was started this session and then **cancelled** after discovering a parallel agent's 20-seed holdout on the same seeds. No science lost; the 20-seed holdout + 20-seed replication covers the 40-seed requirement.
- Mechanism discovery uses held-out hard-margin gradient strictly as a diagnostic oracle. It is never written back into any training direction.

## Next single experiment

**Extend full-horizon training to epoch 200** on the existing 1500-1519 checkpoint protocol (or a fresh 20-seed block), measuring whether the h=40 accuracy peak of shared-credit rules can be recovered, whether hard margins go positive again, and whether the CE/soft cost of local_power saturates or grows. This is the highest-value remaining question because the current results say the entire shared-credit family is a **transient** effect under epoch-0 training.

Secondary (only after 200-epoch data): evaluation-jitter hold-out with a completely different jitter base; biological-strength scale perturbation; temporal Jaccard of selected type-pair groups.

## Artifact index (local)

```
results/artifacts/norm-control-1480-1499/
results/artifacts/full-horizon-1500-1519/
results/artifacts/boundary-geometry-1420-1439/
results/artifacts/mechanism-1540-1559/
results/artifacts/branched-1460-1479/          # downloaded, not re-analyzed this session
results/artifacts/local-power-holdout-1440-1459/ # downloaded, not re-analyzed this session
```
