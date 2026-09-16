# Pure PC-ALM fixed-budget gap (2026-09-16)

This note re-audits the current depth-32 / width-8 matched-credit holdout before treating the 8-lane RTL point as evidence for the main PC-ALM claim. No experiment is rerun here; it uses the existing 20-seed holdout in `pcalm_matched_credit_common_budget_900_919.csv` and `pcalm_matched_credit_summary_900_919.csv`.

## The important distinction

The strongest fixed-budget result currently belongs to the **dual-leak engineering variant**, not unmodified PC-ALM.

At T=128:

| method | useful seeds | useful rate | cosine mean | norm ratio mean | relative error mean |
|---|---:|---:|---:|---:|---:|
| sPC | 0/20 | 0.00 | 0.8646 | 0.0000816 | 0.99993 |
| pure PC-ALM | 10/20 | **0.50** | **0.9592** | 1.4217 | 0.5546 |
| PC-ALM + 0.01 dual leak | 19/20 | **0.95** | 0.9491 | 1.0135 | 0.3412 |

Pure PC-ALM nevertheless reaches the useful-credit criterion at *some* tested budget by T=160 on 17/20 seeds, with median first successful T=96 and IQR 64--112. Therefore the pure method clearly propagates useful deep credit much better than sPC, but it does **not** yet provide a robust common fixed T at the current coefficients.

## Non-monotonicity is the signal

Pure PC-ALM useful rate over the existing common-budget sweep is:

- T=64: 0.30
- T=80: 0.15
- T=96: 0.35
- T=104: 0.30
- T=112: 0.40
- T=120: 0.55
- T=128: 0.50
- T=144: 0.40
- T=160: 0.20

Meanwhile its mean first-layer cosine keeps improving to roughly 0.96, while the mean norm ratio grows from 0.53 at T=64 to 1.66 at T=160. The degradation at long T is therefore not simply a failure to align the gradient direction. The trajectory passes through a useful magnitude regime and then tends to overshoot it. This is consistent with the dual dynamics needing a stopping/stability mechanism rather than simply more iterations.

The leaky variant changes this qualitatively: useful rate rises to 0.95 at T=128 and remains 0.90 at T=144 and T=160, while its norm ratio stays near one (1.0135 at T=128, 1.1126 at T=160).

## Consequence for the hardware gate

The existing 8-lane RTL measurements remain valid engineering measurements for the local dual-update datapath. However, the strongest `T=128 versus sPC >256` system-level break-even statement currently applies to **PC-ALM + dual leak**, not to the unmodified Sakana-style PC-ALM baseline.

For the main research claim, the gate should therefore be split:

1. **Mechanistic gate (pure PC-ALM): passed provisionally.** Pure PC-ALM reaches useful first-layer BP-like credit on 17/20 seeds by T=160 whereas sPC reaches 0/20 by T=256.
2. **Fixed-budget deployment gate (pure PC-ALM): not yet passed.** The best tested common budget is only 11/20 useful at T=120, and performance falls again at larger T.
3. **Engineering deployment gate (dual leak): passed at the current small regime.** T=128 gives 19/20 useful seeds and is compatible with the current low-precision/RTL work.

This prevents the robust leaky result from being silently attributed to pure PC-ALM.

## Highest-value next experiment

Do not spend the next run sweeping more RTL lane counts. The next scientific discriminator should test whether **pure PC-ALM can recover its 17/20 ever-success rate with an observable early-stop rule**, without looking at BP gradients at inference time.

The experiment should record at every outer step, per layer:

- residual norm and relative residual change;
- lambda norm and lambda increment norm;
- augmented-Lagrangian energy (or the same shifted energy already used by the implementation);
- hidden-state update norm;
- finite/saturation indicators.

Candidate stopping rules can then be fitted on a disjoint tuning seed set and evaluated on a fresh holdout. A useful rule must use only quantities available to the PC-ALM dynamics; BP cosine/norm/relative error may be used only as the evaluation target, never as the stopping signal.

If such a rule approaches the 17/20 oracle-by-budget reach rate, pure PC-ALM becomes a stronger hardware candidate without changing its equations. If no observable rule predicts the useful window reliably, dual leak should be treated as a substantive stabilization mechanism rather than a minor implementation tweak, and the main claim should say so explicitly.
