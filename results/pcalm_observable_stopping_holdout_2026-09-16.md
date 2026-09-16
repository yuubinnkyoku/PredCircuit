# Pure PC-ALM observable stopping: fresh 20-seed holdout

Date: 2026-09-16
Source: Actions run 35088765482, commit `0b25f58b89ab99ae8eee9149b801461adab6d323`.
Seeds 920--939, budgets T=48..160 at every step. No dual leak.

## Split

The stopping rule search was restricted to seeds 920--929. Seeds 930--939 were held out until after rule selection. BP quantities are used only to score a selected stopping time, never as stopping inputs.

Observable candidates: residual norm, dual norm, dual-step norm, state-step norm, AL energy, and simple normalized/ratio variants. We searched deliberately simple first-threshold-crossing rules with a minimum-T guard; this is a diagnostic, not a claim that all possible learned stopping policies fail.

## Oracle availability

A useful first-layer-credit point exists somewhere in T=48..160 for:

- train 920--929: 7/10 seeds (failures: 921, 925, 929)
- holdout 930--939: 9/10 seeds (failure: 935)
- all fresh seeds: 16/20

Thus the good-credit window is real on fresh seeds and is not an artifact of the earlier seed set. It is also highly seed-dependent. Example useful windows include 927: T=62..67, 924: 107..160, 931: 61..74, and 934: 124..160.

## Fixed-T baseline on this fresh split

No single T is close to the oracle. On the training half the best fixed-T success count is only 3/10 (many tied T values, including 63--67 and several points around 96--108). Representative holdout counts are also about 1--3/10 at those T values.

This is substantially below the 7/10 train and 9/10 holdout oracle availability.

## Observable stopping diagnostic

The simple hardware-visible first-crossing rules did not close the gap. Across residual norm, dual-step norm, state-step norm, relative dual step, state-step/residual, dual-step/residual, and values normalized to T=48, the best training success was 4/10. The best such rule reached 5/10 on holdout, but this does not rescue the method: it was selected on only ten training seeds and remains far below the 9/10 holdout oracle.

AL energy is especially unhelpful in this experiment: its minimum over the measured interval is T=48 for every seed, so a local/minimum-energy stopping interpretation does not track the useful gradient window. Residual-norm minima are also generally late and inconsistent with the useful window (for example seed 927 is useful only at 62..67 while its residual minimum is at 138; seed 934 is useful at 124..160 while its residual minimum is at 96).

## Interpretation

The earlier hypothesis that pure PC-ALM can recover most of its oracle performance with a very cheap norm-threshold early stopper is not supported by this fresh experiment. The observables do contain dynamics, but the time at which BP-like credit is useful is not aligned by a single simple threshold across seeds.

This strengthens the engineering case for dual leak (or another dynamics-level stabilizer) rather than relying on a cheap early-stop comparator alone. It does **not** prove that observable stopping is impossible: a multivariate/history-based rule could still work, but such a controller has nonzero hardware/state cost and should be compared fairly against the very cheap leaky-dual update already showing much broader fixed-T stability.

The important scientific distinction is therefore:

1. pure PC-ALM frequently *passes through* a useful deep-credit regime (16/20 fresh seeds have at least one useful T in 48..160),
2. that regime is poorly synchronized across seeds,
3. simple BP-free scalar stopping rules recover only part of the oracle gap,
4. dual leak remains the stronger current route to fixed-budget digital hardware.

## Next experiment

Before spending effort on a learned stopping controller, compare leak values around the existing 0.01 setting on the same fresh seeds, recording useful-window width and fixed-T success together with lambda dynamic range. The key question is whether leak broadens the useful window while also reducing lambda range/oscillation. If so, leak helps both algorithmic robustness and fixed-point hardware, which is a stronger result than early stopping alone.
