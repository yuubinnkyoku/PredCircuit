# Multiplier-free PC-ALM state-step budget: T=64 and T=96

Date: 2026-09-20

Configuration: depth=32, width=64, batch=4, `state_lr=0.25` (effective state step = 1.0), state `fixed15_i3`, update `fixed14_i1`, dual `fixed12_i1`, dual leak 0.02. Seeds 845--859. Source artifacts: Actions run 35453887542. These are recovered artifacts; no experiment was rerun for this summary.

Existing useful-credit criterion: finite, first-layer cosine >= 0.9, gradient-norm ratio in [0.5, 2.0], relative error <= 0.6.

## T=64

Useful first-layer credit: **0/15**.

- first-layer BP cosine: mean **0.888623**, sample SD **0.019380**, range **0.861053--0.928545**
- first-layer gradient-norm ratio to BP: mean **0.335981**, SD **0.024748**, range **0.303479--0.379386**
- first-layer relative error to BP: mean **0.717724**, SD **0.025946**, range **0.674614--0.751688**
- late realized/requested state-step norm ratio: mean **1.031839**
- late update zero fraction: mean **0.463385**
- late state zero-step fraction: mean **0.746578**
- residual total: mean **0.060932**

The decisive failure at T=64 is not just cosine: every seed has first-layer gradient norm below half of BP, so the deep credit has not reached sufficient magnitude.

## T=96

Useful first-layer credit: **10/15**.

- first-layer BP cosine: mean **0.908275**, sample SD **0.015504**, range **0.890494--0.943649**
- first-layer gradient-norm ratio to BP: mean **0.776529**, SD **0.052701**, range **0.692889--0.844109**
- first-layer relative error to BP: mean **0.438628**, SD **0.041280**, range **0.351327--0.494431**
- late realized/requested state-step norm ratio: mean **1.014861**
- late update zero fraction: mean **0.428749**
- late state zero-step fraction: mean **0.770261**
- residual total: mean **0.061644**

The five failures are seeds **847, 849, 856, 857, 858**. All five already pass the gradient-norm-ratio and relative-error thresholds; they fail only because first-layer cosine remains just below 0.9 (0.890494--0.896787). Therefore T=96 is close to the robust boundary rather than qualitatively broken.

## Combined interpretation with T=128

The previously recovered T=128 artifacts give 15/15 useful, mean cosine 0.947421 and mean norm ratio 0.876736. Thus the robust multiplier-free budget lies in **96 < T <= 128** for this configuration. T=64 is clearly under-relaxed; T=96 has largely recovered gradient magnitude but five hard seeds still miss the directional-alignment threshold by <0.01 cosine.

A full 15-seed resweep is not the highest-value next experiment. The informative test is to run only the five T=96 failures at intermediate budgets (104, 112, 120), which directly localizes the minimum robust budget while avoiding 30 redundant successful trajectories.
