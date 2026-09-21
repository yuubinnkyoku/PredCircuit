# PC-ALM width-64 dyadic state-step boundary at T=103

Configuration: depth=32, width=64, effective state step=1.0, same fixed-point / dual-leak configuration as the preceding T=96/100/102/104 boundary sweep. These five seeds were selected because they failed the T=96 useful-credit criterion, so this is a hard-seed boundary set, not an independent estimate of population success rate.

| seed | BP cosine | grad-norm ratio | relative error | finite | useful |
|---:|---:|---:|---:|:---:|:---:|
| 847 | 0.90062648 | 0.71355025 | 0.47314835 | yes | yes |
| 849 | 0.90708971 | 0.77362376 | 0.44158965 | yes | yes |
| 856 | 0.90204298 | 0.75795335 | 0.45506069 | yes | yes |
| 857 | 0.90271789 | 0.73683239 | 0.46110576 | yes | yes |
| 858 | 0.90185052 | 0.81421776 | 0.44084576 | yes | yes |

All five pass at T=103. All observed state/update/dual saturation rates are zero. The limiting seed remains 847, only 0.00062648 above the cosine threshold 0.9.

Thus, on this selected hard-seed set, the measured boundary is exactly between T=102 and T=103: T=102 had 4/5 useful while T=103 has 5/5 useful. Relative to the previous conservative T=128 point, T=103 removes 25/128 = 19.53125% of relaxation iterations, and correspondingly the dominant per-relaxation MAC/state-traffic terms, without changing numeric formats.

This does **not** establish T=103 as a robust operating point: the margin is narrow and the five seeds are selection-biased. The next test is an independent 15-seed holdout at T=103 before adopting it as the hardware budget.
