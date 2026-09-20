# RAMB36 legal-shape correction (2026-09-21)

## Why the previous capacity model was optimistic

The previous cyclic-banking analysis divided useful weight bits by 36,864 bits per RAMB36E1. That is a lower bound, not an implementable primitive count for a 14-bit writable weight bank.

For 7-series RAMB36E1 true-dual-port RAM, a 14-bit word occupies the legal 18-bit-wide configuration, whose depth is 2,048 words. Therefore each independent cyclic bank must be built from

```
ceil(bank_depth / 2048)
```

RAMB36E1 primitives. The unused 4 bits in every 18-bit physical word cannot be converted into extra depth for that bank.

The 512x72 simple-dual-port mode is deliberately not used in this model because PredCircuit ultimately needs writable weights and the current banking argument assumes an independently addressable word per bank rather than packing four logical 14-bit banks behind a 72-bit word.

## Corrected 30-layer result

For 30 hidden 64x64 matrices and 14-bit weights:

| P | logical depth / bank | RAMB36 / bank (18x2048) | total RAMB36 | useful-bit efficiency |
|---:|---:|---:|---:|---:|
| 8 | 15,360 | 8 | 64 | 72.92% |
| 16 | 7,680 | 4 | 64 | 72.92% |
| 32 | 3,840 | 2 | 64 | 72.92% |
| 64 | 1,920 | 1 | 64 | 72.92% |

So the earlier 48/48/64/64 estimate was not physically realizable under the assumed 14-bit TDP bank organization. The corrected result is:

```
P = 8,16,32,64  ->  64,64,64,64 RAMB36E1
```

This does **not** invalidate the conflict-free cyclic mapping or the depth-packed address bijection. It only corrects the conversion from logical bank occupancy to FPGA primitives.

## Consequence for the design-space argument

The previously inferred BRAM discontinuity at P=16 -> 32 disappears. Under the corrected primitive model, all four planned parallelisms cost the same 64 RAMB36E1 for these 30 hidden matrices. Thus P=16 no longer has a weight-BRAM-capacity advantage over P=64.

The comparison shifts more cleanly to DSP count, steering/routing cost, achievable Fmax, and cycle reduction. In particular, if a P=64 implementation can route and clock adequately, weight-BRAM count alone no longer argues for P=16.

PC-ALM's measured lambda-state addition (3 RAMB36 in the existing frontend synthesis) remains small relative to this 64-RAMB36 hidden-weight store: 3/64 = 4.6875%. That comparison is still architecture-specific and should not be treated as a device-wide area percentage.

## Remaining caveat

This is now a legal-primitive analytical model, but still not a synthesis result for the depth-packed weight store. The next high-value check is to synthesize the P=16 and P=64 stores and verify that inference actually realizes the expected RAMB36E1 count without unexpected replication or LUTRAM fallback, then measure steering logic and post-route Fmax.
