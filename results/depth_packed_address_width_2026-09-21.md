# Depth-packed weight address width analysis (2026-09-21)

For the planned 30 hidden 64x64 matrices, there are 30*64*64 = 122,880 logical weights. Therefore a flat logical index needs ceil(log2(122,880)) = 17 bits.

Using the conflict-free cyclic mapping

- bank = (row + col) mod P
- local_address = layer*(4096/P) + row*(64/P) + floor(col/P)

with P in {8,16,32,64}, the packed bank depths and minimum coordinate widths are:

| P | bank bits | bank depth | address bits | bank+address bits |
|---:|---:|---:|---:|---:|
| 8 | 3 | 15,360 | 14 | 17 |
| 16 | 4 | 7,680 | 13 | 17 |
| 32 | 5 | 3,840 | 12 | 17 |
| 64 | 6 | 1,920 | 11 | 17 |

Thus increasing spatial parallelism does not increase the minimum information width needed to identify a stored weight: every design point remains exactly 17 bits as a `(bank,address)` coordinate, matching the 17-bit flat-index lower bound. Each doubling of P moves exactly one bit from local address selection into bank selection.

This does **not** prove that routing is free. At P=64, six bank-select bits fan into a 64-way physical memory/MAC structure, so placement/routing and output permutation can still reduce Fmax. But the address representation itself has no bit-width or metadata expansion with P, and no permutation lookup table is required for this aligned power-of-two schedule.

Consequence for the next synthesis experiment: compare P=16 and P=64 while attributing any LUT/Fmax increase to bank steering/routing rather than address-state width. The address generator should be treated as a small affine/bit-slice block, not as a growing memory structure.
