# First-order PC-ALM hardware crossover bound

Date: 2026-09-15

This is a symbolic dependency/work model, not synthesis data. It deliberately separates arithmetic work from layer-dependency waves.

For an equal-width dense network with `H=L-1` hidden states of width `N`, approximate one simultaneous PC-ALM state/dual step as:

- forward local prediction: `H N^2` MACs,
- transpose-neighbour state-gradient term: `H N^2` MACs,
- dual update: `2 H N` scalar multiply/add operations,
- persistent dynamic state: `z` plus `lambda` = `2 H N` words (sPC has `H N`).

A reverse BP/ePC credit sweep needs approximately `H N^2` transpose-matvec MACs for credit propagation alone. Parameter-gradient outer products are omitted from both sides because both learning procedures eventually require them.

Therefore, for `T` PC-ALM relaxation steps:

`total PC-ALM matvec work / one reverse credit sweep ~= 2T`.

This is the key negative bound: spatial parallelism does **not** reduce total arithmetic work. It can only reduce elapsed dependency waves if enough layer hardware is instantiated.

With one dedicated engine per layer:

- PC-ALM: about `T` layer-dependency waves,
- BP/ePC reverse credit: about `H` ordered layer-dependency waves.

Thus the idealized latency crossover requires roughly `T < H` before considering clock-rate/resource differences. For depth 32 (`H=31`), a 96-step PC-ALM run has wave ratio `96/31 = 3.10`; the scalar exact-dual result at 1370 steps has ratio `44.2`. Neither beats a single reverse sweep on this idealized latency metric.

For the project's depth-32,width-8 example:

- one PC-ALM step: ~3968 matvec MACs,
- 96 steps: ~380,928 matvec MACs,
- one reverse credit sweep: ~1,984 matvec MACs,
- PC-ALM dynamic state: 496 words (`z + lambda`), versus 248 words for sPC state alone.

This does **not** rule out an FPGA advantage. It sharply narrows what must provide it: lower-cost fixed-point operators, local/on-chip reuse that avoids repeated global weight/state traffic, overlap with other learning work, sparse/structured weights, or a dynamics/architecture choice that cuts T far below the current exact-dual results. A claim based only on layer parallelism is not supported by this model.

The next hardware model should add explicit weight-memory organization and bandwidth. If each PC-ALM step must reread both W and W^T from BRAM/DRAM, the 2T work penalty is also approximately a traffic penalty; if weights remain resident next to layer engines, external traffic can instead favor the FPGA despite larger on-chip arithmetic work.
