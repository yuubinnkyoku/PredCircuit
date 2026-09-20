# PC-ALM residual transport: streamed schedule vs 64→8 serializer (2026-09-20)

## Status

At `f90bf4a2`, both CI and RTL smoke are green after fixing the serializer testbench sampling edge. The serializer itself is therefore functionally validated; the previous `chunk 1 expected 0` failure was a testbench timing bug, not a datapath failure.

The RTL-smoke XC7 synthesis artifact gives the following mapped result for `residual_serializer_64to8` (DATA_W=12, 64→8):

- estimated logic cells: **836**
- FDRE: **772**
- LUT6: 291
- LUT4: 544
- MUXF7: 640
- MUXF8: 96
- BRAM: **0**
- DSP: **0**

This is unexpectedly expensive compared with the intended role of the block. The reason is structural: the serializer atomically captures all 64×12 = 768 residual bits and then implements a wide variable chunk-select mux. The 768-bit buffer is small in information capacity, but not cheap as a 64-wide-write / 8-wide-read register-and-mux structure on XC7.

## Better schedule: do not create the 64-wide burst

For the first shared-64-MAC architecture, a dense width-64 matrix-vector product does not need to materialize all 64 output residuals in one cycle. A natural schedule assigns the 64 MAC lanes to one 64-element dot product at a time:

- 64 MACs compute one output dot product per cycle (ignoring pipeline fill/drain),
- therefore the matrix side produces about **1 residual scalar/cycle**,
- the selected dual engine can accept **8 residual scalars/cycle**.

Hence the dual consumer has an 8× instantaneous service-rate margin even before using the much larger whole-relaxation average margin already measured.

For a 64×64 matvec, the same 4096 MACs take 64 cycles either way. The difference is only output scheduling:

- burst schedule: hold/collect 64 outputs, then serialize 64→8;
- streamed schedule: emit each completed scalar (or a small fixed group) directly into the dual path as it becomes available.

The streamed schedule removes the need for the synthesized 836-LC serializer entirely. A tiny elastic skid buffer may still be useful for pipeline alignment/backpressure, but its required capacity should be derived from the actual MAC pipeline latency and dual ready/valid trace rather than fixed at 64 scalars.

## Consequence for the architecture decision

The earlier 64-entry FIFO proof remains a valid *upper-bound fallback* for an architecture that really emits a 64-scalar atomic burst. It should no longer be the preferred implementation.

Preferred first design point:

- shared `P_MAC = 64`, scheduled as one width-64 dot product per cycle,
- residual production ≈ 1 scalar/cycle after fill,
- shared `P_lambda = 8`,
- direct streamed residual→dual connection with only minimal elastic buffering,
- persistent lambda store remains 7936×12 = **95,232 bits**.

This strengthens the earlier conclusion that the unavoidable PC-ALM-specific hardware cost is lambda state storage, not dual arithmetic or residual serialization.

## Next falsification target

Build/measure a cycle-accurate producer model (or minimal MAC scheduler stub) that emits the real residual valid pattern and connect it directly to the 8-lane dual path. Record:

1. maximum elastic-buffer occupancy,
2. producer stalls caused by dual backpressure,
3. exact address mapping from scalar residual order to the 992×8 lambda words,
4. pre-dual credit ordering at the streamed interface.

The target is **zero producer stall** with substantially less than 64-scalar buffering. If this fails because the real MAC schedule emits multi-scalar bursts, size the buffer from the measured trace rather than restoring the 768-bit atomic serializer by default.
