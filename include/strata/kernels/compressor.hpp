// include/strata/kernels/compressor.hpp - the DeepSeek-V4 KV compressor's math, docs/DSV4.md P2.
//
// A compressor collapses `ratio` consecutive token rows into ONE pooled row by a softmax over the ROW axis,
// per column: `out[c] = sum_r kv[r][c] * softmax_r(score[:, c])[r]` (FreeToken `kernel/triton/dsv4/
// compress.py::gated_pool`).  The softmax is per COLUMN - `score[:, c]` over r - and not over the columns
// of a row; the parity test asserts the two readings are observably different rather than trusting the
// prose.  Two shapes exist (dsv4_geometry's layer classes):
//
//   * ratio 4, `overlap`: the window is EIGHT rows - the previous block's first-half columns [0, d) and
//     the current block's second-half [d, 2d) - so `wkv`/`wgate` are 2d wide and the halves cross block
//     boundaries.  The rows outside the window carry score -inf and drop out of the softmax.
//   * ratio 128, no overlap: the window is the block itself, d wide.
//
// Between calls a per-layer CARRY REGISTER holds the in-progress block: `ks` rows raw kv, `ss` rows score
// WITH the ape already added (the reference stores it that way, `Compressor.forward` adds `self.ape` at
// every write).  Row layout is [coff*ratio, item]: for overlap the first `ratio` rows are the previous
// block (only their first half is read) and the next `ratio` rows the current one; without overlap just
// `ratio` rows indexed by `pos % ratio`.  A block completes when `(pos + 1) % ratio == 0`; overlap then
// ROLLS the register (`ks[:ratio] = ks[ratio:]`), non-overlap does not - its slots are addressed by
// position and simply overwritten.
//
// What is NOT here: the projections (NativeDense serves `attn_compressor_kv/gate`), the RMSNorm, the RoPE
// at block-start positions and the fp8/fp4 quantization of the pooled row - those are `build_norm`,
// `rope` and `quantize_act`, already existing - and the paged pool addressing, which is the layer's job.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// Pool every COMPLETE block of a prefill chunk into `out[b][0..d)`.
///
/// `kv`/`score` are the chunk's projected rows, `[seqlen, item]` with `item = (overlap ? 2 : 1) * d`.
/// `ape` is `[ratio, item]` and is added to every score row read from `kv`/`score` at its in-block index.
/// For `overlap`, block 0's previous-half rows come from `carry_kv`/`carry_score` (`[ratio, item]`, the
/// score rows ALREADY ape-added - that is how the carry stores them) when non-null, and are -inf/0 when
/// null (a from-scratch prefill has no previous block).  Blocks after 0 take their previous half from the
/// preceding block of this same chunk.  `out` needs `[seqlen / ratio, d]` rows; a chunk shorter than
/// `ratio` launches nothing.
void compressor_prefill(int64_t seqlen, int64_t ratio, bool overlap, int64_t d,
                        const float* kv, const float* score, const float* ape,
                        const float* carry_kv, const float* carry_score,
                        float* out, void* stream);

/// Seed the carry register after a prefill chunk, exactly as `Compressor.forward` leaves it: for overlap,
/// `ks[:ratio]` = the LAST complete block's raw kv, `ss[:ratio]` = its score + ape, the current-block
/// half zeroed/-inf'd, then the `seqlen % ratio` remainder rows written at `[ratio, ...]` (at `[0, ...]`
/// without overlap).  `cutoff = seqlen - seqlen % ratio`; with overlap and `cutoff < ratio` the first part
/// is skipped, as the reference's guard does.
void compressor_seed_carry(int64_t seqlen, int64_t ratio, bool overlap, int64_t item,
                           const float* kv, const float* score, const float* ape,
                           float* ks, float* ss, void* stream);

/// Advance the register by one decode token and, when the block completes, pool it into `compressed`.
///
/// `pos` is the absolute position of THIS token.  The token's raw `kv`/`score` (one `item`-wide row each)
/// is written at `ks[ratio + pos % ratio]` / `ss[...]` (overlap) or `ks[pos % ratio]` (not), the score
/// with `ape[pos % ratio]` added.  When `(pos + 1) % ratio == 0` the window is pooled into
/// `compressed[d]` and, for overlap, the register is rolled; otherwise `compressed` is left untouched.
/// The caller knows `pos` on the host and only scatters `compressed` on completion.
void compressor_decode_step(int64_t pos, int64_t ratio, bool overlap, int64_t d,
                            const float* kv, const float* score, const float* ape,
                            float* ks, float* ss, float* compressed, void* stream);

}  // namespace strata::kernels
