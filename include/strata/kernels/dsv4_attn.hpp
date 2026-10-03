// include/strata/kernels/dsv4_attn.hpp - DeepSeek-V4 attention over the tiered pools, docs/DSV4.md P2.
//
// DSv4 attention is a GATHER, not a scan: each query attends a list of GLOBAL rows - the sliding-window
// ring first (up to `window` rows, the raw per-token KV), then compressed rows (one per compressor
// block) - and the softmax denominator additionally carries the ATTENTION SINK, a null key with logit
// `sinks[h]` and zero value (FreeToken `kernel/triton/dsv4/sparse_attn.py`: `l_i += exp(sink - m_i)`
// after the real rows).  A row serves as BOTH key and value (latent attention: the 512-wide row is the
// key and the value), and rows are stored bf16 already rope-applied and quant-dequant-rounded - the
// pool's contract, not this kernel's concern.
//
// The id lists carry -1 for "no pick" (short windows, padded top-k); those rows are skipped, not read.
// Which ids appear, and how the compressed list was chosen (indexer top-k on r=4 layers, every valid
// block on r=128 layers, none on window-only layers), is the layer's job - this kernel just gathers.
//
// Two shapes: DECODE (one query per head) and PREFILL (n queries, each with its own id lists).
#pragma once

#include <cstdint>

namespace strata::kernels {

struct Dsv4AttnPools {
    const uint16_t* window;  ///< bf16 [window_cap][d] ring
    const uint16_t* cmp;     ///< bf16 [cmp_rows][d]
};

/// `o[h][c] = sum_i softmax_i(scale * q[h].k_i + sink_h) * v_i[c]` over the gathered rows, the sink
/// contributing to the denominator only.  `q` is `[n_heads][d]` f32, `o` the same; `win_ids`/`cmp_ids`
/// are row indices into the two pools, -1 skipped; `sinks` may be null (no sink term).  Online softmax,
/// one block per head.
void dsv4_attn_decode(const float* q, const int32_t* win_ids, int64_t n_win, const int32_t* cmp_ids,
                      int64_t n_cmp, const float* sinks, float scale, const Dsv4AttnPools& pools,
                      int64_t window_cap, int64_t n_heads, int64_t d, float* o, void* stream);

/// The PREFILL shape: `n` queries in one launch.  `q` is [n][n_heads][d] f32 and `o` the same;
/// `win_ids`/`cmp_ids` are per-query lists ([n][n_win] and [n][n_cmp]), -1 skipped as always.
/// Which rows each query may see (its causal window, its indexer picks) is the caller's - this
/// kernel gathers.  One block per (query, head); the math is the decode kernel's exactly, so a
/// prefill row must match a decode call given the same lists.
void dsv4_attn_prefill(const float* q, const int32_t* win_ids, int64_t n_win,
                       const int32_t* cmp_ids, int64_t n_cmp, const float* sinks, float scale,
                       const Dsv4AttnPools& pools, int64_t window_cap, int64_t n, int64_t n_heads,
                       int64_t d, float* o, void* stream);

}  // namespace strata::kernels
