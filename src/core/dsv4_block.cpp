// src/core/dsv4_block.cpp - the join of the two parity-tested halves (see dsv4_block.hpp).
#include "strata/core/dsv4_block.hpp"

#include "strata/core/layout.hpp"

namespace strata::core {

namespace {

// The arena: [attn half][ffn half][staging stream].  256-byte alignment between regions keeps the
// halves' own alignment assumptions (they take float*/int8_t* from one pointer) true at any offset
// that is itself aligned.
int64_t align256(int64_t n) { return (n + 255) / 256 * 256; }

}  // namespace

int64_t dsv4_block_scratch_bytes(const ModelGeometry& g, int64_t layer, int64_t n_stage) {
    const int64_t hc = g.hc;
    return align256(dsv4_attn_scratch_bytes(g, layer, n_stage)) +
           align256(dsv4_ffn_scratch_bytes(g)) + align256(hc * g.n_embd * (int64_t) sizeof(float));
}

void dsv4_block_decode_step(const ModelGeometry& g, int64_t layer, const Dsv4BlockWeights& w,
                            const Dsv4LayerState& st, const float* stream_in, float* stream_out,
                            int64_t pos, int64_t token_id, int64_t n_stage, float* scratch,
                            void* cu_stream) {
    const int64_t hc = g.hc;
    float* p = scratch;
    float* s_attn = p;
    p += align256(dsv4_attn_scratch_bytes(g, layer, n_stage)) / (int64_t) sizeof(float);
    float* s_ffn = p;
    p += align256(dsv4_ffn_scratch_bytes(g)) / (int64_t) sizeof(float);
    float* mid = p;  // [hc * n_embd]: hc_post_combine reads its residual while writing, so the
                     // halves may not share the caller's buffers.

    dsv4_attn_decode_step(g, layer, w.attn, st, stream_in, mid, pos, n_stage, s_attn, cu_stream);
    dsv4_ffn_decode_step(g, layer, w.ffn, mid, stream_out, token_id, s_ffn, cu_stream);
}

}  // namespace strata::core
