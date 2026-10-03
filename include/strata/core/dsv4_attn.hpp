// include/strata/core/dsv4_attn.hpp - the DeepSeek-V4 attention half of one decoder block, one
// decode token, docs/DSV4.md P2 integration.
//
// The transcription of `model.py::Block.decode_step` + `Attention.decode_step` for B=1, from the
// hc=4 residual stream in to the stream out:
//
//   hc_pre(attn) -> attn_norm -> q_a/q_b GEMVs + q_a_norm + per-head norm + rope ->
//   kv GEMV + kv_norm + rope + e4m3 round-trip -> window ring write ->
//   [r>0] compressor decode (kv/gate GEMVs, carry, pool, norm, compress-rope, e4m3) -> cmp write ->
//   [r=4] indexer (q_b GEMV + rope + hadamard + e2m3... e2m1 round-trip, weights GEMV, its own
//   compressor into the idx pool, logits, top-k) ->
//   window ids + cmp ids -> dsv4_attn_decode (sink included) -> inverse rope ->
//   wo_a grouped einsum + wo_b GEMV -> hc_post(attn).
//
// The weights arrive as raw device pointers (`Dsv4AttnWeights`): the production resolver fills it
// from `LayerView`/`WeightRef`, and the parity test fills it with randoms.  Orientations are the
// manifest's (ne0 contiguous), so every GEMV is `bf16_gemv_fp32_mmvf(x_f32, w_bf16, y_f32)`.
// `attn_output_a` is the flattened `wo_a`: row `g * o_lora + r` is group g's output r over the
// group's `heads_per_group * head_dim` slice of `o` - the einsum is one GEMV per group.
//
// `scratch` is caller-owned workspace sized by `dsv4_attn_scratch_bytes`; the pools and carries are
// `Dsv4State` buffers.  `pos` is the absolute position of this token.  Nothing here allocates.
#pragma once

#include "strata/core/dsv4_state.hpp"
#include "strata/core/layout.hpp"

#include <cstdint>

namespace strata::core {

struct Dsv4AttnWeights {
    const uint16_t* q_a = nullptr;    ///< [n_embd, q_lora]
    const uint16_t* q_b = nullptr;    ///< [q_lora, n_head * head_dim]
    const uint16_t* kv = nullptr;     ///< [n_embd, head_dim]
    const uint16_t* out_a = nullptr;  ///< [n_embd, o_groups * o_lora] (flattened wo_a)
    const uint16_t* out_b = nullptr;  ///< [o_groups * o_lora, n_embd]
    const float* norm = nullptr;      ///< attn_norm [n_embd]
    const float* q_a_norm = nullptr;  ///< [q_lora]
    const float* kv_norm = nullptr;   ///< [head_dim]
    const float* sinks = nullptr;     ///< [n_head]
    const float* hc_fn = nullptr;     ///< [mix_hc][hc_dim] f32
    const float* hc_base = nullptr;   ///< [mix_hc]
    const float* hc_scale = nullptr;  ///< [3]
    // compressor, null when ratio == 0
    const uint16_t* comp_kv = nullptr;    ///< [n_embd, item]
    const uint16_t* comp_gate = nullptr;  ///< [n_embd, item]
    const float* ape = nullptr;           ///< [ratio][item]
    const float* comp_norm = nullptr;     ///< [head_dim]
    // indexer, null unless ratio == 4
    const uint16_t* idx_qb = nullptr;       ///< [q_lora, idx_q_heads * idx_key_dim]
    const uint16_t* idx_proj = nullptr;     ///< [n_embd, idx_q_heads]
    const uint16_t* idx_comp_kv = nullptr;  ///< [n_embd, 2 * idx_key_dim]
    const uint16_t* idx_comp_gate = nullptr;
    const float* idx_ape = nullptr;  ///< [4][2 * idx_key_dim]
    const float* idx_comp_norm = nullptr;
};

/// Workspace bytes for one decode step of `layer` with a staging width of `n_stage` cmp columns.
int64_t dsv4_attn_scratch_bytes(const ModelGeometry& g, int64_t layer, int64_t n_stage);

/// One decode token through the attention half.  `stream_in`/`stream_out` are [hc * n_embd] f32
/// (they may alias nothing else); `n_stage` bounds the compressed staging (the caller passes the
/// live count `(pos + 1) / ratio`).
void dsv4_attn_decode_step(const ModelGeometry& g, int64_t layer, const Dsv4AttnWeights& w,
                           const Dsv4LayerState& st, const float* stream_in, float* stream_out,
                           int64_t pos, int64_t n_stage, float* scratch, void* cu_stream);

}  // namespace strata::core
