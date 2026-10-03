// include/strata/core/dsv4_ffn.hpp - the DeepSeek-V4 FFN half of one decoder block, one decode
// token, docs/DSV4.md P2 integration.
//
// The transcription of `model.py::Block.decode_step` (ffn half) + `moe.py::MoE.forward` for B=1,
// from the hc=4 residual stream in to the stream out:
//
//   hc_pre(ffn) -> ffn_norm -> router (bf16 GEMV, then sqrtsoftplus+bias top-k, or the tid2eid
//   table for a hash layer) -> shared expert (bf16 GEMVs + clamped SwiGLU) -> the top-k routed
//   experts (q8_1 activation, the pack's native expert blobs through `iq_mmvq`, clamped SwiGLU,
//   q8_1 again, down) weighted-summed -> shared + routed -> hc_post(ffn).
//
// The shared expert is addressed as bf16 device pointers: the GGUF stores it quantized, and the
// session resolver dequantizes (or serves the same values through the dense path) - the GEMV math
// is value-identical either way.  The routed experts are the pack's own format: `expert_blobs` is
// `n_expert` contiguous blobs in `NativeExpertLayout` order, exactly as `experts.bin` stores them.
//
// `scratch` is caller-owned workspace sized by `dsv4_ffn_scratch_bytes`.  `token_id` selects the
// hash table row on a hash layer and is ignored otherwise.  Nothing here allocates.
#pragma once

#include "strata/core/layout.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cstdint>

namespace strata::core {

struct Dsv4FfnWeights {
    const float* hc_fn = nullptr;     ///< hc_ffn_fn [mix_hc][hc_dim] f32 (the GGUF stores it F32)
    const float* hc_base = nullptr;   ///< [mix_hc]
    const float* hc_scale = nullptr;  ///< [hc-1]
    const float* norm = nullptr;      ///< ffn_norm [n_embd]
    const uint16_t* gate = nullptr;    ///< ffn_gate_inp [n_expert][n_embd] bf16
    const float* probs_b = nullptr;    ///< exp_probs_b [n_expert] f32 (null on a hash layer)
    const int32_t* tid2eid = nullptr;  ///< [n_expert_used][vocab] I32 (null unless a hash layer)
    const uint16_t* sh_gate = nullptr;  ///< ffn_gate_shexp [n_ff][n_embd] bf16
    const uint16_t* sh_up = nullptr;    ///< ffn_up_shexp   [n_ff][n_embd] bf16
    const uint16_t* sh_down = nullptr;  ///< ffn_down_shexp [n_embd][n_ff]  bf16
    kernels::NativeExpertLayout experts;
    const uint8_t* expert_blobs = nullptr;  ///< n_expert * experts.bytes, contiguous
};

/// Workspace bytes for one decode step of `layer`.
int64_t dsv4_ffn_scratch_bytes(const ModelGeometry& g);

/// One decode token through the FFN half.  `stream_in`/`stream_out` are [hc * n_embd] f32.
void dsv4_ffn_decode_step(const ModelGeometry& g, int64_t layer, const Dsv4FfnWeights& w,
                          const float* stream_in, float* stream_out, int64_t token_id, float* scratch,
                          void* cu_stream);

}  // namespace strata::core
