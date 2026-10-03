// include/strata/kernels/dsv4_moe.hpp - DeepSeek-V4 MoE router and clamped SwiGLU, docs/DSV4.md P2.
//
// The router (`moe.py::Gate`; measured config: 256 experts, top 6, sqrtsoftplus, route_scale 1.5,
// renormalize true).  `raw` is the plain bf16 GEMV over `ffn_gate_inp.weight` (the caller runs the
// existing `bf16_gemv`); everything after it is here:
//
//   scores   = sqrt(softplus(raw))            // softplus: x > 20 ? x : log1p(exp(x))
//   original = scores                         // the weights are gathered from the PRE-bias scores
//   scores  += exp_probs_b                    // non-hash layers only - the three hash layers have no bias
//   indices  = topk(scores, 6)                // ties: lowest index first (torch's tie order is unspecified;
//                                             // exact fp32 ties are measure-zero and only permute equal weights)
//   weights  = original[indices]
//   weights /= weights.sum(); weights *= route_scale
//
// A hash layer (0..hash_layers-1) instead reads `indices = tid2eid[token]` - the I32 table is GGML
// [used, vocab], so a token's `used` entries are contiguous at `token * used` - and gathers its weights
// from the same sqrtsoftplus scores (no bias exists to add).
//
// SwiGLU (`swiglu.py::fused_swiglu`, the shared expert's activation):
//   h = silu(min(gate, limit)) * clamp(up, -limit, limit)
// in fp32; `limit` is the layer's own `swiglu_clamp_*` value (measured: 10.0), and limit <= 0 means
// no clamp.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// Score-routed layer: outputs `weights [n][topk]` fp32 and `indices [n][topk]` int32.
void dsv4_router_score(const float* raw, const float* bias, float* weights, int32_t* indices, int64_t n,
                       int64_t n_expert, int64_t topk, float route_scale, void* stream);

/// Hash layer: `tokens` are token ids, `tid2eid` the I32 table ([used, vocab] GGML order).
void dsv4_router_hash(const float* raw, const int32_t* tid2eid, const int64_t* tokens, float* weights,
                      int32_t* indices, int64_t n, int64_t n_expert, int64_t topk, float route_scale,
                      void* stream);

/// `out[i] = silu(min(gate[i], limit)) * clamp(up[i], -limit, limit)`, elementwise over n.
void dsv4_swiglu(const float* gate, const float* up, float* out, int64_t n, float limit, void* stream);

}  // namespace strata::kernels
