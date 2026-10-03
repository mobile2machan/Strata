// src/core/dsv4_ffn.cpp - the DSv4 FFN half of one block, one decode token, docs/DSV4.md P2.
// See the header for the flow; the router, SwiGLU and expert GEMVs are the parity-tested kernels.
#include "strata/core/dsv4_ffn.hpp"

#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/dsv4_hc.hpp"
#include "strata/kernels/dsv4_moe.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <vector>

namespace strata::core {
namespace {

using namespace strata::kernels;

int64_t al(int64_t n) { return (n + 15) / 16 * 16; }
int64_t q8_1_bytes(int64_t n) { return n / 32 * 36; }  // the CUDA block_q8_1: half2 ds (d,sum) + qs[32]

struct Cursor {
    char* p;
    int64_t off = 0;
    bool counting;
    template <typename T>
    T* take(int64_t n) {
        const int64_t bytes = al(n * (int64_t) sizeof(T));
        void* r = counting ? nullptr : (void*) (p + off);
        off += bytes;
        return (T*) r;
    }
};

struct Scratch {
    float* mixes;   // [mix_hc]
    float* pre;     // [hc]
    float* post;    // [hc]
    float* comb;    // [hc * hc]
    float* y;       // [n_embd]
    float* logits;  // [n_expert]
    float* rw;      // [topk] router weights
    int32_t* ids;   // [topk]
    int64_t* tok;   // [1] the token id, for the hash table
    uint8_t* xq;    // q8_1 of y
    float* gu;      // [2 * n_ff] gate|up for one expert
    float* ff;      // [n_ff]
    uint8_t* fq;    // q8_1 of ff
    float* part;    // [n_embd] one expert's down output
    float* y2;      // [n_embd] shared + routed
};

Scratch carve(Cursor& c, const ModelGeometry& g) {
    const int64_t mix_hc = (2 + g.hc) * g.hc, k = g.dsv4.n_expert_used;
    Scratch s;
    s.mixes = c.take<float>(mix_hc);
    s.pre = c.take<float>(g.hc);
    s.post = c.take<float>(g.hc);
    s.comb = c.take<float>(g.hc * g.hc);
    s.y = c.take<float>(g.n_embd);
    s.logits = c.take<float>(g.n_expert);
    s.rw = c.take<float>(k);
    s.ids = c.take<int32_t>(k);
    s.tok = c.take<int64_t>(1);
    s.xq = c.take<uint8_t>(q8_1_bytes(g.n_embd));
    s.gu = c.take<float>(2 * g.n_ff);
    s.ff = c.take<float>(g.n_ff);
    s.fq = c.take<uint8_t>(q8_1_bytes(g.n_ff));
    s.part = c.take<float>(g.n_embd);
    s.y2 = c.take<float>(g.n_embd);
    return s;
}

}  // namespace

int64_t dsv4_ffn_scratch_bytes(const ModelGeometry& g) {
    Cursor c{nullptr, 0, true};
    carve(c, g);
    return c.off;
}

void dsv4_ffn_decode_step(const ModelGeometry& g, int64_t layer, const Dsv4FfnWeights& w,
                          const float* stream_in, float* stream_out, int64_t token_id, float* scratch,
                          void* cu) {
    const int64_t hc = g.hc, dim = g.n_embd, ff = g.n_ff, k = g.dsv4.n_expert_used;
    const int64_t mix_hc = (2 + hc) * g.hc;
    const float eps = g.dsv4.norm_eps;
    const bool hash = dsv4_is_hash_layer(g, layer);
    Cursor cur{(char*) scratch, 0, false};
    Scratch s = carve(cur, g);

    // ---- hc_pre(ffn): mixes GEMV + RMS factor, split/Sinkhorn, collapse the streams.
    kernels::hc_mixes(stream_in, w.hc_fn, s.mixes, 1, hc * dim, mix_hc, g.dsv4.hc_eps, cu);
    kernels::hc_split_sinkhorn(s.mixes, w.hc_scale, w.hc_base, 1, hc, g.dsv4.sinkhorn_iters,
                               g.dsv4.hc_eps, s.pre, s.post, s.comb, cu);
    kernels::hc_pre_combine(stream_in, s.pre, s.y, 1, hc, dim, cu);

    // ---- ffn_norm, then the router.  The hash layers carry the router too - only the INDICES
    // differ (the tid2eid row), the weights still come from the same sqrtsoftplus scores.
    kernels::rms_norm_weighted(s.y, w.norm, 1, dim, eps, cu);
    kernels::bf16_gemv_fp32_mmvf(s.y, w.gate, s.logits, dim, g.n_expert, cu);
    if (hash) {
        cudaMemcpyAsync(s.tok, &token_id, 8, cudaMemcpyHostToDevice, (cudaStream_t) cu);
        kernels::dsv4_router_hash(s.logits, w.tid2eid, s.tok, s.rw, s.ids, 1, g.n_expert, k,
                                  g.dsv4.route_scale, cu);
    } else {
        kernels::dsv4_router_score(s.logits, w.probs_b, s.rw, s.ids, 1, g.n_expert, k,
                                  g.dsv4.route_scale, cu);
    }
    // The picks choose HOST-side blob addresses and the weights scale each expert's output -
    // bring the k picks and k weights back.  The kernel wrappers already synced the stream.
    static thread_local std::vector<int32_t> h_ids;
    static thread_local std::vector<float> h_rw;
    h_ids.resize((size_t) k);
    h_rw.resize((size_t) k);
    cudaMemcpy(h_ids.data(), s.ids, (size_t) k * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rw.data(), s.rw, (size_t) k * 4, cudaMemcpyDeviceToHost);

    // ---- the shared expert: bf16 GEMVs and the clamped SwiGLU, weight 1.
    kernels::bf16_gemv_fp32_mmvf(s.y, w.sh_gate, s.gu, dim, ff, cu);
    kernels::bf16_gemv_fp32_mmvf(s.y, w.sh_up, s.gu + ff, dim, ff, cu);
    kernels::dsv4_swiglu(s.gu, s.gu + ff, s.ff, ff, g.dsv4.swiglu_clamp_shexp[(size_t) layer], cu);
    kernels::bf16_gemv_fp32_mmvf(s.ff, w.sh_down, s.y2, ff, dim, cu);

    // ---- the routed experts: the pack's native blobs through iq_mmvq, q8_1 activations both
    // sides, the same clamped SwiGLU, then the router's weights onto each down output.
    kernels::quantize_q8_1_rows(s.y, 1, dim, s.xq, cu);
    const float gu_limit = g.dsv4.swiglu_clamp_exp[(size_t) layer];
    for (int64_t e = 0; e < k; ++e) {
        const uint8_t* blob = w.expert_blobs + (int64_t) h_ids[(size_t) e] * w.experts.bytes;
        kernels::iq_mmvq(w.experts.gu_type, blob, s.xq, s.gu, (int) dim, (int) ff, 1, cu);
        kernels::iq_mmvq(w.experts.gu_type, blob + w.experts.up_off, s.xq, s.gu + ff, (int) dim,
                         (int) ff, 1, cu);
        kernels::dsv4_swiglu(s.gu, s.gu + ff, s.ff, ff, gu_limit, cu);
        kernels::quantize_q8_1_rows(s.ff, 1, ff, s.fq, cu);
        kernels::iq_mmvq(w.experts.d_type, blob + w.experts.down_off, s.fq, s.part, (int) ff,
                         (int) dim, 1, cu);
        kernels::scale_inplace(s.part, dim, h_rw[(size_t) e], cu);
        kernels::add_inplace(s.y2, s.part, dim, cu);
    }

    // ---- hc_post(ffn).
    kernels::hc_post_combine(s.y2, stream_in, s.post, s.comb, stream_out, 1, hc, dim, cu);
}

}  // namespace strata::core
