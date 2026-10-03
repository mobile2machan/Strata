// src/core/dsv4_ffn.cpp - the DSv4 FFN half of one block, one decode token, docs/DSV4.md P2.
// See the header for the flow; the router, SwiGLU and expert GEMVs are the parity-tested kernels.
#include "strata/core/dsv4_ffn.hpp"

#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/dsv4_hc.hpp"
#include "strata/kernels/dsv4_moe.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
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

bool dsv4_ffn_decode_step(const ModelGeometry& g, int64_t layer, const Dsv4FfnWeights& w,
                          const float* stream_in, float* stream_out, int64_t token_id, float* scratch,
                          void* cu, Dsv4ExpertSource* src) {
    const int64_t hc = g.hc, dim = g.n_embd, ff = g.n_ff, k = g.dsv4.n_expert_used;
    const int64_t mix_hc = (2 + hc) * g.hc;
    const float eps = g.dsv4.norm_eps;
    const bool hash = dsv4_is_hash_layer(g, layer);
    Cursor cur{(char*) scratch, 0, false};
    Scratch s = carve(cur, g);

    const bool dbg = std::getenv("DSV4_DBG") != nullptr && layer == 3;
    auto peek = [&](const char* what, const float* d, int64_t n) {
        if (!dbg) return;
        std::vector<float> h((size_t) n);
        cudaMemcpy(h.data(), d, (size_t) n * 4, cudaMemcpyDeviceToHost);
        float mx = 0; int64_t bad = 0;
        for (float v : h) { if (!std::isfinite(v)) ++bad; else mx = std::max(mx, std::fabs(v)); }
        std::printf("[ffn3] %-10s max %.3e nonfinite %lld\n", what, mx, (long long) bad);
    };

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
    if (dbg) {
        peek("y", s.y, dim);
        peek("logits", s.logits, g.n_expert);
        std::printf("[ffn3] picks:");
        for (int64_t e = 0; e < k; ++e) std::printf(" %d*%.3f", h_ids[(size_t) e], h_rw[(size_t) e]);
        std::printf("\n");
    }

    // ---- the shared expert: native GGUF GEMVs on the q8_1 activation (the same block format the
    // routed path uses, quantized once), and the clamped SwiGLU, weight 1.
    kernels::quantize_q8_1_rows(s.y, 1, dim, s.xq, cu);
    kernels::native_mmvq(w.sh_gate_type, w.sh_gate, s.xq, s.gu, (int) dim, (int) ff, 1, cu);
    kernels::native_mmvq(w.sh_up_type, w.sh_up, s.xq, s.gu + ff, (int) dim, (int) ff, 1, cu);
    kernels::dsv4_swiglu(s.gu, s.gu + ff, s.ff, ff, g.dsv4.swiglu_clamp_shexp[(size_t) layer], cu);
    kernels::quantize_q8_1_rows(s.ff, 1, ff, s.fq, cu);
    kernels::native_mmvq(w.sh_down_type, w.sh_down, s.fq, s.y2, (int) ff, (int) dim, 1, cu);
    peek("y2_shared", s.y2, dim);

    // ---- the routed experts: the pack's native blobs through iq_mmvq, q8_1 activations both
    // sides, the same clamped SwiGLU, then the router's weights onto each down output.  With a
    // source the picked blobs were just staged (pick order); without one the caller already
    // supplied the whole region and the ids index it directly.
    const float gu_limit = g.dsv4.swiglu_clamp_exp[(size_t) layer];
    const uint8_t* base = w.expert_blobs;
    bool staged = false;
    if (src != nullptr) {
        if (!src->stage(h_ids.data(), k, w.experts.bytes, &base)) return false;
        staged = true;
    }
    for (int64_t e = 0; e < k; ++e) {
        const uint8_t* blob = base + (staged ? e : (int64_t) h_ids[(size_t) e]) * w.experts.bytes;
        kernels::iq_mmvq(w.experts.gu_type, blob, s.xq, s.gu, (int) dim, (int) ff, 1, cu);
        kernels::iq_mmvq(w.experts.gu_type, blob + w.experts.up_off, s.xq, s.gu + ff, (int) dim,
                         (int) ff, 1, cu);
        if (dbg && e == 0) { peek("gu", s.gu, 2 * ff); std::printf("[ffn3] blob_off %lld id %d bytes %lld\n", (long long) (blob - w.expert_blobs), h_ids[(size_t) e], (long long) w.experts.bytes); }
        kernels::dsv4_swiglu(s.gu, s.gu + ff, s.ff, ff, gu_limit, cu);
        kernels::quantize_q8_1_rows(s.ff, 1, ff, s.fq, cu);
        kernels::iq_mmvq(w.experts.d_type, blob + w.experts.down_off, s.fq, s.part, (int) ff,
                         (int) dim, 1, cu);
        kernels::scale_inplace(s.part, dim, h_rw[(size_t) e], cu);
        kernels::add_inplace(s.y2, s.part, dim, cu);
        if (dbg) { peek("part", s.part, dim); peek("y2", s.y2, dim); }
    }

    // ---- hc_post(ffn).
    kernels::hc_post_combine(s.y2, stream_in, s.post, s.comb, stream_out, 1, hc, dim, cu);
    peek("out", stream_out, hc * dim);
    return true;
}

namespace {

struct PScratch {
    float* mixes;   // [n][mix_hc]
    float* pre;     // [n][hc]
    float* post;    // [n][hc]
    float* comb;    // [n][hc][hc]
    float* y;       // [n][dim]
    float* logits;  // [n][n_expert]
    float* rw;      // [n][topk]
    int32_t* ids;   // [n][topk]
    int64_t* tok;   // [n]
    uint8_t* xq;    // n q8_1 columns of dim
    float* gg;      // [n][n_ff] gate
    float* gu;      // [n][n_ff] up
    float* ff;      // [n][n_ff]
    uint8_t* fq;    // n q8_1 columns of n_ff
    float* part;    // [dim] one (token, expert) down output
    float* y2;      // [n][dim]
};

PScratch pcarve(Cursor& c, const ModelGeometry& g, int64_t n) {
    const int64_t mix_hc = (2 + g.hc) * g.hc, k = g.dsv4.n_expert_used;
    PScratch s;
    s.mixes = c.take<float>(n * mix_hc);
    s.pre = c.take<float>(n * g.hc);
    s.post = c.take<float>(n * g.hc);
    s.comb = c.take<float>(n * g.hc * g.hc);
    s.y = c.take<float>(n * g.n_embd);
    s.logits = c.take<float>(n * g.n_expert);
    s.rw = c.take<float>(n * k);
    s.ids = c.take<int32_t>(n * k);
    s.tok = c.take<int64_t>(n);
    s.xq = c.take<uint8_t>(n * q8_1_bytes(g.n_embd));
    s.gg = c.take<float>(n * g.n_ff);
    s.gu = c.take<float>(n * g.n_ff);
    s.ff = c.take<float>(n * g.n_ff);
    s.fq = c.take<uint8_t>(n * q8_1_bytes(g.n_ff));
    s.part = c.take<float>(g.n_embd);
    s.y2 = c.take<float>(n * g.n_embd);
    return s;
}

}  // namespace

int64_t dsv4_ffn_prefill_scratch_bytes(const ModelGeometry& g, int64_t n) {
    Cursor c{nullptr, 0, true};
    pcarve(c, g, n);
    return c.off;
}

bool dsv4_ffn_prefill_step(const ModelGeometry& g, int64_t layer, const Dsv4FfnWeights& w,
                           const float* stream_in, float* stream_out, const int64_t* token_ids,
                           int64_t n, float* scratch, void* cu, Dsv4ExpertSource* src) {
    if (n < 1 || n > 8) return false;  // the native GEMV column limit
    const int64_t hc = g.hc, dim = g.n_embd, ff = g.n_ff, k = g.dsv4.n_expert_used;
    const int64_t mix_hc = (2 + hc) * g.hc;
    const float eps = g.dsv4.norm_eps;
    const bool hash = dsv4_is_hash_layer(g, layer);
    Cursor cur{(char*) scratch, 0, false};
    PScratch s = pcarve(cur, g, n);

    kernels::hc_mixes(stream_in, w.hc_fn, s.mixes, n, hc * dim, mix_hc, g.dsv4.hc_eps, cu);
    kernels::hc_split_sinkhorn(s.mixes, w.hc_scale, w.hc_base, n, hc, g.dsv4.sinkhorn_iters,
                               g.dsv4.hc_eps, s.pre, s.post, s.comb, cu);
    kernels::hc_pre_combine(stream_in, s.pre, s.y, n, hc, dim, cu);
    kernels::rms_norm_weighted(s.y, w.norm, n, dim, eps, cu);
    kernels::bf16_gemv_fp32_mmvf_multi(s.y, dim, w.gate, s.logits, g.n_expert, dim, g.n_expert,
                                       (int) n, cu);
    if (hash) {
        cudaMemcpyAsync(s.tok, token_ids, (size_t) n * 8, cudaMemcpyHostToDevice, (cudaStream_t) cu);
        kernels::dsv4_router_hash(s.logits, w.tid2eid, s.tok, s.rw, s.ids, n, g.n_expert, k,
                                 g.dsv4.route_scale, cu);
    } else {
        kernels::dsv4_router_score(s.logits, w.probs_b, s.rw, s.ids, n, g.n_expert, k,
                                  g.dsv4.route_scale, cu);
    }
    static thread_local std::vector<int32_t> h_ids;
    static thread_local std::vector<float> h_rw;
    h_ids.resize((size_t) n * k);
    h_rw.resize((size_t) n * k);
    cudaMemcpy(h_ids.data(), s.ids, (size_t) n * k * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rw.data(), s.rw, (size_t) n * k * 4, cudaMemcpyDeviceToHost);

    // shared expert: one n-column quantization, two n-column GEMVs (output column t lands at
    // y + t * n_out, so the gate and up buffers are separate [n][n_ff] slabs).
    kernels::quantize_q8_1_rows(s.y, n, dim, s.xq, cu);
    kernels::native_mmvq(w.sh_gate_type, w.sh_gate, s.xq, s.gg, (int) dim, (int) ff, (int) n, cu);
    kernels::native_mmvq(w.sh_up_type, w.sh_up, s.xq, s.gu, (int) dim, (int) ff, (int) n, cu);
    kernels::dsv4_swiglu(s.gg, s.gu, s.ff, n * ff, g.dsv4.swiglu_clamp_shexp[(size_t) layer], cu);
    kernels::quantize_q8_1_rows(s.ff, n, ff, s.fq, cu);
    kernels::native_mmvq(w.sh_down_type, w.sh_down, s.fq, s.y2, (int) ff, (int) dim, (int) n, cu);

    const float gu_limit = g.dsv4.swiglu_clamp_exp[(size_t) layer];
    const uint8_t* base = w.expert_blobs;
    bool staged = false;
    if (src != nullptr) {
        if (!src->stage(h_ids.data(), n * k, w.experts.bytes, &base)) return false;
        staged = true;
    }
    const int64_t xq_col = q8_1_bytes(dim), fq_col = q8_1_bytes(ff);
    for (int64_t t = 0; t < n; ++t) {
        for (int64_t e = 0; e < k; ++e) {
            const int64_t p = t * k + e;
            const uint8_t* blob = base + (staged ? p : (int64_t) h_ids[(size_t) p]) * w.experts.bytes;
            kernels::iq_mmvq(w.experts.gu_type, blob, s.xq + t * xq_col, s.gg + t * ff, (int) dim,
                             (int) ff, 1, cu);
            kernels::iq_mmvq(w.experts.gu_type, blob + w.experts.up_off, s.xq + t * xq_col,
                             s.gu + t * ff, (int) dim, (int) ff, 1, cu);
            kernels::dsv4_swiglu(s.gg + t * ff, s.gu + t * ff, s.ff + t * ff, ff, gu_limit, cu);
            kernels::quantize_q8_1_rows(s.ff + t * ff, 1, ff, s.fq + t * fq_col, cu);
            kernels::iq_mmvq(w.experts.d_type, blob + w.experts.down_off, s.fq + t * fq_col, s.part,
                             (int) ff, (int) dim, 1, cu);
            kernels::scale_inplace(s.part, dim, h_rw[(size_t) p], cu);
            kernels::add_inplace(s.y2 + t * dim, s.part, dim, cu);
        }
    }

    kernels::hc_post_combine(s.y2, stream_in, s.post, s.comb, stream_out, n, hc, dim, cu);
    return true;
}

}  // namespace strata::core
