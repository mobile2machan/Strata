// src/core/dsv4_attn.cpp - the DSv4 attention half of one block, one decode token, docs/DSV4.md P2.
// See the header for the flow; the math itself lives in the parity-tested kernels.
#include "strata/core/dsv4_attn.hpp"

#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/compressor.hpp"
#include "strata/kernels/dsv4_attn.hpp"
#include "strata/kernels/dsv4_hc.hpp"
#include "strata/kernels/dsv4_indexer.hpp"
#include "strata/kernels/dsv4_quant.hpp"
#include "strata/kernels/dsv4_rope.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::core {
namespace {

using namespace strata::kernels;

int64_t al(int64_t n) { return (n + 15) / 16 * 16; }

/// A cursor that either hands out real pointers or just counts bytes (for the size query).
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
    float* mixes;      // [mix_hc]
    float* pre;        // [hc]
    float* post;       // [hc]
    float* comb;       // [hc * hc]
    float* y;          // [n_embd]
    float* qr;         // [q_lora]
    float* q;          // [n_head * head_dim]
    float* kv;         // [head_dim]
    uint16_t* kv_bf;   // [head_dim]
    float* ckv;        // [2 * head_dim]
    float* csc;        // [2 * head_dim]
    float* cmp_f;      // [head_dim]
    uint16_t* cmp_bf;  // [head_dim]
    float* iq;         // [idx_q_heads * idx_key_dim]
    float* iq_h;       // (hadamard out)
    uint16_t* iq_bf;
    float* iq_w;       // [idx_q_heads]
    float* ikv;        // [2 * idx_key_dim]
    float* isc;        // [2 * idx_key_dim]
    float* icmp_f;     // [idx_key_dim]
    float* icmp_h;     // (hadamard out)
    uint16_t* icmp_bf;
    float* logits;     // [n_stage]
    int32_t* ids;      // [n_stage] (indexer pool ids)
    int32_t* win_ids;  // [window]
    int32_t* cmp_ids;  // [max(n_stage, idx_topk)]
    float* o;          // [n_head * head_dim]
    float* wo_tmp;     // [o_groups * o_lora]
    uint8_t* xq_dim;   // q8_1 of an [n_embd] (or wo_a group-slice) activation
    uint8_t* xq_qr;    // q8_1 of a [q_lora] activation
    uint8_t* xq_ob;    // q8_1 of an [o_groups * o_lora] activation
};

Scratch carve(Cursor& c, const ModelGeometry& g, int64_t n_stage) {
    const int64_t mix_hc = (2 + g.hc) * g.hc;
    Scratch s;
    s.mixes = c.take<float>(mix_hc);
    s.pre = c.take<float>(g.hc);
    s.post = c.take<float>(g.hc);
    s.comb = c.take<float>(g.hc * g.hc);
    s.y = c.take<float>(g.n_embd);
    s.qr = c.take<float>(g.dsv4.q_lora);
    s.q = c.take<float>(g.n_head * g.head_dim);
    s.kv = c.take<float>(g.head_dim);
    s.kv_bf = c.take<uint16_t>(g.head_dim);
    s.ckv = c.take<float>(2 * g.head_dim);
    s.csc = c.take<float>(2 * g.head_dim);
    s.cmp_f = c.take<float>(g.head_dim);
    s.cmp_bf = c.take<uint16_t>(g.head_dim);
    s.iq = c.take<float>(g.idx_q_heads * g.idx_key_dim);
    s.iq_h = c.take<float>(g.idx_q_heads * g.idx_key_dim);
    s.iq_bf = c.take<uint16_t>(g.idx_q_heads * g.idx_key_dim);
    s.iq_w = c.take<float>(g.idx_q_heads);
    s.ikv = c.take<float>(2 * g.idx_key_dim);
    s.isc = c.take<float>(2 * g.idx_key_dim);
    s.icmp_f = c.take<float>(g.idx_key_dim);
    s.icmp_h = c.take<float>(g.idx_key_dim);
    s.icmp_bf = c.take<uint16_t>(g.idx_key_dim);
    s.logits = c.take<float>(n_stage);
    s.ids = c.take<int32_t>(n_stage);
    s.win_ids = c.take<int32_t>(g.dsv4.sliding_window);
    s.cmp_ids = c.take<int32_t>(n_stage > g.dsv4.idx_topk ? n_stage : g.dsv4.idx_topk);
    s.o = c.take<float>(g.n_head * g.head_dim);
    s.wo_tmp = c.take<float>(g.dsv4.o_groups * g.dsv4.o_lora);
    const int64_t hpg_in = (g.n_head / g.dsv4.o_groups) * g.head_dim;
    s.xq_dim = c.take<uint8_t>((int64_t) native_q8_1_bytes((int) (g.n_embd > hpg_in ? g.n_embd : hpg_in), 1));
    s.xq_qr = c.take<uint8_t>((int64_t) native_q8_1_bytes((int) g.dsv4.q_lora, 1));
    s.xq_ob = c.take<uint8_t>((int64_t) native_q8_1_bytes((int) (g.dsv4.o_groups * g.dsv4.o_lora), 1));
    return s;
}

}  // namespace

int64_t dsv4_attn_scratch_bytes(const ModelGeometry& g, int64_t layer, int64_t n_stage) {
    (void) layer;
    Cursor c{nullptr, 0, true};
    carve(c, g, n_stage);
    return c.off;
}

namespace {

struct PScratch {
    float* mixes;   // [n][mix_hc]
    float* pre;     // [n][hc]
    float* post;    // [n][hc]
    float* comb;    // [n][hc][hc]
    float* y;       // [n][dim]
    float* qr;      // [n][q_lora]
    float* q;       // [n][heads*hd]
    float* kv;      // [n][hd]
    uint16_t* kv_bf;
    float* ckv;     // [n][item]
    float* csc;
    float* cmp_f;   // [n][hd]
    uint16_t* cmp_bf;
    float* iq;      // [n][iqh*ikd]
    float* iq_h;
    uint16_t* iq_bf;
    float* iq_w;    // [n][iqh]
    float* ikv;     // [n][2*ikd]
    float* isc;
    float* icmp_f;  // [n][ikd]
    float* icmp_h;
    uint16_t* icmp_bf;
    float* logits;  // [n][n_stage]
    int32_t* ids;   // [n][n_stage]
    int32_t* win_ids;  // [n][window]
    int32_t* cmp_ids;  // [n][max(n_stage, topk)]
    float* o;       // [n][heads*hd]
    float* wo_tmp;  // [n][o_groups*o_lora]
    uint8_t* xq_dim;
    uint8_t* xq_qr;
    uint8_t* xq_ob;
};

PScratch pcarve(Cursor& c, const ModelGeometry& g, int64_t n, int64_t n_stage) {
    const int64_t mix_hc = (2 + g.hc) * g.hc, hd = g.head_dim, ikd = g.idx_key_dim;
    const int64_t item = 2 * hd, iitem = 2 * ikd;
    PScratch s;
    s.mixes = c.take<float>(n * mix_hc);
    s.pre = c.take<float>(n * g.hc);
    s.post = c.take<float>(n * g.hc);
    s.comb = c.take<float>(n * g.hc * g.hc);
    s.y = c.take<float>(n * g.n_embd);
    s.qr = c.take<float>(n * g.dsv4.q_lora);
    s.q = c.take<float>(n * g.n_head * hd);
    s.kv = c.take<float>(n * hd);
    s.kv_bf = c.take<uint16_t>(n * hd);
    s.ckv = c.take<float>(n * item);
    s.csc = c.take<float>(n * item);
    s.cmp_f = c.take<float>(n * hd);
    s.cmp_bf = c.take<uint16_t>(n * hd);
    s.iq = c.take<float>(n * g.idx_q_heads * ikd);
    s.iq_h = c.take<float>(n * g.idx_q_heads * ikd);
    s.iq_bf = c.take<uint16_t>(n * g.idx_q_heads * ikd);
    s.iq_w = c.take<float>(n * g.idx_q_heads);
    s.ikv = c.take<float>(n * iitem);
    s.isc = c.take<float>(n * iitem);
    s.icmp_f = c.take<float>(n * ikd);
    s.icmp_h = c.take<float>(n * ikd);
    s.icmp_bf = c.take<uint16_t>(n * ikd);
    s.logits = c.take<float>(n * n_stage);
    s.ids = c.take<int32_t>(n * n_stage);
    s.win_ids = c.take<int32_t>(n * g.dsv4.sliding_window);
    s.cmp_ids = c.take<int32_t>(n * (n_stage > g.dsv4.idx_topk ? n_stage : g.dsv4.idx_topk));
    s.o = c.take<float>(n * g.n_head * hd);
    s.wo_tmp = c.take<float>(n * g.dsv4.o_groups * g.dsv4.o_lora);
    const int64_t hpg_in = (g.n_head / g.dsv4.o_groups) * hd;
    const int64_t wide = g.n_embd > hpg_in ? g.n_embd : hpg_in;
    s.xq_dim = c.take<uint8_t>((int64_t) n * native_q8_1_bytes((int) wide, 1));
    s.xq_qr = c.take<uint8_t>((int64_t) n * native_q8_1_bytes((int) g.dsv4.q_lora, 1));
    s.xq_ob = c.take<uint8_t>((int64_t) n * native_q8_1_bytes((int) (g.dsv4.o_groups * g.dsv4.o_lora), 1));
    return s;
}

}  // namespace

int64_t dsv4_attn_prefill_scratch_bytes(const ModelGeometry& g, int64_t layer, int64_t n,
                                        int64_t n_stage) {
    (void) layer;
    Cursor c{nullptr, 0, true};
    pcarve(c, g, n, n_stage);
    return c.off;
}

bool dsv4_attn_prefill_step(const ModelGeometry& g, int64_t layer, const Dsv4AttnWeights& w,
                            Dsv4LayerState& st, const float* stream_in, float* stream_out,
                            int64_t pos0, int64_t n, int64_t n_stage, float* scratch, void* cu,
                            float* carry_snap, int64_t carry_stride) {
    if (n < 1 || n > 8) return false;
    const int64_t hc = g.hc, dim = g.n_embd, hd = g.head_dim, rd = g.dsv4.rope_dim;
    const int64_t ratio = g.dsv4.compress_ratios[(size_t) layer];
    const bool compressed = ratio > 0, indexed = ratio == 4;
    const int64_t mix_hc = (2 + hc) * g.hc;
    const float eps = g.dsv4.norm_eps;
    // The per-position carry snapshot's row layout matches `Dsv4State`'s arena span exactly -
    // [align64(ks+ss bytes)][align64(iks+iss bytes)] - so a snapshot row is a `carry_restore` source.
    const Dsv4PoolSizes psz = dsv4_pool_sizes(g, layer, 1);
    const int64_t snap_main = (psz.carry_floats * 4 + 63) / 64 * 64 / 4;
    const int64_t snap_row = carry_stride > 0 ? carry_stride
                                              : snap_main + (psz.idx_carry_floats * 4 + 63) / 64 * 64 / 4;
    Cursor cur{(char*) scratch, 0, false};
    PScratch s = pcarve(cur, g, n, n_stage);

    auto gemm = [&](const float* x, int type, const uint8_t* wgt, float* y, int64_t n_in,
                    int64_t n_out, uint8_t* xq) {
        kernels::quantize_q8_1_rows(x, n, n_in, xq, cu);
        kernels::native_mmvq(type, wgt, xq, y, (int) n_in, (int) n_out, (int) n, cu);
    };

    kernels::hc_mixes(stream_in, w.hc_fn, s.mixes, n, hc * dim, mix_hc, g.dsv4.hc_eps, cu);
    kernels::hc_split_sinkhorn(s.mixes, w.hc_scale, w.hc_base, n, hc, g.dsv4.sinkhorn_iters,
                               g.dsv4.hc_eps, s.pre, s.post, s.comb, cu);
    kernels::hc_pre_combine(stream_in, s.pre, s.y, n, hc, dim, cu);

    kernels::rms_norm_weighted(s.y, w.norm, n, dim, eps, cu);
    gemm(s.y, w.q_a_type, w.q_a, s.qr, dim, g.dsv4.q_lora, s.xq_dim);
    kernels::rms_norm_weighted(s.qr, w.q_a_norm, n, g.dsv4.q_lora, eps, cu);
    gemm(s.qr, w.q_b_type, w.q_b, s.q, g.dsv4.q_lora, g.n_head * hd, s.xq_qr);
    kernels::rms_norm_weighted(s.q, nullptr, n * g.n_head, hd, eps, cu);
    const int64_t attn_orig = compressed ? g.dsv4.yarn_orig : 0;
    for (int64_t t = 0; t < n; ++t)
        kernels::dsv4_rope_f32(s.q + t * g.n_head * hd, g.n_head, hd, rd, pos0 + t,
                               g.dsv4.rope_theta, g.dsv4.yarn_factor, attn_orig,
                               g.dsv4.yarn_beta_fast, g.dsv4.yarn_beta_slow, false, cu);

    gemm(s.y, w.kv_type, w.kv, s.kv, dim, hd, s.xq_dim);
    kernels::rms_norm_weighted(s.kv, w.kv_norm, n, hd, eps, cu);
    for (int64_t t = 0; t < n; ++t) {
        const int64_t pos = pos0 + t;
        kernels::dsv4_rope_f32(s.kv + t * hd, 1, hd, rd, pos, g.dsv4.rope_theta, g.dsv4.yarn_factor,
                               attn_orig, g.dsv4.yarn_beta_fast, g.dsv4.yarn_beta_slow, false, cu);
        kernels::roundtrip_fp8_e4m3(s.kv + t * hd, s.kv_bf + t * hd, hd - rd, 64, cu);
        kernels::f32_to_bf16_bulk(s.kv + t * hd + (hd - rd), s.kv_bf + t * hd + (hd - rd), rd, cu);
        cudaMemcpyAsync(st.window + dsv4_window_slot(pos, g.dsv4.sliding_window) * hd,
                        s.kv_bf + t * hd, (size_t) hd * 2, cudaMemcpyDeviceToDevice,
                        (cudaStream_t) cu);
    }

    if (compressed) {
        const int64_t item = (indexed ? 2 : 1) * hd;
        gemm(s.y, w.comp_kv_type, w.comp_kv, s.ckv, dim, item, s.xq_dim);
        gemm(s.y, w.comp_gate_type, w.comp_gate, s.csc, dim, item, s.xq_dim);
        for (int64_t t = 0; t < n; ++t) {
            const int64_t pos = pos0 + t;
            kernels::compressor_decode_step(pos, ratio, indexed, hd, s.ckv + t * item,
                                            s.csc + t * item, w.ape, st.ks, st.ss,
                                            s.cmp_f + t * hd, cu);
            if (carry_snap != nullptr)
                cudaMemcpyAsync(carry_snap + t * snap_row, st.ks, (size_t) psz.carry_floats * 4,
                                cudaMemcpyDeviceToDevice, (cudaStream_t) cu);
            if ((pos + 1) % ratio == 0) {
                kernels::rms_norm_weighted(s.cmp_f + t * hd, w.comp_norm, 1, hd, eps, cu);
                const int64_t bpos = pos + 1 - ratio > 0 ? pos + 1 - ratio : 0;
                kernels::dsv4_rope_f32(s.cmp_f + t * hd, 1, hd, rd, bpos,
                                       g.dsv4.compress_rope_theta, g.dsv4.yarn_factor,
                                       g.dsv4.yarn_orig, g.dsv4.yarn_beta_fast,
                                       g.dsv4.yarn_beta_slow, false, cu);
                kernels::roundtrip_fp8_e4m3(s.cmp_f + t * hd, s.cmp_bf + t * hd, hd - rd, 64, cu);
                kernels::f32_to_bf16_bulk(s.cmp_f + t * hd + (hd - rd), s.cmp_bf + t * hd + (hd - rd),
                                          rd, cu);
                cudaMemcpyAsync(st.cmp + dsv4_cmp_row(pos, ratio) * hd, s.cmp_bf + t * hd,
                                (size_t) hd * 2, cudaMemcpyDeviceToDevice, (cudaStream_t) cu);
            }
        }
    }

    int64_t n_cmp = 0;
    // the gather kernel reads cmp_ids at stride n_cmp, so that IS the stride.
    const int64_t cmp_stride = indexed ? g.dsv4.idx_topk : n_stage;
    if (indexed) {
        gemm(s.qr, w.idx_qb_type, w.idx_qb, s.iq, g.dsv4.q_lora,
             g.idx_q_heads * g.idx_key_dim, s.xq_qr);
        for (int64_t t = 0; t < n; ++t) {
            kernels::dsv4_rope_f32(s.iq + t * g.idx_q_heads * g.idx_key_dim, g.idx_q_heads,
                                   g.idx_key_dim, rd, pos0 + t, g.dsv4.compress_rope_theta,
                                   g.dsv4.yarn_factor, g.dsv4.yarn_orig, g.dsv4.yarn_beta_fast,
                                   g.dsv4.yarn_beta_slow, false, cu);
            kernels::hadamard(s.iq + t * g.idx_q_heads * g.idx_key_dim,
                              s.iq_h + t * g.idx_q_heads * g.idx_key_dim, g.idx_q_heads,
                              g.idx_key_dim, cu);
            kernels::roundtrip_fp4_e2m1(s.iq_h + t * g.idx_q_heads * g.idx_key_dim,
                                        s.iq_bf + t * g.idx_q_heads * g.idx_key_dim,
                                        g.idx_q_heads * g.idx_key_dim, 32, cu);
        }
        kernels::bf16_gemv_fp32_mmvf_multi(s.y, dim, w.idx_proj, s.iq_w, g.idx_q_heads, dim,
                                           g.idx_q_heads, (int) n, cu);
        const float idx_scale = (float)(1.0 / std::sqrt((double) g.idx_key_dim) /
                                        std::sqrt((double) g.idx_q_heads));
        kernels::scale_inplace(s.iq_w, n * g.idx_q_heads, idx_scale, cu);
        const int64_t iitem = 2 * g.idx_key_dim;
        gemm(s.y, w.idx_comp_kv_type, w.idx_comp_kv, s.ikv, dim, iitem, s.xq_dim);
        gemm(s.y, w.idx_comp_gate_type, w.idx_comp_gate, s.isc, dim, iitem, s.xq_dim);
        for (int64_t t = 0; t < n; ++t) {
            const int64_t pos = pos0 + t;
            kernels::compressor_decode_step(pos, 4, true, g.idx_key_dim, s.ikv + t * iitem,
                                            s.isc + t * iitem, w.idx_ape, st.iks, st.iss,
                                            s.icmp_f + t * g.idx_key_dim, cu);
            if (carry_snap != nullptr)
                cudaMemcpyAsync(carry_snap + t * snap_row + snap_main, st.iks,
                                (size_t) psz.idx_carry_floats * 4, cudaMemcpyDeviceToDevice,
                                (cudaStream_t) cu);
            if ((pos + 1) % 4 == 0) {
                kernels::rms_norm_weighted(s.icmp_f + t * g.idx_key_dim, w.idx_comp_norm, 1,
                                           g.idx_key_dim, eps, cu);
                const int64_t bpos = pos + 1 - 4 > 0 ? pos + 1 - 4 : 0;
                kernels::dsv4_rope_f32(s.icmp_f + t * g.idx_key_dim, 1, g.idx_key_dim, rd, bpos,
                                       g.dsv4.compress_rope_theta, g.dsv4.yarn_factor,
                                       g.dsv4.yarn_orig, g.dsv4.yarn_beta_fast,
                                       g.dsv4.yarn_beta_slow, false, cu);
                kernels::hadamard(s.icmp_f + t * g.idx_key_dim, s.icmp_h + t * g.idx_key_dim, 1,
                                  g.idx_key_dim, cu);
                kernels::roundtrip_fp4_e2m1(s.icmp_h + t * g.idx_key_dim,
                                            s.icmp_bf + t * g.idx_key_dim, g.idx_key_dim, 32, cu);
                cudaMemcpyAsync(st.idx + dsv4_cmp_row(pos, 4) * g.idx_key_dim,
                                s.icmp_bf + t * g.idx_key_dim, (size_t) g.idx_key_dim * 2,
                                cudaMemcpyDeviceToDevice, (cudaStream_t) cu);
            }
        }
        static thread_local std::vector<int32_t> h_ids;
        h_ids.resize((size_t) n * n_stage);
        for (int64_t t = 0; t < n; ++t)
            for (int64_t i = 0; i < n_stage; ++i)
                h_ids[(size_t) t * n_stage + i] = (int32_t) i;
        cudaMemcpyAsync(s.ids, h_ids.data(), (size_t) n * n_stage * 4, cudaMemcpyHostToDevice,
                        (cudaStream_t) cu);
        for (int64_t t = 0; t < n; ++t) {
            const int64_t pos = pos0 + t;
            const int64_t valid = dsv4_cmp_valid(pos, 4);
            kernels::dsv4_indexer_logits(s.iq_bf + t * g.idx_q_heads * g.idx_key_dim,
                                         s.iq_w + t * g.idx_q_heads, st.idx,
                                         s.ids + t * n_stage, n_stage, valid, g.idx_q_heads,
                                         g.idx_key_dim, s.logits + t * n_stage, cu);
            kernels::dsv4_indexer_select(s.logits + t * n_stage, n_stage, g.dsv4.idx_topk,
                                         s.cmp_ids + t * cmp_stride, cu);
        }
        n_cmp = g.dsv4.idx_topk;
    } else if (compressed) {
        static thread_local std::vector<int32_t> h_cmp;
        h_cmp.assign((size_t) n * cmp_stride, -1);
        for (int64_t t = 0; t < n; ++t) {
            const int64_t valid = dsv4_cmp_valid(pos0 + t, ratio);
            for (int64_t i = 0; i < valid && i < n_stage; ++i)
                h_cmp[(size_t) t * cmp_stride + i] = (int32_t) i;
        }
        cudaMemcpyAsync(s.cmp_ids, h_cmp.data(), (size_t) n * cmp_stride * 4, cudaMemcpyHostToDevice,
                        (cudaStream_t) cu);
        n_cmp = n_stage;
    }

    {
        static thread_local std::vector<int32_t> h_win;
        h_win.resize((size_t) n * g.dsv4.sliding_window);
        for (int64_t t = 0; t < n; ++t)
            dsv4_window_ids(pos0 + t, g.dsv4.sliding_window,
                            h_win.data() + (size_t) t * g.dsv4.sliding_window);
        cudaMemcpyAsync(s.win_ids, h_win.data(), (size_t) n * g.dsv4.sliding_window * 4,
                        cudaMemcpyHostToDevice, (cudaStream_t) cu);
    }
    kernels::Dsv4AttnPools pools{st.window, st.cmp};
    kernels::dsv4_attn_prefill(s.q, s.win_ids, g.dsv4.sliding_window,
                               compressed ? s.cmp_ids : nullptr, n_cmp, w.sinks,
                               (float)(1.0 / std::sqrt((double) hd)), pools, g.dsv4.sliding_window,
                               n, g.n_head, hd, s.o, cu);
    for (int64_t t = 0; t < n; ++t)
        kernels::dsv4_rope_f32(s.o + t * g.n_head * hd, g.n_head, hd, rd, pos0 + t,
                               g.dsv4.rope_theta, g.dsv4.yarn_factor, attn_orig,
                               g.dsv4.yarn_beta_fast, g.dsv4.yarn_beta_slow, true, cu);

    const int64_t hpg = g.n_head / g.dsv4.o_groups;
    const int64_t oa_row_bytes = (int64_t) native_mmvq_weight_bytes(w.out_a_type, (int) (hpg * hd), 1);
    // the group slice of token t is at o + t * heads * hd, not at t * hpg * hd - the rows are
    // strided, so this GEMV stays one token at a time (multi-exact makes it bitwise the same).
    for (int64_t gr = 0; gr < g.dsv4.o_groups; ++gr)
        for (int64_t t = 0; t < n; ++t) {
            kernels::quantize_q8_1_rows(s.o + t * g.n_head * hd + gr * hpg * hd, 1, hpg * hd,
                                        s.xq_dim, cu);
            kernels::native_mmvq(w.out_a_type, w.out_a + gr * g.dsv4.o_lora * oa_row_bytes,
                                 s.xq_dim, s.wo_tmp + t * g.dsv4.o_groups * g.dsv4.o_lora +
                                                 gr * g.dsv4.o_lora,
                                 (int) (hpg * hd), (int) g.dsv4.o_lora, 1, cu);
        }
    gemm(s.wo_tmp, w.out_b_type, w.out_b, s.y, g.dsv4.o_groups * g.dsv4.o_lora, dim, s.xq_ob);

    kernels::hc_post_combine(s.y, stream_in, s.post, s.comb, stream_out, n, hc, dim, cu);
    return true;
}

void dsv4_attn_decode_step(const ModelGeometry& g, int64_t layer, const Dsv4AttnWeights& w,
                           const Dsv4LayerState& st, const float* stream_in, float* stream_out,
                           int64_t pos, int64_t n_stage, float* scratch, void* cu) {
    const int64_t hc = g.hc, dim = g.n_embd, hd = g.head_dim, rd = g.dsv4.rope_dim;
    const int64_t ratio = g.dsv4.compress_ratios[(size_t) layer];
    const bool compressed = ratio > 0, indexed = ratio == 4;
    const int64_t mix_hc = (2 + hc) * g.hc;
    const float eps = g.dsv4.norm_eps;
    Cursor cur{(char*) scratch, 0, false};
    Scratch s = carve(cur, g, n_stage);

    // A native GEMV: quantize the activation to q8_1 (the same block format the FFN half and the
    // qwen dense path use), then the GGUF-block kernel for this tensor's type.  One q8_1 buffer
    // per input width, reused across the GEMVs that share it.
    auto gemm = [&](const float* x, int type, const uint8_t* wgt, float* y, int64_t n_in,
                    int64_t n_out, uint8_t* xq) {
        kernels::quantize_q8_1_rows(x, 1, n_in, xq, cu);
        kernels::native_mmvq(type, wgt, xq, y, (int) n_in, (int) n_out, 1, cu);
    };

    const bool dbg = std::getenv("DSV4_DBG") != nullptr && layer == 3;
    auto peek = [&](const char* what, const float* d, int64_t n) {
        if (!dbg) return;
        std::vector<float> h((size_t) n);
        cudaMemcpy(h.data(), d, (size_t) n * 4, cudaMemcpyDeviceToHost);
        float mx = 0; int64_t bad = 0;
        for (float v : h) { if (!std::isfinite(v)) ++bad; else mx = std::max(mx, std::fabs(v)); }
        std::printf("[attn3] %-10s max %.3e nonfinite %lld\n", what, mx, (long long) bad);
    };

    // ---- hc_pre(attn): mixes GEMV + RMS factor, split/Sinkhorn, collapse the streams.
    kernels::hc_mixes(stream_in, w.hc_fn, s.mixes, 1, hc * dim, mix_hc, g.dsv4.hc_eps, cu);
    kernels::hc_split_sinkhorn(s.mixes, w.hc_scale, w.hc_base, 1, hc, g.dsv4.sinkhorn_iters,
                               g.dsv4.hc_eps, s.pre, s.post, s.comb, cu);
    kernels::hc_pre_combine(stream_in, s.pre, s.y, 1, hc, dim, cu);

    // ---- attn_norm, then the LoRA-split Q.
    kernels::rms_norm_weighted(s.y, w.norm, 1, dim, eps, cu);
    peek("y", s.y, dim);
    gemm(s.y, w.q_a_type, w.q_a, s.qr, dim, g.dsv4.q_lora, s.xq_dim);
    kernels::rms_norm_weighted(s.qr, w.q_a_norm, 1, g.dsv4.q_lora, eps, cu);
    gemm(s.qr, w.q_b_type, w.q_b, s.q, g.dsv4.q_lora, g.n_head * hd, s.xq_qr);
    kernels::rms_norm_weighted(s.q, nullptr, g.n_head, hd, eps, cu);
    // attention rope: YaRN only on compressed layers (the reference's original_seq_len is 0 for
    // r=0 layers).
    const int64_t attn_orig = compressed ? g.dsv4.yarn_orig : 0;
    kernels::dsv4_rope_f32(s.q, g.n_head, hd, rd, pos, g.dsv4.rope_theta, g.dsv4.yarn_factor,
                           attn_orig, g.dsv4.yarn_beta_fast, g.dsv4.yarn_beta_slow, false, cu);
    peek("q", s.q, g.n_head * hd);

    // ---- kv: one head, norm, rope, e4m3 round-trip on the non-rope part, ring write.
    gemm(s.y, w.kv_type, w.kv, s.kv, dim, hd, s.xq_dim);
    kernels::rms_norm_weighted(s.kv, w.kv_norm, 1, hd, eps, cu);
    kernels::dsv4_rope_f32(s.kv, 1, hd, rd, pos, g.dsv4.rope_theta, g.dsv4.yarn_factor, attn_orig,
                           g.dsv4.yarn_beta_fast, g.dsv4.yarn_beta_slow, false, cu);
    kernels::roundtrip_fp8_e4m3(s.kv, s.kv_bf, hd - rd, 64, cu);
    kernels::f32_to_bf16_bulk(s.kv + (hd - rd), s.kv_bf + (hd - rd), rd, cu);
    cudaMemcpyAsync(st.window + dsv4_window_slot(pos, g.dsv4.sliding_window) * hd, s.kv_bf,
                    (size_t) hd * 2, cudaMemcpyDeviceToDevice, (cudaStream_t) cu);

    // ---- compressor decode + cmp pool write.
    if (compressed) {
        const int64_t item = (indexed ? 2 : 1) * hd;
        gemm(s.y, w.comp_kv_type, w.comp_kv, s.ckv, dim, item, s.xq_dim);
        gemm(s.y, w.comp_gate_type, w.comp_gate, s.csc, dim, item, s.xq_dim);
        peek("ckv", s.ckv, item);
        peek("csc", s.csc, item);
        kernels::compressor_decode_step(pos, ratio, indexed, hd, s.ckv, s.csc, w.ape, st.ks, st.ss,
                                        s.cmp_f, cu);
        if ((pos + 1) % ratio == 0) {
            kernels::rms_norm_weighted(s.cmp_f, w.comp_norm, 1, hd, eps, cu);
            const int64_t bpos = pos + 1 - ratio > 0 ? pos + 1 - ratio : 0;
            kernels::dsv4_rope_f32(s.cmp_f, 1, hd, rd, bpos, g.dsv4.compress_rope_theta,
                                   g.dsv4.yarn_factor, g.dsv4.yarn_orig, g.dsv4.yarn_beta_fast,
                                   g.dsv4.yarn_beta_slow, false, cu);
            kernels::roundtrip_fp8_e4m3(s.cmp_f, s.cmp_bf, hd - rd, 64, cu);
            kernels::f32_to_bf16_bulk(s.cmp_f + (hd - rd), s.cmp_bf + (hd - rd), rd, cu);
            cudaMemcpyAsync(st.cmp + dsv4_cmp_row(pos, ratio) * hd, s.cmp_bf, (size_t) hd * 2,
                            cudaMemcpyDeviceToDevice, (cudaStream_t) cu);
        }
    }

    // ---- compressed ids.
    const int64_t valid = dsv4_cmp_valid(pos, ratio > 0 ? ratio : 1);
    int64_t n_cmp = 0;
    if (indexed) {
        // indexer q: rope, hadamard, e2m1 round-trip; weights GEMV with the folded scale.
        gemm(s.qr, w.idx_qb_type, w.idx_qb, s.iq, g.dsv4.q_lora,
             g.idx_q_heads * g.idx_key_dim, s.xq_qr);
        kernels::dsv4_rope_f32(s.iq, g.idx_q_heads, g.idx_key_dim, rd, pos,
                               g.dsv4.compress_rope_theta, g.dsv4.yarn_factor, g.dsv4.yarn_orig,
                               g.dsv4.yarn_beta_fast, g.dsv4.yarn_beta_slow, false, cu);
        kernels::hadamard(s.iq, s.iq_h, g.idx_q_heads, g.idx_key_dim, cu);
        kernels::roundtrip_fp4_e2m1(s.iq_h, s.iq_bf, g.idx_q_heads * g.idx_key_dim, 32, cu);
        kernels::bf16_gemv_fp32_mmvf(s.y, w.idx_proj, s.iq_w, dim, g.idx_q_heads, cu);
        peek("iq", s.iq, g.idx_q_heads * g.idx_key_dim);
        peek("iq_w", s.iq_w, g.idx_q_heads);
        const float idx_scale = (float)(1.0 / std::sqrt((double) g.idx_key_dim) /
                                        std::sqrt((double) g.idx_q_heads));
        kernels::scale_inplace(s.iq_w, g.idx_q_heads, idx_scale, cu);
        // the indexer's own compressor, rotate=True: hadamard + e2m1 over the whole row.
        const int64_t iitem = 2 * g.idx_key_dim;
        gemm(s.y, w.idx_comp_kv_type, w.idx_comp_kv, s.ikv, dim, iitem, s.xq_dim);
        gemm(s.y, w.idx_comp_gate_type, w.idx_comp_gate, s.isc, dim, iitem, s.xq_dim);
        kernels::compressor_decode_step(pos, 4, true, g.idx_key_dim, s.ikv, s.isc, w.idx_ape, st.iks,
                                        st.iss, s.icmp_f, cu);
        if ((pos + 1) % 4 == 0) {
            kernels::rms_norm_weighted(s.icmp_f, w.idx_comp_norm, 1, g.idx_key_dim, eps, cu);
            const int64_t bpos = pos + 1 - 4 > 0 ? pos + 1 - 4 : 0;
            kernels::dsv4_rope_f32(s.icmp_f, 1, g.idx_key_dim, rd, bpos, g.dsv4.compress_rope_theta,
                                   g.dsv4.yarn_factor, g.dsv4.yarn_orig, g.dsv4.yarn_beta_fast,
                                   g.dsv4.yarn_beta_slow, false, cu);
            kernels::hadamard(s.icmp_f, s.icmp_h, 1, g.idx_key_dim, cu);
            kernels::roundtrip_fp4_e2m1(s.icmp_h, s.icmp_bf, g.idx_key_dim, 32, cu);
            cudaMemcpyAsync(st.idx + dsv4_cmp_row(pos, 4) * g.idx_key_dim, s.icmp_bf,
                            (size_t) g.idx_key_dim * 2, cudaMemcpyDeviceToDevice, (cudaStream_t) cu);
        }
        // ids [0..stage) and the window list are host-built arithmetic; stage them and copy.
        static thread_local std::vector<int32_t> h_ids;
        h_ids.resize((size_t) n_stage);
        for (int64_t t = 0; t < n_stage; ++t) h_ids[(size_t) t] = (int32_t) t;
        cudaMemcpyAsync(s.ids, h_ids.data(), (size_t) n_stage * 4, cudaMemcpyHostToDevice,
                        (cudaStream_t) cu);
        if (n_stage > 0) {
            kernels::dsv4_indexer_logits(s.iq_bf, s.iq_w, st.idx, s.ids, n_stage, valid, g.idx_q_heads,
                                         g.idx_key_dim, s.logits, cu);
            kernels::dsv4_indexer_select(s.logits, n_stage, g.dsv4.idx_topk, s.cmp_ids, cu);
        } else {
            // nothing scored yet: the whole pick list is padding.
            static thread_local std::vector<int32_t> h_pad;
            h_pad.assign((size_t) g.dsv4.idx_topk, -1);
            cudaMemcpyAsync(s.cmp_ids, h_pad.data(), (size_t) g.dsv4.idx_topk * 4,
                            cudaMemcpyHostToDevice, (cudaStream_t) cu);
        }
        n_cmp = g.dsv4.idx_topk;
    } else if (compressed) {
        static thread_local std::vector<int32_t> h_cmp;
        h_cmp.resize((size_t) n_stage);
        for (int64_t t = 0; t < n_stage; ++t) h_cmp[(size_t) t] = t < valid ? (int32_t) t : -1;
        cudaMemcpyAsync(s.cmp_ids, h_cmp.data(), (size_t) n_stage * 4, cudaMemcpyHostToDevice,
                        (cudaStream_t) cu);
        n_cmp = n_stage;
    }

    // ---- window ids, gather, inverse rope.
    {
        static thread_local std::vector<int32_t> h_win;
        h_win.resize((size_t) g.dsv4.sliding_window);
        dsv4_window_ids(pos, g.dsv4.sliding_window, h_win.data());
        cudaMemcpyAsync(s.win_ids, h_win.data(), (size_t) g.dsv4.sliding_window * 4,
                        cudaMemcpyHostToDevice, (cudaStream_t) cu);
    }
    kernels::Dsv4AttnPools pools{st.window, st.cmp};
    kernels::dsv4_attn_decode(s.q, s.win_ids, g.dsv4.sliding_window,
                             compressed ? s.cmp_ids : nullptr, n_cmp, w.sinks,
                             (float)(1.0 / std::sqrt((double) hd)), pools, g.dsv4.sliding_window,
                             g.n_head, hd, s.o, cu);
    kernels::dsv4_rope_f32(s.o, g.n_head, hd, rd, pos, g.dsv4.rope_theta, g.dsv4.yarn_factor,
                           attn_orig, g.dsv4.yarn_beta_fast, g.dsv4.yarn_beta_slow, true, cu);
    peek("o", s.o, g.n_head * hd);

    // ---- wo: the grouped einsum is one GEMV per group over the flattened wo_a, then wo_b.
    // The group offset is in BYTES: a quantized row is not `dim` bytes wide.
    const int64_t hpg = g.n_head / g.dsv4.o_groups;
    const int64_t oa_row_bytes = (int64_t) native_mmvq_weight_bytes(w.out_a_type, (int) (hpg * hd), 1);
    for (int64_t gr = 0; gr < g.dsv4.o_groups; ++gr)
        gemm(s.o + gr * hpg * hd, w.out_a_type, w.out_a + gr * g.dsv4.o_lora * oa_row_bytes,
             s.wo_tmp + gr * g.dsv4.o_lora, hpg * hd, g.dsv4.o_lora, s.xq_dim);
    gemm(s.wo_tmp, w.out_b_type, w.out_b, s.y, g.dsv4.o_groups * g.dsv4.o_lora, dim, s.xq_ob);
    peek("wo", s.y, dim);

    // ---- hc_post(attn).
    kernels::hc_post_combine(s.y, stream_in, s.post, s.comb, stream_out, 1, hc, dim, cu);
    peek("out", stream_out, hc * dim);
}

}  // namespace strata::core
