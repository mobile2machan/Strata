// tests/core/dsv4_block_layer_parity.cpp - docs/DSV4.md P2 integration: the JOIN of the two
// parity-tested halves into one full decoder block (dsv4_block.cpp).
//
// Each half is already checked against a float64 reference by dsv4_attn_layer_parity and
// dsv4_ffn_layer_parity; re-transcribing the whole block here would test the copy, not the join.
// What the join itself can get wrong - the arena split between the halves, the staging stream
// (hc_post_combine reads its residual while writing, so the halves may not share buffers), and the
// order - is caught exactly: the block must be BIT-IDENTICAL to running the two halves in sequence
// with separate workspaces, and its pools must match too.  Plus the sanity a bit-identity cannot
// show: outputs finite, and the r=4 layer's pools actually fill.
#include "strata/core/dsv4_block.hpp"
#include "strata/core/dsv4_state.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "ggml.h"
#include "ggml-cpu.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace strata;
using strata::kernels::bf16_from_f32;

int failures = 0;
void expect(bool ok, const char* what) {
    std::printf("    %-46s %s\n", what, ok ? "ok" : "*** FAIL ***");
    if (!ok) ++failures;
}
void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

std::mt19937 rng(20261004u);
double nrm() {
    static std::normal_distribution<double> nd(0.0, 1.0);
    return nd(rng);
}
std::vector<uint16_t> rand_bf16(int64_t n, double sd) {
    std::vector<uint16_t> v((size_t) n);
    for (int64_t i = 0; i < n; ++i) v[(size_t) i] = bf16_from_f32((float)(nrm() * sd));
    return v;
}
std::vector<float> rand_f32(int64_t n, double sd) {
    std::vector<float> v((size_t) n);
    for (int64_t i = 0; i < n; ++i) v[(size_t) i] = (float)(nrm() * sd);
    return v;
}
std::vector<float> ones_f32(int64_t n, double jitter) {
    std::vector<float> v((size_t) n);
    for (int64_t i = 0; i < n; ++i) v[(size_t) i] = (float)(1.0 + jitter * nrm());
    return v;
}

void* dalloc(size_t bytes) {
    void* p = nullptr;
    check(cudaMalloc(&p, bytes), "dalloc");
    return p;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) != "--selftest") {
        std::fprintf(stderr, "usage: %s --selftest\n", argv[0]);
        return 2;
    }
    std::printf("dsv4_block_layer_parity: full-block join, block == halves in sequence\n");
    ggml_cpu_init();
    // native_mmvq requires an explicit non-null stream.
    static cudaStream_t g_stream = [] { cudaStream_t s; cudaStreamCreate(&s); return s; }();

    core::ModelGeometry g;
    g.n_layers = 2;
    g.n_embd = 4096;
    g.n_ff = 2048;
    g.n_expert = 8;
    g.n_head = 64;
    g.head_dim = 512;
    g.hc = 4;
    g.idx_q_heads = 64;
    g.idx_key_dim = 128;
    g.dsv4.q_lora = 1024;
    g.dsv4.o_lora = 1024;
    g.dsv4.o_groups = 8;
    g.dsv4.sliding_window = 128;
    g.dsv4.idx_topk = 1;
    g.dsv4.n_expert_used = 2;
    g.dsv4.hash_layers = 1;
    g.dsv4.sinkhorn_iters = 20;
    g.dsv4.hc_eps = 1e-6f;
    g.dsv4.norm_eps = 1e-6f;
    g.dsv4.route_scale = 1.5f;
    g.dsv4.swiglu_clamp_exp = {10.0f, 10.0f};
    g.dsv4.swiglu_clamp_shexp = {10.0f, 10.0f};
    g.dsv4.rope_theta = 10000.0;
    g.dsv4.compress_rope_theta = 160000.0;
    g.dsv4.rope_dim = 64;
    g.dsv4.yarn_factor = 16.0;
    g.dsv4.yarn_orig = 65536;
    g.dsv4.yarn_beta_fast = 32;
    g.dsv4.yarn_beta_slow = 1;
    // layer 0: r=0 (window only) + hash routing; layer 1: r=4 + score routing.
    g.dsv4.compress_ratios = {0, 4};
    const int64_t dim = g.n_embd, ff = g.n_ff, hd = g.head_dim, hc = g.hc, k = g.dsv4.n_expert_used;
    const int64_t mix_hc = (2 + hc) * hc;
    const int64_t vocab = 16, ntok = 4;

    // ---- weights (values; the engine gets device copies)
    std::vector<core::Dsv4AttnWeights> wa((size_t) g.n_layers);
    std::vector<core::Dsv4FfnWeights> wf((size_t) g.n_layers);
    std::vector<void*> keep;
    auto to_dev = [&](const auto& host, auto* dp) {
        void* p = nullptr;
        check(cudaMalloc(&p, host.size() * sizeof(host[0])), "weights");
        cudaMemcpy(p, host.data(), host.size() * sizeof(host[0]), cudaMemcpyHostToDevice);
        keep.push_back(p);
        using T = std::remove_reference_t<decltype(*dp)>;
        *dp = (T) p;
    };
    for (int64_t L = 0; L < g.n_layers; ++L) {
        auto& a = wa[(size_t) L];
        auto& f = wf[(size_t) L];
        // attention half: native Q8_0 blobs, like the real pack serves them
        auto set_gemm = [&](int64_t rows, int64_t cols, double sd, auto* dp, int* tp) {
            std::vector<float> wt((size_t)(rows * cols));
            for (auto& v : wt) v = (float)(nrm() * sd);
            std::vector<uint8_t> b((size_t)(rows * cols / 32 * 34));
            ggml_quantize_chunk(GGML_TYPE_Q8_0, wt.data(), b.data(), 0, rows, cols, nullptr);
            void* p = nullptr;
            check(cudaMalloc(&p, b.size()), "weights");
            cudaMemcpy(p, b.data(), b.size(), cudaMemcpyHostToDevice);
            keep.push_back(p);
            *dp = (const uint8_t*) p;
            *tp = GGML_TYPE_Q8_0;
        };
        // rows = n_out, cols = n_in: the GEMV reads each output row as one contiguous run.
        set_gemm(g.dsv4.q_lora, dim, 0.02, &a.q_a, &a.q_a_type);
        set_gemm(g.n_head * hd, g.dsv4.q_lora, 0.02, &a.q_b, &a.q_b_type);
        set_gemm(hd, dim, 0.02, &a.kv, &a.kv_type);
        set_gemm(g.dsv4.o_groups * g.dsv4.o_lora, (g.n_head / g.dsv4.o_groups) * hd, 0.02, &a.out_a,
                 &a.out_a_type);
        set_gemm(dim, g.dsv4.o_groups * g.dsv4.o_lora, 0.02, &a.out_b, &a.out_b_type);
        set_gemm(2 * hd, dim, 0.02, &a.comp_kv, &a.comp_kv_type);
        set_gemm(2 * hd, dim, 0.02, &a.comp_gate, &a.comp_gate_type);
        set_gemm(g.idx_q_heads * g.idx_key_dim, g.dsv4.q_lora, 0.02, &a.idx_qb, &a.idx_qb_type);
        set_gemm(2 * g.idx_key_dim, dim, 0.02, &a.idx_comp_kv, &a.idx_comp_kv_type);
        set_gemm(2 * g.idx_key_dim, dim, 0.02, &a.idx_comp_gate, &a.idx_comp_gate_type);
        to_dev(rand_bf16(dim * g.idx_q_heads, 0.02), &a.idx_proj);
        to_dev(ones_f32(dim, 0.1), &a.norm);
        to_dev(ones_f32(g.dsv4.q_lora, 0.1), &a.q_a_norm);
        to_dev(ones_f32(hd, 0.1), &a.kv_norm);
        to_dev(rand_f32(g.n_head, 1.0), &a.sinks);
        to_dev(rand_f32(mix_hc * hc * dim, 0.02), &a.hc_fn);
        to_dev(rand_f32(mix_hc, 1.0), &a.hc_base);
        to_dev(std::vector<float>(3, 0.5f), &a.hc_scale);
        to_dev(rand_f32(4 * 2 * hd, 0.5), &a.ape);
        to_dev(ones_f32(hd, 0.1), &a.comp_norm);
        to_dev(rand_f32(4 * 2 * g.idx_key_dim, 0.5), &a.idx_ape);
        to_dev(ones_f32(g.idx_key_dim, 0.1), &a.idx_comp_norm);
        // FFN half (its own hc buffers - the real artifact has separate hc_attn_fn / hc_ffn_fn)
        to_dev(rand_f32(mix_hc * hc * dim, 0.02), &f.hc_fn);
        to_dev(rand_f32(mix_hc, 1.0), &f.hc_base);
        to_dev(std::vector<float>(hc - 1, 0.5f), &f.hc_scale);
        to_dev(ones_f32(dim, 0.1), &f.norm);
        to_dev(rand_bf16(g.n_expert * dim, 0.02), &f.gate);
        if (L >= g.dsv4.hash_layers) to_dev(rand_f32(g.n_expert, 2.0), &f.probs_b);
        set_gemm(ff, dim, 0.15, &f.sh_gate, &f.sh_gate_type);
        set_gemm(ff, dim, 0.15, &f.sh_up, &f.sh_up_type);
        set_gemm(dim, ff, 0.02, &f.sh_down, &f.sh_down_type);
    }
    std::vector<int32_t> tid2eid((size_t) k * vocab);
    for (auto& v : tid2eid) v = (int32_t)(rng() % (unsigned) g.n_expert);
    to_dev(tid2eid, &wf[0].tid2eid);

    kernels::NativeExpertLayout EL = kernels::native_expert_layout(GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS, dim, ff);
    std::vector<uint8_t> blobs((size_t) g.n_expert * EL.bytes);
    {
        std::mt19937 brng(777u);
        std::normal_distribution<float> nd(0.f, 1.f);
        auto quant = [&](int type, int64_t rows, int64_t cols, uint8_t* dst) {
            std::vector<float> w((size_t)(rows * cols));
            for (int64_t r = 0; r < rows; ++r) {
                const float sc = 0.15f * (0.5f + (float)(r % 7) / 7.0f);
                for (int64_t c = 0; c < cols; ++c) w[(size_t)(r * cols + c)] = sc * nd(brng);
            }
            const std::vector<float> imatrix((size_t) cols, 1.0f);
            ggml_quantize_chunk((ggml_type) type, w.data(), dst, 0, rows, cols,
                                ggml_quantize_requires_imatrix((ggml_type) type) ? imatrix.data() : nullptr);
        };
        for (int64_t e = 0; e < g.n_expert; ++e) {
            uint8_t* b = blobs.data() + (size_t) e * EL.bytes;
            quant(EL.gu_type, ff, dim, b);
            quant(EL.gu_type, ff, dim, b + EL.up_off);
            quant(EL.d_type, dim, ff, b + EL.down_off);
        }
    }
    void* d_blobs = dalloc(blobs.size());
    cudaMemcpy(d_blobs, blobs.data(), blobs.size(), cudaMemcpyHostToDevice);
    for (auto& f : wf) {
        f.experts = EL;
        f.expert_blobs = (const uint8_t*) d_blobs;
    }

    // ---- two independent states: one for the block, one for the manual sequence
    core::Dsv4State state_blk, state_seq;
    std::string err;
    if (!state_blk.init(g, 64, err) || !state_seq.init(g, 64, err)) {
        std::fprintf(stderr, "state init: %s\n", err.c_str());
        return 1;
    }

    float* stream_in = (float*) dalloc(hc * dim * 4);
    float* stream_out = (float*) dalloc(hc * dim * 4);
    float* seq_out = (float*) dalloc(hc * dim * 4);
    float* mid = (float*) dalloc(hc * dim * 4);
    std::vector<float> seed_h((size_t) hc * dim);
    for (auto& v : seed_h) v = (float) nrm();

    for (int64_t layer = 1; layer >= 0; --layer) {
        const int64_t ratio = g.dsv4.compress_ratios[(size_t) layer];
        const int64_t max_stage = ratio > 0 ? (ntok + ratio - 1) / ratio : 0;
        const int64_t blk_bytes = core::dsv4_block_scratch_bytes(g, layer, max_stage);
        const int64_t attn_bytes = core::dsv4_attn_scratch_bytes(g, layer, max_stage);
        const int64_t ffn_bytes = core::dsv4_ffn_scratch_bytes(g);
        std::printf("  layer %lld (r=%lld): block scratch %lld bytes\n", (long long) layer,
                    (long long) ratio, (long long) blk_bytes);
        expect(blk_bytes >= attn_bytes + ffn_bytes, "block scratch covers both halves");
        float* s_blk = (float*) dalloc((size_t) blk_bytes);
        float* s_attn = (float*) dalloc((size_t) attn_bytes);
        float* s_ffn = (float*) dalloc((size_t) ffn_bytes);

        core::Dsv4BlockWeights w;
        w.attn = wa[(size_t) layer];
        w.ffn = wf[(size_t) layer];
        const core::Dsv4LayerState st_b = state_blk.layer(layer);
        const core::Dsv4LayerState st_s = state_seq.layer(layer);

        cudaMemcpy(stream_in, seed_h.data(), (size_t) hc * dim * 4, cudaMemcpyHostToDevice);
        for (int64_t pos = 0; pos < ntok; ++pos) {
            const int64_t n_stage = ratio > 0 ? (pos + 1) / ratio : 0;
            core::dsv4_block_decode_step(g, layer, w, st_b, stream_in, stream_out, pos, pos, n_stage,
                                         s_blk, g_stream);
            core::dsv4_attn_decode_step(g, layer, w.attn, st_s, stream_in, mid, pos, n_stage, s_attn,
                                        g_stream);
            core::dsv4_ffn_decode_step(g, layer, w.ffn, mid, seq_out, pos, s_ffn, g_stream);
            std::vector<float> a((size_t) hc * dim), b((size_t) hc * dim);
            cudaMemcpy(a.data(), stream_out, (size_t) hc * dim * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(b.data(), seq_out, (size_t) hc * dim * 4, cudaMemcpyDeviceToHost);
            bool finite = true;
            for (float v : a) if (!std::isfinite(v)) finite = false;
            if (pos == ntok - 1) {
                expect(finite, "block output finite");
                expect(std::memcmp(a.data(), b.data(), (size_t) hc * dim * 4) == 0,
                       "block bit-identical to halves in sequence");
            }
            cudaMemcpy(stream_in, stream_out, (size_t) hc * dim * 4, cudaMemcpyDeviceToHost);
        }

        // pools must match too (same inputs, two states) - and the r=4 layer's must have filled.
        const core::Dsv4PoolSizes ps = core::dsv4_pool_sizes(g, layer, 64);
        std::vector<uint8_t> pa((size_t) ps.window_bytes), pb((size_t) ps.window_bytes);
        cudaMemcpy(pa.data(), st_b.window, (size_t) ps.window_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(pb.data(), st_s.window, (size_t) ps.window_bytes, cudaMemcpyDeviceToHost);
        expect(std::memcmp(pa.data(), pb.data(), (size_t) ps.window_bytes) == 0, "window pools match");
        bool win_filled = false;
        for (uint8_t v : pa) if (v) { win_filled = true; break; }
        expect(win_filled, "window pool written");
        if (ratio > 0) {
            std::vector<uint8_t> ca((size_t) ps.cmp_bytes), cb((size_t) ps.cmp_bytes);
            cudaMemcpy(ca.data(), st_b.cmp, (size_t) ps.cmp_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(cb.data(), st_s.cmp, (size_t) ps.cmp_bytes, cudaMemcpyDeviceToHost);
            expect(std::memcmp(ca.data(), cb.data(), (size_t) ps.cmp_bytes) == 0, "cmp pools match");
            bool cmp_filled = false;
            for (uint8_t v : ca) if (v) { cmp_filled = true; break; }
            expect(cmp_filled, "cmp pool written (compressor ran)");
            std::vector<uint8_t> ia((size_t) ps.idx_bytes), ib((size_t) ps.idx_bytes);
            cudaMemcpy(ia.data(), st_b.idx, (size_t) ps.idx_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(ib.data(), st_s.idx, (size_t) ps.idx_bytes, cudaMemcpyDeviceToHost);
            expect(std::memcmp(ia.data(), ib.data(), (size_t) ps.idx_bytes) == 0, "idx pools match");
        }
        cudaFree(s_blk);
        cudaFree(s_attn);
        cudaFree(s_ffn);
    }

    std::printf("dsv4_block_layer_parity: %s\n", failures ? "*** FAIL ***" : "ok");
    return failures ? 1 : 0;
}
