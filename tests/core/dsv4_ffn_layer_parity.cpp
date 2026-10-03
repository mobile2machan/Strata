// tests/core/dsv4_ffn_layer_parity.cpp - docs/DSV4.md P2 integration: the DSv4 FFN HALF of one
// decoder block, wired end to end (mHC pre -> ffn_norm -> router -> shared expert -> the top-k
// packed experts -> weighted sum -> mHC post), against a float64 transcription of the reference.
//
// Layer 1 is a score-routed layer (sqrtsoftplus + exp_probs_b top-k); layer 0 is a hash layer
// (indices from the tid2eid table, weights still from the scores).  The routed experts are the
// pack's own format: ggml-quantized IQ2_XXS/IQ3_XXS blobs in NativeExpertLayout order through
// `iq_mmvq`, exactly the real artifact's pair at the real 4096/2048.  The reference dequantizes
// those blobs and the q8_1 activations with the SAME kernels the engine's parity tests already
// vouch for, then does the arithmetic in float64.
//
// Three wrong wirings must come out observably apart:
//   1. route_scale dropped (1.0 instead of 1.5).
//   2. weights gathered from the POST-bias scores (the reference gathers them from the pre-bias
//      scores; the bias only chooses).
//   3. the SwiGLU clamp dropped (the weights here put gate/up past +-10 on purpose).
#include "strata/core/dsv4_ffn.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace strata;
using strata::kernels::bf16_from_f32;
using strata::kernels::f32_from_bf16;

int failures = 0;
void expect(bool ok, const char* what) {
    std::printf("    %-46s %s\n", what, ok ? "ok" : "*** FAIL ***");
    if (!ok) ++failures;
}
void check(cudaError_t e, const char* what) {
    if (e != cudaError::cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
double rel_l1(const std::vector<double>& a, const std::vector<double>& b) {
    double d = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += std::fabs(a[i] - b[i]);
        m += std::fabs(a[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
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
// exact dequantization of a Q8_0 blob (half d + int8 qs[32], 34 B per 32)
std::vector<float> q8_deq(const std::vector<uint8_t>& b, int64_t n) {
    std::vector<float> v((size_t) n);
    for (int64_t blk = 0; blk < n / 32; ++blk) {
        const uint16_t hb = (uint16_t) b[(size_t) blk * 34] |
                            ((uint16_t) b[(size_t) blk * 34 + 1] << 8);
        const float d = kernels::f32_from_f16(hb);
        for (int i = 0; i < 32; ++i)
            v[(size_t) blk * 32 + i] = d * (float)(int8_t) b[(size_t) blk * 34 + 2 + i];
    }
    return v;
}

void* dalloc(size_t bytes) {
    void* p = nullptr;
    check(cudaMalloc(&p, bytes), "dalloc");
    return p;
}

// ---- float64 reference pieces (same transcriptions as the per-kernel parity tests)
double sqrtsoftplus(double x) { return std::sqrt(x > 20.0 ? x : std::log1p(std::exp(x))); }

struct Trap {
    double scale = 1.5;
    bool gather_biased = false;  // weights from the POST-bias scores - wrong, and always visible
    double clamp = 10.0;         // <= 0 means no clamp
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) != "--selftest") {
        std::fprintf(stderr, "usage: %s --selftest\n", argv[0]);
        return 2;
    }
    std::printf("dsv4_ffn_layer_parity: FFN-half layer wiring, router + hash + packed experts\n");
    ggml_cpu_init();
    // native_mmvq requires an explicit non-null stream.
    static cudaStream_t g_stream = [] { cudaStream_t s; cudaStreamCreate(&s); return s; }();

    core::ModelGeometry g;
    g.n_layers = 2;
    g.n_embd = 4096;
    g.n_ff = 2048;
    g.n_expert = 8;
    g.hc = 4;
    g.dsv4.n_expert_used = 2;
    g.dsv4.hash_layers = 1;
    g.dsv4.sinkhorn_iters = 20;
    g.dsv4.hc_eps = 1e-6f;
    g.dsv4.norm_eps = 1e-6f;
    g.dsv4.route_scale = 1.5f;
    g.dsv4.swiglu_clamp_exp = {10.0f, 10.0f};
    g.dsv4.swiglu_clamp_shexp = {10.0f, 10.0f};
    g.dsv4.compress_ratios = {0, 0};
    const int64_t dim = g.n_embd, ff = g.n_ff, hc = g.hc, k = g.dsv4.n_expert_used;
    const int64_t mix_hc = (2 + hc) * hc;
    const int64_t vocab = 16;

    // ---- weights (host values; the engine gets device copies).  The shared expert is Q8_0 blobs
    // (the pack's native form; the real artifact's Q5_K/Q6_K are the same native path, vouched by
    // the qwen dense parity), the reference dots with their exact dequantization.
    std::vector<std::vector<float>> hc_fn((size_t) g.n_layers), hc_base((size_t) g.n_layers),
        hc_scale((size_t) g.n_layers), norm((size_t) g.n_layers), probs_b((size_t) g.n_layers);
    std::vector<std::vector<float>> sh_g((size_t) g.n_layers), sh_u((size_t) g.n_layers),
        sh_d((size_t) g.n_layers);
    std::vector<std::vector<uint8_t>> sh_blob((size_t) g.n_layers * 3);
    std::vector<std::vector<uint16_t>> gate((size_t) g.n_layers);
    std::vector<int32_t> tid2eid((size_t) k * vocab);
    for (auto& v : tid2eid) v = (int32_t)(rng() % (unsigned) g.n_expert);
    for (int64_t L = 0; L < g.n_layers; ++L) {
        hc_fn[(size_t) L] = rand_f32(mix_hc * hc * dim, 0.02);
        hc_base[(size_t) L] = rand_f32(mix_hc, 1.0);
        hc_scale[(size_t) L] = std::vector<float>((size_t) (hc - 1), 0.5f);
        norm[(size_t) L] = ones_f32(dim, 0.1);
        gate[(size_t) L] = rand_bf16(g.n_expert * dim, 0.02);
        probs_b[(size_t) L] = rand_f32(g.n_expert, 2.0);  // big enough that dropping it flips the picks
        // sd 0.15 over 4096 inputs puts the shared expert's gate/up past the +-10 clamp on purpose.
        auto q8 = [&](int64_t rows, int64_t cols, double sd, int idx, std::vector<float>& refv) {
            std::vector<float> w((size_t)(rows * cols));
            for (auto& v : w) v = (float)(nrm() * sd);
            sh_blob[(size_t) idx].resize((size_t)(rows * cols / 32 * 34));
            ggml_quantize_chunk(GGML_TYPE_Q8_0, w.data(), sh_blob[(size_t) idx].data(), 0, rows, cols,
                                nullptr);
            refv = q8_deq(sh_blob[(size_t) idx], rows * cols);
        };
        q8(ff, dim, 0.15, (int) L * 3 + 0, sh_g[(size_t) L]);
        q8(ff, dim, 0.15, (int) L * 3 + 1, sh_u[(size_t) L]);
        q8(dim, ff, 0.02, (int) L * 3 + 2, sh_d[(size_t) L]);
    }

    // ---- expert blobs: the real pair at the real dims, ggml-quantized from random weights
    kernels::NativeExpertLayout L = kernels::native_expert_layout(GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS, dim, ff);
    expect(kernels::native_expert_supported(L.gu_type, L.d_type, dim, ff), "IQ2_XXS/IQ3_XXS 4096/2048 supported");
    std::vector<uint8_t> blobs((size_t) g.n_expert * L.bytes);
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
            uint8_t* b = blobs.data() + (size_t) e * L.bytes;
            quant(L.gu_type, ff, dim, b);
            quant(L.gu_type, ff, dim, b + L.up_off);
            quant(L.d_type, dim, ff, b + L.down_off);
        }
    }

    // ---- device copies
    auto to_dev = [&](const auto& host) {
        void* p = nullptr;
        check(cudaMalloc(&p, host.size() * sizeof(host[0])), "weights");
        cudaMemcpy(p, host.data(), host.size() * sizeof(host[0]), cudaMemcpyHostToDevice);
        return p;
    };
    std::vector<core::Dsv4FfnWeights> w((size_t) g.n_layers);
    std::vector<void*> keep;
    auto keep_dev = [&](const auto& host, auto* dp) {
        void* p = to_dev(host);
        keep.push_back(p);
        using T = std::remove_reference_t<decltype(*dp)>;
        *dp = (T) p;
    };
    void* d_blobs = dalloc(blobs.size());
    cudaMemcpy(d_blobs, blobs.data(), blobs.size(), cudaMemcpyHostToDevice);
    void* d_tid = to_dev(tid2eid);
    for (int64_t L2 = 0; L2 < g.n_layers; ++L2) {
        auto& ww = w[(size_t) L2];
        keep_dev(hc_fn[(size_t) L2], &ww.hc_fn);
        keep_dev(hc_base[(size_t) L2], &ww.hc_base);
        keep_dev(hc_scale[(size_t) L2], &ww.hc_scale);
        keep_dev(norm[(size_t) L2], &ww.norm);
        keep_dev(gate[(size_t) L2], &ww.gate);
        if (L2 >= g.dsv4.hash_layers) keep_dev(probs_b[(size_t) L2], &ww.probs_b);
        for (int j = 0; j < 3; ++j) {
            void* p = to_dev(sh_blob[(size_t) L2 * 3 + (size_t) j]);
            keep.push_back(p);
            const uint8_t* dp = (const uint8_t*) p;
            if (j == 0) { ww.sh_gate = dp; ww.sh_gate_type = GGML_TYPE_Q8_0; }
            if (j == 1) { ww.sh_up = dp; ww.sh_up_type = GGML_TYPE_Q8_0; }
            if (j == 2) { ww.sh_down = dp; ww.sh_down_type = GGML_TYPE_Q8_0; }
        }
        ww.experts = L;
        ww.expert_blobs = (const uint8_t*) d_blobs;
        if (L2 < g.dsv4.hash_layers) ww.tid2eid = (const int32_t*) d_tid;
    }
    const int64_t scratch_bytes = core::dsv4_ffn_scratch_bytes(g);
    float* scratch = (float*) dalloc((size_t) scratch_bytes);
    std::printf("  scratch %lld bytes\n", (long long) scratch_bytes);

    // ---- one reference token through layer `layer`, in float64.  The packed experts and the q8_1
    // activations are read through the engine's own kernels (single-sourced), then dequantized
    // exactly and dotted in float64.
    auto ref_token = [&](int64_t layer, const std::vector<double>& stream, int64_t token,
                         const Trap& trap, std::vector<double>& out) {
        const double eps = g.dsv4.norm_eps;
        const auto& hwf = hc_fn[(size_t) layer];
        const auto& hwb = hc_base[(size_t) layer];
        const auto& hws = hc_scale[(size_t) layer];
        std::vector<double> mixes((size_t) mix_hc);
        double sq = 0;
        for (double v : stream) sq += v * v;
        const double rms_inv = 1.0 / std::sqrt(sq / (double)(hc * dim) + g.dsv4.hc_eps);
        for (int64_t o = 0; o < mix_hc; ++o) {
            double acc = 0;
            for (int64_t i = 0; i < hc * dim; ++i)
                acc += stream[(size_t) i] * (double) hwf[(size_t) o * hc * dim + i];
            mixes[(size_t) o] = acc * rms_inv;
        }
        std::vector<double> pre((size_t) hc), post((size_t) hc), comb((size_t) hc * hc);
        for (int64_t h = 0; h < hc; ++h)
            pre[(size_t) h] = 1.0 / (1.0 + std::exp(-(mixes[(size_t) h] * hws[0] + hwb[(size_t) h]))) + g.dsv4.hc_eps;
        for (int64_t h = 0; h < hc; ++h)
            post[(size_t) h] = 2.0 / (1.0 + std::exp(-(mixes[(size_t)(hc + h)] * hws[1] + hwb[(size_t)(hc + h)])));
        std::vector<std::vector<double>> c((size_t) hc, std::vector<double>((size_t) hc));
        for (int64_t p = 0; p < hc; ++p) {
            double mx = -INFINITY;
            for (int64_t q = 0; q < hc; ++q)
                mx = std::max(mx, mixes[(size_t)(2 * hc + p * hc + q)] * hws[2] + hwb[(size_t)(2 * hc + p * hc + q)]);
            double s = 0;
            for (int64_t q = 0; q < hc; ++q) {
                c[p][q] = std::exp(mixes[(size_t)(2 * hc + p * hc + q)] * hws[2] +
                                   hwb[(size_t)(2 * hc + p * hc + q)] - mx);
                s += c[p][q];
            }
            for (int64_t q = 0; q < hc; ++q) c[p][q] = c[p][q] / s + g.dsv4.hc_eps;
        }
        auto ncol = [&]() {
            for (int64_t q = 0; q < hc; ++q) {
                double s = g.dsv4.hc_eps;
                for (int64_t p = 0; p < hc; ++p) s += c[p][q];
                for (int64_t p = 0; p < hc; ++p) c[p][q] /= s;
            }
        };
        auto nrow = [&]() {
            for (int64_t p = 0; p < hc; ++p) {
                double s = g.dsv4.hc_eps;
                for (int64_t q = 0; q < hc; ++q) s += c[p][q];
                for (int64_t q = 0; q < hc; ++q) c[p][q] /= s;
            }
        };
        ncol();
        for (int64_t it = 1; it < g.dsv4.sinkhorn_iters; ++it) { nrow(); ncol(); }
        for (int64_t p = 0; p < hc; ++p)
            for (int64_t q = 0; q < hc; ++q) comb[(size_t) p * hc + q] = c[p][q];
        std::vector<double> y((size_t) dim);
        for (int64_t cc = 0; cc < dim; ++cc) {
            double acc = 0;
            for (int64_t h = 0; h < hc; ++h) acc += pre[(size_t) h] * stream[(size_t)(h * dim + cc)];
            y[(size_t) cc] = acc;
        }
        // ffn_norm
        {
            double s2 = 0;
            for (double v : y) s2 += v * v;
            const double inv = 1.0 / std::sqrt(s2 / (double) dim + eps);
            for (int64_t cc = 0; cc < dim; ++cc) y[(size_t) cc] *= inv * (double) norm[(size_t) layer][(size_t) cc];
        }
        // router
        const auto& wg = gate[(size_t) layer];
        std::vector<double> raw((size_t) g.n_expert);
        for (int64_t e = 0; e < g.n_expert; ++e) {
            double acc = 0;
            for (int64_t i = 0; i < dim; ++i) acc += y[(size_t) i] * (double) f32_from_bf16(wg[(size_t) e * dim + i]);
            raw[(size_t) e] = acc;
        }
        std::vector<double> rw((size_t) k);
        std::vector<int64_t> ids((size_t) k);
        if (layer < g.dsv4.hash_layers) {
            double sum = 0;
            for (int64_t j = 0; j < k; ++j) {
                ids[(size_t) j] = tid2eid[(size_t) token * k + j];
                rw[(size_t) j] = sqrtsoftplus(raw[(size_t) ids[(size_t) j]]);
                sum += rw[(size_t) j];
            }
            for (auto& v : rw) v *= trap.scale / sum;
        } else {
            std::vector<double> orig((size_t) g.n_expert), biased((size_t) g.n_expert);
            for (int64_t e = 0; e < g.n_expert; ++e) {
                orig[(size_t) e] = sqrtsoftplus(raw[(size_t) e]);
                biased[(size_t) e] = orig[(size_t) e] + (double) probs_b[(size_t) layer][(size_t) e];
            }
            double sum = 0;
            for (int64_t j = 0; j < k; ++j) {
                int best = 0;
                double bv = biased[0];
                for (int e = 1; e < (int) g.n_expert; ++e)
                    if (biased[(size_t) e] > bv) { bv = biased[(size_t) e]; best = e; }
                biased[(size_t) best] = -INFINITY;
                ids[(size_t) j] = best;
                rw[(size_t) j] = trap.gather_biased ? bv : orig[(size_t) best];
                sum += rw[(size_t) j];
            }
            for (auto& v : rw) v *= trap.scale / sum;
        }
        // shared expert (fp64 on the bf16 values) + clamped SwiGLU
        std::vector<double> acc_out((size_t) dim, 0.0);
        {
            const auto& sg = sh_g[(size_t) layer];
            const auto& su = sh_u[(size_t) layer];
            const auto& sd = sh_d[(size_t) layer];
            std::vector<double> h((size_t) ff);
            for (int64_t r = 0; r < ff; ++r) {
                double dg = 0, du = 0;
                for (int64_t i = 0; i < dim; ++i) {
                    dg += y[(size_t) i] * (double) sg[(size_t) r * dim + i];
                    du += y[(size_t) i] * (double) su[(size_t) r * dim + i];
                }
                if (trap.clamp > 0) {
                    if (dg > trap.clamp) dg = trap.clamp;
                    if (du > trap.clamp) du = trap.clamp;
                    if (du < -trap.clamp) du = -trap.clamp;
                }
                h[(size_t) r] = dg / (1.0 + std::exp(-dg)) * du;
            }
            for (int64_t cc = 0; cc < dim; ++cc) {
                double a2 = 0;
                for (int64_t r = 0; r < ff; ++r) a2 += h[(size_t) r] * (double) sd[(size_t) cc * ff + r];
                acc_out[(size_t) cc] = a2;
            }
        }
        // routed experts: q8_1 activation through the engine's kernel, blob dequant through the
        // engine's dequantizer, then float64 arithmetic.
        std::vector<float> y_f((size_t) dim);
        for (int64_t i = 0; i < dim; ++i) y_f[(size_t) i] = (float) y[(size_t) i];
        const int64_t xq_bytes = dim / 32 * 36;
        float* d_y = (float*) dalloc((size_t) dim * 4);
        uint8_t* d_xq = (uint8_t*) dalloc((size_t) xq_bytes);
        cudaMemcpy(d_y, y_f.data(), (size_t) dim * 4, cudaMemcpyHostToDevice);
        kernels::quantize_q8_1_rows(d_y, 1, dim, d_xq, nullptr);
        std::vector<uint8_t> xq((size_t) xq_bytes);
        cudaMemcpy(xq.data(), d_xq, (size_t) xq_bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_y);
        cudaFree(d_xq);
        auto q8_decode = [&](const std::vector<uint8_t>& q, int64_t n) {
            std::vector<double> v((size_t) n);
            for (int64_t b = 0; b < n / 32; ++b) {
                const uint16_t hb = (uint16_t) q[(size_t) b * 36] | ((uint16_t) q[(size_t) b * 36 + 1] << 8);
                const double d = kernels::f32_from_f16(hb);
                for (int i = 0; i < 32; ++i)
                    v[(size_t) b * 32 + i] = d * (double)(int8_t) q[(size_t) b * 36 + 4 + i];
            }
            return v;
        };
        std::vector<double> xq_d = q8_decode(xq, dim);
        for (int64_t j = 0; j < k; ++j) {
            const int64_t e = ids[(size_t) j];
            const uint8_t* blob = blobs.data() + (size_t) e * L.bytes;
            // dequantize this expert's gate/up/down with the engine's dequantizer
            auto deq = [&](int type, const uint8_t* src, int64_t rows, int64_t cols) {
                const int64_t n = rows * cols;
                void* d_src = dalloc(rows * ggml_row_size((ggml_type) type, cols));
                float* d_dst = (float*) dalloc((size_t) n * 4);
                cudaMemcpy(d_src, src, (size_t) ggml_row_size((ggml_type) type, cols) * rows, cudaMemcpyHostToDevice);
                kernels::iq_dequant_f32(type, d_src, n, d_dst, nullptr);
                std::vector<float> h((size_t) n);
                cudaMemcpy(h.data(), d_dst, (size_t) n * 4, cudaMemcpyDeviceToHost);
                cudaFree(d_src);
                cudaFree(d_dst);
                return h;
            };
            std::vector<float> gw = deq(L.gu_type, blob, ff, dim);
            std::vector<float> uw = deq(L.gu_type, blob + L.up_off, ff, dim);
            std::vector<float> dw = deq(L.d_type, blob + L.down_off, dim, ff);
            std::vector<double> h((size_t) ff);
            for (int64_t r = 0; r < ff; ++r) {
                double dg = 0, du = 0;
                for (int64_t i = 0; i < dim; ++i) {
                    dg += xq_d[(size_t) i] * (double) gw[(size_t) r * dim + i];
                    du += xq_d[(size_t) i] * (double) uw[(size_t) r * dim + i];
                }
                if (trap.clamp > 0) {
                    if (dg > trap.clamp) dg = trap.clamp;
                    if (du > trap.clamp) du = trap.clamp;
                    if (du < -trap.clamp) du = -trap.clamp;
                }
                h[(size_t) r] = dg / (1.0 + std::exp(-dg)) * du;
            }
            // quantize h the same way the engine does, then dot the dequantized down rows
            std::vector<float> h_f((size_t) ff);
            for (int64_t r = 0; r < ff; ++r) h_f[(size_t) r] = (float) h[(size_t) r];
            const int64_t fq_bytes = ff / 32 * 36;
            float* d_h = (float*) dalloc((size_t) ff * 4);
            uint8_t* d_fq = (uint8_t*) dalloc((size_t) fq_bytes);
            cudaMemcpy(d_h, h_f.data(), (size_t) ff * 4, cudaMemcpyHostToDevice);
            kernels::quantize_q8_1_rows(d_h, 1, ff, d_fq, nullptr);
            std::vector<uint8_t> fq((size_t) fq_bytes);
            cudaMemcpy(fq.data(), d_fq, (size_t) fq_bytes, cudaMemcpyDeviceToHost);
            cudaFree(d_h);
            cudaFree(d_fq);
            std::vector<double> fq_d = q8_decode(fq, ff);
            for (int64_t cc = 0; cc < dim; ++cc) {
                double a2 = 0;
                for (int64_t r = 0; r < ff; ++r) a2 += fq_d[(size_t) r] * (double) dw[(size_t) cc * ff + r];
                acc_out[(size_t) cc] += rw[(size_t) j] * a2;
            }
        }
        // hc_post
        out.assign((size_t) hc * dim, 0.0);
        for (int64_t q = 0; q < hc; ++q)
            for (int64_t cc = 0; cc < dim; ++cc) {
                double a2 = post[(size_t) q] * acc_out[(size_t) cc];
                for (int64_t p = 0; p < hc; ++p) a2 += comb[(size_t) p * hc + q] * stream[(size_t)(p * dim + cc)];
                out[(size_t)(q * dim + cc)] = a2;
            }
    };

    struct Case { int64_t layer; int64_t token; };
    const Case cases[] = {{1, 5}, {1, 6}, {0, 7}};
    for (const Case& cse : cases) {
        std::vector<double> stream((size_t) hc * dim);
        for (auto& v : stream) v = nrm();
        float* d_in = (float*) dalloc((size_t) stream.size() * 4);
        float* d_out = (float*) dalloc((size_t) stream.size() * 4);
        std::vector<float> in_f(stream.size());
        for (size_t i = 0; i < in_f.size(); ++i) in_f[i] = (float) stream[i];
        cudaMemcpy(d_in, in_f.data(), in_f.size() * 4, cudaMemcpyHostToDevice);
        core::dsv4_ffn_decode_step(g, cse.layer, w[(size_t) cse.layer], d_in, d_out, cse.token, scratch,
                                   g_stream);
        std::vector<float> out_f(in_f.size());
        cudaMemcpy(out_f.data(), d_out, out_f.size() * 4, cudaMemcpyDeviceToHost);
        cudaFree(d_in);
        cudaFree(d_out);
        std::vector<double> eng(out_f.begin(), out_f.end()), ref_o;
        ref_token(cse.layer, stream, cse.token, Trap{}, ref_o);
        const double r = rel_l1(ref_o, eng);
        std::printf("  layer %lld token %lld rel_l1 %.3e\n", (long long) cse.layer, (long long) cse.token, r);
        expect(r < 2e-3, "stream out matches reference");

        Trap t1; t1.scale = 1.0;
        std::vector<double> o1;
        ref_token(cse.layer, stream, cse.token, t1, o1);
        expect(rel_l1(o1, eng) > 1e-2, "route_scale dropped is apart");
        Trap t2; t2.gather_biased = true;
        std::vector<double> o2;
        ref_token(cse.layer, stream, cse.token, t2, o2);
        if (cse.layer >= g.dsv4.hash_layers) expect(rel_l1(o2, eng) > 1e-2, "post-bias weight gather is apart");
        Trap t3; t3.clamp = 0.0;
        std::vector<double> o3;
        ref_token(cse.layer, stream, cse.token, t3, o3);
        expect(rel_l1(o3, eng) > 1e-2, "clamp dropped is apart");
    }

    std::printf("dsv4_ffn_layer_parity: %s\n", failures == 0 ? "ok" : "*** FAIL ***");
    return failures == 0 ? 0 : 1;
}
