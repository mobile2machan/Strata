// tests/core/dsv4_attn_layer_parity.cpp - docs/DSV4.md P2 integration: the DSv4 attention HALF of
// one decoder block, wired end to end (mHC pre -> Q/kv -> window ring -> compressor -> indexer ->
// tiered gather -> wo -> mHC post), over eight consecutive decode tokens through an r=4 layer.
//
// The reference is the same transcription of `model.py::Block.decode_step` +
// `Attention.decode_step` the engine is, but in float64 on the host, EXCEPT for the quantization
// round-trips and the Hadamard, which BOTH sides get from the GPU kernels - those are parity-tested
// on their own (dsv4_quant_parity), and re-implementing them here would test the copy, not the
// wiring.  Dimensions are the real artifact's; `idx_topk` is 1 so the indexer's selection actually
// discards a scored row from token 7 on.
//
// Three wrong wirings must come out observably apart, not within tolerance:
//   1. no sink term in the gather.
//   2. the compressed rows dropped from the gather (window-only attention).
//   3. no indexer top-k: every scored row used instead of the best one.
#include "strata/core/dsv4_attn.hpp"
#include "strata/core/dsv4_state.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/dsv4_quant.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
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
    if (e != cudaSuccess) {
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

std::mt19937 rng(20261003u);
double nrm() {
    static std::normal_distribution<double> nd(0.0, 1.0);
    return nd(rng);
}

// ---- host weight set (values; the engine gets Q8_0 blobs / f32 device copies of the same numbers)
struct HostW {
    std::vector<float> q_a, q_b, kv, out_a, out_b, comp_kv, comp_gate, idx_qb, idx_comp_kv,
        idx_comp_gate;  // dequantized Q8_0 values - what the kernel sees
    std::vector<uint16_t> idx_proj;
    std::vector<float> norm, q_a_norm, kv_norm, sinks, hc_fn, hc_base, hc_scale, ape, comp_norm,
        idx_ape, idx_comp_norm;
};

// A Q8_0 blob of random weights, and its exact dequantization (the reference dots with the values
// the native kernel sees, not the pre-quantization ones - same contract as the FFN test).
std::vector<uint8_t> q8_blob(int64_t rows, int64_t cols, double sd) {
    std::vector<float> w((size_t)(rows * cols));
    for (auto& v : w) v = (float)(nrm() * sd);
    std::vector<uint8_t> b((size_t)(rows * cols / 32 * 34));
    ggml_quantize_chunk(GGML_TYPE_Q8_0, w.data(), b.data(), 0, rows, cols, nullptr);
    return b;
}
std::vector<float> q8_deq(const std::vector<uint8_t>& b, int64_t n) {
    std::vector<float> v((size_t) n);
    for (int64_t blk = 0; blk < n / 32; ++blk) {
        const uint16_t hb = (uint16_t) b[(size_t) blk * 34] |
                            ((uint16_t) b[(size_t) blk * 34 + 1] << 8);
        const float d = strata::kernels::f32_from_f16(hb);
        for (int i = 0; i < 32; ++i)
            v[(size_t) blk * 32 + i] = d * (float)(int8_t) b[(size_t) blk * 34 + 2 + i];
    }
    return v;
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

// ---- GPU helpers the reference borrows (quantization / Hadamard are single-sourced)
void* dalloc(size_t bytes) {
    void* p = nullptr;
    check(cudaMalloc(&p, bytes), "dalloc");
    return p;
}

/// `y = e4m3 round-trip of x[0..n)` (block `block`), via the GPU kernel.
std::vector<uint16_t> gpu_e4m3(const std::vector<double>& x, int64_t n, int64_t block) {
    std::vector<float> xf((size_t) n);
    for (int64_t i = 0; i < n; ++i) xf[(size_t) i] = (float) x[(size_t) i];
    float* dx = (float*) dalloc((size_t) n * 4);
    uint16_t* dy = (uint16_t*) dalloc((size_t) n * 2);
    cudaMemcpy(dx, xf.data(), (size_t) n * 4, cudaMemcpyHostToDevice);
    kernels::roundtrip_fp8_e4m3(dx, dy, n, block, nullptr);
    std::vector<uint16_t> y((size_t) n);
    cudaMemcpy(y.data(), dy, (size_t) n * 2, cudaMemcpyDeviceToHost);
    cudaFree(dx);
    cudaFree(dy);
    return y;
}
std::vector<uint16_t> gpu_e2m1(const std::vector<double>& x, int64_t n, int64_t block) {
    std::vector<float> xf((size_t) n);
    for (int64_t i = 0; i < n; ++i) xf[(size_t) i] = (float) x[(size_t) i];
    float* dx = (float*) dalloc((size_t) n * 4);
    uint16_t* dy = (uint16_t*) dalloc((size_t) n * 2);
    cudaMemcpy(dx, xf.data(), (size_t) n * 4, cudaMemcpyHostToDevice);
    kernels::roundtrip_fp4_e2m1(dx, dy, n, block, nullptr);
    std::vector<uint16_t> y((size_t) n);
    cudaMemcpy(y.data(), dy, (size_t) n * 2, cudaMemcpyDeviceToHost);
    cudaFree(dx);
    cudaFree(dy);
    return y;
}
std::vector<double> gpu_hadamard(const std::vector<double>& x, int64_t rows, int64_t d) {
    std::vector<float> xf((size_t) x.size());
    for (size_t i = 0; i < x.size(); ++i) xf[i] = (float) x[i];
    float* dx = (float*) dalloc(xf.size() * 4);
    float* dy = (float*) dalloc(xf.size() * 4);
    cudaMemcpy(dx, xf.data(), xf.size() * 4, cudaMemcpyHostToDevice);
    kernels::hadamard(dx, dy, rows, d, nullptr);
    std::vector<float> yf(xf.size());
    cudaMemcpy(yf.data(), dy, yf.size() * 4, cudaMemcpyDeviceToHost);
    cudaFree(dx);
    cudaFree(dy);
    return std::vector<double>(yf.begin(), yf.end());
}

// ---- float64 reference pieces, copied from the per-kernel parity tests
constexpr double PI = 3.14159265358979323846;

std::vector<double> ref_freqs(int rd, double theta, double factor, int64_t orig, int64_t bf,
                              int64_t bs) {
    std::vector<double> f((size_t) rd / 2);
    for (int j = 0; j < rd / 2; ++j) f[(size_t) j] = std::pow(theta, -(double)(2 * j) / (double) rd);
    if (orig > 0) {
        auto corr = [&](double rot) {
            return (double) rd * std::log((double) orig / (rot * 2.0 * PI)) / (2.0 * std::log(theta));
        };
        int64_t low = std::max<int64_t>((int64_t) std::floor(corr((double) bf)), 0);
        int64_t high = std::min<int64_t>((int64_t) std::ceil(corr((double) bs)), rd - 1);
        for (int j = 0; j < rd / 2; ++j) {
            double lo = (double) low, hi = (double) high;
            if (lo == hi) hi = lo + 0.001;
            double t = ((double) j - lo) / (hi - lo);
            t = t < 0 ? 0 : (t > 1 ? 1 : t);
            const double smooth = 1.0 - t;
            f[(size_t) j] = f[(size_t) j] / factor * (1.0 - smooth) + f[(size_t) j] * smooth;
        }
    }
    return f;
}
void ref_rope(std::vector<double>& x, int64_t n_rows, int64_t row_dim, int rd, int64_t pos,
              const std::vector<double>& f, bool inverse) {
    for (int64_t r = 0; r < n_rows; ++r) {
        size_t base = (size_t) (r * row_dim + row_dim - rd);
        for (int j = 0; j < rd / 2; ++j) {
            size_t i0 = base + (size_t)(2 * j), i1 = base + (size_t)(2 * j + 1);
            double s = std::sin((double) pos * f[(size_t) j]), c = std::cos((double) pos * f[(size_t) j]);
            if (inverse) s = -s;
            const double x0 = x[i0], x1 = x[i1];
            x[i0] = x0 * c - x1 * s;
            x[i1] = x0 * s + x1 * c;
        }
    }
}
void ref_split(const std::vector<double>& mixes, const std::vector<double>& scale,
               const std::vector<double>& base, int64_t hc, int64_t iters, double eps,
               std::vector<double>& pre, std::vector<double>& post, std::vector<double>& comb) {
    pre.assign((size_t) hc, 0.0);
    post.assign((size_t) hc, 0.0);
    comb.assign((size_t) hc * hc, 0.0);
    for (int64_t h = 0; h < hc; ++h)
        pre[(size_t) h] = 1.0 / (1.0 + std::exp(-(mixes[(size_t) h] * scale[0] + base[(size_t) h]))) + eps;
    for (int64_t h = 0; h < hc; ++h)
        post[(size_t) h] = 2.0 / (1.0 + std::exp(-(mixes[(size_t) (hc + h)] * scale[1] + base[(size_t) (hc + h)])));
    std::vector<std::vector<double>> c((size_t) hc, std::vector<double>((size_t) hc));
    for (int64_t p = 0; p < hc; ++p) {
        double mx = -INFINITY;
        for (int64_t q = 0; q < hc; ++q)
            mx = std::max(mx, mixes[(size_t) (2 * hc + p * hc + q)] * scale[2] + base[(size_t) (2 * hc + p * hc + q)]);
        double s = 0;
        for (int64_t q = 0; q < hc; ++q) {
            c[p][q] = std::exp(mixes[(size_t) (2 * hc + p * hc + q)] * scale[2] +
                               base[(size_t) (2 * hc + p * hc + q)] - mx);
            s += c[p][q];
        }
        for (int64_t q = 0; q < hc; ++q) c[p][q] = c[p][q] / s + eps;
    }
    auto norm_cols = [&]() {
        for (int64_t q = 0; q < hc; ++q) {
            double s = eps;
            for (int64_t p = 0; p < hc; ++p) s += c[p][q];
            for (int64_t p = 0; p < hc; ++p) c[p][q] /= s;
        }
    };
    auto norm_rows = [&]() {
        for (int64_t p = 0; p < hc; ++p) {
            double s = eps;
            for (int64_t q = 0; q < hc; ++q) s += c[p][q];
            for (int64_t q = 0; q < hc; ++q) c[p][q] /= s;
        }
    };
    norm_cols();
    for (int64_t it = 1; it < iters; ++it) { norm_rows(); norm_cols(); }
    for (int64_t p = 0; p < hc; ++p)
        for (int64_t q = 0; q < hc; ++q) comb[(size_t) p * hc + q] = c[p][q];
}
void ref_rms(std::vector<double>& x, const std::vector<float>* w, int64_t rows, int64_t cols, double eps) {
    for (int64_t r = 0; r < rows; ++r) {
        double sq = 0;
        for (int64_t c = 0; c < cols; ++c) sq += x[(size_t) r * cols + c] * x[(size_t) r * cols + c];
        const double inv = 1.0 / std::sqrt(sq / (double) cols + eps);
        for (int64_t c = 0; c < cols; ++c) {
            double v = x[(size_t) r * cols + c] * inv;
            if (w) v *= (double) (*w)[(size_t) c];
            x[(size_t) r * cols + c] = v;
        }
    }
}
void ref_gemv(const std::vector<double>& x, const std::vector<uint16_t>& w, int64_t n_in,
              int64_t n_out, std::vector<double>& y) {
    y.assign((size_t) n_out, 0.0);
    for (int64_t o = 0; o < n_out; ++o) {
        const uint16_t* wr = &w[(size_t) o * n_in];
        double acc = 0;
        for (int64_t i = 0; i < n_in; ++i) acc += x[(size_t) i] * (double) f32_from_bf16(wr[i]);
        y[(size_t) o] = acc;
    }
}
void ref_gemv(const std::vector<double>& x, const std::vector<float>& w, int64_t n_in,
              int64_t n_out, std::vector<double>& y) {
    y.assign((size_t) n_out, 0.0);
    for (int64_t o = 0; o < n_out; ++o) {
        const float* wr = &w[(size_t) o * n_in];
        double acc = 0;
        for (int64_t i = 0; i < n_in; ++i) acc += x[(size_t) i] * (double) wr[i];
        y[(size_t) o] = acc;
    }
}
struct Shape {
    int64_t ratio;
    bool overlap;
    int64_t d;
    int64_t item() const { return (overlap ? 2 : 1) * d; }
    int64_t rows() const { return (overlap ? 2 : 1) * ratio; }
};
void ref_decode_token(const Shape& sh, int64_t pos, const std::vector<double>& kv,
                      const std::vector<double>& score, const std::vector<float>& ape,
                      std::vector<double>& ks, std::vector<double>& ss, std::vector<double>* compressed) {
    const int64_t item = sh.item(), d = sh.d;
    const int64_t idx = pos % sh.ratio, slot = sh.overlap ? sh.ratio + idx : idx;
    for (int64_t i = 0; i < item; ++i) {
        ks[(size_t) slot * item + i] = kv[(size_t) i];
        ss[(size_t) slot * item + i] = score[(size_t) i] + (double) ape[(size_t) idx * item + i];
    }
    if ((pos + 1) % sh.ratio != 0) return;
    compressed->assign((size_t) d, 0.0);
    for (int64_t c = 0; c < d; ++c) {
        double mx = -INFINITY;
        for (int64_t j = 0; j < sh.rows(); ++j) {
            const int64_t col = (sh.overlap && j >= sh.ratio) ? d + c : c;
            const double s = ss[(size_t) j * item + col];
            if (s > mx) mx = s;
        }
        double den = 0, acc = 0;
        for (int64_t j = 0; j < sh.rows(); ++j) {
            const int64_t col = (sh.overlap && j >= sh.ratio) ? d + c : c;
            const double s = ss[(size_t) j * item + col];
            const double w = (s == -INFINITY) ? 0.0 : std::exp(s - mx);
            den += w;
            acc += w * ks[(size_t) j * item + col];
        }
        (*compressed)[(size_t) c] = den > 0 ? acc / den : 0.0;
    }
    if (sh.overlap) {
        for (int64_t row = 0; row < sh.ratio; ++row)
            for (int64_t col = 0; col < item; ++col) {
                ks[(size_t) row * item + col] = ks[(size_t) (row + sh.ratio) * item + col];
                ss[(size_t) row * item + col] = ss[(size_t) (row + sh.ratio) * item + col];
            }
    }
}

struct Trap {
    bool no_sink = false;
    bool drop_cmp = false;
    bool no_select = false;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) != "--selftest") {
        std::fprintf(stderr, "usage: %s --selftest\n", argv[0]);
        return 2;
    }
    std::printf("dsv4_attn_parity: attention-half layer wiring, 8 tokens, r=4\n");
    ggml_cpu_init();
    // native_mmvq requires an explicit non-null stream.
    static cudaStream_t g_stream = [] { cudaStream_t s; cudaStreamCreate(&s); return s; }();

    // ---- geometry: the real artifact's dimensions; idx_topk 1 so selection discards a real row.
    core::ModelGeometry g;
    g.n_layers = 1;
    g.n_embd = 4096;
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
    g.dsv4.sinkhorn_iters = 20;
    g.dsv4.hc_eps = 1e-6f;
    g.dsv4.norm_eps = 1e-6f;
    g.dsv4.rope_theta = 10000.0;
    g.dsv4.compress_rope_theta = 160000.0;
    g.dsv4.rope_dim = 64;
    g.dsv4.yarn_factor = 16.0;
    g.dsv4.yarn_orig = 65536;
    g.dsv4.yarn_beta_fast = 32;
    g.dsv4.yarn_beta_slow = 1;
    g.dsv4.compress_ratios = {4};
    const int64_t dim = g.n_embd, hd = g.head_dim, rd = g.dsv4.rope_dim, hc = g.hc;
    const int64_t ntok = 8;

    // ---- weights: Q8_0 blobs for the engine (the real pack's native form), their dequantized
    // values for the reference.
    HostW hw;
    core::Dsv4AttnWeights w;
    std::vector<std::vector<uint8_t>> blob(10);
    auto set_gemm = [&](int i, int64_t rows, int64_t cols, double sd, std::vector<float>& refv,
                        auto* dp, int* tp) {
        blob[(size_t) i] = q8_blob(rows, cols, sd);
        refv = q8_deq(blob[(size_t) i], rows * cols);
        void* p = nullptr;
        check(cudaMalloc(&p, blob[(size_t) i].size()), "weights");
        cudaMemcpy(p, blob[(size_t) i].data(), blob[(size_t) i].size(), cudaMemcpyHostToDevice);
        *dp = (const uint8_t*) p;
        *tp = GGML_TYPE_Q8_0;
    };
    // rows = n_out, cols = n_in: the GEMV reads each output row as one contiguous quantized run.
    set_gemm(0, g.dsv4.q_lora, dim, 0.02, hw.q_a, &w.q_a, &w.q_a_type);
    set_gemm(1, g.n_head * hd, g.dsv4.q_lora, 0.02, hw.q_b, &w.q_b, &w.q_b_type);
    set_gemm(2, hd, dim, 0.02, hw.kv, &w.kv, &w.kv_type);
    set_gemm(3, g.dsv4.o_groups * g.dsv4.o_lora, (g.n_head / g.dsv4.o_groups) * hd, 0.02, hw.out_a,
             &w.out_a, &w.out_a_type);
    set_gemm(4, dim, g.dsv4.o_groups * g.dsv4.o_lora, 0.02, hw.out_b, &w.out_b, &w.out_b_type);
    set_gemm(5, 2 * hd, dim, 0.02, hw.comp_kv, &w.comp_kv, &w.comp_kv_type);
    set_gemm(6, 2 * hd, dim, 0.02, hw.comp_gate, &w.comp_gate, &w.comp_gate_type);
    set_gemm(7, g.idx_q_heads * g.idx_key_dim, g.dsv4.q_lora, 0.02, hw.idx_qb, &w.idx_qb, &w.idx_qb_type);
    set_gemm(8, 2 * g.idx_key_dim, dim, 0.02, hw.idx_comp_kv, &w.idx_comp_kv, &w.idx_comp_kv_type);
    set_gemm(9, 2 * g.idx_key_dim, dim, 0.02, hw.idx_comp_gate, &w.idx_comp_gate, &w.idx_comp_gate_type);
    hw.idx_proj = rand_bf16(dim * g.idx_q_heads, 0.02);
    hw.norm = ones_f32(dim, 0.1);
    hw.q_a_norm = ones_f32(g.dsv4.q_lora, 0.1);
    hw.kv_norm = ones_f32(hd, 0.1);
    hw.sinks = rand_f32(g.n_head, 1.0);
    hw.hc_fn = rand_f32((2 + hc) * hc * hc * dim, 0.02);
    hw.hc_base = rand_f32((2 + hc) * hc, 1.0);
    hw.hc_scale = rand_f32(3, 0.0);
    for (auto& v : hw.hc_scale) v = 0.5f;
    hw.ape = rand_f32(4 * 2 * hd, 0.5);
    hw.comp_norm = ones_f32(hd, 0.1);
    hw.idx_ape = rand_f32(4 * 2 * g.idx_key_dim, 0.5);
    hw.idx_comp_norm = ones_f32(g.idx_key_dim, 0.1);

    auto to_dev = [&](const auto& host, auto* dp) {
        using T = std::remove_reference_t<decltype(*dp)>;
        void* p = nullptr;
        check(cudaMalloc(&p, host.size() * sizeof(host[0])), "weights");
        cudaMemcpy(p, host.data(), host.size() * sizeof(host[0]), cudaMemcpyHostToDevice);
        *dp = (T) p;
    };
    to_dev(hw.idx_proj, &w.idx_proj);
    to_dev(hw.norm, &w.norm);
    to_dev(hw.q_a_norm, &w.q_a_norm);
    to_dev(hw.kv_norm, &w.kv_norm);
    to_dev(hw.sinks, &w.sinks);
    to_dev(hw.hc_fn, &w.hc_fn);
    to_dev(hw.hc_base, &w.hc_base);
    to_dev(hw.hc_scale, &w.hc_scale);
    to_dev(hw.ape, &w.ape);
    to_dev(hw.comp_norm, &w.comp_norm);
    to_dev(hw.idx_ape, &w.idx_ape);
    to_dev(hw.idx_comp_norm, &w.idx_comp_norm);

    // ---- engine state
    core::Dsv4State state;
    std::string err;
    if (!state.init(g, 64, err)) {
        std::fprintf(stderr, "state init: %s\n", err.c_str());
        return 1;
    }
    const core::Dsv4LayerState st = state.layer(0);
    const int64_t max_stage = (ntok + 3) / 4;
    const int64_t scratch_bytes = core::dsv4_attn_scratch_bytes(g, 0, max_stage);
    float* scratch = (float*) dalloc((size_t) scratch_bytes);
    std::printf("  scratch %lld bytes\n", (long long) scratch_bytes);

    // ---- reference state
    std::vector<uint16_t> ref_win((size_t) g.dsv4.sliding_window * hd, 0),
        ref_cmp((size_t) (64 / 4) * hd, 0), ref_idx((size_t) (64 / 4) * g.idx_key_dim, 0);
    Shape shc{4, true, hd}, shi{4, true, g.idx_key_dim};
    std::vector<double> ks((size_t) shc.rows() * shc.item(), 0.0),
        ss((size_t) shc.rows() * shc.item(), 0.0), iks((size_t) shi.rows() * shi.item(), 0.0),
        iss((size_t) shi.rows() * shi.item(), 0.0);
    const std::vector<double> f_attn = ref_freqs((int) rd, g.dsv4.rope_theta, g.dsv4.yarn_factor,
                                                 g.dsv4.yarn_orig, g.dsv4.yarn_beta_fast, g.dsv4.yarn_beta_slow);
    const std::vector<double> f_cmp = ref_freqs((int) rd, g.dsv4.compress_rope_theta, g.dsv4.yarn_factor,
                                                g.dsv4.yarn_orig, g.dsv4.yarn_beta_fast, g.dsv4.yarn_beta_slow);

    // One reference token; returns the stream out.  `trap` only changes the gather, so re-running
    // with a trap leaves the pools identical.
    auto ref_token = [&](const std::vector<double>& stream, int64_t pos, const Trap& trap,
                         std::vector<double>& out) {
        const double eps = g.dsv4.norm_eps;
        const int64_t mix_hc = (2 + hc) * hc;
        std::vector<double> mixes((size_t) mix_hc);
        double sq = 0;
        for (double v : stream) sq += v * v;
        const double rms_inv = 1.0 / std::sqrt(sq / (double) (hc * dim) + g.dsv4.hc_eps);
        for (int64_t o = 0; o < mix_hc; ++o) {
            double acc = 0;
            for (int64_t i = 0; i < hc * dim; ++i) acc += stream[(size_t) i] * (double) hw.hc_fn[(size_t) o * hc * dim + i];
            mixes[(size_t) o] = acc * rms_inv;
        }
        std::vector<double> pre, post, comb;
        ref_split(mixes, std::vector<double>(hw.hc_scale.begin(), hw.hc_scale.end()),
                  std::vector<double>(hw.hc_base.begin(), hw.hc_base.end()), hc, g.dsv4.sinkhorn_iters,
                  g.dsv4.hc_eps, pre, post, comb);
        std::vector<double> y((size_t) dim);
        for (int64_t c = 0; c < dim; ++c) {
            double acc = 0;
            for (int64_t h = 0; h < hc; ++h) acc += pre[(size_t) h] * stream[(size_t) (h * dim + c)];
            y[(size_t) c] = acc;
        }
        ref_rms(y, &hw.norm, 1, dim, eps);
        std::vector<double> qr, q, kvv;
        ref_gemv(y, hw.q_a, dim, g.dsv4.q_lora, qr);
        ref_rms(qr, &hw.q_a_norm, 1, g.dsv4.q_lora, eps);
        ref_gemv(qr, hw.q_b, g.dsv4.q_lora, g.n_head * hd, q);
        ref_rms(q, nullptr, g.n_head, hd, eps);
        ref_rope(q, g.n_head, hd, (int) rd, pos, f_attn, false);
        ref_gemv(y, hw.kv, dim, hd, kvv);
        ref_rms(kvv, &hw.kv_norm, 1, hd, eps);
        ref_rope(kvv, 1, hd, (int) rd, pos, f_attn, false);
        {
            std::vector<uint16_t> row = gpu_e4m3(kvv, hd - rd, 64);
            for (int64_t c = hd - rd; c < hd; ++c) row.push_back(bf16_from_f32((float) kvv[(size_t) c]));
            const size_t slot = (size_t) core::dsv4_window_slot(pos, g.dsv4.sliding_window);
            std::copy(row.begin(), row.end(), ref_win.begin() + (ptrdiff_t)(slot * hd));
        }
        std::vector<double> ckv, csc, cmp;
        ref_gemv(y, hw.comp_kv, dim, 2 * hd, ckv);
        ref_gemv(y, hw.comp_gate, dim, 2 * hd, csc);
        ref_decode_token(shc, pos, ckv, csc, hw.ape, ks, ss, &cmp);
        if ((pos + 1) % 4 == 0) {
            ref_rms(cmp, &hw.comp_norm, 1, hd, eps);
            const int64_t bpos = pos + 1 - 4 > 0 ? pos + 1 - 4 : 0;
            ref_rope(cmp, 1, hd, (int) rd, bpos, f_cmp, false);
            std::vector<uint16_t> row = gpu_e4m3(cmp, hd - rd, 64);
            for (int64_t c = hd - rd; c < hd; ++c) row.push_back(bf16_from_f32((float) cmp[(size_t) c]));
            const size_t r = (size_t) core::dsv4_cmp_row(pos, 4);
            std::copy(row.begin(), row.end(), ref_cmp.begin() + (ptrdiff_t)(r * hd));
        }
        // indexer
        std::vector<double> iq, iq_h, iq_w, ikv, isc, icmp;
        ref_gemv(qr, hw.idx_qb, g.dsv4.q_lora, g.idx_q_heads * g.idx_key_dim, iq);
        ref_rope(iq, g.idx_q_heads, g.idx_key_dim, (int) rd, pos, f_cmp, false);
        iq_h = gpu_hadamard(iq, g.idx_q_heads, g.idx_key_dim);
        std::vector<uint16_t> iq_bf = gpu_e2m1(iq_h, g.idx_q_heads * g.idx_key_dim, 32);
        ref_gemv(y, hw.idx_proj, dim, g.idx_q_heads, iq_w);
        const double idx_scale = 1.0 / std::sqrt((double) g.idx_key_dim) / std::sqrt((double) g.idx_q_heads);
        for (auto& v : iq_w) v *= idx_scale;
        ref_gemv(y, hw.idx_comp_kv, dim, 2 * g.idx_key_dim, ikv);
        ref_gemv(y, hw.idx_comp_gate, dim, 2 * g.idx_key_dim, isc);
        ref_decode_token(shi, pos, ikv, isc, hw.idx_ape, iks, iss, &icmp);
        if ((pos + 1) % 4 == 0) {
            ref_rms(icmp, &hw.idx_comp_norm, 1, g.idx_key_dim, eps);
            const int64_t bpos = pos + 1 - 4 > 0 ? pos + 1 - 4 : 0;
            ref_rope(icmp, 1, g.idx_key_dim, (int) rd, bpos, f_cmp, false);
            std::vector<double> ih = gpu_hadamard(icmp, 1, g.idx_key_dim);
            std::vector<uint16_t> row = gpu_e2m1(ih, g.idx_key_dim, 32);
            const size_t r = (size_t) core::dsv4_cmp_row(pos, 4);
            std::copy(row.begin(), row.end(), ref_idx.begin() + (ptrdiff_t)(r * g.idx_key_dim));
        }
        const int64_t valid = core::dsv4_cmp_valid(pos, 4);
        std::vector<int32_t> ids((size_t) valid);
        for (int64_t t = 0; t < valid; ++t) ids[(size_t) t] = (int32_t) t;
        std::vector<double> logits((size_t) valid, -INFINITY);
        for (int64_t t = 0; t < valid; ++t) {
            const uint16_t* k = &ref_idx[(size_t) ids[(size_t) t] * g.idx_key_dim];
            double acc = 0;
            for (int64_t h = 0; h < g.idx_q_heads; ++h) {
                double dot = 0;
                for (int64_t c = 0; c < g.idx_key_dim; ++c)
                    dot += (double) f32_from_bf16(iq_bf[(size_t) h * g.idx_key_dim + c]) *
                           (double) f32_from_bf16(k[c]);
                if (dot < 0.0) dot = 0.0;
                acc += dot * iq_w[(size_t) h];
            }
            logits[(size_t) t] = acc;
        }
        std::vector<int32_t> cmp_ids;
        if (trap.no_select) {
            cmp_ids = ids;
        } else {
            for (int64_t j = 0; j < g.dsv4.idx_topk; ++j) {
                double best = -INFINITY;
                int bidx = -1;
                for (int64_t t = 0; t < valid; ++t) {
                    const double v = logits[(size_t) t];
                    if (v > best || (v == best && v != -INFINITY && (bidx < 0 || t < bidx))) {
                        best = v;
                        bidx = (int) t;
                    }
                }
                cmp_ids.push_back(bidx);
                if (bidx >= 0) logits[(size_t) bidx] = -INFINITY;
            }
        }
        // gather
        std::vector<int32_t> win_ids((size_t) g.dsv4.sliding_window);
        core::dsv4_window_ids(pos, g.dsv4.sliding_window, win_ids.data());
        std::vector<double> o((size_t) g.n_head * hd, 0.0);
        const double scale = 1.0 / std::sqrt((double) hd);
        for (int64_t h = 0; h < g.n_head; ++h) {
            std::vector<double> s;
            std::vector<const uint16_t*> rows;
            for (int32_t id : win_ids) {
                if (id < 0) continue;
                rows.push_back(&ref_win[(size_t) id * hd]);
            }
            if (!trap.drop_cmp)
                for (int32_t id : cmp_ids) {
                    if (id < 0) continue;
                    rows.push_back(&ref_cmp[(size_t) id * hd]);
                }
            for (const uint16_t* row : rows) {
                double dot = 0;
                for (int64_t c = 0; c < hd; ++c)
                    dot += q[(size_t) h * hd + c] * (double) f32_from_bf16(row[c]);
                s.push_back(dot * scale);
            }
            double mx = -INFINITY;
            for (double v : s) mx = std::max(mx, v);
            if (!trap.no_sink) mx = std::max(mx, (double) hw.sinks[(size_t) h]);
            double den = 0;
            std::vector<double> p;
            for (double v : s) {
                const double wt = std::exp(v - mx);
                den += wt;
                p.push_back(wt);
            }
            if (!trap.no_sink) den += std::exp((double) hw.sinks[(size_t) h] - mx);
            if (den == 0) continue;
            for (size_t i = 0; i < rows.size(); ++i)
                for (int64_t c = 0; c < hd; ++c)
                    o[(size_t) h * hd + c] += p[i] / den * (double) f32_from_bf16(rows[i][c]);
        }
        ref_rope(o, g.n_head, hd, (int) rd, pos, f_attn, true);
        std::vector<double> wo((size_t) g.dsv4.o_groups * g.dsv4.o_lora);
        const int64_t hpg = g.n_head / g.dsv4.o_groups;
        for (int64_t gr = 0; gr < g.dsv4.o_groups; ++gr) {
            std::vector<double> slice(o.begin() + (ptrdiff_t)(gr * hpg * hd),
                                     o.begin() + (ptrdiff_t)((gr + 1) * hpg * hd));
            std::vector<double> part;
            ref_gemv(slice, std::vector<float>(
                                 hw.out_a.begin() + (ptrdiff_t)(gr * g.dsv4.o_lora * dim),
                                 hw.out_a.begin() + (ptrdiff_t)((gr + 1) * g.dsv4.o_lora * dim)),
                     hpg * hd, g.dsv4.o_lora, part);
            std::copy(part.begin(), part.end(), wo.begin() + (ptrdiff_t)(gr * g.dsv4.o_lora));
        }
        std::vector<double> y2;
        ref_gemv(wo, hw.out_b, g.dsv4.o_groups * g.dsv4.o_lora, dim, y2);
        out.assign((size_t) hc * dim, 0.0);
        for (int64_t qh = 0; qh < hc; ++qh)
            for (int64_t c = 0; c < dim; ++c) {
                double acc = post[(size_t) qh] * y2[(size_t) c];
                for (int64_t p = 0; p < hc; ++p)
                    acc += comb[(size_t) p * hc + qh] * stream[(size_t) (p * dim + c)];
                out[(size_t) (qh * dim + c)] = acc;
            }
    };

    // ---- run both sides over the tokens
    std::vector<std::vector<double>> engine_out((size_t) ntok), ref_out((size_t) ntok);
    std::vector<std::vector<double>> streams((size_t) ntok);
    for (int64_t pos = 0; pos < ntok; ++pos) {
        streams[(size_t) pos] = std::vector<double>((size_t) hc * dim);
        for (auto& v : streams[(size_t) pos]) v = nrm();
        float* d_in = (float*) dalloc((size_t) hc * dim * 4);
        float* d_out = (float*) dalloc((size_t) hc * dim * 4);
        std::vector<float> in_f((size_t) streams[(size_t) pos].size());
        for (size_t i = 0; i < in_f.size(); ++i) in_f[i] = (float) streams[(size_t) pos][i];
        cudaMemcpy(d_in, in_f.data(), in_f.size() * 4, cudaMemcpyHostToDevice);
        const int64_t valid = core::dsv4_cmp_valid(pos, 4);
        core::dsv4_attn_decode_step(g, 0, w, st, d_in, d_out, pos, valid, scratch, g_stream);
        std::vector<float> out_f(in_f.size());
        cudaMemcpy(out_f.data(), d_out, out_f.size() * 4, cudaMemcpyDeviceToHost);
        engine_out[(size_t) pos].assign(out_f.begin(), out_f.end());
        ref_token(streams[(size_t) pos], pos, Trap{}, ref_out[(size_t) pos]);
        cudaFree(d_in);
        cudaFree(d_out);
        const double r = rel_l1(ref_out[(size_t) pos], engine_out[(size_t) pos]);
        std::printf("  token %lld rel_l1 %.3e\n", (long long) pos, r);
        // The native path quantizes every GEMV activation to q8_1 (one scale per 32 elements);
        // that noise measures ~5e-3 per GEMV standalone and ~2e-2 accumulated through the
        // attention chain.  Wrong wiring is O(1) - see the traps below.
        expect(r < 3e-2, "stream out matches reference");
    }

    // ---- pools must match too (the addressing is only observable through them)
    {
        std::vector<uint16_t> eng_win(ref_win.size()), eng_cmp(ref_cmp.size()), eng_idx(ref_idx.size());
        cudaMemcpy(eng_win.data(), st.window, eng_win.size() * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(eng_cmp.data(), st.cmp, eng_cmp.size() * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(eng_idx.data(), st.idx, eng_idx.size() * 2, cudaMemcpyDeviceToHost);
        auto cmp_pool = [&](const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
            std::vector<double> da(a.size()), db(b.size());
            for (size_t i = 0; i < a.size(); ++i) { da[i] = f32_from_bf16(a[i]); db[i] = f32_from_bf16(b[i]); }
            return rel_l1(da, db);
        };
        expect(cmp_pool(ref_win, eng_win) < 1e-2, "window pool matches");
        expect(cmp_pool(ref_cmp, eng_cmp) < 1e-2, "cmp pool matches");
        expect(cmp_pool(ref_idx, eng_idx) < 1e-2, "idx pool matches");
    }

    // ---- traps: the same reference with one wrong wiring must be observably apart
    std::printf("  traps (last token)\n");
    auto mk_trap = [](bool no_sink, bool drop_cmp, bool no_select) {
        Trap t;
        t.no_sink = no_sink;
        t.drop_cmp = drop_cmp;
        t.no_select = no_select;
        return t;
    };
    const std::pair<const char*, Trap> traps[] = {
        {"no sink", mk_trap(true, false, false)},
        {"window-only (cmp dropped)", mk_trap(false, true, false)},
        {"no indexer top-k", mk_trap(false, false, true)}};
    for (const auto& kv : traps) {
        std::vector<double> t_out;
        ref_token(streams[(size_t) (ntok - 1)], ntok - 1, kv.second, t_out);
        const double r = rel_l1(t_out, engine_out[(size_t) (ntok - 1)]);
        std::printf("    %-30s apart by %.3e\n", kv.first, r);
        expect(r > 1e-1, kv.first);
    }

    std::printf("dsv4_attn_parity: %s\n", failures == 0 ? "ok" : "*** FAIL ***");
    return failures == 0 ? 0 : 1;
}
