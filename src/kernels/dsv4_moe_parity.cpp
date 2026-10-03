// src/kernels/dsv4_moe_parity.cpp - docs/DSV4.md P2: the sqrtsoftplus/hash router and clamped SwiGLU.
//
// The reference is `moe.py::Gate` and `swiglu.py::fused_swiglu` read in float64:
//   scores = sqrt(softplus(raw)); weights gathered from those PRE-bias scores; topk over scores+bias;
//   renormalize; * route_scale.  Hash layers read indices from the tid2eid table instead.
//   swiglu: silu(min(gate, limit)) * clamp(up, -limit, limit).
// Measured config: 256 experts, top 6, route_scale 1.5, clamp 10.0.  Wrong readings asserted
// observably apart: dropping the bias, skipping the renormalization, softmax instead of sqrtsoftplus,
// gathering weights from the POST-bias scores, and an unclamped swiglu.
#include "strata/kernels/dsv4_moe.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

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

std::vector<double> to_d(const std::vector<float>& v) { return std::vector<double>(v.begin(), v.end()); }

double sqrtsoftplus(double x) { return std::sqrt(x > 20.0 ? x : std::log1p(std::exp(x))); }

// reference router; variants for the traps
void ref_router(const std::vector<double>& raw, const std::vector<double>& bias, int64_t n, int64_t ne,
                int64_t topk, double scale, std::vector<double>& weights, std::vector<int64_t>& indices,
                bool use_bias = true, bool renorm = true, bool softmax_instead = false,
                bool gather_biased = false) {
    weights.assign((size_t) n * topk, 0.0);
    indices.assign((size_t) n * topk, 0);
    for (int64_t t = 0; t < n; ++t) {
        std::vector<double> orig((size_t) ne), biased((size_t) ne);
        for (int64_t e = 0; e < ne; ++e) {
            const double x = raw[(size_t) t * ne + e];
            orig[(size_t) e] = softmax_instead ? 1.0 / (1.0 + std::exp(-x)) : sqrtsoftplus(x);
            biased[(size_t) e] = orig[(size_t) e] + (use_bias ? bias[(size_t) e] : 0.0);
        }
        double sum = 0;
        for (int64_t j = 0; j < topk; ++j) {
            int best = 0;
            double bv = biased[0];
            for (int e = 1; e < (int) ne; ++e)
                if (biased[(size_t) e] > bv) { bv = biased[(size_t) e]; best = e; }
            biased[(size_t) best] = -INFINITY;
            indices[(size_t) t * topk + j] = best;
            weights[(size_t) t * topk + j] = (gather_biased ? bv : orig[(size_t) best]);
            sum += weights[(size_t) t * topk + j];
        }
        if (renorm)
            for (int64_t j = 0; j < topk; ++j) weights[(size_t) t * topk + j] *= scale / sum;
        else
            for (int64_t j = 0; j < topk; ++j) weights[(size_t) t * topk + j] *= scale;
    }
}

void ref_hash(const std::vector<double>& raw, const std::vector<int32_t>& tid2eid,
              const std::vector<int64_t>& tokens, int64_t n, int64_t topk, double scale,
              std::vector<double>& weights, std::vector<int64_t>& indices) {
    weights.assign((size_t) n * topk, 0.0);
    indices.assign((size_t) n * topk, 0);
    for (int64_t t = 0; t < n; ++t) {
        double sum = 0;
        for (int64_t j = 0; j < topk; ++j) {
            const int32_t e = tid2eid[(size_t) tokens[(size_t) t] * topk + j];
            indices[(size_t) t * topk + j] = e;
            weights[(size_t) t * topk + j] = sqrtsoftplus(raw[(size_t) t * raw.size() / n + e]);
            sum += weights[(size_t) t * topk + j];
        }
        for (int64_t j = 0; j < topk; ++j) weights[(size_t) t * topk + j] *= scale / sum;
    }
}

int failures = 0;
void expect(bool ok, const char* what) {
    std::printf("    %-46s %s\n", what, ok ? "ok" : "*** FAIL ***");
    if (!ok) ++failures;
}

void run_score_case(int64_t n, int64_t ne, int64_t topk, double scale, std::mt19937_64& rng) {
    std::printf("    case n=%lld experts=%lld topk=%lld\n", (long long) n, (long long) ne, (long long) topk);
    std::normal_distribution<double> nd(0.0, 2.0), bd(0.0, 0.5);
    std::vector<double> raw((size_t) n * ne), bias((size_t) ne);
    for (auto& x : raw) x = nd(rng);
    for (auto& x : bias) x = bd(rng);
    std::vector<float> d_raw, d_bias;
    for (double x : raw) d_raw.push_back((float) x);
    for (double x : bias) d_bias.push_back((float) x);

    std::vector<double> want_w;
    std::vector<int64_t> want_i;
    ref_router(raw, bias, n, ne, topk, scale, want_w, want_i);

    float *d_rw, *d_rb, *d_gw;
    int32_t* d_gi;
    check(cudaMalloc(&d_rw, d_raw.size() * 4), "alloc raw");
    check(cudaMalloc(&d_rb, d_bias.size() * 4), "alloc bias");
    check(cudaMalloc(&d_gw, (size_t) n * topk * 4), "alloc got w");
    check(cudaMalloc(&d_gi, (size_t) n * topk * 4), "alloc got i");
    check(cudaMemcpy(d_rw, d_raw.data(), d_raw.size() * 4, cudaMemcpyHostToDevice), "copy raw");
    check(cudaMemcpy(d_rb, d_bias.data(), d_bias.size() * 4, cudaMemcpyHostToDevice), "copy bias");
    std::vector<float> got_w((size_t) n * topk);
    std::vector<int32_t> got_i((size_t) n * topk);
    strata::kernels::dsv4_router_score(d_rw, d_rb, d_gw, d_gi, n, ne, topk, (float) scale, nullptr);
    check(cudaMemcpy(got_w.data(), d_gw, got_w.size() * 4, cudaMemcpyDeviceToHost), "read w");
    check(cudaMemcpy(got_i.data(), d_gi, got_i.size() * 4, cudaMemcpyDeviceToHost), "read i");
    bool idx_ok = true;
    for (size_t j = 0; j < want_i.size(); ++j)
        if (want_i[j] != got_i[j]) idx_ok = false;
    expect(rel_l1(want_w, to_d(got_w)) < 1e-6 && idx_ok, "sqrtsoftplus+topk+renorm vs reference");

    // traps
    std::vector<double> t1, t2, t3, t4;
    std::vector<int64_t> ti;
    ref_router(raw, bias, n, ne, topk, scale, t1, ti, false);
    ref_router(raw, bias, n, ne, topk, scale, t2, ti, true, false);
    ref_router(raw, bias, n, ne, topk, scale, t3, ti, true, true, true);
    ref_router(raw, bias, n, ne, topk, scale, t4, ti, true, true, false, true);
    const double d1 = rel_l1(want_w, t1), d2 = rel_l1(want_w, t2), d3 = rel_l1(want_w, t3),
               d4 = rel_l1(want_w, t4);
    std::printf("    %-46s no-bias %.2e, no-renorm %.2e, softmax %.2e, post-bias gather %.2e\n",
                "wrong readings observably apart", d1, d2, d3, d4);
    expect(d1 > 1e-3 && d2 > 1e-3 && d3 > 1e-3 && d4 > 1e-3, "traps observably apart");
    cudaFree(d_rw);
    cudaFree(d_rb);
    cudaFree(d_gw);
    cudaFree(d_gi);
}

void run_hash_case(int64_t n, int64_t ne, int64_t topk, int64_t vocab, double scale, std::mt19937_64& rng) {
    std::normal_distribution<double> nd(0.0, 2.0);
    std::vector<double> raw((size_t) n * ne);
    for (auto& x : raw) x = nd(rng);
    std::vector<float> d_raw;
    for (double x : raw) d_raw.push_back((float) x);
    std::vector<int32_t> table((size_t) vocab * topk);
    std::uniform_int_distribution<int32_t> ed(0, (int32_t) ne - 1);
    for (auto& x : table) x = ed(rng);
    std::vector<int64_t> tokens((size_t) n);
    std::uniform_int_distribution<int64_t> td(0, vocab - 1);
    for (auto& x : tokens) x = td(rng);

    std::vector<double> want_w;
    std::vector<int64_t> want_i;
    ref_hash(raw, table, tokens, n, topk, scale, want_w, want_i);

    float* d_rw;
    int32_t *d_tab, *d_gi;
    int64_t* d_tok;
    float* d_gw;
    check(cudaMalloc(&d_rw, d_raw.size() * 4), "alloc raw");
    check(cudaMalloc(&d_tab, table.size() * 4), "alloc table");
    check(cudaMalloc(&d_tok, tokens.size() * 8), "alloc tokens");
    check(cudaMalloc(&d_gw, (size_t) n * topk * 4), "alloc got w");
    check(cudaMalloc(&d_gi, (size_t) n * topk * 4), "alloc got i");
    check(cudaMemcpy(d_rw, d_raw.data(), d_raw.size() * 4, cudaMemcpyHostToDevice), "copy raw");
    check(cudaMemcpy(d_tab, table.data(), table.size() * 4, cudaMemcpyHostToDevice), "copy table");
    check(cudaMemcpy(d_tok, tokens.data(), tokens.size() * 8, cudaMemcpyHostToDevice), "copy tokens");
    std::vector<float> got_w((size_t) n * topk);
    std::vector<int32_t> got_i((size_t) n * topk);
    strata::kernels::dsv4_router_hash(d_rw, d_tab, d_tok, d_gw, d_gi, n, ne, topk, (float) scale, nullptr);
    check(cudaMemcpy(got_w.data(), d_gw, got_w.size() * 4, cudaMemcpyDeviceToHost), "read w");
    check(cudaMemcpy(got_i.data(), d_gi, got_i.size() * 4, cudaMemcpyDeviceToHost), "read i");
    bool idx_ok = true;
    for (size_t j = 0; j < want_i.size(); ++j)
        if (want_i[j] != got_i[j]) idx_ok = false;
    expect(rel_l1(want_w, to_d(got_w)) < 1e-6 && idx_ok, "tid2eid routing vs reference");
    cudaFree(d_rw);
    cudaFree(d_tab);
    cudaFree(d_tok);
    cudaFree(d_gw);
    cudaFree(d_gi);
}

void run_swiglu_case(int64_t n, float limit, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> ud(-30.0, 30.0);
    std::vector<double> g((size_t) n), u((size_t) n);
    for (auto& x : g) x = ud(rng);
    for (auto& x : u) x = ud(rng);
    std::vector<float> d_g, d_u;
    for (double x : g) d_g.push_back((float) x);
    for (double x : u) d_u.push_back((float) x);
    std::vector<double> want((size_t) n), unclamped((size_t) n);
    for (int64_t i = 0; i < n; ++i) {
        const double gc = std::min(g[(size_t) i], (double) limit),
                     uc = std::max(-(double) limit, std::min(u[(size_t) i], (double) limit));
        want[(size_t) i] = gc / (1.0 + std::exp(-gc)) * uc;
        unclamped[(size_t) i] = g[(size_t) i] / (1.0 + std::exp(-g[(size_t) i])) * u[(size_t) i];
    }
    float *d_gg, *d_uu, *d_out;
    check(cudaMalloc(&d_gg, d_g.size() * 4), "alloc g");
    check(cudaMalloc(&d_uu, d_u.size() * 4), "alloc u");
    check(cudaMalloc(&d_out, d_g.size() * 4), "alloc out");
    check(cudaMemcpy(d_gg, d_g.data(), d_g.size() * 4, cudaMemcpyHostToDevice), "copy g");
    check(cudaMemcpy(d_uu, d_u.data(), d_u.size() * 4, cudaMemcpyHostToDevice), "copy u");
    std::vector<float> got(d_g.size());
    strata::kernels::dsv4_swiglu(d_gg, d_uu, d_out, n, limit, nullptr);
    check(cudaMemcpy(got.data(), d_out, got.size() * 4, cudaMemcpyDeviceToHost), "read out");
    expect(rel_l1(want, to_d(got)) < 1e-6, "clamped swiglu vs reference");
    const double dnc = rel_l1(want, unclamped);
    std::printf("    %-46s unclamped %.2e\n", "wrong readings observably apart", dnc);
    expect(dnc > 1e-3, "clamps observably matter");
    cudaFree(d_gg);
    cudaFree(d_uu);
    cudaFree(d_out);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) != "--selftest") {
        std::fprintf(stderr, "usage: %s --selftest\n", argv[0]);
        return 2;
    }
    std::mt19937_64 rng(20261003);
    std::printf("dsv4_moe_parity: sqrtsoftplus/hash router + clamped swiglu\n");
    run_score_case(1, 256, 6, 1.5, rng);
    run_score_case(3, 256, 6, 1.5, rng);
    run_hash_case(2, 256, 6, 1000, 1.5, rng);
    run_swiglu_case(4096, 10.0f, rng);
    std::printf("dsv4_moe_parity: %s\n", failures == 0 ? "ok" : "*** FAIL ***");
    return failures == 0 ? 0 : 1;
}
