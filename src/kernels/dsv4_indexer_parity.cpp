// src/kernels/dsv4_indexer_parity.cpp - docs/DSV4.md P2: the indexer's head-reduced logits.
//
// The reference is `indexer.py`'s own equivalence: `einsum("bhd,btd->bht", q, k).relu_() *
// weights[..., None]` summed over heads, -inf past the live count.  Three readings are asserted
// observably apart:
//
//   1. no relu.  Random q/k make most head dots negative; without the rectifier they CANCEL across
//      heads instead of dropping out, and the block ordering the top-k sees changes.
//   2. weights ignored (every head weight 1).  The per-head gate is the indexer's learned selection
//      signal; a flat sum is a different, plausible score.
//   3. the valid bound dropped.  Scoring staged-but-not-live blocks gives the top-k real-looking
//      logits for keys that do not exist yet.
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/dsv4_indexer.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

using strata::kernels::bf16_from_f32;
using strata::kernels::f32_from_bf16;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void ref_logits(int64_t H, int64_t d, const std::vector<uint16_t>& q, const std::vector<float>& w,
                const std::vector<uint16_t>& pool, const std::vector<int32_t>& ids, int64_t n_stage,
                int64_t valid, std::vector<double>& out, bool no_relu = false, bool flat_weights = false,
                bool ignore_valid = false) {
    out.assign((size_t) n_stage, -INFINITY);
    for (int64_t t = 0; t < n_stage; ++t) {
        if (!ignore_valid && t >= valid) continue;
        if (ids[(size_t) t] < 0) continue;
        const uint16_t* k = &pool[(size_t) ids[(size_t) t] * d];
        double acc = 0;
        for (int64_t h = 0; h < H; ++h) {
            double dot = 0;
            for (int64_t c = 0; c < d; ++c)
                dot += (double) f32_from_bf16(q[(size_t) h * d + c]) * f32_from_bf16(k[c]);
            if (!no_relu && dot < 0.0) dot = 0.0;
            acc += dot * (flat_weights ? 1.0 : (double) w[(size_t) h]);
        }
        out[(size_t) t] = acc;
    }
}

double compare(const std::vector<double>& a, const std::vector<float>& b) {
    double d = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::isinf(a[i]) || std::isinf(b[i])) {
            if (std::isinf(a[i]) != std::isinf(b[i])) return 1.0;
            continue;
        }
        d += std::fabs(a[i] - (double) b[i]);
        m += std::fabs(a[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
}

void run_case(int64_t H, int64_t d, int64_t n_stage, int64_t valid, uint32_t seed, int& bad) {
    std::printf("  heads %lld d %lld n_stage %lld valid %lld\n", (long long) H, (long long) d,
                (long long) n_stage, (long long) valid);
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<uint16_t> q((size_t) H * d), pool((size_t) n_stage * d);
    std::vector<float> w((size_t) H), ws((size_t) H);
    for (auto& v : q) v = bf16_from_f32(g(rng));
    for (auto& v : pool) v = bf16_from_f32(g(rng));
    for (auto& v : w) v = g(rng) * 0.3f;
    std::vector<int32_t> ids((size_t) n_stage);
    for (int64_t t = 0; t < n_stage; ++t) ids[(size_t) t] = (t % 17 == 3) ? -1 : (int32_t) t;

    uint16_t *d_q = nullptr, *d_pool = nullptr;
    float *d_w = nullptr, *d_out = nullptr;
    int32_t* d_ids = nullptr;
    check(cudaMalloc(&d_q, q.size() * 2), "q");
    check(cudaMalloc(&d_pool, pool.size() * 2), "pool");
    check(cudaMalloc(&d_w, w.size() * 4), "w");
    check(cudaMalloc(&d_out, n_stage * 4), "out");
    check(cudaMalloc(&d_ids, ids.size() * 4), "ids");
    check(cudaMemcpy(d_q, q.data(), q.size() * 2, cudaMemcpyHostToDevice), "mq");
    check(cudaMemcpy(d_pool, pool.data(), pool.size() * 2, cudaMemcpyHostToDevice), "mpool");
    check(cudaMemcpy(d_w, w.data(), w.size() * 4, cudaMemcpyHostToDevice), "mw");
    check(cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "mids");
    strata::kernels::dsv4_indexer_logits(d_q, d_w, d_pool, d_ids, n_stage, valid, H, d, d_out, nullptr);
    std::vector<float> got((size_t) n_stage);
    check(cudaMemcpy(got.data(), d_out, n_stage * 4, cudaMemcpyDeviceToHost), "mo");

    std::vector<double> want;
    ref_logits(H, d, q, w, pool, ids, n_stage, valid, want);
    const double rel = compare(want, got);
    std::printf("    %-46s rel %.3e\n", "logits vs the reference", rel);
    if (!(rel <= 1e-5)) { std::printf("      *** over 1e-5 ***\n"); ++bad; }

    std::vector<double> t1, t2, t3;
    ref_logits(H, d, q, w, pool, ids, n_stage, valid, t1, true);
    ref_logits(H, d, q, w, pool, ids, n_stage, valid, t2, false, true);
    ref_logits(H, d, q, w, pool, ids, n_stage, valid, t3, false, false, true);
    auto cmpd = [](const std::vector<double>& a, const std::vector<double>& b) {
        double d = 0, m = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::isinf(a[i]) || std::isinf(b[i])) {
                if (std::isinf(a[i]) != std::isinf(b[i])) return 1.0;
                continue;
            }
            d += std::fabs(a[i] - b[i]);
            m += std::fabs(a[i]);
        }
        return d / (m > 1e-30 ? m : 1e-30);
    };
    // the ignore-valid trap needs a staged-but-not-live tail to hide behind; a fully-live fixture
    // cannot observe it and saying so is better than a vacuous pass
    const bool valid_trap_live = valid < n_stage;
    std::printf("    %-46s no-relu %.2e, flat-weights %.2e, ignore-valid %s\n",
                "wrong readings observably apart", cmpd(want, t1), cmpd(want, t2),
                valid_trap_live ? (std::to_string(cmpd(want, t3))).c_str() : "n/a (all live)");
    if (!(cmpd(want, t1) > 1e-3 && cmpd(want, t2) > 1e-3 && (!valid_trap_live || cmpd(want, t3) > 1e-3))) {
        std::printf("      *** a trap is not observable ***\n");
        ++bad;
    }

    cudaFree(d_q); cudaFree(d_pool); cudaFree(d_w); cudaFree(d_out); cudaFree(d_ids);
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: dsv4_indexer_parity [--selftest]\n"); return 2; }
    }
    int bad = 0;
    run_case(64, 128, 200, 150, 31, bad);  // the real index shape: 64 heads, 128-wide keys
    run_case(64, 128, 64, 64, 32, bad);    // fully live
    std::printf("dsv4_indexer_parity: %s\n", bad ? "*** FAIL ***" : "ok");
    return bad ? 1 : 0;
}
