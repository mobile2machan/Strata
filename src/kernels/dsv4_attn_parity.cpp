// src/kernels/dsv4_attn_parity.cpp - docs/DSV4.md P2: the tiered-pool gather + sink.
//
// The reference is the semantics `sparse_attn.py` states in its own docstring: gather the real rows
// (window-first, -1 skipped), softmax over `scale * q.k` with the SINK as a null key - logit
// `sinks[h]`, zero value - in the denominator.  Three readings are asserted observably apart:
//
//   1. no sink term.  The sink is invisible in `o`'s SHAPE and only shrinks every weight by the same
//      factor against the rest - a plausible attention output, wrong.
//   2. -1 ids read row 0 instead of being skipped.  Padding is half the fixture; a kernel that reads
//      it attends tokens that do not exist.
//   3. the softmax scale dropped.  `head_dim ** -0.5` over a 512-wide row is not a rounding detail.
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/dsv4_attn.hpp"

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

double rel_l1(const std::vector<double>& a, const std::vector<double>& b) {
    double d = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += std::fabs(a[i] - b[i]);
        m += std::fabs(a[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
}

void ref_attn(int64_t n_heads, int64_t d, const std::vector<float>& q, const std::vector<uint16_t>& win,
              const std::vector<uint16_t>& cmp, const std::vector<int32_t>& win_ids,
              const std::vector<int32_t>& cmp_ids, const std::vector<float>* sinks, double scale,
              std::vector<double>& o, bool no_sink = false, bool minus_one_reads_zero = false,
              int64_t window_cap = 128) {
    o.assign((size_t) n_heads * d, 0.0);
    for (int64_t h = 0; h < n_heads; ++h) {
        std::vector<double> s;
        std::vector<const uint16_t*> rows;
        for (int32_t id : win_ids) {
            if (id < 0 && !minus_one_reads_zero) continue;
            if (id < 0) id = 0;
            rows.push_back(&win[(size_t) id * d]);
        }
        for (int32_t id : cmp_ids) {
            if (id < 0 && !minus_one_reads_zero) continue;
            if (id < 0) id = 0;
            rows.push_back(&cmp[(size_t) id * d]);
        }
        (void) window_cap;
        for (const uint16_t* row : rows) {
            double dot = 0;
            for (int64_t c = 0; c < d; ++c) dot += (double) q[(size_t) h * d + c] * f32_from_bf16(row[c]);
            s.push_back(dot * scale);
        }
        double mx = -INFINITY;
        for (double v : s) mx = std::max(mx, v);
        if (!no_sink && sinks) mx = std::max(mx, (double) (*sinks)[(size_t) h]);
        double den = 0;
        std::vector<double> p;
        for (double v : s) {
            const double w = std::exp(v - mx);
            den += w;
            p.push_back(w);
        }
        if (!no_sink && sinks) den += std::exp((double) (*sinks)[(size_t) h] - mx);
        if (den == 0) continue;
        for (size_t i = 0; i < rows.size(); ++i)
            for (int64_t c = 0; c < d; ++c)
                o[(size_t) h * d + c] += p[i] / den * f32_from_bf16(rows[i][c]);
    }
}

void run_case(int64_t n_heads, int64_t d, int64_t n_win, int64_t n_cmp, bool with_sink, uint32_t seed,
              int& bad) {
    std::printf("  heads %lld d %lld win %lld cmp %lld sink %d\n", (long long) n_heads, (long long) d,
                (long long) n_win, (long long) n_cmp, (int) with_sink);
    const int64_t wcap = 128, cmp_cap = n_cmp > 0 ? n_cmp : 1;
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> q((size_t) n_heads * d), sinks((size_t) n_heads);
    std::vector<uint16_t> win((size_t) wcap * d), cmp((size_t) cmp_cap * d);
    for (auto& v : q) v = g(rng);
    for (auto& v : sinks) v = g(rng);
    for (auto& v : win) v = bf16_from_f32(g(rng));
    for (auto& v : cmp) v = bf16_from_f32(g(rng));
    std::vector<int32_t> win_ids((size_t) n_win), cmp_ids((size_t) n_cmp);
    for (int64_t i = 0; i < n_win; ++i) win_ids[(size_t) i] = (i % 11 == 7) ? -1 : (int32_t) i;
    for (int64_t i = 0; i < n_cmp; ++i) cmp_ids[(size_t) i] = (i % 13 == 5) ? -1 : (int32_t) i;
    const float scale = 1.0f / std::sqrt((float) d);

    float *d_q = nullptr, *d_o = nullptr, *d_sink = nullptr;
    uint16_t *d_win = nullptr, *d_cmp = nullptr;
    int32_t *d_wid = nullptr, *d_cid = nullptr;
    check(cudaMalloc(&d_q, q.size() * 4), "q");
    check(cudaMalloc(&d_o, q.size() * 4), "o");
    check(cudaMalloc(&d_win, win.size() * 2), "win");
    check(cudaMalloc(&d_cmp, cmp.size() * 2), "cmp");
    check(cudaMalloc(&d_wid, win_ids.size() * 4), "wid");
    check(cudaMalloc(&d_cid, cmp_ids.size() * 4), "cid");
    if (with_sink) check(cudaMalloc(&d_sink, sinks.size() * 4), "sink");
    check(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice), "mq");
    check(cudaMemcpy(d_win, win.data(), win.size() * 2, cudaMemcpyHostToDevice), "mwin");
    check(cudaMemcpy(d_cmp, cmp.data(), cmp.size() * 2, cudaMemcpyHostToDevice), "mcmp");
    check(cudaMemcpy(d_wid, win_ids.data(), win_ids.size() * 4, cudaMemcpyHostToDevice), "mwid");
    check(cudaMemcpy(d_cid, cmp_ids.data(), cmp_ids.size() * 4, cudaMemcpyHostToDevice), "mcid");
    if (with_sink) check(cudaMemcpy(d_sink, sinks.data(), sinks.size() * 4, cudaMemcpyHostToDevice), "msink");

    strata::kernels::Dsv4AttnPools pools;
    pools.window = d_win;
    pools.cmp = d_cmp;
    strata::kernels::dsv4_attn_decode(d_q, d_wid, n_win, d_cid, n_cmp, with_sink ? d_sink : nullptr, scale,
                                      pools, wcap, n_heads, d, d_o, nullptr);
    std::vector<float> got(q.size());
    check(cudaMemcpy(got.data(), d_o, got.size() * 4, cudaMemcpyDeviceToHost), "go");
    std::vector<double> want;
    ref_attn(n_heads, d, q, win, cmp, win_ids, cmp_ids, with_sink ? &sinks : nullptr, scale, want);
    std::vector<double> gd(got.begin(), got.end());
    const double rel = rel_l1(want, gd);
    std::printf("    %-46s rel %.3e\n", "gather vs the reference", rel);
    if (!(rel <= 1e-5)) { std::printf("      *** over 1e-5 ***\n"); ++bad; }

    // traps
    std::vector<double> t1, t2, t3;
    ref_attn(n_heads, d, q, win, cmp, win_ids, cmp_ids, with_sink ? &sinks : nullptr, scale, t1, true);
    ref_attn(n_heads, d, q, win, cmp, win_ids, cmp_ids, with_sink ? &sinks : nullptr, scale, t2, false, true);
    ref_attn(n_heads, d, q, win, cmp, win_ids, cmp_ids, with_sink ? &sinks : nullptr, 1.0, t3);
    const bool t1_live = !with_sink || rel_l1(want, t1) > 1e-3;
    std::printf("    %-46s no-sink %.2e, -1-as-row0 %.2e, no-scale %.2e\n", "wrong readings observably apart",
                with_sink ? rel_l1(want, t1) : 0.0, rel_l1(want, t2), rel_l1(want, t3));
    if (!(t1_live && rel_l1(want, t2) > 1e-3 && rel_l1(want, t3) > 1e-3)) {
        std::printf("      *** a trap is not observable ***\n");
        ++bad;
    }

    cudaFree(d_q); cudaFree(d_o); cudaFree(d_win); cudaFree(d_cmp);
    cudaFree(d_wid); cudaFree(d_cid);
    if (d_sink) cudaFree(d_sink);
}

// The prefill shape: n queries, each with its own causal window list and its own cmp list.  The
// oracle is the decode kernel's own reference run once per query with that query's lists - the
// two kernels must agree exactly, and a query that got its neighbour's list must not.
void run_prefill_case(int64_t n, int64_t n_heads, int64_t d, int64_t n_win, int64_t n_cmp,
                      uint32_t seed, int& bad) {
    std::printf("  prefill n %lld heads %lld d %lld win %lld cmp %lld\n", (long long) n,
                (long long) n_heads, (long long) d, (long long) n_win, (long long) n_cmp);
    const int64_t wcap = n_win, cmp_cap = n_cmp > 0 ? n_cmp : 1;
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> q((size_t) n * n_heads * d), sinks((size_t) n_heads);
    std::vector<uint16_t> win((size_t) wcap * d), cmp((size_t) cmp_cap * d);
    for (auto& v : q) v = g(rng);
    for (auto& v : sinks) v = g(rng);
    for (auto& v : win) v = bf16_from_f32(g(rng));
    for (auto& v : cmp) v = bf16_from_f32(g(rng));
    std::vector<int32_t> win_ids((size_t) n * n_win, -1), cmp_ids((size_t) n * n_cmp, -1);
    for (int64_t t = 0; t < n; ++t) {
        for (int64_t i = 0; i <= t && i < n_win; ++i) win_ids[(size_t) t * n_win + i] = (int32_t) i;
        const int64_t valid = (t + 1) / 4;
        for (int64_t i = 0; i < valid && i < n_cmp; ++i) cmp_ids[(size_t) t * n_cmp + i] = (int32_t) i;
    }
    const float scale = 1.0f / std::sqrt((float) d);

    float *d_q = nullptr, *d_o = nullptr, *d_sink = nullptr;
    uint16_t *d_win = nullptr, *d_cmp = nullptr;
    int32_t *d_wid = nullptr, *d_cid = nullptr;
    check(cudaMalloc(&d_q, q.size() * 4), "pq");
    check(cudaMalloc(&d_o, q.size() * 4), "po");
    check(cudaMalloc(&d_win, win.size() * 2), "pwin");
    check(cudaMalloc(&d_cmp, cmp.size() * 2), "pcmp");
    check(cudaMalloc(&d_wid, win_ids.size() * 4), "pwid");
    check(cudaMalloc(&d_cid, cmp_ids.size() * 4), "pcid");
    check(cudaMalloc(&d_sink, sinks.size() * 4), "psink");
    check(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice), "pmq");
    check(cudaMemcpy(d_win, win.data(), win.size() * 2, cudaMemcpyHostToDevice), "pmwin");
    check(cudaMemcpy(d_cmp, cmp.data(), cmp.size() * 2, cudaMemcpyHostToDevice), "pmcmp");
    check(cudaMemcpy(d_wid, win_ids.data(), win_ids.size() * 4, cudaMemcpyHostToDevice), "pmwid");
    check(cudaMemcpy(d_cid, cmp_ids.data(), cmp_ids.size() * 4, cudaMemcpyHostToDevice), "pmcid");
    check(cudaMemcpy(d_sink, sinks.data(), sinks.size() * 4, cudaMemcpyHostToDevice), "pmsink");

    strata::kernels::Dsv4AttnPools pools;
    pools.window = d_win;
    pools.cmp = d_cmp;
    strata::kernels::dsv4_attn_prefill(d_q, d_wid, n_win, d_cid, n_cmp, d_sink, scale, pools, wcap, n,
                                       n_heads, d, d_o, nullptr);
    std::vector<float> got(q.size());
    check(cudaMemcpy(got.data(), d_o, got.size() * 4, cudaMemcpyDeviceToHost), "pgo");

    double worst = 0, shifted = 0;
    for (int64_t t = 0; t < n; ++t) {
        std::vector<float> qt(q.begin() + (size_t) t * n_heads * d,
                             q.begin() + (size_t) (t + 1) * n_heads * d);
        std::vector<int32_t> w(win_ids.begin() + (size_t) t * n_win,
                               win_ids.begin() + (size_t) (t + 1) * n_win);
        std::vector<int32_t> c(cmp_ids.begin() + (size_t) t * n_cmp,
                               cmp_ids.begin() + (size_t) (t + 1) * n_cmp);
        std::vector<double> want;
        ref_attn(n_heads, d, qt, win, cmp, w, c, &sinks, scale, want);
        std::vector<double> gd(got.begin() + (size_t) t * n_heads * d,
                               got.begin() + (size_t) (t + 1) * n_heads * d);
        worst = std::max(worst, rel_l1(want, gd));
        if (t + 1 < n) {  // the neighbour's list must be observably different
            std::vector<double> w2;
            ref_attn(n_heads, d, qt, win, cmp,
                     std::vector<int32_t>(win_ids.begin() + (size_t) (t + 1) * n_win,
                                          win_ids.begin() + (size_t) (t + 2) * n_win),
                     std::vector<int32_t>(cmp_ids.begin() + (size_t) (t + 1) * n_cmp,
                                          cmp_ids.begin() + (size_t) (t + 2) * n_cmp),
                     &sinks, scale, w2);
            shifted = std::max(shifted, rel_l1(w2, gd));
        }
    }
    std::printf("    %-46s rel %.3e, neighbour-list %.2e\n", "per-query rows vs the reference", worst,
                shifted);
    if (!(worst <= 1e-5)) { std::printf("      *** over 1e-5 ***\n"); ++bad; }
    if (!(shifted > 1e-3)) { std::printf("      *** a swapped list is not observable ***\n"); ++bad; }

    cudaFree(d_q); cudaFree(d_o); cudaFree(d_win); cudaFree(d_cmp);
    cudaFree(d_wid); cudaFree(d_cid); cudaFree(d_sink);
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: dsv4_attn_parity [--selftest]\n"); return 2; }
    }
    int bad = 0;
    run_case(64, 512, 128, 64, true, 21, bad);   // r=128 layer: full window + all valid blocks
    run_case(64, 512, 128, 0, true, 22, bad);     // window-only layer
    run_case(64, 512, 128, 16, false, 23, bad);   // no-sink path (sinks null)
    run_prefill_case(8, 8, 512, 8, 4, 24, bad);   // prefill: causal per-query lists
    std::printf("dsv4_attn_parity: %s\n", bad ? "*** FAIL ***" : "ok");
    return bad ? 1 : 0;
}
