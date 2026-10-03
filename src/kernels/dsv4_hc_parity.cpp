// src/kernels/dsv4_hc_parity.cpp - docs/DSV4.md P2: the mHC split/Sinkhorn and stream mixing.
//
// The reference is `sinkhorn.py` + `hc.py` read in float64, operation order included: row-softmax,
// +eps, ONE column normalization, then (iters-1) rounds of row-then-column, each dividing by the sum
// PLUS eps (not after dividing by it).  The real config is hc=4, iters=20, eps=1e-6.  Four readings
// are asserted observably apart:
//
//   1. post_combine reducing over comb's SECOND axis instead of the first.  comb is doubly stochastic
//      but NOT symmetric; the axis swap is invisible in every shape.
//   2. Sinkhorn starting with a row normalization (no initial column pass).  The fixed point is the
//      same; the 20-iterate path is not.
//   3. the post gate without its factor of 2.
//   4. pre_combine with uniform weights (ignoring pre).
#include "strata/kernels/dsv4_hc.hpp"

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

void ref_split(const std::vector<double>& mixes, const std::vector<double>& scale,
               const std::vector<double>& base, int64_t hc, int64_t iters, double eps,
               std::vector<double>& pre, std::vector<double>& post, std::vector<double>& comb) {
    pre.resize((size_t) hc);
    post.resize((size_t) hc);
    comb.resize((size_t) hc * hc);
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
    // the reference's order: ONE column normalization, then (iters-1) rounds of row-then-column.
    // (At the real iters=20 the order of the FIRST pass is numerically irrelevant - the iteration has
    // converged either way, ~1e-14 apart; the iteration COUNT is what matters, and the trap below
    // measures that instead.)
    norm_cols();
    for (int64_t it = 1; it < iters; ++it) { norm_rows(); norm_cols(); }
    for (int64_t p = 0; p < hc; ++p)
        for (int64_t q = 0; q < hc; ++q) comb[(size_t) p * hc + q] = c[p][q];
}

void run_case(int64_t hc, int64_t iters, double eps, int64_t M, int64_t d, uint32_t seed, int& bad) {
    std::printf("  hc %lld iters %lld eps %.0e M %lld d %lld\n", (long long) hc, (long long) iters, eps,
                (long long) M, (long long) d);
    const int64_t mix_hc = (2 + hc) * hc;
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> mixes((size_t) M * mix_hc), scale(3), base((size_t) mix_hc);
    std::vector<float> x((size_t) M * hc * d), a((size_t) M * d);
    for (auto& v : mixes) v = g(rng) * 3.0f;
    for (auto& v : scale) v = g(rng);
    for (auto& v : base) v = g(rng);
    for (auto& v : x) v = g(rng);
    for (auto& v : a) v = g(rng);

    float *d_mix = nullptr, *d_sc = nullptr, *d_base = nullptr, *d_pre = nullptr, *d_post = nullptr,
          *d_comb = nullptr, *d_x = nullptr, *d_a = nullptr, *d_y1 = nullptr, *d_y2 = nullptr;
    check(cudaMalloc(&d_mix, mixes.size() * 4), "mix");
    check(cudaMalloc(&d_sc, scale.size() * 4), "sc");
    check(cudaMalloc(&d_base, base.size() * 4), "base");
    check(cudaMalloc(&d_pre, (size_t) M * hc * 4), "pre");
    check(cudaMalloc(&d_post, (size_t) M * hc * 4), "post");
    check(cudaMalloc(&d_comb, (size_t) M * hc * hc * 4), "comb");
    check(cudaMalloc(&d_x, x.size() * 4), "x");
    check(cudaMalloc(&d_a, a.size() * 4), "a");
    check(cudaMalloc(&d_y1, x.size() * 4), "y1");
    check(cudaMalloc(&d_y2, x.size() * 4), "y2");
    for (auto [dst, src, n] : std::vector<std::tuple<float*, const float*, size_t>>{
             {d_mix, mixes.data(), mixes.size()}, {d_sc, scale.data(), scale.size()},
             {d_base, base.data(), base.size()}, {d_x, x.data(), x.size()}, {d_a, a.data(), a.size()}}) {
        check(cudaMemcpy(dst, src, n * 4, cudaMemcpyHostToDevice), "copy");
    }
    strata::kernels::hc_split_sinkhorn(d_mix, d_sc, d_base, M, hc, iters, (float) eps, d_pre, d_post, d_comb,
                                       nullptr);
    strata::kernels::hc_pre_combine(d_x, d_pre, d_y1, M, hc, d, nullptr);
    strata::kernels::hc_post_combine(d_a, d_x, d_post, d_comb, d_y2, M, hc, d, nullptr);

    std::vector<float> got_pre((size_t) M * hc), got_post((size_t) M * hc), got_comb((size_t) M * hc * hc),
            got_y1(x.size()), got_y2(x.size());
    check(cudaMemcpy(got_pre.data(), d_pre, got_pre.size() * 4, cudaMemcpyDeviceToHost), "gp");
    check(cudaMemcpy(got_post.data(), d_post, got_post.size() * 4, cudaMemcpyDeviceToHost), "gq");
    check(cudaMemcpy(got_comb.data(), d_comb, got_comb.size() * 4, cudaMemcpyDeviceToHost), "gc");
    check(cudaMemcpy(got_y1.data(), d_y1, got_y1.size() * 4, cudaMemcpyDeviceToHost), "g1");
    check(cudaMemcpy(got_y2.data(), d_y2, got_y2.size() * 4, cudaMemcpyDeviceToHost), "g2");

    double wpre = 0, wpost = 0, wcomb = 0;
    std::vector<double> want_y1((size_t) M * d), want_y2(x.size());
    for (int64_t m = 0; m < M; ++m) {
        std::vector<double> pre, post, comb;
        ref_split(std::vector<double>(mixes.begin() + m * mix_hc, mixes.begin() + (m + 1) * mix_hc),
                  std::vector<double>(scale.begin(), scale.end()),
                  std::vector<double>(base.begin(), base.end()), hc, iters, eps, pre, post, comb);
        wpre = std::max(wpre, rel_l1(pre, std::vector<double>(got_pre.begin() + m * hc, got_pre.begin() + (m + 1) * hc)));
        wpost = std::max(wpost, rel_l1(post, std::vector<double>(got_post.begin() + m * hc, got_post.begin() + (m + 1) * hc)));
        wcomb = std::max(wcomb, rel_l1(comb, std::vector<double>(got_comb.begin() + m * hc * hc,
                                                                 got_comb.begin() + (m + 1) * hc * hc)));
        for (int64_t c = 0; c < d; ++c) {
            double acc = 0;
            for (int64_t h = 0; h < hc; ++h) acc += pre[(size_t) h] * x[((size_t) (m * hc + h) * d) + c];
            want_y1[(size_t) m * d + c] = acc;
        }
        for (int64_t q = 0; q < hc; ++q)
            for (int64_t c = 0; c < d; ++c) {
                double acc = post[(size_t) q] * a[(size_t) m * d + c];
                for (int64_t p = 0; p < hc; ++p)
                    acc += comb[(size_t) p * hc + q] * x[((size_t) (m * hc + p) * d) + c];
                want_y2[(size_t) (m * hc + q) * d + c] = acc;
            }
    }
    std::printf("    %-46s pre %.2e post %.2e comb %.2e\n", "split+sinkhorn vs the reference", wpre, wpost,
                wcomb);
    if (!(wpre <= 1e-6 && wpost <= 1e-6 && wcomb <= 1e-6)) { std::printf("      *** over 1e-6 ***\n"); ++bad; }
    const double r1 = rel_l1(want_y1, to_d(got_y1)), r2 = rel_l1(want_y2, to_d(got_y2));
    std::printf("    %-46s pre_combine %.2e post_combine %.2e\n", "combines vs the reference", r1, r2);
    if (!(r1 <= 1e-6 && r2 <= 1e-6)) { std::printf("      *** over 1e-6 ***\n"); ++bad; }

    // traps
    double t_axis = 0, t_iters = 0, t_half = 0, t_uniform = 0;
    for (int64_t m = 0; m < M; ++m) {
        std::vector<double> pre, post, comb;
        ref_split(std::vector<double>(mixes.begin() + m * mix_hc, mixes.begin() + (m + 1) * mix_hc),
                  std::vector<double>(scale.begin(), scale.end()),
                  std::vector<double>(base.begin(), base.end()), hc, iters, eps, pre, post, comb);
        std::vector<double> pre_1, post_1, comb_1;
        ref_split(std::vector<double>(mixes.begin() + m * mix_hc, mixes.begin() + (m + 1) * mix_hc),
                  std::vector<double>(scale.begin(), scale.end()),
                  std::vector<double>(base.begin(), base.end()), hc, 1, eps, pre_1, post_1, comb_1);
        t_iters = std::max(t_iters, rel_l1(comb, comb_1));
        std::vector<double> y_axis(x.size()), y_half(x.size()), y_uni(x.size());
        for (int64_t q = 0; q < hc; ++q)
            for (int64_t c = 0; c < d; ++c) {
                double ax = post[(size_t) q] * a[(size_t) m * d + c];
                double hf = 0.5 * post[(size_t) q] * a[(size_t) m * d + c];
                double un = 0.0;
                for (int64_t p = 0; p < hc; ++p) {
                    ax += comb[(size_t) q * hc + p] * x[((size_t) (m * hc + p) * d) + c];  // swapped axis
                    hf += comb[(size_t) p * hc + q] * x[((size_t) (m * hc + p) * d) + c];
                    un += (1.0 / (double) hc) * x[((size_t) (m * hc + p) * d) + c];
                }
                y_axis[(size_t) (m * hc + q) * d + c] = ax;
                y_half[(size_t) (m * hc + q) * d + c] = hf;
                y_uni[(size_t) m * d + c] = un;
            }
        t_axis = std::max(t_axis, rel_l1(want_y2, y_axis));
        t_half = std::max(t_half, rel_l1(want_y2, y_half));
        t_uniform = std::max(t_uniform, rel_l1(want_y1, y_uni));
    }
    std::printf("    %-46s axis %.2e, iters=1 %.2e, half-post %.2e, uniform-pre %.2e\n",
                "wrong readings observably apart", t_axis, t_iters, t_half, t_uniform);
    if (!(t_axis > 1e-3 && t_iters > 1e-3 && t_half > 1e-3 && t_uniform > 1e-3)) {
        std::printf("      *** a trap is not observable ***\n");
        ++bad;
    }

    cudaFree(d_mix); cudaFree(d_sc); cudaFree(d_base); cudaFree(d_pre); cudaFree(d_post); cudaFree(d_comb);
    cudaFree(d_x); cudaFree(d_a); cudaFree(d_y1); cudaFree(d_y2);
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: dsv4_hc_parity [--selftest]\n"); return 2; }
    }
    int bad = 0;
    run_case(4, 20, 1e-6, 1, 512, 61, bad);  // decode, a d slice for speed
    run_case(4, 20, 1e-6, 3, 256, 62, bad);   // prefill-shaped, M=3
    std::printf("dsv4_hc_parity: %s\n", bad ? "*** FAIL ***" : "ok");
    return bad ? 1 : 0;
}
