// src/kernels/dsv4_rope_parity.cpp - docs/DSV4.md P2: the DSv4 interleaved rope + YaRN.
//
// The reference is `ops.py::precompute_freqs_cis` + `apply_rotary_emb` read in float64:
// interleaved pairs on the LAST rd dims, the YaRN blend exactly as the reference writes it
// (floor/ceil correction range, linear ramp, freqs blended before the outer product).  The two
// measured regimes are exercised (theta 10000 YaRN-on, theta 160000 YaRN-on) plus the r=0
// regime (YaRN off) and the inverse (o-path) rotation.  Traps asserted observably apart:
// neox half-split pairing, skipping the YaRN blend, and a wrong theta.
#include "strata/kernels/dsv4_rope.hpp"
#include "strata/kernels/bf16_bits.hpp"

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

std::vector<double> to_d(const std::vector<float>& v) { return std::vector<double>(v.begin(), v.end()); }

// ops.py::precompute_freqs_cis freqs[j], float64
constexpr double PI = 3.14159265358979323846;
std::vector<double> ref_freqs(int rd, double theta, double factor, int64_t orig, int64_t bf, int64_t bs) {
    std::vector<double> f((size_t) rd / 2);
    for (int j = 0; j < rd / 2; ++j) f[(size_t) j] = std::pow(theta, -(double)(2 * j) / (double) rd);
    if (orig > 0) {
        auto corr = [&](double rot) {
            return (double) rd * std::log((double) orig / (rot * 2.0 * PI)) / (2.0 * std::log(theta));
        };
        int64_t low = (int64_t) std::floor(corr((double) bf));
        int64_t high = (int64_t) std::ceil(corr((double) bs));
        low = std::max<int64_t>(low, 0);
        high = std::min<int64_t>(high, rd - 1);
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

// rotate the last rd dims of each row, interleaved (or neox for the trap)
void ref_rope(std::vector<double>& x, int64_t n_rows, int64_t row_dim, int rd, int64_t pos,
              const std::vector<double>& f, bool inverse, bool neox = false) {
    for (int64_t r = 0; r < n_rows; ++r) {
        size_t base = (size_t) (r * row_dim + row_dim - rd);
        for (int j = 0; j < rd / 2; ++j) {
            size_t i0 = base + (neox ? (size_t) j : (size_t)(2 * j));
            size_t i1 = base + (neox ? (size_t)(rd / 2 + j) : (size_t)(2 * j + 1));
            double s = std::sin((double) pos * f[(size_t) j]);
            double c = std::cos((double) pos * f[(size_t) j]);
            if (inverse) s = -s;
            const double x0 = x[i0], x1 = x[i1];
            x[i0] = x0 * c - x1 * s;
            x[i1] = x0 * s + x1 * c;
        }
    }
}

int failures = 0;
void expect(bool ok, const char* what) {
    std::printf("    %-46s %s\n", what, ok ? "ok" : "*** FAIL ***");
    if (!ok) ++failures;
}

void run_case(const char* name, int64_t n_rows, int64_t row_dim, int rd, int64_t pos, double theta,
              double factor, int64_t orig, int64_t bf, int64_t bs, bool inverse, bool bf16,
              std::mt19937_64& rng) {
    std::printf("    %s rows=%lld dim=%lld pos=%lld\n", name, (long long) n_rows, (long long) row_dim,
                (long long) pos);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> src((size_t) n_rows * row_dim);
    for (auto& v : src) v = nd(rng);
    std::vector<float> h((size_t) src.size());
    std::vector<uint16_t> hb((size_t) src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        h[i] = (float) src[i];
        hb[i] = bf16_from_f32((float) src[i]);
    }

    const auto f = ref_freqs(rd, theta, factor, orig, bf, bs);
    std::vector<double> want = src;
    ref_rope(want, n_rows, row_dim, rd, pos, f, inverse);

    uint16_t* d_bf = nullptr;
    float* d_f = nullptr;
    if (bf16) {
        check(cudaMalloc(&d_bf, hb.size() * 2), "alloc");
        check(cudaMemcpy(d_bf, hb.data(), hb.size() * 2, cudaMemcpyHostToDevice), "copy");
        strata::kernels::dsv4_rope_bf16(d_bf, n_rows, row_dim, rd, pos, theta, factor, orig, bf, bs,
                                        inverse, nullptr);
        // re-read as bf16
        std::vector<uint16_t> got(hb.size());
        check(cudaMemcpy(got.data(), d_bf, got.size() * 2, cudaMemcpyDeviceToHost), "read2");
        std::vector<double> g(got.size());
        for (size_t i = 0; i < got.size(); ++i) g[i] = f32_from_bf16(got[i]);
        const double d = rel_l1(want, g);
        std::printf("      rope vs reference                          rel %.2e\n", d);
        expect(d < 2e-2, "interleaved rope vs reference");
        cudaFree(d_bf);
    } else {
        check(cudaMalloc(&d_f, h.size() * 4), "alloc");
        check(cudaMemcpy(d_f, h.data(), h.size() * 4, cudaMemcpyHostToDevice), "copy");
        strata::kernels::dsv4_rope_f32(d_f, n_rows, row_dim, rd, pos, theta, factor, orig, bf, bs,
                                       inverse, nullptr);
        check(cudaMemcpy(h.data(), d_f, h.size() * 4, cudaMemcpyDeviceToHost), "read");
        std::vector<double> g(h.size());
        for (size_t i = 0; i < h.size(); ++i) g[i] = h[i];
        const double d = rel_l1(want, g);
        std::printf("      rope vs reference                          rel %.2e\n", d);
        expect(d < 1e-5, "interleaved rope vs reference");

        // traps
        std::vector<double> t_neox = src;
        ref_rope(t_neox, n_rows, row_dim, rd, pos, f, inverse, true);
        std::vector<double> f_plain = ref_freqs(rd, theta, factor, 0, bf, bs);
        std::vector<double> t_noyarn = src;
        ref_rope(t_noyarn, n_rows, row_dim, rd, pos, f_plain, inverse);
        std::vector<double> f_wrong = ref_freqs(rd, theta == 10000.0 ? 160000.0 : 10000.0, factor, orig, bf, bs);
        std::vector<double> t_theta = src;
        ref_rope(t_theta, n_rows, row_dim, rd, pos, f_wrong, inverse);
        const double d1 = rel_l1(want, t_neox), d2 = rel_l1(want, t_noyarn), d3 = rel_l1(want, t_theta);
        std::printf("      wrong readings observably apart            neox %.2e, no-yaarn %.2e, theta %.2e\n",
                    d1, d2, d3);
        if (orig > 0) expect(d1 > 1e-3 && d2 > 1e-3 && d3 > 1e-3, "traps observably apart");
        else expect(d1 > 1e-3 && d3 > 1e-3, "traps observably apart (no-yaarn n/a)");
        cudaFree(d_f);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) != "--selftest") {
        std::fprintf(stderr, "usage: %s --selftest\n", argv[0]);
        return 2;
    }
    std::mt19937_64 rng(20261003);
    std::printf("dsv4_rope_parity: interleaved rope + reference YaRN\n");
    run_case("attn q (theta 1e4, yarn)", 64, 512, 64, 100000, 10000.0, 16.0, 65536, 32, 1, false, false, rng);
    run_case("kv (theta 1.6e5, yarn, bf16)", 1, 512, 64, 7, 160000.0, 16.0, 65536, 32, 1, false, true, rng);
    run_case("idx q (theta 1.6e5, bf16)", 64, 128, 64, 3, 160000.0, 16.0, 65536, 32, 1, false, true, rng);
    run_case("o inverse (theta 1e4, yarn)", 64, 512, 64, 100000, 10000.0, 16.0, 65536, 32, 1, true, false, rng);
    run_case("r=0 layer (yarn off)", 64, 512, 64, 5, 10000.0, 16.0, 0, 32, 1, false, false, rng);
    std::printf("dsv4_rope_parity: %s\n", failures == 0 ? "ok" : "*** FAIL ***");
    return failures == 0 ? 0 : 1;
}
