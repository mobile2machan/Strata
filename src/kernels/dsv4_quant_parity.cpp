// src/kernels/dsv4_quant_parity.cpp - docs/DSV4.md P2: the pool round-trips and the WHT.
//
// The round-trips are asserted BIT-EXACT, not within a tolerance: their outputs live on the e4m3 /
// e2m1 grids times a power of two, which bf16 represents exactly, so any disagreement with the
// reference is a real difference, not rounding noise.  The e4m3 reference enumerates the grid and
// rounds nearest-ties-to-even-CODE; the e2m1 reference is `_round_fp4`'s comparison chain, and the
// fixture carries the exact tie points (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0) because random draws
// never hit them and the tie table is the part that is easy to get backwards.
//
// Traps asserted observably apart: a LINEAR scale (amax/448, not the power of two) for fp8, a flipped
// tie rule for fp4, and an UNNORMALIZED Hadamard (the d**-0.5 is what keeps the fp4 quantizer's
// 6.0 ceiling meaningful after rotation).
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/dsv4_quant.hpp"

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

// ---- e4m3 grid, enumerated: code -> value (positive half; 0x7f is NaN)
std::vector<std::pair<double, int>> e4m3_grid() {
    std::vector<std::pair<double, int>> g;
    for (int code = 0; code <= 126; ++code) {
        const int e = code >> 3, m = code & 7;
        double v;
        if (e == 0) v = m * std::pow(2.0, -9);
        else v = (1.0 + m / 8.0) * std::pow(2.0, e - 7);
        g.push_back({v, code});
    }
    return g;
}

double round_e4m3_ref(double x, const std::vector<std::pair<double, int>>& g) {
    const double a = std::fabs(x);
    double best = 0.0;
    int best_code = 0;
    double best_d = 1e300;
    for (const auto& [v, code] : g) {
        const double d = std::fabs(a - v);
        if (d < best_d || (d == best_d && (code & 1) == 0 && (best_code & 1) != 0)) {
            best = v;
            best_code = code;
            best_d = d;
        }
    }
    return x < 0 ? -best : best;
}

double round_e2m1_ref(double x) {
    const double sign = x < 0 ? -1.0 : 1.0;
    const double a = std::fabs(x);
    double r;
    if (a <= 0.25) r = 0.0;
    else if (a < 0.75) r = 0.5;
    else if (a <= 1.25) r = 1.0;
    else if (a < 1.75) r = 1.5;
    else if (a <= 2.5) r = 2.0;
    else if (a < 3.5) r = 3.0;
    else if (a <= 5.0) r = 4.0;
    else r = 6.0;
    return sign * r;
}

// the kernel's own bit-trick on the FLOAT product - the reference triton computes the scale in
// float32 too, so the reference must not compute it in double and land on the other side of a
// power-of-two boundary
double pow2_scale_ref(float amax, float inv_max) {
    const float v = amax * inv_max;
    uint32_t b;
    std::memcpy(&b, &v, 4);
    const int e = (int) ((b >> 23) & 0xFFu) - 127 + (((b & 0x7FFFFFu) != 0u) ? 1 : 0);
    return std::pow(2.0, e);
}

void run_roundtrip(bool fp8, int64_t block, uint32_t seed, int& bad) {
    const int64_t rows = 64, n = rows * block;
    std::printf("  %s block %lld\n", fp8 ? "fp8 e4m3" : "fp4 e2m1", (long long) block);
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> x((size_t) n);
    for (auto& v : x) v = g(rng) * (1.0f + 100.0f * (float) (rng() % 4));
    // exact tie points, filling whole groups whose amax forces s = 1 (fp4: amax in (3,6]; fp8: amax
    // = 448) - random draws never hit a midpoint and the tie table is the part easy to get backwards
    const float inv_max = fp8 ? 1.0f / 448.0f : 1.0f / 6.0f;
    const double amax_floor = fp8 ? (double) 1e-4f : 6.0 * std::pow(2.0, -126);
    if (fp8) {
        const double ties8[] = {448.0, 446.0, 2.0, 1.9375, 0.0019531250};  // incl. a subnormal-region step
        for (int64_t i = 0; i < 2 * block; ++i) x[(size_t) i] = (float) ties8[i % 5];
        for (int64_t i = 2 * block; i < 3 * block; ++i) x[(size_t) i] = 0.0f;  // the amax floor path
    } else {
        const double ties4[] = {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0};
        for (int64_t i = 0; i < 2 * block; ++i) x[(size_t) i] = (float) ties4[i % 8];
        for (int64_t i = 2 * block; i < 3 * block; ++i) x[(size_t) i] = 0.0f;
    }

    float* d_x = nullptr;
    uint16_t* d_y = nullptr;
    check(cudaMalloc(&d_x, n * 4), "x");
    check(cudaMalloc(&d_y, n * 2), "y");
    check(cudaMemcpy(d_x, x.data(), n * 4, cudaMemcpyHostToDevice), "mx");
    if (fp8) strata::kernels::roundtrip_fp8_e4m3(d_x, d_y, n, block, nullptr);
    else strata::kernels::roundtrip_fp4_e2m1(d_x, d_y, n, block, nullptr);
    std::vector<uint16_t> got((size_t) n);
    check(cudaMemcpy(got.data(), d_y, n * 2, cudaMemcpyDeviceToHost), "my");

    const auto grid = e4m3_grid();
    std::vector<uint16_t> want((size_t) n), trap((size_t) n);
    int mismatch = 0, trap_diff = 0;
    for (int64_t r = 0; r < rows; ++r) {
        double amax = amax_floor;
        for (int64_t i = 0; i < block; ++i) amax = std::max(amax, (double) std::fabs(x[(size_t) r * block + i]));
        const double s = pow2_scale_ref((float) std::max(amax, amax_floor), inv_max);
        const double s_lin = std::max(amax, amax_floor) * inv_max;  // the non-power-of-two trap
        for (int64_t i = 0; i < block; ++i) {
            const double q = std::min(std::max((double) x[(size_t) r * block + i] / s, -1.0 / inv_max), 1.0 / inv_max);
            const double v = fp8 ? round_e4m3_ref(q, grid) : round_e2m1_ref(q);
            want[(size_t) r * block + i] = bf16_from_f32((float) (v * s));
            const double ql = std::min(std::max((double) x[(size_t) r * block + i] / s_lin, -1.0 / inv_max), 1.0 / inv_max);
            const double vl = fp8 ? round_e4m3_ref(ql, grid) : round_e2m1_ref(ql);
            trap[(size_t) r * block + i] = bf16_from_f32((float) (vl * s_lin));
            if (got[(size_t) r * block + i] != want[(size_t) r * block + i]) ++mismatch;
            if (got[(size_t) r * block + i] != trap[(size_t) r * block + i]) ++trap_diff;
        }
    }
    std::printf("    %-46s %d of %lld differ from the reference\n", "round-trip bit-exact", mismatch,
                (long long) n);
    if (mismatch) ++bad;
    std::printf("    %-46s %d of %lld differ\n", "linear-scale trap observably apart", trap_diff,
                (long long) n);
    if (trap_diff == 0) { std::printf("      *** trap not observable ***\n"); ++bad; }

    // the tie table itself, read off the fixture's first group (s == 1 by construction)
    if (!fp8) {
        const double ties4[] = {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0};
        const double want4[] = {0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0};
        int wrong = 0;
        for (size_t i = 0; i < 8; ++i)
            if (f32_from_bf16(got[i]) != (float) want4[i]) ++wrong;
        std::printf("    %-46s %d of 8 tie points wrong\n", "e2m1 tie table (0.75->1, 3.5->4, ...)", wrong);
        if (wrong) ++bad;
    }

    cudaFree(d_x);
    cudaFree(d_y);
}

void run_hadamard(int64_t d, int64_t rows, int& bad) {
    std::printf("  hadamard d %lld\n", (long long) d);
    std::mt19937 rng(41);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> x((size_t) rows * d);
    for (auto& v : x) v = g(rng);
    float *d_x = nullptr, *d_y = nullptr;
    check(cudaMalloc(&d_x, x.size() * 4), "x");
    check(cudaMalloc(&d_y, x.size() * 4), "y");
    check(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "mx");
    strata::kernels::hadamard(d_x, d_y, rows, d, nullptr);
    std::vector<float> got(x.size());
    check(cudaMemcpy(got.data(), d_y, x.size() * 4, cudaMemcpyDeviceToHost), "my");

    // Sylvester matrix by construction: H_2k = [[H_k, H_k], [H_k, -H_k]] over every 2k block
    std::vector<double> H((size_t) d * d, 0.0);
    H[0] = 1.0;
    for (size_t k = 1; k < (size_t) d; k <<= 1)
        for (size_t bi = 0; bi < (size_t) d; bi += 2 * k)
            for (size_t bj = 0; bj < (size_t) d; bj += 2 * k)
                for (size_t i = 0; i < k; ++i)
                    for (size_t j = 0; j < k; ++j) {
                        const double v = H[(bi + i) * d + bj + j];
                        H[(bi + k + i) * d + bj + j] = v;
                        H[(bi + i) * d + bj + k + j] = v;
                        H[(bi + k + i) * d + bj + k + j] = -v;
                    }
    double dnum = 0, dden = 0, tnum = 0;
    const double scale = std::pow((double) d, -0.5);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < d; ++c) {
            double acc = 0;
            for (int64_t k = 0; k < d; ++k) acc += (double) x[(size_t) r * d + k] * H[(size_t) k * d + c];
            acc *= scale;
            dnum += std::fabs(acc - (double) got[(size_t) r * d + c]);
            dden += std::fabs(acc);
            tnum += std::fabs(acc / scale - (double) got[(size_t) r * d + c]);  // the unnormalized trap
        }
    }
    const double rel = dnum / dden, trap = tnum / dden;
    std::printf("    %-46s rel %.3e\n", "hadamard vs the fp64 Sylvester transform", rel);
    if (!(rel <= 1e-6)) { std::printf("      *** over 1e-6 ***\n"); ++bad; }
    std::printf("    %-46s rel %.2e\n", "unnormalized trap observably apart", trap);
    if (!(trap > 1e-3)) { std::printf("      *** trap not observable ***\n"); ++bad; }
    cudaFree(d_x);
    cudaFree(d_y);
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: dsv4_quant_parity [--selftest]\n"); return 2; }
    }
    int bad = 0;
    run_roundtrip(true, 64, 51, bad);   // the window / compressor KV round-trip
    run_roundtrip(false, 32, 52, bad);  // the indexer q / key round-trip
    run_hadamard(128, 8, bad);          // the indexer head_dim
    std::printf("dsv4_quant_parity: %s\n", bad ? "*** FAIL ***" : "ok");
    return bad ? 1 : 0;
}
