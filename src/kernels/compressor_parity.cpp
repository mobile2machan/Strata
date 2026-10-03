// src/kernels/compressor_parity.cpp - docs/DSV4.md P2: the compressor's pooling and carry.
//
// The reference is FreeToken `models/deepseek_v4/compress.py` read in float64: `overlap_transform` +
// `(kv * score.softmax(dim=2)).sum(dim=2)` for prefill, `decode_step`'s scatter/roll for the register.
// Three readings are asserted OBSERVABLY different rather than assumed, because each produces a plausible
// tensor:
//
//   1. the softmax is over the ROW axis per column (`softmax(score[:, c])`), not over the columns of a
//      row.  A row-softmax pools every window to a convex combination of its own rows with weights that
//      sum to 1 ACROSS COLUMNS - plausible-looking, wrong.
//   2. the overlap window takes the PREVIOUS block's FIRST-half columns and the current block's SECOND
//      half.  Swapping which half of which block feeds which slot keeps every shape legal.
//   3. block 0 of a from-scratch prefill has NO previous block: its extra rows are -inf and drop out.
//      Zeroing their scores instead gives them real weight and a different pooled row.
#include "strata/kernels/compressor.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

double rel_l1_range(const std::vector<double>& a, const std::vector<double>& b, size_t lo, size_t hi) {
    double d = 0, m = 0;
    for (size_t i = lo; i < hi; ++i) {
        d += std::fabs(a[i] - b[i]);
        m += std::fabs(a[i]);
    }
    return d / (m > 1e-30 ? m : 1e-30);
}

std::vector<double> to_d(const std::vector<float>& v) { return std::vector<double>(v.begin(), v.end()); }

struct Shape {
    int64_t ratio;
    bool overlap;
    int64_t d;
    int64_t item() const { return (overlap ? 2 : 1) * d; }
    int64_t rows() const { return (overlap ? 2 : 1) * ratio; }
};

// ---- the float64 reference, mirroring compress.py row for row

// effective window of block b: (k, s) per row j; s = -inf means "no source"
void window_row_ref(const Shape& sh, const std::vector<float>& kv, const std::vector<float>& score,
                    const std::vector<float>& ape, const double* carry_k, const double* carry_s, int64_t b,
                    int64_t j, int64_t c, double& k, double& s) {
    const int64_t d = sh.d, item = sh.item();
    if (sh.overlap && j < sh.ratio) {
        if (b == 0) {
            if (carry_s == nullptr) { k = 0.0; s = -INFINITY; return; }
            k = carry_k[j * item + c];
            s = carry_s[j * item + c];
            return;
        }
        const int64_t row = (b - 1) * sh.ratio + j;
        k = kv[row * item + c];
        s = score[row * item + c] + ape[j * item + c];
        return;
    }
    const int64_t jj = sh.overlap ? j - sh.ratio : j;
    const int64_t row = b * sh.ratio + jj;
    const int64_t col = sh.overlap ? d + c : c;
    k = kv[row * item + col];
    s = score[row * item + col] + ape[jj * item + col];
}

void ref_prefill(const Shape& sh, int64_t seqlen, const std::vector<float>& kv,
                 const std::vector<float>& score, const std::vector<float>& ape,
                 const double* carry_k, const double* carry_s, std::vector<double>& out,
                 bool row_softmax_trap = false, bool wrong_half_trap = false, bool zero_instead_of_ninf = false) {
    const int64_t nb = seqlen / sh.ratio, d = sh.d, R = sh.rows();
    out.assign((size_t) nb * d, 0.0);
    for (int64_t b = 0; b < nb; ++b) {
        std::vector<double> K((size_t) R * d), S((size_t) R * d);
        for (int64_t j = 0; j < R; ++j) {
            for (int64_t c = 0; c < d; ++c) {
                double k, s;
                if (wrong_half_trap && sh.overlap && j < sh.ratio && b > 0) {
                    const int64_t row = (b - 1) * sh.ratio + j;
                    k = kv[row * sh.item() + sh.d + c];
                    s = score[row * sh.item() + sh.d + c] + ape[j * sh.item() + sh.d + c];
                } else {
                    window_row_ref(sh, kv, score, ape, carry_k, carry_s, b, j, c, k, s);
                }
                if (zero_instead_of_ninf && s == -INFINITY) s = 0.0;
                K[(size_t) j * d + c] = k;
                S[(size_t) j * d + c] = s;
            }
        }
        if (row_softmax_trap) {
            // softmax over the window's COLUMNS per row - the wrong axis
            for (int64_t j = 0; j < R; ++j) {
                double mx = -INFINITY;
                for (int64_t c = 0; c < d; ++c) mx = std::max(mx, S[(size_t) j * d + c]);
                double den = 0;
                for (int64_t c = 0; c < d; ++c)
                    den += (S[(size_t) j * d + c] == -INFINITY) ? 0.0 : std::exp(S[(size_t) j * d + c] - mx);
                for (int64_t c = 0; c < d; ++c) {
                    const double s = S[(size_t) j * d + c];
                    const double w = (s == -INFINITY || den == 0.0) ? 0.0 : std::exp(s - mx) / den;
                    out[(size_t) b * d + c] += K[(size_t) j * d + c] * w;
                }
            }
            continue;
        }
        for (int64_t c = 0; c < d; ++c) {
            double mx = -INFINITY;
            for (int64_t j = 0; j < R; ++j) mx = std::max(mx, S[(size_t) j * d + c]);
            double den = 0, acc = 0;
            for (int64_t j = 0; j < R; ++j) {
                const double s = S[(size_t) j * d + c];
                const double w = (s == -INFINITY) ? 0.0 : std::exp(s - mx);
                den += w;
                acc += w * K[(size_t) j * d + c];
            }
            out[(size_t) b * d + c] = den > 0 ? acc / den : 0.0;
        }
    }
}

void ref_seed_carry(const Shape& sh, int64_t seqlen, const std::vector<float>& kv,
                    const std::vector<float>& score, const std::vector<float>& ape, std::vector<double>& ks,
                    std::vector<double>& ss) {
    const int64_t item = sh.item(), cutoff = seqlen - seqlen % sh.ratio, rem = seqlen % sh.ratio;
    ks.assign((size_t) sh.rows() * item, 0.0);
    ss.assign((size_t) sh.rows() * item, -INFINITY);
    for (int64_t row = 0; row < sh.rows(); ++row) {
        int64_t src = -1, arow = -1;
        if (sh.overlap) {
            if (row < sh.ratio) {
                if (cutoff >= sh.ratio) { src = cutoff - sh.ratio + row; arow = row; }
            } else if (row - sh.ratio < rem) {
                src = cutoff + (row - sh.ratio); arow = row - sh.ratio;
            }
        } else if (row < rem) {
            src = cutoff + row; arow = row;
        }
        if (src < 0) continue;
        for (int64_t col = 0; col < item; ++col) {
            ks[(size_t) row * item + col] = kv[(size_t) src * item + col];
            ss[(size_t) row * item + col] = score[(size_t) src * item + col] + ape[(size_t) arow * item + col];
        }
    }
}

void ref_decode_token(const Shape& sh, int64_t pos, const std::vector<float>& kv,
                      const std::vector<float>& score, const std::vector<float>& ape, std::vector<double>& ks,
                      std::vector<double>& ss, std::vector<double>* compressed) {
    const int64_t item = sh.item(), d = sh.d;
    const int64_t idx = pos % sh.ratio, slot = sh.overlap ? sh.ratio + idx : idx;
    for (int64_t i = 0; i < item; ++i) {
        ks[(size_t) slot * item + i] = kv[(size_t) i];
        ss[(size_t) slot * item + i] = score[(size_t) i] + ape[(size_t) idx * item + i];
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

void run_shape(const Shape& sh, int64_t seqlen, uint32_t seed, int& bad) {
    const int64_t item = sh.item(), d = sh.d;
    std::printf("  ratio %lld overlap %d d %lld\n", (long long) sh.ratio, (int) sh.overlap, (long long) d);
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f), a(0.0f, 0.5f);
    std::vector<float> kv((size_t) seqlen * item), score((size_t) seqlen * item), ape((size_t) sh.ratio * item);
    for (auto& v : kv) v = g(rng);
    for (auto& v : score) v = g(rng) * 2.0f;  // non-uniform softmax weights
    for (auto& v : ape) v = a(rng);

    float *d_kv = nullptr, *d_sc = nullptr, *d_ape = nullptr, *d_out = nullptr, *d_ks = nullptr, *d_ss = nullptr,
          *d_cmp = nullptr;
    const int64_t reg = sh.rows() * item;
    check(cudaMalloc(&d_kv, (size_t) seqlen * item * 4), "kv");
    check(cudaMalloc(&d_sc, (size_t) seqlen * item * 4), "sc");
    check(cudaMalloc(&d_ape, (size_t) sh.ratio * item * 4), "ape");
    check(cudaMalloc(&d_out, (size_t) (seqlen / sh.ratio) * d * 4), "out");
    check(cudaMalloc(&d_ks, (size_t) reg * 4), "ks");
    check(cudaMalloc(&d_ss, (size_t) reg * 4), "ss");
    check(cudaMalloc(&d_cmp, (size_t) d * 4), "cmp");
    check(cudaMemcpy(d_kv, kv.data(), (size_t) seqlen * item * 4, cudaMemcpyHostToDevice), "mkv");
    check(cudaMemcpy(d_sc, score.data(), (size_t) seqlen * item * 4, cudaMemcpyHostToDevice), "msc");
    check(cudaMemcpy(d_ape, ape.data(), (size_t) sh.ratio * item * 4, cudaMemcpyHostToDevice), "mape");

    // ---- 1. prefill (from scratch: no carry)
    strata::kernels::compressor_prefill(seqlen, sh.ratio, sh.overlap, d, d_kv, d_sc, d_ape, nullptr, nullptr,
                                        d_out, nullptr);
    std::vector<float> got_out((size_t) (seqlen / sh.ratio) * d);
    check(cudaMemcpy(got_out.data(), d_out, got_out.size() * 4, cudaMemcpyDeviceToHost), "gout");
    std::vector<double> want;
    ref_prefill(sh, seqlen, kv, score, ape, nullptr, nullptr, want);
    const double rel = rel_l1(want, to_d(got_out));
    std::printf("    %-46s rel %.3e\n", "prefill vs the reference", rel);
    if (!(rel <= 1e-6)) { std::printf("      *** over 1e-6 ***\n"); ++bad; }

    // ---- 2. carry seed
    strata::kernels::compressor_seed_carry(seqlen, sh.ratio, sh.overlap, item, d_kv, d_sc, d_ape, d_ks, d_ss,
                                           nullptr);
    std::vector<float> got_ks((size_t) reg), got_ss((size_t) reg);
    check(cudaMemcpy(got_ks.data(), d_ks, (size_t) reg * 4, cudaMemcpyDeviceToHost), "gks");
    check(cudaMemcpy(got_ss.data(), d_ss, (size_t) reg * 4, cudaMemcpyDeviceToHost), "gss");
    std::vector<double> rks, rss;
    ref_seed_carry(sh, seqlen, kv, score, ape, rks, rss);
    const double rel_c = std::max(rel_l1(rks, to_d(got_ks)),
                                  rel_l1(rss, to_d(got_ss)));
    std::printf("    %-46s rel %.3e\n", "carry seed vs the reference", rel_c);
    if (!(rel_c <= 1e-6)) { std::printf("      *** over 1e-6 ***\n"); ++bad; }

    // ---- 3. decode sequence across block boundaries
    std::vector<double> ref_cmp;
    int completions = 0, mismatches = 0;
    for (int64_t pos = seqlen; pos < seqlen + 3 * sh.ratio + 2; ++pos) {
        std::vector<float> tkv((size_t) item), tsc((size_t) item);
        for (auto& v : tkv) v = g(rng);
        for (auto& v : tsc) v = g(rng) * 2.0f;
        float *d_tkv, *d_tsc;
        check(cudaMalloc(&d_tkv, (size_t) item * 4), "tkv");
        check(cudaMalloc(&d_tsc, (size_t) item * 4), "tsc");
        check(cudaMemcpy(d_tkv, tkv.data(), (size_t) item * 4, cudaMemcpyHostToDevice), "mtkv");
        check(cudaMemcpy(d_tsc, tsc.data(), (size_t) item * 4, cudaMemcpyHostToDevice), "mtsc");
        strata::kernels::compressor_decode_step(pos, sh.ratio, sh.overlap, d, d_tkv, d_tsc, d_ape, d_ks, d_ss,
                                                d_cmp, nullptr);
        cudaFree(d_tkv);
        cudaFree(d_tsc);
        ref_decode_token(sh, pos, tkv, tsc, ape, rks, rss, &ref_cmp);
        if ((pos + 1) % sh.ratio == 0) {
            ++completions;
            std::vector<float> got_cmp((size_t) d);
            check(cudaMemcpy(got_cmp.data(), d_cmp, (size_t) d * 4, cudaMemcpyDeviceToHost), "gcmp");
            const double r = rel_l1(ref_cmp, to_d(got_cmp));
            if (!(r <= 1e-6)) {
                std::printf("      *** completion at pos %lld rel %.3e ***\n", (long long) pos, r);
                ++mismatches;
            }
        }
    }
    check(cudaMemcpy(got_ks.data(), d_ks, (size_t) reg * 4, cudaMemcpyDeviceToHost), "gks2");
    check(cudaMemcpy(got_ss.data(), d_ss, (size_t) reg * 4, cudaMemcpyDeviceToHost), "gss2");
    const double rel_d = std::max(rel_l1(rks, to_d(got_ks)), rel_l1(rss, to_d(got_ss)));
    std::printf("    %-46s %d completions, %d off, carry rel %.3e\n", "decode sequence", completions,
                mismatches, rel_d);
    if (mismatches || !(rel_d <= 1e-6)) ++bad;

    // ---- 4. prefill WITH a carry (the extend path): block 0's previous half comes from the register
    {
        std::vector<double> ck = rks, cs = rss;  // the seeded register as the carry source
        // only the first `ratio` rows are read as the previous block; for non-overlap there is no carry
        if (sh.overlap) {
            std::vector<float> carry_k((size_t) sh.ratio * item), carry_s((size_t) sh.ratio * item);
            for (size_t i = 0; i < (size_t) sh.ratio * item; ++i) {
                carry_k[i] = (float) ck[i];
                carry_s[i] = std::isinf(cs[i]) ? -FLT_MAX : (float) cs[i];
            }
            float *d_ck, *d_cs;
            check(cudaMalloc(&d_ck, (size_t) sh.ratio * item * 4), "ck");
            check(cudaMalloc(&d_cs, (size_t) sh.ratio * item * 4), "cs");
            check(cudaMemcpy(d_ck, carry_k.data(), (size_t) sh.ratio * item * 4, cudaMemcpyHostToDevice), "mck");
            check(cudaMemcpy(d_cs, carry_s.data(), (size_t) sh.ratio * item * 4, cudaMemcpyHostToDevice), "mcs");
            strata::kernels::compressor_prefill(seqlen, sh.ratio, sh.overlap, d, d_kv, d_sc, d_ape, d_ck,
                                                d_cs, d_out, nullptr);
            check(cudaMemcpy(got_out.data(), d_out, got_out.size() * 4, cudaMemcpyDeviceToHost), "gout2");
            std::vector<double> want2;
            std::vector<double> cs_f(cs.size());
            for (size_t i = 0; i < cs.size(); ++i) cs_f[i] = std::isinf(cs[i]) ? -1e30 : cs[i];
            ref_prefill(sh, seqlen, kv, score, ape, ck.data(), cs_f.data(), want2);
            const double r = rel_l1(want2, to_d(got_out));
            std::printf("    %-46s rel %.3e\n", "prefill with carry vs the reference", r);
            if (!(r <= 1e-6)) { std::printf("      *** over 1e-6 ***\n"); ++bad; }
            cudaFree(d_ck);
            cudaFree(d_cs);
        }
    }

    // ---- 5. traps: the wrong readings must be OBSERVABLY different.  Each is measured WHERE IT ACTS:
    // the wrong-half swap only touches blocks after 0, and the -inf masking only touches block 0 - a
    // whole-output measure dilutes them under the noise of the blocks they do not change.
    if (sh.overlap) {
        const size_t total = want.size(), dw = (size_t) sh.d;
        std::vector<double> t1, t2, t3;
        ref_prefill(sh, seqlen, kv, score, ape, nullptr, nullptr, t1, true);
        ref_prefill(sh, seqlen, kv, score, ape, nullptr, nullptr, t2, false, true);
        ref_prefill(sh, seqlen, kv, score, ape, nullptr, nullptr, t3, false, false, true);
        std::printf("    %-46s row-softmax %.2e, wrong-half %.2e, 0-for--inf %.2e\n",
                    "wrong readings observably apart", rel_l1(want, t1),
                    rel_l1_range(want, t2, dw, total), rel_l1_range(want, t3, 0, dw));
        if (!(rel_l1(want, t1) > 1e-3 && rel_l1_range(want, t2, dw, total) > 1e-3 &&
              rel_l1_range(want, t3, 0, dw) > 1e-3)) {
            std::printf("      *** a trap is not observable ***\n");
            ++bad;
        }
    }

    cudaFree(d_kv); cudaFree(d_sc); cudaFree(d_ape); cudaFree(d_out);
    cudaFree(d_ks); cudaFree(d_ss); cudaFree(d_cmp);
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: compressor_parity [--selftest]\n"); return 2; }
    }
    int bad = 0;
    // the real shapes: attention compressor r=4 (d = kv_lora_rank 512) and r=128, indexer r=4 (d 128)
    run_shape({4, true, 512}, 1000, 11, bad);
    run_shape({128, false, 512}, 1000, 12, bad);
    run_shape({4, true, 128}, 1000, 13, bad);
    std::printf("compressor_parity: %s\n", bad ? "*** FAIL ***" : "ok");
    return bad ? 1 : 0;
}
