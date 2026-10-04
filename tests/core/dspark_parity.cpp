// tests/core/dspark_parity.cpp - the DSpark drafter's own math, docs/DSV4.md P4.
//
// The drafter's three stages are the r=0 dsv4 block, already parity-tested (`dsv4_block_layer_parity`),
// and the borrowed head is the target's.  What is NEW here is the wiring around them, and this test
// checks exactly that against an independent float64 reference over the REAL sidecar weights:
//
//   1. `hc_mean` - the capture the drafter reads (the mean over the target's hc streams).
//   2. `inject`  - the fc fusion + `enc.output_norm` + each stage's kv projection, norm, rope and
//                  ring write.
//   3. `draft`   - the Markov chain and the confidence head, against a float64 chain recomputed
//                  from the drafter's own base logits.
//
// Needs the GPU and the real files:
//   dspark_parity --sidecar-pack <pack> --sidecar-gguf <sidecar.gguf> --target-shard1 <shard1.gguf>
#include "strata/artifact/dflash_geometry.hpp"
#include "strata/artifact/dsv4_geometry.hpp"
#include "strata/artifact/dequant.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/gguf_split.hpp"
#include "strata/core/dspark_forward.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/dsv4_hc.hpp"
#include "strata/kernels/dsv4_rope.hpp"
#include "strata/kernels/dsv4_quant.hpp"
#include "strata/kernels/elementwise.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace strata::core;
using namespace strata::kernels;

namespace {

int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-72s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok) ++g_fail;
}

/// Relative L1 between two vectors - the same comparison shape the P2 parity tests use.
double rel_l1(const float* a, const float* b, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        num += std::fabs((double) a[i] - (double) b[i]);
        den += std::fabs((double) b[i]);
    }
    return den > 0.0 ? num / den : (num > 0.0 ? 1.0 : 0.0);
}

std::set<std::string> skip_rows(const std::string& pack) {
    std::set<std::string> skip;
    std::ifstream idx(pack + "/index.txt");
    std::string line;
    while (std::getline(idx, line)) {
        if (line.empty() || line[0] == '#') continue;
        char name[256] = {0};
        int file = 0, kind = 0, bits = 0;
        unsigned long long dummy = 0, dst_bytes = 0;
        long long ne = 0;
        if (std::sscanf(line.c_str(), "%255s %d %d %llu %llu %llu %llu %lld %lld %d", name, &file, &kind,
                        &dummy, &dummy, &dummy, &dst_bytes, &ne, &ne, &bits) != 10)
            continue;
        if (kind == 0 && bits != 0 && dst_bytes == 0) skip.insert(name);
    }
    return skip;
}

/// One Q8_0 row of a GGUF tensor, dequantized to float (ne0 contiguous, 32 elements per 34-byte block).
void dequant_q8_0_row(const uint8_t* base, int64_t ne0, int64_t row, std::vector<float>& out) {
    out.assign((size_t) ne0, 0.0f);
    const int64_t blocks = ne0 / 32;
    for (int64_t b = 0; b < blocks; ++b)
        strata::dequantize_q8_0(base + (row * blocks + b) * 34, out.data() + (size_t) b * 32);
}

void rms_ref(const float* x, const float* w, int64_t n, float eps, std::vector<double>& out) {
    double ss = 0.0;
    for (int64_t i = 0; i < n; ++i) ss += (double) x[i] * (double) x[i];
    const double inv = 1.0 / std::sqrt(ss / (double) n + (double) eps);
    out.assign((size_t) n, 0.0);
    for (int64_t i = 0; i < n; ++i) out[(size_t) i] = (double) x[i] * (double) w[i] * inv;
}

const strata::TensorInfo* find_tensor(const strata::GgufFile& f, const std::string& name) {
    for (const auto& t : f.tensors())
        if (t.name == name) return &t;
    return nullptr;
}

float* dalloc(size_t bytes) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) throw std::runtime_error("cudaMalloc");
    return (float*) p;
}

}  // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("dspark_parity: no CUDA device, skipped\n");
        return 77;
    }
    std::string sidecar_pack, sidecar_gguf, target_shard1;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--sidecar-pack") && i + 1 < argc) sidecar_pack = argv[++i];
        else if (!std::strcmp(argv[i], "--sidecar-gguf") && i + 1 < argc) sidecar_gguf = argv[++i];
        else if (!std::strcmp(argv[i], "--target-shard1") && i + 1 < argc) target_shard1 = argv[++i];
    }
    if (sidecar_pack.empty() || sidecar_gguf.empty() || target_shard1.empty()) {
        std::printf("usage: dspark_parity --sidecar-pack P --sidecar-gguf G --target-shard1 S\n");
        return 2;
    }

    std::string err;
    ModelGeometry gd;
    {
        strata::GgufFile fs(sidecar_gguf);
        if (!strata::artifact::dflash_geometry(fs, gd, err)) {
            std::printf("dspark_parity: sidecar geometry: %s\n", err.c_str());
            return 1;
        }
    }
    ModelGeometry gt;
    {
        strata::GgufFile ft(target_shard1);
        if (!strata::artifact::deepseek4_geometry(ft, gt, err)) {
            std::printf("dspark_parity: target geometry: %s\n", err.c_str());
            return 1;
        }
    }
    gd.dsv4.n_vocab = gt.dsv4.n_vocab;
    const int64_t dim = gd.n_embd, hd = gd.head_dim, rd = gd.dsv4.rope_dim, hc = gd.hc;
    const int64_t cap = (int64_t) gd.dflash.target_layers.size();
    const int64_t n_vocab = gd.dsv4.n_vocab;

    const std::set<std::string> skip = skip_rows(sidecar_pack);
    uint64_t pool = 0;
    if (!WeightTable::pool_bytes(sidecar_pack, pool, err, &skip)) {
        std::printf("dspark_parity: pool: %s\n", err.c_str());
        return 1;
    }
    void* arena = nullptr;
    if (cudaMalloc(&arena, (size_t) pool) != cudaSuccess) { std::printf("dspark_parity: no VRAM\n"); return 1; }
    WeightTable wt;
    if (!wt.load(sidecar_pack, arena, pool, err, &skip)) { std::printf("dspark_parity: load: %s\n", err.c_str()); return 1; }
    const std::vector<std::string> dshards = strata::gguf_split_paths(sidecar_gguf);
    NativeDense dense;
    if (!dense.load(dshards, wt, err, false)) { std::printf("dspark_parity: native dense: %s\n", err.c_str()); return 1; }

    cudaStream_t cu = nullptr;
    if (cudaStreamCreate(&cu) != cudaSuccess) { std::printf("dspark_parity: stream\n"); return 1; }

    // ---- 1: hc_mean, the capture the drafter reads --------------------------------------------
    std::printf("1. hc_mean (the mean over the target's hc streams)\n");
    {
        const int64_t n = 2, d = 64;
        std::vector<float> h((size_t) n * hc * d);
        uint64_t s = 12345;
        for (auto& v : h) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; v = (float)((int64_t)(s >> 33) % 2000) / 500.0f - 2.0f; }
        std::vector<float> want((size_t) n * d);
        for (int64_t m = 0; m < n; ++m)
            for (int64_t j = 0; j < d; ++j) {
                double acc = 0.0;
                for (int64_t h2 = 0; h2 < hc; ++h2) acc += (double) h[(size_t) (m * hc + h2) * d + j];
                want[(size_t) m * d + j] = (float) (acc / (double) hc);
            }
        float* din = dalloc(h.size() * 4);
        float* dout = dalloc(want.size() * 4);
        cudaMemcpy(din, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
        hc_mean(din, n, hc, d, dout, (void*) cu);
        std::vector<float> got(want.size());
        cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost);
        cudaFree(din); cudaFree(dout);
        double worst = 0.0;
        for (size_t i = 0; i < got.size(); ++i)
            worst = std::max(worst, std::fabs((double) got[i] - (double) want[i]) /
                                    (1.0 + std::fabs((double) want[i])));
        check(worst < 1e-5, "every captured row is the mean of the hc streams");
        std::printf("    worst abs err %.3e\n", worst);
    }

    // ---- 2: inject, the encoder path over the real sidecar weights ---------------------------
    std::printf("2. inject (fc fusion + enc norm + each stage's kv/norm/rope/ring write)\n");
    NativeEmbed embed;  // not loaded: inject() does not touch the borrowed tables
    NativeHead head;
    DsparkForward drafter;
    if (!drafter.init(gd, sidecar_pack, dshards, wt, embed, head, err, 256)) {
        std::printf("dspark_parity: drafter init: %s\n", err.c_str());
        return 1;
    }
    {
        const int64_t n = 2, pos0 = 5;
        std::vector<float> fused((size_t) n * cap * dim);
        uint64_t s = 999;
        for (auto& v : fused) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; v = (float)((int64_t)(s >> 35) % 400) / 200.0f - 1.0f; }
        float* dfused = dalloc(fused.size() * 4);
        cudaMemcpy(dfused, fused.data(), fused.size() * 4, cudaMemcpyHostToDevice);
        if (!drafter.inject(dfused, pos0, n, err)) { std::printf("  inject failed: %s\n", err.c_str()); return 1; }

        // The reference: fc (Q8_0, dequantized) -> enc.output_norm -> each stage's attn_kv (Q8_0)
        // -> attn_kv_a_norm.  The rope and the e4m3 round-trip are the engine's own kernels -
        // both are parity-tested in P2 - applied to the reference values, so what this compares is
        // the fusion, the norms, the projection and the ring addressing.
        strata::GgufFile fs(sidecar_gguf);
        const strata::TensorInfo* t_fc = find_tensor(fs, "fc.weight");
        const strata::TensorInfo* t_enc = find_tensor(fs, "enc.output_norm.weight");
        if (!t_fc || !t_enc) { std::printf("  fc / enc.output_norm not in the sidecar\n"); return 1; }
        std::vector<float> enc_w(dim);
        strata::dequantize_f32(fs.tensor_data(*t_enc), enc_w.data(), (int) dim);
        std::vector<float> fc_row((size_t) cap * dim);
        for (int64_t l = 0; l < gd.n_layers; ++l) {
            const std::string kvn = "blk." + std::to_string(l) + ".attn_kv.weight";
            const std::string knn = "blk." + std::to_string(l) + ".attn_kv_a_norm.weight";
            const strata::TensorInfo* t_kv = find_tensor(fs, kvn);
            const strata::TensorInfo* t_kn = find_tensor(fs, knn);
            if (!t_kv || !t_kn) { std::printf("  %s missing\n", kvn.c_str()); return 1; }
            std::vector<float> knorm(hd);
            strata::dequantize_f32(fs.tensor_data(*t_kn), knorm.data(), (int) hd);
            std::vector<float> kv_row((size_t) dim);
            for (int64_t t = 0; t < n; ++t) {
                // recompute this token's fused -> enc norm -> kv -> kv norm in float64
                std::vector<float> xrow((size_t) cap * dim);
                std::memcpy(xrow.data(), fused.data() + (size_t) t * cap * dim, xrow.size() * 4);
                std::vector<float> fc_out((size_t) dim);
                for (int64_t j = 0; j < dim; ++j) {
                    dequant_q8_0_row(fs.tensor_data(*t_fc), cap * dim, j, fc_row);
                    double acc = 0.0;
                    for (int64_t i = 0; i < cap * dim; ++i) acc += (double) fc_row[(size_t) i] * (double) xrow[(size_t) i];
                    fc_out[(size_t) j] = (float) acc;
                }
                std::vector<double> normed;
                rms_ref(fc_out.data(), enc_w.data(), dim, gd.dsv4.norm_eps, normed);
                std::vector<float> normed_f((size_t) dim);
                for (int64_t i = 0; i < dim; ++i) normed_f[(size_t) i] = (float) normed[(size_t) i];
                std::vector<float> kv((size_t) hd);
                for (int64_t j = 0; j < hd; ++j) {
                    dequant_q8_0_row(fs.tensor_data(*t_kv), dim, j, kv_row);
                    double acc = 0.0;
                    for (int64_t i = 0; i < dim; ++i) acc += (double) kv_row[(size_t) i] * (double) normed_f[(size_t) i];
                    kv[(size_t) j] = (float) acc;
                }
                std::vector<double> kn;
                rms_ref(kv.data(), knorm.data(), hd, gd.dsv4.norm_eps, kn);
                std::vector<float> ref((size_t) hd);
                for (int64_t i = 0; i < hd; ++i) ref[(size_t) i] = (float) kn[(size_t) i];
                // the engine's rope + e4m3 round-trip, applied to the reference ON THE DEVICE
                // (these are device kernels; feeding them a host pointer is an illegal access)
                float* d_ref = nullptr;
                uint16_t* d_ref_bf = nullptr;
                cudaMalloc((void**) &d_ref, (size_t) hd * 4);
                cudaMalloc((void**) &d_ref_bf, (size_t) hd * 2);
                cudaMemcpy(d_ref, ref.data(), (size_t) hd * 4, cudaMemcpyHostToDevice);
                dsv4_rope_f32(d_ref, 1, hd, rd, pos0 + t, gd.dsv4.rope_theta, gd.dsv4.yarn_factor,
                              0, gd.dsv4.yarn_beta_fast, gd.dsv4.yarn_beta_slow, false, (void*) cu);
                roundtrip_fp8_e4m3(d_ref, d_ref_bf, hd - rd, 64, (void*) cu);
                f32_to_bf16_bulk(d_ref + (hd - rd), d_ref_bf + (hd - rd), rd, (void*) cu);
                cudaStreamSynchronize(cu);
                std::vector<uint16_t> ref_bf((size_t) hd);
                cudaMemcpy(ref_bf.data(), d_ref_bf, (size_t) hd * 2, cudaMemcpyDeviceToHost);
                cudaFree(d_ref);
                cudaFree(d_ref_bf);
                std::vector<uint16_t> got((size_t) hd);
                cudaMemcpy(got.data(), drafter.ring_row(l, pos0 + t), (size_t) hd * 2, cudaMemcpyDeviceToHost);
                std::vector<float> gf((size_t) hd), rf((size_t) hd);
                for (int64_t i = 0; i < hd; ++i) { gf[(size_t) i] = f32_from_bf16(got[(size_t) i]); rf[(size_t) i] = f32_from_bf16(ref_bf[(size_t) i]); }
                const double e = rel_l1(gf.data(), rf.data(), (size_t) hd);
                char msg[128];
                std::snprintf(msg, sizeof msg, "stage %lld position %lld ring row", (long long) l, (long long) (pos0 + t));
                check(e < 2e-2, msg);
                std::printf("    rel_l1 %.3e\n", e);
            }
        }
        cudaFree(dfused);
    }

    // ---- 3: the Markov chain and the confidence head -------------------------------------------
    std::printf("3. draft (Markov chain + confidence head, against a float64 chain)\n");
    {
        const std::vector<std::string> tshards = strata::gguf_split_paths(target_shard1);
        NativeEmbed temb;
        NativeHead thead;
        if (!temb.load(tshards, gt.n_embd, gt.dsv4.n_vocab, err)) { std::printf("  target embed: %s\n", err.c_str()); return 1; }
        if (!thead.load(tshards, gt.n_embd, gt.dsv4.n_vocab, err)) { std::printf("  target head: %s\n", err.c_str()); return 1; }
        DsparkForward d2;
        if (!d2.init(gd, sidecar_pack, dshards, wt, temb, thead, err, 256)) {
            std::printf("  drafter init with the borrowed head: %s\n", err.c_str());
            return 1;
        }
        // Fill the drafter's ring for the positions before the block, as the serve loop would.
        const int64_t pre = 8;
        std::vector<float> fused((size_t) pre * cap * dim);
        uint64_t s = 7;
        for (auto& v : fused) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; v = (float)((int64_t)(s >> 35) % 400) / 200.0f - 1.0f; }
        float* dfused = dalloc(fused.size() * 4);
        cudaMemcpy(dfused, fused.data(), fused.size() * 4, cudaMemcpyHostToDevice);
        if (!d2.inject(dfused, 0, pre, err)) { std::printf("  inject: %s\n", err.c_str()); return 1; }
        cudaFree(dfused);

        const int64_t anchor = 1000, pos = pre, depth = 3;
        std::vector<int64_t> toks;
        std::vector<float> conf;
        if (!d2.draft(anchor, pos, depth, toks, conf, err)) { std::printf("  draft: %s\n", err.c_str()); return 1; }

        // The reference chain: the drafter's own base logits, plus the Markov bias and the
        // confidence dot computed in float64 straight out of the sidecar's BF16 tensors.
        std::vector<float> base((size_t) depth * n_vocab);
        cudaMemcpy(base.data(), d2.base_logits(), base.size() * 4, cudaMemcpyDeviceToHost);
        std::vector<float> temb_h((size_t) depth * dim);
        cudaMemcpy(temb_h.data(), d2.pre_norm_hidden(), temb_h.size() * 4, cudaMemcpyDeviceToHost);
        std::vector<uint16_t> w1((size_t) d2.markov_rank() * n_vocab), w2(w1.size());
        cudaMemcpy(w1.data(), d2.markov_w1(), w1.size() * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(w2.data(), d2.markov_w2(), w2.size() * 2, cudaMemcpyDeviceToHost);
        std::vector<uint16_t> cp((size_t) (dim + d2.markov_rank()));
        cudaMemcpy(cp.data(), d2.conf_proj(), cp.size() * 2, cudaMemcpyDeviceToHost);

        int64_t prev = anchor;
        bool same_tokens = true;
        double worst_conf = 0.0;
        for (int64_t i = 0; i < depth; ++i) {
            std::vector<double> bias((size_t) n_vocab, 0.0);
            for (int64_t v = 0; v < n_vocab; ++v) {
                double acc = 0.0;
                const uint16_t* wr = w2.data() + (size_t) v * d2.markov_rank();
                for (int64_t r = 0; r < d2.markov_rank(); ++r)
                    acc += (double) f32_from_bf16(wr[(size_t) r]) * (double) f32_from_bf16(w1[(size_t) prev * d2.markov_rank() + r]);
                bias[(size_t) v] = acc;
            }
            int64_t best = 0;
            double bestv = -1e300;
            for (int64_t v = 0; v < n_vocab; ++v) {
                const double s2 = (double) base[(size_t) i * n_vocab + v] + bias[(size_t) v];
                if (s2 > bestv) { bestv = s2; best = v; }
            }
            if (best != toks[(size_t) i]) same_tokens = false;
            double dot = 0.0;
            for (int64_t k = 0; k < dim; ++k)
                dot += (double) temb_h[(size_t) i * dim + k] * (double) f32_from_bf16(cp[(size_t) k]);
            for (int64_t k = 0; k < d2.markov_rank(); ++k)
                dot += (double) f32_from_bf16(w1[(size_t) prev * d2.markov_rank() + k]) *
                       (double) f32_from_bf16(cp[(size_t) (dim + k)]);
            const double c = 1.0 / (1.0 + std::exp(-dot));
            worst_conf = std::max(worst_conf, std::fabs(c - (double) conf[(size_t) i]));
            std::printf("    pos %lld: engine %lld reference %lld, conf engine %.6f reference %.6f\n",
                        (long long) (pos + i), (long long) toks[(size_t) i], (long long) best,
                        (double) conf[(size_t) i], c);
            prev = best;
        }
        check(same_tokens, "every draft token is the float64 chain's argmax");
        check(worst_conf < 1e-4, "every confidence matches the float64 confidence head");
        std::printf("    worst confidence diff %.3e\n", worst_conf);
    }

    std::printf(g_fail ? "dspark_parity: %d FAILED\n" : "dspark_parity: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
