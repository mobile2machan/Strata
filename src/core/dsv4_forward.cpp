// src/core/dsv4_forward.cpp - see include/strata/core/dsv4_forward.hpp.
#include "strata/core/dsv4_forward.hpp"

#include "strata/kernels/dsv4_hc.hpp"
#include "strata/kernels/elementwise.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace strata::core {
namespace {

bool parse_layer_experts(const std::string& path, int64_t n_layers, int64_t n_expert, int64_t dim,
                         int64_t ff, std::vector<int64_t>& off, int64_t& max_blob,
                         std::vector<kernels::NativeExpertLayout>& lay, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open " + path; return false; }
    std::vector<int64_t> blob((size_t) n_layers, 0);
    std::vector<int> gu((size_t) n_layers, -1), d((size_t) n_layers, -1);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        int layer, g, dd;
        long long offset, bl;
        if (!(ss >> layer >> g >> dd >> offset >> bl)) continue;
        if (layer < 0 || layer >= n_layers) continue;
        gu[(size_t) layer] = g;
        d[(size_t) layer] = dd;
        blob[(size_t) layer] = bl;
    }
    int64_t cur = 0;
    max_blob = 0;
    for (int64_t l = 0; l < n_layers; ++l) {
        if (gu[(size_t) l] < 0) { err = "native_experts.txt has no row for layer " + std::to_string(l); return false; }
        const kernels::NativeExpertLayout nl = kernels::native_expert_layout(gu[(size_t) l], d[(size_t) l], dim, ff);
        if (nl.bytes != (size_t) blob[(size_t) l]) {
            err = "layer " + std::to_string(l) + ": native_expert_layout bytes != pack blob_bytes";
            return false;
        }
        off.push_back(cur);
        max_blob = std::max(max_blob, (int64_t) nl.bytes);
        lay.push_back(nl);
        cur += blob[(size_t) l] * n_expert;
    }
    return true;
}

}  // namespace

Dsv4Forward::~Dsv4Forward() {
    cudaFree(d_a_); cudaFree(d_b_); cudaFree(d_mixes_); cudaFree(d_pre_); cudaFree(d_y_);
    cudaFree(d_logits_); cudaFree(d_scratch_); cudaFree(d_blobs_);
    if (h_stage_ != nullptr) {
        cudaHostUnregister(h_stage_);
        std::free(h_stage_);
    }
    if (experts_file_ != nullptr) std::fclose(experts_file_);
    if (stream_ != nullptr) cudaStreamDestroy((cudaStream_t) stream_);
}

bool Dsv4Forward::init(const ModelGeometry& g, const std::string& pack_dir,
                       const std::vector<std::string>& shards, WeightTable& wt, std::string& err,
                       int64_t max_seq) {
    g_ = g;
    wt_ = &wt;
    if (!embed_.load(shards, g.n_embd, g.dsv4.n_vocab, err)) return false;
    if (!head_.load(shards, g.n_embd, g.dsv4.n_vocab, err)) return false;
    if (!resolver_.init(g, wt, shards, err)) return false;
    if (!state_.init(g, max_seq, err)) return false;

    auto resident = [&](const char* name, const float*& out) -> bool {
        const WeightRef* ref = wt.find(std::string(name));
        if (!ref || ref->kind != WeightKind::F32 || ref->data == nullptr) {
            err = std::string(name) + " is not a resident F32 tensor";
            return false;
        }
        out = (const float*) ref->data;
        return true;
    };
    if (!resident("output_hc_fn.weight", hc_head_fn_)) return false;
    if (!resident("output_hc_base.weight", hc_head_base_)) return false;
    if (!resident("output_hc_scale.weight", hc_head_scale_)) return false;
    if (!resident("output_norm.weight", out_norm_)) return false;

    int64_t max_blob = 0;
    if (!parse_layer_experts(pack_dir + "/native_experts.txt", g.n_layers, g.n_expert, g.n_embd,
                             g.n_ff, layer_off_, max_blob, layer_layout_, err))
        return false;
    experts_file_ = std::fopen((pack_dir + "/experts.bin").c_str(), "rb");
    if (experts_file_ == nullptr) { err = "cannot open " + pack_dir + "/experts.bin"; return false; }
    const int64_t stage_bytes = max_blob * g.dsv4.n_expert_used;
    h_stage_ = (uint8_t*) std::malloc((size_t) stage_bytes);
    if (h_stage_ == nullptr) { err = "staging malloc"; return false; }
    if (cudaHostRegister(h_stage_, (size_t) stage_bytes, cudaHostRegisterMapped) != cudaSuccess)
        (void) cudaGetLastError();  // pageable staging is legal, just slower
    if (cudaMalloc((void**) &d_blobs_, (size_t) stage_bytes) != cudaSuccess) { err = "cudaMalloc blobs"; return false; }

    const int64_t dim = g.n_embd, hc = g.hc;
    cudaMalloc((void**) &d_a_, (size_t) hc * dim * 4);
    cudaMalloc((void**) &d_b_, (size_t) hc * dim * 4);
    cudaMalloc((void**) &d_mixes_, (size_t) 6 * hc * 4);
    cudaMalloc((void**) &d_pre_, (size_t) hc * 4);
    cudaMalloc((void**) &d_y_, (size_t) dim * 4);
    cudaMalloc((void**) &d_logits_, (size_t) g.dsv4.n_vocab * 4);
    const int64_t max_stage = max_seq / 4 + 2;
    int64_t scratch = 0;
    for (int64_t l = 0; l < g.n_layers; ++l)
        scratch = std::max(scratch, dsv4_block_scratch_bytes(g, l, max_stage));
    cudaMalloc((void**) &d_scratch_, (size_t) scratch);
    cudaStream_t s = nullptr;
    if (cudaStreamCreate(&s) != cudaSuccess) { err = "cudaStreamCreate"; return false; }
    stream_ = (void*) s;
    return true;
}

bool Dsv4Forward::decode(int64_t token, int64_t pos, std::string& err) {
    const bool dbg = std::getenv("DSV4_DBG") != nullptr;
    auto nonfinite = [&](const float* d, int64_t n) -> std::pair<int64_t, float> {
        std::vector<float> h((size_t) n);
        cudaMemcpy(h.data(), d, (size_t) n * 4, cudaMemcpyDeviceToHost);
        int64_t bad = 0;
        float mx = 0;
        for (float v : h) { if (!std::isfinite(v)) ++bad; else mx = std::max(mx, std::fabs(v)); }
        return {bad, mx};
    };
    const int64_t dim = g_.n_embd, hc = g_.hc;
    for (int64_t h = 0; h < hc; ++h) embed_.gather_one(token, d_a_ + h * dim, stream_);
    if (dbg) {
        const auto [bad, mx] = nonfinite(d_a_, hc * dim);
        std::printf("[dbg] after embed: max %.3e nonfinite %lld\n", mx, (long long) bad);
    }
    float* in = d_a_;
    float* out = d_b_;
    const int64_t n_stage = pos / 4 + 2;
    for (int64_t l = 0; l < g_.n_layers; ++l) {
        stager_.off = layer_off_[(size_t) l];
        Dsv4BlockWeights w;
        if (!resolver_.resolve(g_, *wt_, l, layer_layout_[(size_t) l], nullptr, w, err)) return false;
        if (!dsv4_block_decode_step(g_, l, w, state_.layer(l), in, out, pos, token, n_stage,
                                    d_scratch_, stream_, &stager_)) {
            err = "block step failed at layer " + std::to_string(l);
            return false;
        }
        std::swap(in, out);
        if (dbg) {
            const auto [bad, mx] = nonfinite(in, hc * dim);
            std::printf("[dbg] after layer %lld: max %.3e nonfinite %lld\n", (long long) l, mx, (long long) bad);
            if (bad != 0) break;
        }
    }
    // ---- the head's collapse: hc_mixes, sigmoid pre, combine, output_norm, the native head.
    kernels::hc_mixes(in, hc_head_fn_, d_mixes_, 1, hc * dim, hc, g_.dsv4.hc_eps, stream_);
    kernels::hc_head_pre(d_mixes_, hc_head_scale_, hc_head_base_, d_pre_, 1, hc, g_.dsv4.hc_eps, stream_);
    kernels::hc_pre_combine(in, d_pre_, d_y_, 1, hc, dim, stream_);
    kernels::rms_norm_weighted(d_y_, out_norm_, 1, dim, g_.dsv4.norm_eps, stream_);
    if (dbg) std::printf("[dbg] y: max %.3e nonfinite %lld\n", (double) nonfinite(d_y_, dim).second, (long long) nonfinite(d_y_, dim).first);
    if (!head_.run(d_y_, d_logits_, stream_, err)) return false;
    if (dbg) std::printf("[dbg] logits nonfinite %lld\n", (long long) nonfinite(d_logits_, g_.dsv4.n_vocab).first);
    return true;
}

bool Dsv4Forward::Stager::stage(const int32_t* ids, int64_t n, int64_t bytes, const uint8_t** out) {
    for (int64_t i = 0; i < n; ++i) {
        const int64_t at = off + (int64_t) ids[i] * bytes;
#ifdef _WIN32
        if (_fseeki64(f->experts_file_, (__int64) at, SEEK_SET) != 0) return false;
#else
        if (fseeko(f->experts_file_, (off_t) at, SEEK_SET) != 0) return false;
#endif
        if (std::fread(f->h_stage_ + (size_t) i * bytes, 1, (size_t) bytes, f->experts_file_) !=
            (size_t) bytes)
            return false;
    }
    cudaMemcpyAsync(f->d_blobs_, f->h_stage_, (size_t) n * bytes, cudaMemcpyHostToDevice,
                    (cudaStream_t) f->stream_);
    cudaStreamSynchronize((cudaStream_t) f->stream_);  // the slots are reused by the next call
    *out = f->d_blobs_;
    return true;
}

}  // namespace strata::core
