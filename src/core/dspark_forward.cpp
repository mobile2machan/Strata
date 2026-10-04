// src/core/dspark_forward.cpp - see include/strata/core/dspark_forward.hpp.
#include "strata/core/dspark_forward.hpp"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/dsv4_hc.hpp"
#include "strata/kernels/dsv4_quant.hpp"
#include "strata/kernels/dsv4_rope.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

namespace strata::core {
namespace {

// Same table the target's forward parses (the sidecar pack writes its own native_experts.txt).
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

DsparkForward::~DsparkForward() {
    cudaFree(d_pa_); cudaFree(d_pb_); cudaFree(d_ptok_); cudaFree(d_mixes_); cudaFree(d_pre_);
    cudaFree(d_y_); cudaFree(d_normed_); cudaFree(d_logits_); cudaFree(d_enc_); cudaFree(d_kv_);
    cudaFree(d_kv_bf_); cudaFree(d_xq_enc_); cudaFree(d_xq_dim_); cudaFree(d_w1row_);
    cudaFree(d_bias_); cudaFree(d_scratch_); cudaFree(d_blobs_);
    if (h_stage_ != nullptr) {
        cudaHostUnregister(h_stage_);
        std::free(h_stage_);
    }
    if (experts_file_ != nullptr) std::fclose(experts_file_);
    if (stream_ != nullptr) cudaStreamDestroy((cudaStream_t) stream_);
}

bool DsparkForward::init(const ModelGeometry& g, const std::string& pack_dir,
                         const std::vector<std::string>& shards, WeightTable& wt,
                         const NativeEmbed& embed, const NativeHead& head, std::string& err,
                         int64_t max_seq) {
    g_ = g;
    wt_ = &wt;
    embed_ = &embed;
    head_ = &head;
    if (g_.arch != ModelArch::DFlash || g_.dflash.target_layers.empty() ||
        g_.dflash.mask_token_id < 0) {
        err = "dspark: the geometry is not a complete dflash geometry";
        return false;
    }
    if (!resolver_.init(g, wt, shards, err)) return false;
    if (!state_.init(g, max_seq, err)) return false;

    auto resident = [&](const char* name, int64_t elements, const float*& out) -> bool {
        const WeightRef* ref = wt.find(std::string(name));
        if (!ref || ref->kind != WeightKind::F32 || ref->elements != elements || ref->data == nullptr) {
            err = std::string(name) + " is not a resident F32 tensor";
            return false;
        }
        out = (const float*) ref->data;
        return true;
    };
    if (!resident("enc.output_norm.weight", g.n_embd, fc_norm_)) return false;
    if (!resident("output_hc_fn.weight", g.hc_dim() * g.hc, hc_head_fn_)) return false;
    if (!resident("output_hc_base.weight", g.hc, hc_head_base_)) return false;
    if (!resident("output_hc_scale.weight", 1, hc_head_scale_)) return false;
    if (!resident("output_norm.weight", g.n_embd, out_norm_)) return false;

    {
        const WeightRef* ref = wt.find("fc.weight");
        if (!ref || ref->ne0 != (int64_t) g.dflash.target_layers.size() * g.n_embd ||
            ref->ne1 != g.n_embd || ref->native_data == nullptr || ref->native_type < 0) {
            err = "fc.weight is not a native GEMV of [target_layers * n_embd, n_embd]";
            return false;
        }
        fc_type_ = ref->native_type;
        fc_ = (const uint8_t*) ref->native_data;
    }
    auto bf16 = [&](const char* name, int64_t n0, int64_t n1, const uint16_t*& out) -> bool {
        const WeightRef* ref = wt.find(std::string(name));
        if (!ref || ref->kind != WeightKind::Bf16InF32 || ref->ne0 != n0 || ref->ne1 != n1 ||
            ref->data == nullptr) {
            err = std::string(name) + " is not resident raw BF16 [" + std::to_string(n0) + "," +
                  std::to_string(n1) + "]";
            return false;
        }
        out = (const uint16_t*) ref->data;
        return true;
    };
    const int64_t rank0 = g.n_embd;  // placeholder; the real rank is the tensor's own ne0, pinned below
    (void) rank0;
    {
        const WeightRef* ref = wt.find("markov_w1.weight");
        if (!ref || ref->kind != WeightKind::Bf16InF32 || ref->data == nullptr ||
            ref->ne0 <= 0 || ref->ne1 != g.dsv4.n_vocab) {
            err = "markov_w1.weight is not resident raw BF16 [rank, n_vocab]";
            return false;
        }
        markov_rank_ = ref->ne0;
        markov_w1_ = (const uint16_t*) ref->data;
    }
    const int64_t rank = markov_rank_;
    if (!bf16("markov_w2.weight", rank, g.dsv4.n_vocab, markov_w2_)) return false;
    if (!bf16("conf_proj.weight", g.n_embd + rank, 1, conf_proj_)) return false;
    h_conf_proj_.assign((size_t) (g.n_embd + rank), 0);
    if (cudaMemcpy(h_conf_proj_.data(), conf_proj_, (size_t) (g.n_embd + rank) * 2,
                  cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = "copy conf_proj to host";
        return false;
    }
    h_markov_row_.assign((size_t) rank, 0);

    int64_t max_blob = 0;
    if (!parse_layer_experts(pack_dir + "/native_experts.txt", g.n_layers, g.n_expert, g.n_embd,
                             g.n_ff, layer_off_, max_blob, layer_layout_, err))
        return false;
    experts_file_ = std::fopen((pack_dir + "/experts.bin").c_str(), "rb");
    if (experts_file_ == nullptr) { err = "cannot open " + pack_dir + "/experts.bin"; return false; }
    const int64_t stage_bytes = max_blob * g.dsv4.n_expert_used * 8;
    h_stage_ = (uint8_t*) std::malloc((size_t) stage_bytes);
    if (h_stage_ == nullptr) { err = "staging malloc"; return false; }
    if (cudaHostRegister(h_stage_, (size_t) stage_bytes, cudaHostRegisterMapped) != cudaSuccess)
        (void) cudaGetLastError();
    if (cudaMalloc((void**) &d_blobs_, (size_t) stage_bytes) != cudaSuccess) { err = "cudaMalloc blobs"; return false; }

    const int64_t dim = g.n_embd, hc = g.hc, hd = g.head_dim;
    const int64_t cap = (int64_t) g.dflash.target_layers.size();
    const int64_t mix_hc = (2 + hc) * hc;
    const int64_t ntok = 8;
    cudaMalloc((void**) &d_pa_, (size_t) ntok * hc * dim * 4);
    cudaMalloc((void**) &d_pb_, (size_t) ntok * hc * dim * 4);
    cudaMalloc((void**) &d_ptok_, (size_t) ntok * sizeof(int64_t));
    cudaMalloc((void**) &d_mixes_, (size_t) ntok * mix_hc * 4);
    cudaMalloc((void**) &d_pre_, (size_t) ntok * hc * 4);
    cudaMalloc((void**) &d_y_, (size_t) ntok * dim * 4);
    cudaMalloc((void**) &d_normed_, (size_t) ntok * dim * 4);
    cudaMalloc((void**) &d_logits_, (size_t) ntok * g.dsv4.n_vocab * 4);
    cudaMalloc((void**) &d_enc_, (size_t) ntok * dim * 4);
    cudaMalloc((void**) &d_kv_, (size_t) ntok * hd * 4);
    cudaMalloc((void**) &d_kv_bf_, (size_t) ntok * hd * 2);
    cudaMalloc((void**) &d_xq_enc_, (size_t) ntok * kernels::native_q8_1_bytes((int) (cap * dim), 1));
    cudaMalloc((void**) &d_xq_dim_, (size_t) ntok * kernels::native_q8_1_bytes((int) dim, 1));
    cudaMalloc((void**) &d_w1row_, (size_t) rank * 2);
    cudaMalloc((void**) &d_bias_, (size_t) g.dsv4.n_vocab * 4);
    int64_t scratch = 0;
    for (int64_t l = 0; l < g.n_layers; ++l)
        scratch = std::max(scratch, dsv4_block_prefill_scratch_bytes(g, l, ntok, 1));
    cudaMalloc((void**) &d_scratch_, (size_t) scratch);
    cudaStream_t s = nullptr;
    if (cudaStreamCreate(&s) != cudaSuccess) { err = "cudaStreamCreate"; return false; }
    stream_ = (void*) s;
    h_logits_.resize((size_t) ntok * g.dsv4.n_vocab);
    h_bias_.resize((size_t) g.dsv4.n_vocab);
    h_tembd_.resize((size_t) dim);
    return true;
}

bool DsparkForward::inject(const float* d_fused, int64_t pos0, int64_t n, std::string& err) {
    if (n < 1 || n > 8) { err = "dspark inject: n must be 1..8"; return false; }
    const int64_t dim = g_.n_embd, hd = g_.head_dim, rd = g_.dsv4.rope_dim;
    const int64_t cap = (int64_t) g_.dflash.target_layers.size();
    const float eps = g_.dsv4.norm_eps;
    void* cu = stream_;

    // fc + the encoder's output norm.
    kernels::quantize_q8_1_rows(d_fused, n, cap * dim, d_xq_enc_, cu);
    kernels::native_mmvq(fc_type_, fc_, d_xq_enc_, d_enc_, (int) (cap * dim), (int) dim, (int) n, cu);
    kernels::rms_norm_weighted(d_enc_, fc_norm_, n, dim, eps, cu);
    kernels::quantize_q8_1_rows(d_enc_, n, dim, d_xq_dim_, cu);

    // Each stage's own kv projection, norm, rope (the r=0 parameters: plain theta on the trailing
    // dims), e4m3 round-trip, ring write - the same sequence the target's attention writes its
    // window with, so the drafter's ring and its attention agree.
    for (int64_t l = 0; l < g_.n_layers; ++l) {
        Dsv4BlockWeights w;
        if (!resolver_.resolve(g_, *wt_, l, layer_layout_[(size_t) l], nullptr, w, err)) return false;
        kernels::native_mmvq(w.attn.kv_type, w.attn.kv, d_xq_dim_, d_kv_, (int) dim, (int) hd, (int) n, cu);
        kernels::rms_norm_weighted(d_kv_, w.attn.kv_norm, n, hd, eps, cu);
        for (int64_t t = 0; t < n; ++t) {
            const int64_t pos = pos0 + t;
            kernels::dsv4_rope_f32(d_kv_ + t * hd, 1, hd, rd, pos, g_.dsv4.rope_theta,
                                   g_.dsv4.yarn_factor, 0, g_.dsv4.yarn_beta_fast,
                                   g_.dsv4.yarn_beta_slow, false, cu);
            kernels::roundtrip_fp8_e4m3(d_kv_ + t * hd, d_kv_bf_ + t * hd, hd - rd, 64, cu);
            kernels::f32_to_bf16_bulk(d_kv_ + t * hd + (hd - rd), d_kv_bf_ + t * hd + (hd - rd), rd, cu);
            cudaMemcpyAsync(state_.layer(l).window +
                                dsv4_window_slot(pos, g_.dsv4.sliding_window) * hd,
                            d_kv_bf_ + t * hd, (size_t) hd * 2, cudaMemcpyDeviceToDevice,
                            (cudaStream_t) cu);
        }
    }
    return true;
}

bool DsparkForward::draft(int64_t anchor, int64_t pos, int64_t d, std::vector<int64_t>& tokens,
                          std::vector<float>& conf, std::string& err) {
    if (d < 1 || d > 8) { err = "dspark draft: d must be 1..8"; return false; }
    const int64_t dim = g_.n_embd, hc = g_.hc, rank = markov_rank_;
    const int64_t n_vocab = g_.dsv4.n_vocab;
    void* cu = stream_;

    // The block [anchor, mask * (d-1)] at positions pos..pos+d-1, the embedding replicated across
    // the hc streams exactly as the target's decode does.
    std::vector<int64_t> toks((size_t) d);
    toks[0] = anchor;
    for (int64_t i = 1; i < d; ++i) toks[(size_t) i] = g_.dflash.mask_token_id;
    for (int64_t t = 0; t < d; ++t)
        for (int64_t h = 0; h < hc; ++h)
            embed_->gather_one(toks[(size_t) t], d_pa_ + (t * hc + h) * dim, cu);
    cudaMemcpyAsync(d_ptok_, toks.data(), (size_t) d * sizeof(int64_t), cudaMemcpyHostToDevice,
                    (cudaStream_t) cu);

    float* in = d_pa_;
    float* out = d_pb_;
    for (int64_t l = 0; l < g_.n_layers; ++l) {
        stager_.off = layer_off_[(size_t) l];
        Dsv4BlockWeights w;
        if (!resolver_.resolve(g_, *wt_, l, layer_layout_[(size_t) l], nullptr, w, err)) return false;
        if (!dsv4_block_prefill(g_, l, w, state_.layer(l), in, out, d_ptok_, pos, d, 1, d_scratch_,
                                cu, &stager_)) {
            err = "dspark block step failed at stage " + std::to_string(l);
            return false;
        }
        std::swap(in, out);
    }

    // The head's collapse; d_y_ stays the PRE-NORM hidden - the confidence head reads it before
    // output_norm, exactly as the reference graph does.
    kernels::hc_mixes(in, hc_head_fn_, d_mixes_, d, hc * dim, hc, g_.dsv4.hc_eps, cu);
    kernels::hc_head_pre(d_mixes_, hc_head_scale_, hc_head_base_, d_pre_, d, hc, g_.dsv4.hc_eps, cu);
    kernels::hc_pre_combine(in, d_pre_, d_y_, d, hc, dim, cu);
    cudaMemcpyAsync(d_normed_, d_y_, (size_t) d * dim * 4, cudaMemcpyDeviceToDevice, (cudaStream_t) cu);
    kernels::rms_norm_weighted(d_normed_, out_norm_, d, dim, g_.dsv4.norm_eps, cu);
    for (int64_t t = 0; t < d; ++t)
        if (!head_->run(d_normed_ + t * dim, d_logits_ + t * n_vocab, cu, err)) return false;

    // The Markov chain, greedy from the anchor, and the confidence head.  The bias GEMV runs on the
    // GPU (33M MACs per position); the argmax and the 4352-element confidence dot are host work on
    // a handful of rows.
    tokens.assign((size_t) d, 0);
    conf.assign((size_t) d, 0.0f);
    int64_t prev = anchor;
    if (!g_.dflash.sample_from_anchor) { err = "dspark: sample_from_anchor=false is not implemented"; return false; }
    for (int64_t i = 0; i < d; ++i) {
        cudaMemcpyAsync(d_w1row_, markov_w1_ + prev * rank, (size_t) rank * 2,
                        cudaMemcpyDeviceToDevice, (cudaStream_t) cu);
        kernels::bf16_gemv(d_w1row_, markov_w2_, d_bias_, rank, n_vocab, cu);
        cudaMemcpy(h_bias_.data(), d_bias_, (size_t) n_vocab * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_logits_.data(), d_logits_ + i * n_vocab, (size_t) n_vocab * 4,
                   cudaMemcpyDeviceToHost);
        int64_t best = 0;
        float bestv = -1e30f;
        for (int64_t v = 0; v < n_vocab; ++v) {
            const float s = h_logits_[(size_t) v] + h_bias_[(size_t) v];
            if (s > bestv) { bestv = s; best = v; }
        }
        tokens[(size_t) i] = best;

        cudaMemcpy(h_tembd_.data(), d_y_ + i * dim, (size_t) dim * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_markov_row_.data(), markov_w1_ + prev * rank, (size_t) rank * 2,
                   cudaMemcpyDeviceToHost);
        double dot = 0.0;
        for (int64_t k = 0; k < dim; ++k)
            dot += (double) h_tembd_[(size_t) k] *
                   (double) kernels::f32_from_bf16(h_conf_proj_[(size_t) k]);
        for (int64_t k = 0; k < rank; ++k)
            dot += (double) kernels::f32_from_bf16(h_markov_row_[(size_t) k]) *
                   (double) kernels::f32_from_bf16(h_conf_proj_[(size_t) (dim + k)]);
        conf[(size_t) i] = (float) (1.0 / (1.0 + std::exp(-dot)));
        prev = best;
    }
    return true;
}

bool DsparkForward::Stager::stage(const int32_t* ids, int64_t n, int64_t bytes, const uint8_t** out) {
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
    cudaStreamSynchronize((cudaStream_t) f->stream_);
    *out = f->d_blobs_;
    return true;
}

}  // namespace strata::core
