// src/core/dsv4_weights.cpp - see include/strata/core/dsv4_weights.hpp.
#include "strata/core/dsv4_weights.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/elementwise.hpp"

#include <cuda_runtime.h>
#include <memory>

namespace strata::core {
namespace {

std::string lname(int64_t layer, const char* suffix) {
    return "blk." + std::to_string(layer) + "." + suffix;
}

}  // namespace

Dsv4WeightResolver::~Dsv4WeightResolver() {
    for (auto& kv : idx_proj_) cudaFree(kv.second);
    for (auto& kv : tid2eid_) cudaFree(kv.second);
}

bool Dsv4WeightResolver::init(const ModelGeometry& g, const WeightTable& tables,
                              const std::vector<std::string>& shards, std::string& err) {
    // indexer.proj: F32 in the arena -> bf16 device copy, once per indexed layer.
    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (dsv4_ratio(g, l) != 4) continue;
        const WeightRef* ref = tables.find(lname(l, "indexer.proj.weight"));
        if (!ref || ref->kind != WeightKind::F32 || ref->data == nullptr) {
            err = lname(l, "indexer.proj.weight") + " is not a resident F32 tensor";
            return false;
        }
        const int64_t n = ref->elements;
        uint16_t* dst = nullptr;
        if (cudaMalloc(&dst, (size_t) n * 2) != cudaSuccess) { err = "cudaMalloc indexer.proj"; return false; }
        kernels::f32_to_bf16_bulk((const float*) ref->data, dst, n, nullptr);
        idx_proj_[l] = dst;
    }
    // ffn_gate_tid2eid: I32 the pack skips; read it from the shard that holds it.
    std::unique_ptr<GgufModel> model;
    try {
        model = std::make_unique<GgufModel>(shards);
    } catch (const std::exception& e) {
        err = std::string("GGUF: ") + e.what();
        return false;
    }
    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (!dsv4_is_hash_layer(g, l)) continue;
        const std::string name = lname(l, "ffn_gate_tid2eid.weight");
        size_t shard = 0;
        const TensorInfo* t = model->find(name, &shard);
        if (t == nullptr) { err = name + " is not in the model's GGUF shards"; return false; }
        if (t->type != 26 /* I32 */) { err = name + " is GGUF type " + std::to_string(t->type) + ", expected I32"; return false; }
        const int64_t n = (int64_t) t->elements();
        int32_t* dst = nullptr;
        if (cudaMalloc(&dst, (size_t) n * 4) != cudaSuccess) { err = "cudaMalloc tid2eid"; return false; }
        if (cudaMemcpy(dst, model->shard(shard).tensor_data(*t), (size_t) n * 4,
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(dst);
            err = "cudaMemcpy tid2eid " + name;
            return false;
        }
        tid2eid_[l] = dst;
    }
    return true;
}

bool Dsv4WeightResolver::resolve(const ModelGeometry& g, const WeightTable& tables, int64_t layer,
                                 const kernels::NativeExpertLayout& experts, const uint8_t* expert_blobs,
                                 Dsv4BlockWeights& w, std::string& err) const {
    const LayerView v(tables, layer);
    const int64_t r = dsv4_ratio(g, layer);
    const bool indexed = r == 4;
    const bool compressed = r > 0;
    const bool hash = dsv4_is_hash_layer(g, layer);

    auto native = [&](const char* suffix, int64_t n_in, int64_t n_out, int& type,
                      const uint8_t*& data) -> bool {
        const WeightRef* ref = v.get(suffix);
        if (!ref) { err = v.name(suffix) + " is missing"; return false; }
        if (ref->ne0 != n_in || ref->ne1 != n_out) {
            err = v.name(suffix) + " is [" + std::to_string(ref->ne0) + "," + std::to_string(ref->ne1) +
                  "], expected [" + std::to_string(n_in) + "," + std::to_string(n_out) + "]";
            return false;
        }
        if (ref->native_data == nullptr || ref->native_type < 0) {
            err = v.name(suffix) + " is not served by NativeDense";
            return false;
        }
        type = ref->native_type;
        data = (const uint8_t*) ref->native_data;
        return true;
    };
    auto resident = [&](const char* suffix, int64_t elements, WeightKind kind,
                        const float*& data) -> bool {
        const WeightRef* ref = v.get(suffix);
        if (!ref) { err = v.name(suffix) + " is missing"; return false; }
        if (ref->elements != elements || ref->kind != kind || ref->data == nullptr) {
            err = v.name(suffix) + " is not a resident " + std::to_string((int) elements) +
                  "-element form-" + std::to_string((int) kind) + " tensor";
            return false;
        }
        data = (const float*) ref->data;
        return true;
    };

    Dsv4AttnWeights& a = w.attn;
    if (!native("attn_q_a.weight", g.n_embd, g.dsv4.q_lora, a.q_a_type, a.q_a)) return false;
    if (!native("attn_q_b.weight", g.dsv4.q_lora, g.n_head * g.head_dim, a.q_b_type, a.q_b)) return false;
    if (!native("attn_kv.weight", g.n_embd, g.n_head_kv * g.head_dim, a.kv_type, a.kv)) return false;
    if (!native("attn_output_a.weight", g.n_embd, g.dsv4.o_groups * g.dsv4.o_lora, a.out_a_type, a.out_a))
        return false;
    if (!native("attn_output_b.weight", g.dsv4.o_groups * g.dsv4.o_lora, g.n_embd, a.out_b_type, a.out_b))
        return false;
    if (!resident("attn_norm.weight", g.n_embd, WeightKind::F32, a.norm)) return false;
    if (!resident("attn_q_a_norm.weight", g.dsv4.q_lora, WeightKind::F32, a.q_a_norm)) return false;
    if (!resident("attn_kv_a_norm.weight", g.head_dim, WeightKind::F32, a.kv_norm)) return false;
    if (!resident("attn_sinks.weight", g.n_head, WeightKind::F32, a.sinks)) return false;
    if (!resident("hc_attn_fn.weight", g.hc_dim() * 6 * g.hc, WeightKind::F32, a.hc_fn)) return false;
    if (!resident("hc_attn_base.weight", 6 * g.hc, WeightKind::F32, a.hc_base)) return false;
    if (!resident("hc_attn_scale.weight", g.hc - 1, WeightKind::F32, a.hc_scale)) return false;
    if (compressed) {
        const int64_t comp_w = indexed ? 2 * g.head_dim : g.head_dim;
        if (!native("attn_compressor_kv.weight", g.n_embd, comp_w, a.comp_kv_type, a.comp_kv)) return false;
        if (!native("attn_compressor_gate.weight", g.n_embd, comp_w, a.comp_gate_type, a.comp_gate)) return false;
        if (!resident("attn_compressor_ape.weight", comp_w * r, WeightKind::F32, a.ape)) return false;
        if (!resident("attn_compressor_norm.weight", g.head_dim, WeightKind::F32, a.comp_norm)) return false;
    }
    if (indexed) {
        if (!native("indexer.attn_q_b.weight", g.dsv4.q_lora, g.idx_q_heads * g.idx_key_dim, a.idx_qb_type,
                     a.idx_qb))
            return false;
        const auto it = idx_proj_.find(layer);
        if (it == idx_proj_.end()) { err = "indexer.proj for layer " + std::to_string(layer) + " was not converted"; return false; }
        a.idx_proj = it->second;
        if (!native("indexer_compressor_kv.weight", g.n_embd, 2 * g.idx_key_dim, a.idx_comp_kv_type,
                     a.idx_comp_kv))
            return false;
        if (!native("indexer_compressor_gate.weight", g.n_embd, 2 * g.idx_key_dim, a.idx_comp_gate_type,
                     a.idx_comp_gate))
            return false;
        if (!resident("indexer_compressor_ape.weight", 2 * g.idx_key_dim * 4, WeightKind::F32, a.idx_ape))
            return false;
        if (!resident("indexer_compressor_norm.weight", g.idx_key_dim, WeightKind::F32, a.idx_comp_norm))
            return false;
    }

    Dsv4FfnWeights& f = w.ffn;
    if (!resident("hc_ffn_fn.weight", g.hc_dim() * 6 * g.hc, WeightKind::F32, f.hc_fn)) return false;
    if (!resident("hc_ffn_base.weight", 6 * g.hc, WeightKind::F32, f.hc_base)) return false;
    if (!resident("hc_ffn_scale.weight", g.hc - 1, WeightKind::F32, f.hc_scale)) return false;
    if (!resident("ffn_norm.weight", g.n_embd, WeightKind::F32, f.norm)) return false;
    {
        const WeightRef* ref = v.get("ffn_gate_inp.weight");
        if (!ref || ref->kind != WeightKind::Bf16InF32 || ref->data == nullptr) {
            err = v.name("ffn_gate_inp.weight") + " is not resident raw BF16";
            return false;
        }
        f.gate = (const uint16_t*) ref->data;
    }
    if (hash) {
        const auto it = tid2eid_.find(layer);
        if (it == tid2eid_.end()) { err = "tid2eid for layer " + std::to_string(layer) + " was not loaded"; return false; }
        f.tid2eid = it->second;
    } else if (!resident("exp_probs_b.bias", g.n_expert, WeightKind::F32, f.probs_b)) {
        return false;
    }
    if (!native("ffn_gate_shexp.weight", g.n_embd, g.n_ff, f.sh_gate_type, f.sh_gate)) return false;
    if (!native("ffn_up_shexp.weight", g.n_embd, g.n_ff, f.sh_up_type, f.sh_up)) return false;
    if (!native("ffn_down_shexp.weight", g.n_ff, g.n_embd, f.sh_down_type, f.sh_down)) return false;
    f.experts = experts;
    f.expert_blobs = expert_blobs;
    return true;
}

}  // namespace strata::core
