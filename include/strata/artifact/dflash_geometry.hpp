// include/strata/artifact/dflash_geometry.hpp - read a dflash (DSpark drafter) pack's geometry out of its
// OWN metadata (docs/DSV4.md P4).
//
// The drafter's blocks are deepseek4 blocks - every one of them the r=0 window-only class - so the block
// parameters are read into `g.dsv4` exactly as `dsv4_geometry` reads them, and `check_one_dsv4` serves the
// layers.  What this reader adds is the attachment to the target: `dflash.block_size` and
// `dflash.target_layers`, measured from the sidecar's own metadata (block_size 5, target_layers [41,42,43]).
//
// Two things the deepseek4 reader requires are deliberately NOT required here:
//   * the indexer keys.  The measured sidecar carries `dflash.attention.indexer.*` metadata, but with every
//     layer r=0 no kernel reads it and no tensor backs it - a future sidecar may drop the keys.
//   * the vocab.  The drafter ships no token embeddings and no output head; it borrows the target's, so the
//     vocab is the TARGET's number.  The Markov head's vocab dimension is pinned against the target by the
//     drafter loader, which knows it.
//
// Header-only like the rest of strata_artifact: it parses metadata the reader already parsed.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/layout.hpp"

#include <string>
#include <vector>

namespace strata::artifact {

/// Fill `g` from `f`'s `dflash.*` metadata.  Returns false with a message naming the MISSING or
/// IMPLAUSIBLE key - same rule as the deepseek4 reader: a geometry that silently defaults is a geometry
/// that loads the wrong shapes.
inline bool dflash_geometry(const GgufFile& f, core::ModelGeometry& g, std::string& err) {
    const auto& meta = f.metadata();
    auto arch = meta.find("general.architecture");
    if (arch == meta.end() || arch->second.s != "dflash") {
        err = "general.architecture is not dflash";
        return false;
    }
    auto num = [&](const char* key, int64_t& out) -> bool {
        auto it = meta.find(key);
        if (it == meta.end() || !it->second.is_num()) { err = std::string("missing ") + key; return false; }
        out = (int64_t) it->second.num();
        return true;
    };
    auto dnum = [&](const char* key, double& out) -> bool {
        auto it = meta.find(key);
        if (it == meta.end() || !it->second.is_num()) { err = std::string("missing ") + key; return false; }
        out = it->second.num();
        return true;
    };
    auto pos = [&](const char* key, int64_t& out) -> bool {
        if (!num(key, out)) return false;
        if (out <= 0) { err = std::string(key) + " is " + std::to_string(out) + ", must be positive"; return false; }
        return true;
    };

    core::ModelGeometry m;
    m.arch = core::ModelArch::DFlash;
    if (!pos("dflash.block_count", m.n_layers)) return false;
    if (!pos("dflash.embedding_length", m.n_embd)) return false;
    if (!pos("dflash.expert_count", m.n_expert)) return false;
    if (!pos("dflash.expert_used_count", m.dsv4.n_expert_used)) return false;
    if (!pos("dflash.expert_feed_forward_length", m.n_ff)) return false;
    int64_t shared = 0;
    if (!pos("dflash.expert_shared_count", shared)) return false;
    if (shared != 1) { err = "expert_shared_count is " + std::to_string(shared) +
                             ", the engine's shared-expert path serves exactly one"; return false; }
    // The drafter has no hash layers: the measured sidecar carries no `ffn_gate_tid2eid` on any layer, and
    // every layer carries `exp_probs_b`.  A nonzero count here would ask the layer checks for a table the
    // file does not have.
    int64_t hash = 0;
    if (!num("dflash.hash_layer_count", hash)) return false;
    if (hash != 0) {
        err = "hash_layer_count is " + std::to_string(hash) +
              ", the drafter's layers all route through a router (measured: 0)";
        return false;
    }
    m.dsv4.hash_layers = 0;
    if (!pos("dflash.hyper_connection.count", m.hc)) return false;
    if (!pos("dflash.hyper_connection.sinkhorn_iterations", m.dsv4.sinkhorn_iters)) return false;
    double eps = 0, scale = 0;
    if (!dnum("dflash.hyper_connection.epsilon", eps)) return false;
    if (eps <= 0) { err = "hyper_connection.epsilon is not positive"; return false; }
    m.dsv4.hc_eps = (float) eps;
    if (!pos("dflash.expert_gating_func", m.dsv4.gating_func)) return false;
    if (m.dsv4.gating_func != 4) {
        err = "expert_gating_func is " + std::to_string(m.dsv4.gating_func) +
              ", the engine implements only sqrtsoftplus (4)";
        return false;
    }
    if (!dnum("dflash.expert_weights_scale", scale)) return false;
    if (scale <= 0) { err = "expert_weights_scale is not positive"; return false; }
    m.dsv4.route_scale = (float) scale;
    int64_t norm_i = 0;
    if (!num("dflash.expert_weights_norm", norm_i)) return false;
    if (norm_i != 1) {
        err = "expert_weights_norm is not true, the router always renormalizes the top-k weights";
        return false;
    }
    m.dsv4.weights_norm = true;
    if (!pos("dflash.attention.head_count", m.n_head)) return false;
    if (!pos("dflash.attention.head_count_kv", m.n_head_kv)) return false;
    if (!pos("dflash.attention.key_length", m.head_dim)) return false;
    int64_t value_length = 0;
    if (!pos("dflash.attention.value_length", value_length)) return false;
    if (value_length != m.head_dim) {
        err = "value_length " + std::to_string(value_length) + " != key_length " + std::to_string(m.head_dim) +
              ", the KV pool holds one width";
        return false;
    }
    if (!pos("dflash.attention.q_lora_rank", m.dsv4.q_lora)) return false;
    if (!pos("dflash.attention.output_lora_rank", m.dsv4.o_lora)) return false;
    if (!pos("dflash.attention.output_group_count", m.dsv4.o_groups)) return false;
    // The r=0 window IS the drafter's whole attention: it keeps its own KV and attends inside this window
    // and nothing else.  Without it the drafter's attention is undefined, so it is required.
    if (!pos("dflash.attention.sliding_window", m.dsv4.sliding_window)) return false;

    double eps_v = 0, theta = 0, ctheta = 0, yfactor = 0, bfast = 0, bslow = 0;
    if (!dnum("dflash.attention.layer_norm_rms_epsilon", eps_v)) return false;
    if (eps_v <= 0) { err = "layer_norm_rms_epsilon is not positive"; return false; }
    m.dsv4.norm_eps = (float) eps_v;
    if (!dnum("dflash.rope.freq_base", theta)) return false;
    if (theta <= 1.0) { err = "rope.freq_base is not above 1"; return false; }
    m.dsv4.rope_theta = theta;
    if (!pos("dflash.rope.dimension_count", m.dsv4.rope_dim)) return false;
    if (m.dsv4.rope_dim % 2 != 0 || m.dsv4.rope_dim > m.head_dim) {
        err = "rope.dimension_count " + std::to_string(m.dsv4.rope_dim) + " is not a valid even part of key_length";
        return false;
    }
    if (!dnum("dflash.rope.scaling.factor", yfactor)) return false;
    if (!num("dflash.rope.scaling.original_context_length", m.dsv4.yarn_orig)) return false;
    if (!dnum("dflash.rope.scaling.yarn_beta_fast", bfast)) return false;
    if (!dnum("dflash.rope.scaling.yarn_beta_slow", bslow)) return false;
    m.dsv4.yarn_factor = yfactor;
    m.dsv4.yarn_beta_fast = (int64_t) bfast;
    m.dsv4.yarn_beta_slow = (int64_t) bslow;
    if (!dnum("dflash.attention.compress_rope_freq_base", ctheta)) return false;
    m.dsv4.compress_rope_theta = ctheta;

    // The layer classes: the measured sidecar's array is [0, 0, 0].  A nonzero ratio here would name a
    // class (compressor, indexer) whose tensors the drafter does not carry, so refuse rather than let
    // `check_one_dsv4` report a missing tensor for a class the reader invented.
    auto it = meta.find("dflash.attention.compress_ratios");
    if (it == meta.end() || it->second.type != MetaType::ARRAY) { err = "missing dflash.attention.compress_ratios"; return false; }
    if ((int64_t) it->second.count < m.n_layers) {
        err = "compress_ratios has " + std::to_string(it->second.count) + " entries, block_count is " +
              std::to_string(m.n_layers);
        return false;
    }
    if ((int64_t) it->second.items.size() < m.n_layers) {
        err = "compress_ratios: only " + std::to_string(it->second.items.size()) +
              " of the " + std::to_string(m.n_layers) + " needed entries were kept";
        return false;
    }
    for (int64_t l = 0; l < m.n_layers; ++l) {
        const int64_t r = (int64_t) it->second.items[(size_t) l].num();
        if (r != 0) {
            err = "layer " + std::to_string(l) + ": compress_ratio " + std::to_string(r) +
                  ", the drafter implements only the r=0 window class (measured: all 0)";
            return false;
        }
        m.dsv4.compress_ratios.push_back(0);
    }

    auto clamps = [&](const char* key, std::vector<float>& out) -> bool {
        auto cit = meta.find(key);
        if (cit == meta.end() || cit->second.type != MetaType::ARRAY) { err = std::string("missing ") + key; return false; }
        if ((int64_t) cit->second.count < m.n_layers || (int64_t) cit->second.items.size() < m.n_layers) {
            err = std::string(key) + " does not carry " + std::to_string(m.n_layers) + " entries";
            return false;
        }
        for (int64_t l = 0; l < m.n_layers; ++l) out.push_back((float) cit->second.items[(size_t) l].num());
        return true;
    };
    if (!clamps("dflash.swiglu_clamp_exp", m.dsv4.swiglu_clamp_exp)) return false;
    if (!clamps("dflash.swiglu_clamp_shexp", m.dsv4.swiglu_clamp_shexp)) return false;

    // The attachment to the target.  `block_size` is the trained draft block - the depth the drafter was
    // trained to draft in one pass - and `target_layers` names the target hidden states `fc` consumes.
    // Their range against the TARGET's layer count is the drafter loader's check, not this one's: the
    // sidecar does not know the target's geometry.
    if (!pos("dflash.block_size", m.dflash.block_size)) return false;
    auto tl = meta.find("dflash.target_layers");
    if (tl == meta.end() || tl->second.type != MetaType::ARRAY) { err = "missing dflash.target_layers"; return false; }
    if (tl->second.count == 0 || (int64_t) tl->second.items.size() < (int64_t) tl->second.count) {
        err = "target_layers is empty or was truncated";
        return false;
    }
    for (const auto& v : tl->second.items) {
        const int64_t l = (int64_t) v.num();
        if (l < 0) { err = "target_layers contains " + std::to_string(l); return false; }
        m.dflash.target_layers.push_back(l);
    }
    // The block the drafter feeds is `[anchor, mask * (d-1)]` - without the mask token the block is
    // undefined, so the tokenizer's mask id is required, not defaulted.  `sample_from_anchor` is the
    // SpecForge export flag; the measured sidecar omits it and llama.cpp's default is true.
    if (!pos("tokenizer.ggml.mask_token_id", m.dflash.mask_token_id)) return false;
    if (auto sa = meta.find("dflash.sample_from_anchor"); sa != meta.end() && sa->second.is_num()) {
        m.dflash.sample_from_anchor = sa->second.num() != 0;
    }
    g = m;
    return true;
}

}  // namespace strata::artifact
