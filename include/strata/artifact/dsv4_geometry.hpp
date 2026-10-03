// include/strata/artifact/dsv4_geometry.hpp - read a deepseek4 pack's geometry out of its OWN metadata.
//
// The Qwen path hard-codes its geometry in `generate.cpp` because there is exactly one Qwen model and its
// numbers are in `docs/semantics.md`.  deepseek4 (docs/DSV4.md) gets a reader instead: the numbers are
// measured from the artifact (UD-IQ2_XXS, 2026-07), and a second DeepSeek-V4 variant - GLM-5.3-Flash
// shares the family (docs/DSV4.md P5) - must not need an engine edit to be read.
//
// Header-only like the rest of strata_artifact: it parses metadata the reader already parsed, and a second
// parser would be a second place to disagree about what a GGUF array is.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/layout.hpp"

#include <string>
#include <vector>

namespace strata::artifact {

/// Fill `g` from `f`'s `deepseek4.*` metadata.  Returns false with a message naming the MISSING or
/// IMPLAUSIBLE key - a geometry that silently defaults is a geometry that loads the wrong shapes.
/// Only the model's own keys are read; the tokenizer arrays are P3 (docs/DSV4.md), except their COUNT,
/// which is the only number here that is not a `deepseek4.*` key: `ffn_gate_tid2eid` is [used, vocab].
inline bool deepseek4_geometry(const GgufFile& f, core::ModelGeometry& g, std::string& err) {
    const auto& meta = f.metadata();
    auto arch = meta.find("general.architecture");
    if (arch == meta.end() || arch->second.s != "deepseek4") {
        err = "general.architecture is not deepseek4";
        return false;
    }
    auto num = [&](const char* key, int64_t& out) -> bool {
        auto it = meta.find(key);
        if (it == meta.end() || !it->second.is_num()) { err = std::string("missing ") + key; return false; }
        out = (int64_t) it->second.num();
        return true;
    };
    auto pos = [&](const char* key, int64_t& out) -> bool {
        if (!num(key, out)) return false;
        if (out <= 0) { err = std::string(key) + " is " + std::to_string(out) + ", must be positive"; return false; }
        return true;
    };

    core::ModelGeometry m;
    m.arch = core::ModelArch::DeepSeek4;
    if (!pos("deepseek4.block_count", m.n_layers)) return false;
    if (!pos("deepseek4.embedding_length", m.n_embd)) return false;
    if (!pos("deepseek4.expert_count", m.n_expert)) return false;
    if (!pos("deepseek4.expert_used_count", m.dsv4.n_expert_used)) return false;
    if (!pos("deepseek4.expert_feed_forward_length", m.n_ff)) return false;
    int64_t shared = 0;
    if (!pos("deepseek4.expert_shared_count", shared)) return false;
    if (shared != 1) { err = "expert_shared_count is " + std::to_string(shared) +
                             ", the engine's shared-expert path serves exactly one"; return false; }
    if (!num("deepseek4.hash_layer_count", m.dsv4.hash_layers)) return false;   // 0 is a legal value
    if (m.dsv4.hash_layers < 0 || m.dsv4.hash_layers > m.n_layers) {
        err = "hash_layer_count is " + std::to_string(m.dsv4.hash_layers) + ", outside 0.." +
              std::to_string(m.n_layers);
        return false;
    }
    if (!pos("deepseek4.hyper_connection.count", m.hc)) return false;
    if (!pos("deepseek4.hyper_connection.sinkhorn_iterations", m.dsv4.sinkhorn_iters)) return false;
    if (!pos("deepseek4.attention.head_count", m.n_head)) return false;
    if (!pos("deepseek4.attention.head_count_kv", m.n_head_kv)) return false;
    if (!pos("deepseek4.attention.key_length", m.head_dim)) return false;
    int64_t value_length = 0;
    if (!pos("deepseek4.attention.value_length", value_length)) return false;
    if (value_length != m.head_dim) {
        err = "value_length " + std::to_string(value_length) + " != key_length " + std::to_string(m.head_dim) +
              ", the KV pool holds one width";
        return false;
    }
    if (!pos("deepseek4.attention.q_lora_rank", m.dsv4.q_lora)) return false;
    if (!pos("deepseek4.attention.output_lora_rank", m.dsv4.o_lora)) return false;
    if (!pos("deepseek4.attention.output_group_count", m.dsv4.o_groups)) return false;
    if (!pos("deepseek4.attention.sliding_window", m.dsv4.sliding_window)) return false;
    if (!pos("deepseek4.attention.indexer.head_count", m.idx_q_heads)) return false;
    if (!pos("deepseek4.attention.indexer.key_length", m.idx_key_dim)) return false;
    if (!pos("deepseek4.attention.indexer.top_k", m.dsv4.idx_topk)) return false;

    auto it = meta.find("deepseek4.attention.compress_ratios");
    if (it == meta.end() || it->second.type != MetaType::ARRAY) { err = "missing deepseek4.attention.compress_ratios"; return false; }
    if ((int64_t) it->second.count < m.n_layers) {
        err = "compress_ratios has " + std::to_string(it->second.count) + " entries, block_count is " +
              std::to_string(m.n_layers);
        return false;
    }
    // The reader keeps a 64-item SAMPLE of long arrays; the measured artifact's array is 46 entries, so the
    // n_layers entries this geometry needs are all in `items`.  Refuse rather than guess past the sample.
    if ((int64_t) it->second.items.size() < m.n_layers) {
        err = "compress_ratios: only " + std::to_string(it->second.items.size()) +
              " of the " + std::to_string(m.n_layers) + " needed entries were kept";
        return false;
    }
    for (int64_t l = 0; l < m.n_layers; ++l)
        m.dsv4.compress_ratios.push_back((int64_t) it->second.items[(size_t) l].num());

    // vocab: the tokenizer array's COUNT (its items are P3's business).  A pack without a tokenizer still
    // has the array; a pack without it cannot serve tid2eid, and only a hash layer needs vocab - so ask
    // only when there is one.
    if (m.dsv4.hash_layers > 0) {
        auto tok = meta.find("tokenizer.ggml.tokens");
        if (tok == meta.end() || tok->second.type != MetaType::ARRAY) {
            err = "hash layers need vocab: tokenizer.ggml.tokens is absent";
            return false;
        }
        m.dsv4.n_vocab = (int64_t) tok->second.count;
    }

    g = m;
    return true;
}

}  // namespace strata::artifact
