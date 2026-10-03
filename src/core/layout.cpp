// src/core/layout.cpp - the shape checks.  See the header for why this exists.
#include "strata/core/layout.hpp"

#include <cstdio>
#include <vector>

namespace strata::core {
namespace {

/// A required 2-D shape.  `ne0` is the CONTIGUOUS axis, matching the manifest and `s_gemv`'s convention
/// (`y[o] = sum_i x[i]*W[i][o]`, row o contiguous of length ne0).
struct Want2 {
    const char* suffix;
    int64_t ne0;
    int64_t ne1;
    bool qsa_only;
    bool gdn_only;
};

/// A required element count and ENGINE FORM for a 1-D tensor.
///
/// `kind` is not decoration.  The engine form of a 1-D tensor is whatever `WeightKind` the loader applied -
/// `F32` copies 4 B/elem, `Bf16InF32` re-rounds to 2 - and a consumer that casts either one to `const float*`
/// reads 2x its length.  **THAT IS NOT HYPOTHETICAL: `ffn_gate_inp_shexp.weight` is BF16, so the arena holds
/// 5120 B for 2560 elements, and `shared_expert` read it as f32.  Every one of the 2560 MoE outputs came out
/// non-finite** - a wrong answer loud enough to notice, which is the lucky version.  Reading 5120 B past the
/// end of a tensor inside a 4.5 GiB arena does not fault, so the same mistake on a tensor followed by
/// plausible bytes would have produced plausible logits.
struct Want1 {
    const char* suffix;
    int64_t elements;
    WeightKind kind;
    bool qsa_only;
    bool gdn_only;
};

bool fail(std::string& err, const LayerView& v, const char* suffix, const char* what, int64_t got,
          int64_t want) {
    char buf[512];
    std::snprintf(buf, sizeof buf, "layer %lld: %s %s is %lld, the kernels require %lld",
                  (long long) v.layer(), v.name(suffix).c_str(), what, (long long) got, (long long) want);
    err = buf;
    return false;
}

// ---- deepseek4 (docs/DSV4.md).  Every shape below was read off the UD-IQ2_XXS artifact's own tensor
// directory, not derived from the metadata: the two compressor widths (2x key for r=4, key for r=128) and
// the mHC parameter widths (6*hc and hc-1) are exactly the kind of number this file exists to pin down,
// because nothing in the metadata states them and a wrong one decodes to plausible bytes.
bool check_one_dsv4(const WeightTable& t, const ModelGeometry& g, int64_t layer, std::string& err) {
    const LayerView v(t, layer);
    const ModelGeometry::Dsv4& d = g.dsv4;
    const int64_t r = dsv4_ratio(g, layer);
    const bool indexed = r == 4;
    const bool compressed = r > 0;
    const bool hash = dsv4_is_hash_layer(g, layer);
    // the overlapping variant (r=4) pools over 2x-wide input; the plain one (r=128) over key_length.
    const int64_t comp_w = indexed ? 2 * g.head_dim : g.head_dim;
    const int64_t mhc_fn = 6 * g.hc;   // hc_*_fn is [hc_dim, 6*hc], hc_*_base is (6*hc), hc_*_scale is (hc-1)

    std::vector<Want2> want2 = {
        // attention, EVERY layer: the LoRA-split Q (q_a then q_b), one KV head, the LoRA-split O.
        {"attn_q_a.weight", g.n_embd, d.q_lora, false, false},
        {"attn_q_b.weight", d.q_lora, g.n_head * g.head_dim, false, false},
        {"attn_kv.weight", g.n_embd, g.n_head_kv * g.head_dim, false, false},
        {"attn_output_a.weight", g.n_embd, d.o_groups * d.o_lora, false, false},
        {"attn_output_b.weight", d.o_groups * d.o_lora, g.n_embd, false, false},
        // mHC, EVERY layer (the GR plumbing's deepseek4 counterpart: a learned per-stream mixing)
        {"hc_attn_fn.weight", g.hc_dim(), mhc_fn, false, false},
        {"hc_ffn_fn.weight", g.hc_dim(), mhc_fn, false, false},
        // MoE, EVERY layer - including the hash layers, which carry a router ALONGSIDE the tid2eid table
        {"ffn_gate_inp.weight", g.n_embd, g.n_expert, false, false},
        {"ffn_gate_shexp.weight", g.n_embd, g.n_ff, false, false},
        {"ffn_up_shexp.weight", g.n_embd, g.n_ff, false, false},
        {"ffn_down_shexp.weight", g.n_ff, g.n_embd, false, false},
    };
    if (compressed) {
        want2.push_back({"attn_compressor_gate.weight", g.n_embd, comp_w, false, false});
        want2.push_back({"attn_compressor_kv.weight", g.n_embd, comp_w, false, false});
        want2.push_back({"attn_compressor_ape.weight", comp_w, r, false, false});
    }
    if (indexed) {
        want2.push_back({"indexer.attn_q_b.weight", d.q_lora, g.idx_q_heads * g.idx_key_dim, false, false});
        want2.push_back({"indexer.proj.weight", g.n_embd, g.idx_q_heads, false, false});
        want2.push_back({"indexer_compressor_gate.weight", g.n_embd, 2 * g.idx_key_dim, false, false});
        want2.push_back({"indexer_compressor_kv.weight", g.n_embd, 2 * g.idx_key_dim, false, false});
        want2.push_back({"indexer_compressor_ape.weight", 2 * g.idx_key_dim, 4, false, false});
    }
    if (hash) want2.push_back({"ffn_gate_tid2eid.weight", d.n_expert_used, d.n_vocab, false, false});
    for (const Want2& w : want2) {
        const WeightRef* ref = v.get(w.suffix);
        if (!ref) {
            err = "layer " + std::to_string(layer) + ": missing " + v.name(w.suffix);
            return false;
        }
        if (ref->ne0 != w.ne0) return fail(err, v, w.suffix, "ne0", ref->ne0, w.ne0);
        if (ref->ne1 != w.ne1) return fail(err, v, w.suffix, "ne1", ref->ne1, w.ne1);
    }
    // The router's FORM, not just its shape: the pack stores it as raw BF16 (index kind 4, 2 B/elem) and
    // `moe_route` reads BF16 logits.  A row that said F32 would hand the GEMV 2x the elements it walks.
    if (const WeightRef* ref = v.get("ffn_gate_inp.weight")) {
        if (ref->kind != WeightKind::Bf16InF32) {
            char buf[512];
            std::snprintf(buf, sizeof buf, "layer %lld: %s is engine form %d, the router reads it as %d",
                          (long long) layer, v.name("ffn_gate_inp.weight").c_str(), (int) ref->kind,
                          (int) WeightKind::Bf16InF32);
            err = buf;
            return false;
        }
    }

    std::vector<Want1> want1 = {
        {"attn_norm.weight", g.n_embd, WeightKind::F32, false, false},
        {"ffn_norm.weight", g.n_embd, WeightKind::F32, false, false},
        {"attn_q_a_norm.weight", d.q_lora, WeightKind::F32, false, false},
        {"attn_kv_a_norm.weight", g.head_dim, WeightKind::F32, false, false},
        {"attn_sinks.weight", g.n_head, WeightKind::F32, false, false},   // one learned sink logit per head
        {"hc_attn_base.weight", mhc_fn, WeightKind::F32, false, false},
        {"hc_attn_scale.weight", g.hc - 1, WeightKind::F32, false, false},
        {"hc_ffn_base.weight", mhc_fn, WeightKind::F32, false, false},
        {"hc_ffn_scale.weight", g.hc - 1, WeightKind::F32, false, false},
    };
    // exp_probs_b is the learned per-expert bias the sqrtsoftplus router adds.  The three hash layers do
    // not have it (measured: 40 of 43) - their routing is the static tid2eid table.
    if (!hash) want1.push_back({"exp_probs_b.bias", g.n_expert, WeightKind::F32, false, false});
    if (compressed) want1.push_back({"attn_compressor_norm.weight", g.head_dim, WeightKind::F32, false, false});
    if (indexed) want1.push_back({"indexer_compressor_norm.weight", g.idx_key_dim, WeightKind::F32, false, false});
    for (const Want1& w : want1) {
        const WeightRef* ref = v.get(w.suffix);
        if (!ref) {
            err = "layer " + std::to_string(layer) + ": missing " + v.name(w.suffix);
            return false;
        }
        if (ref->elements != w.elements) return fail(err, v, w.suffix, "elements", ref->elements, w.elements);
        if (ref->kind != w.kind) {
            const uint64_t want_bytes = (uint64_t) w.elements *
                                        (w.kind == WeightKind::Bf16InF32 ? 2u : 4u);
            char buf[512];
            std::snprintf(buf, sizeof buf,
                          "layer %lld: %s is engine form %d (%llu B), the kernels read it as form %d (%llu B)",
                          (long long) layer, v.name(w.suffix).c_str(), (int) ref->kind,
                          (unsigned long long) ref->bytes, (int) w.kind, (unsigned long long) want_bytes);
            err = buf;
            return false;
        }
    }
    return true;
}

bool check_one(const WeightTable& t, const ModelGeometry& g, int64_t layer, std::string& err) {
    if (g.arch == ModelArch::DeepSeek4) return check_one_dsv4(t, g, layer, err);
    const LayerView v(t, layer);
    const bool qsa = is_qsa_layer(g, layer);

    // ---- the 2-D tensor set, per layer family
    const Want2 want2[] = {
        // gated residual, EVERY layer - `gr_read` takes w_down (hc_lr, hc_dim) and w_up (hc_lr, hc_dim)
        // after its own transpose, so the pack's orientation is [hc_dim, hc_lr] and [hc_lr, hc_dim].
        {"hc_attn_down.weight", g.hc_dim(), g.hc_lr, false, false},
        {"hc_attn_up.weight", g.hc_lr, g.hc_dim(), false, false},
        {"hc_attn_inject.weight", g.hc_dim(), g.hc, false, false},
        {"hc_ffn_down.weight", g.hc_dim(), g.hc_lr, false, false},
        {"hc_ffn_up.weight", g.hc_lr, g.hc_dim(), false, false},
        {"hc_ffn_inject.weight", g.hc_dim(), g.hc, false, false},
        // MoE, EVERY layer
        {"ffn_gate_inp.weight", g.n_embd, g.n_expert, false, false},
        {"ffn_gate_shexp.weight", g.n_embd, g.n_ff, false, false},
        {"ffn_up_shexp.weight", g.n_embd, g.n_ff, false, false},
        {"ffn_down_shexp.weight", g.n_ff, g.n_embd, false, false},
        // GDN only
        {"attn_qkv.weight", g.n_embd, g.ssm_conv_channels, false, true},
        {"attn_gate.weight", g.n_embd, g.ssm_value_dim, false, true},
        {"ssm_out.weight", g.ssm_value_dim, g.n_embd, false, true},
        {"ssm_conv1d.weight", g.ssm_d_conv, g.ssm_conv_channels, false, true},
        {"ssm_alpha.weight", g.n_embd, g.ssm_v_heads, false, true},
        {"ssm_beta.weight", g.n_embd, g.ssm_v_heads, false, true},
        // QSA only
        {"attn_q.weight", g.n_embd, 2 * g.n_head * g.head_dim, true, false},
        {"attn_k.weight", g.n_embd, g.n_head_kv * g.head_dim, true, false},
        {"attn_v.weight", g.n_embd, g.n_head_kv * g.head_dim, true, false},
        {"attn_output.weight", g.n_head * g.head_dim, g.n_embd, true, false},
        // THE INDEXER.  `q_proj` is the QUERY count and `k_proj` is the KEY width, and they are different
        // numbers - conflating them is what made the planner's indexer term 4x too big in round 194.
        {"indexer.q_proj.weight", g.n_embd, g.idx_q_heads * g.idx_key_dim, true, false},
        {"indexer.k_proj.weight", g.n_embd, g.idx_key_dim, true, false},
    };
    for (const Want2& w : want2) {
        if (w.qsa_only && !qsa) continue;
        if (w.gdn_only && qsa) continue;
        const WeightRef* r = v.get(w.suffix);
        if (!r) {
            err = "layer " + std::to_string(layer) + ": missing " + v.name(w.suffix);
            return false;
        }
        if (r->ne0 != w.ne0) return fail(err, v, w.suffix, "ne0", r->ne0, w.ne0);
        if (r->ne1 != w.ne1) return fail(err, v, w.suffix, "ne1", r->ne1, w.ne1);
    }

    // ---- the 1-D set.  EVERY ONE OF THESE IS `F32` EXCEPT the shared expert's scalar gate, which is BF16 -
    // and that single exception is the one a reader would not guess, so it is written down rather than
    // inferred from the tensor count.
    const Want1 want1[] = {
        {"hc_attn_norm.weight", g.hc_dim(), WeightKind::F32, false, false},
        {"hc_ffn_norm.weight", g.hc_dim(), WeightKind::F32, false, false},
        {"ffn_gate_inp_shexp.weight", g.n_embd, WeightKind::Bf16InF32, false, false},
        {"ssm_a", g.ssm_v_heads, WeightKind::F32, false, true},
        {"ssm_dt.bias", g.ssm_v_heads, WeightKind::F32, false, true},
        {"ssm_norm.weight", g.ssm_state_size, WeightKind::F32, false, true},
        {"attn_q_norm.weight", g.head_dim, WeightKind::F32, true, false},
        {"attn_k_norm.weight", g.head_dim, WeightKind::F32, true, false},
        {"indexer.q_norm.weight", g.idx_key_dim, WeightKind::F32, true, false},
        {"indexer.k_norm.weight", g.idx_key_dim, WeightKind::F32, true, false},
    };
    for (const Want1& w : want1) {
        if (w.qsa_only && !qsa) continue;
        if (w.gdn_only && qsa) continue;
        const WeightRef* r = v.get(w.suffix);
        if (!r) {
            err = "layer " + std::to_string(layer) + ": missing " + v.name(w.suffix);
            return false;
        }
        if (r->elements != w.elements) return fail(err, v, w.suffix, "elements", r->elements, w.elements);
        if (r->kind != w.kind) {
            // The byte count is what makes this a REAL check and not a label: a 4 B/elem tensor holds
            // `elements * 4`, a 2 B/elem one holds `elements * 2`, and a consumer reading it as the wrong one
            // walks off the end.
            const uint64_t want_bytes = (uint64_t) w.elements *
                                        (w.kind == WeightKind::Bf16InF32 ? 2u : 4u);
            char buf[512];
            std::snprintf(buf, sizeof buf,
                          "layer %lld: %s is engine form %d (%llu B), the kernels read it as form %d (%llu B)",
                          (long long) layer, v.name(w.suffix).c_str(), (int) r->kind,
                          (unsigned long long) r->bytes, (int) w.kind, (unsigned long long) want_bytes);
            err = buf;
            return false;
        }
    }
    return true;
}

}  // namespace

std::string LayerView::name(const char* suffix) const {
    return "blk." + std::to_string(layer_) + "." + suffix;
}

bool check_layer(const WeightTable& table, const ModelGeometry& g, int64_t layer, std::string& err) {
    if (layer < 0 || layer >= g.n_layers) {
        err = "layer " + std::to_string(layer) + " is outside 0.." + std::to_string(g.n_layers - 1);
        return false;
    }
    return check_one(table, g, layer, err);
}

bool check_all(const WeightTable& table, const ModelGeometry& g, std::string& err) {
    if (g.arch == ModelArch::DeepSeek4) {
        // The layer classes come from the artifact's own array; the cross-layer check is that the array
        // COVERS every layer and only names classes the pack can serve.  The per-layer presence checks in
        // `check_one_dsv4` already tie each class to its tensors, so a class the pack does not have fails
        // there, not here.
        if ((int64_t) g.dsv4.compress_ratios.size() < g.n_layers) {
            err = "compress_ratios covers " + std::to_string(g.dsv4.compress_ratios.size()) +
                  " layers, the model has " + std::to_string(g.n_layers);
            return false;
        }
        if (g.dsv4.hash_layers < 0 || g.dsv4.hash_layers > g.n_layers) {
            err = "hash_layer_count " + std::to_string(g.dsv4.hash_layers) + " is outside 0.." +
                  std::to_string(g.n_layers);
            return false;
        }
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const int64_t r = dsv4_ratio(g, l);
            if (r != 0 && r != 4 && r != 128) {
                err = "layer " + std::to_string(l) + ": compress_ratio " + std::to_string(r) +
                      " is a class no kernel implements (measured artifact: 0, 4, 128 only)";
                return false;
            }
            if (!check_one(table, g, l, err)) return false;
        }
        return true;
    }
    int64_t n_qsa = 0, n_gdn = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (!check_one(table, g, l, err)) return false;
        if (is_qsa_layer(g, l)) {
            ++n_qsa;
        } else {
            ++n_gdn;
        }
    }
    // The split is derived twice over in `docs/semantics.md` - the interval and an explicit ratio array -
    // and both give 36 GDN and 12 QSA.  Counting them here is the check that the LAYER TYPE PREDICATE and
    // the pack agree, which no per-tensor shape check can see.
    if (n_qsa != g.n_qsa_layers() || n_gdn != g.n_gdn_layers()) {
        char buf[256];
        std::snprintf(buf, sizeof buf, "layer split is %lld QSA / %lld GDN, the geometry says %lld / %lld",
                      (long long) n_qsa, (long long) n_gdn, (long long) g.n_qsa_layers(),
                      (long long) g.n_gdn_layers());
        err = buf;
        return false;
    }
    return true;
}

}  // namespace strata::core
