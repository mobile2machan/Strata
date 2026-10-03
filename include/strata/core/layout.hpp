// include/strata/core/layout.hpp - the pack's LAYOUT, resolved by name and checked against the kernels.
//
// `WeightTable` answers "where are the bytes".  This answers "is this the tensor I think it is", and it is
// the layer where a pack and a kernel disagree - which is the failure this project keeps paying for.  Two
// examples, both real:
//
//   * round 194 sized the indexer key store from `indexer.head_count = 4`, which counts QUERY heads.  The
//     cached key is ONE shared head of 128 (`indexer.k_proj` is [2560, 128]), so the term was 4x too big and
//     the error survived a round because it moved in the direction that TIGHTENS the budget.
//   * `docs/semantics.md` and the tensor manifest agree on every dimension here, but nothing CHECKED that a
//     named tensor had the shape the kernel reading it assumes.
//
// So this header does two things: it names the tensors per layer type, and it asserts every shape the
// kernels depend on.  A mismatch is reported with the tensor name, the shape found and the shape required,
// at LOAD time - not as a wrong number inside a GEMV at token 4000.
#pragma once

#include "strata/core/weights.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// Which model family a pack is.  The engine had exactly one (`qwen4exp`) until `deepseek4` (docs/DSV4.md);
/// the reader that fills the geometry dispatches on `general.architecture`, and so does `check_layer`.
enum class ModelArch : int { Qwen4Exp = 0, DeepSeek4 = 1 };

/// The model's geometry, taken from `docs/semantics.md` and the artifact's own metadata.  Every field here
/// is a number a kernel depends on, so a change is a change to a kernel contract and not a tuning knob.
struct ModelGeometry {
    ModelArch arch = ModelArch::Qwen4Exp;
    int64_t n_embd = 2560;
    int64_t n_layers = 48;
    int64_t qsa_interval = 4;      ///< every 4th layer is full attention: layers 3, 7, ... 47

    // GDN (36 layers)
    int64_t ssm_state_size = 128;
    int64_t ssm_k_heads = 16;
    int64_t ssm_v_heads = 48;
    int64_t ssm_d_conv = 4;
    int64_t ssm_conv_channels = 10240;   ///< 2*128*16 + 128*48
    int64_t ssm_value_dim = 6144;        ///< 128 * 48

    // QSA (12 layers)
    int64_t n_head = 24;
    int64_t n_head_kv = 2;
    int64_t head_dim = 256;
    int64_t idx_q_heads = 4;
    int64_t idx_key_dim = 128;

    // gated residual, on every layer
    int64_t hc = 4;
    int64_t hc_lr = 320;

    // MoE, on every layer
    int64_t n_expert = 512;
    int64_t n_ff = 640;

    /// deepseek4 only (docs/DSV4.md).  The shared scalars above are reused as-is - for the measured
    /// UD-IQ2_XXS artifact they read n_embd 4096, n_layers 43, n_head 64, n_head_kv 1, head_dim 512
    /// (key_length = value_length), idx_q_heads 64, idx_key_dim 128, hc 4, n_expert 256, n_ff 2048 -
    /// and the GDN/QSA fields are not read.  Everything here is a number the deepseek4 kernels will
    /// depend on, measured from the artifact's own metadata, not guessed.
    struct Dsv4 {
        int64_t n_vocab = 0;           ///< from the tokenizer array; `ffn_gate_tid2eid` is [used, vocab]
        int64_t q_lora = 1024;         ///< attn_q_a is [n_embd, q_lora], attn_q_b is [q_lora, heads*dim]
        int64_t o_lora = 1024;         ///< attn_output_a is [n_embd, o_groups*o_lora]
        int64_t o_groups = 8;
        int64_t sliding_window = 128;  ///< the r=0 layers attend inside this and nothing else
        int64_t idx_topk = 512;
        int64_t n_expert_used = 6;
        int64_t hash_layers = 3;       ///< layers 0..hash_layers-1 route through tid2eid instead of a router
        int64_t sinkhorn_iters = 20;   ///< mHC mixing, `hyper_connection.sinkhorn_iterations`
        float hc_eps = 1e-6f;          ///< `hyper_connection.epsilon`, the split/Sinkhorn epsilon
        int64_t gating_func = 4;       ///< `expert_gating_func`; measured 4 = sqrtsoftplus, the only one implemented
        float route_scale = 1.5f;      ///< `expert_weights_scale`, multiplied onto the renormalized top-k weights
        bool weights_norm = true;      ///< `expert_weights_norm`, renormalize the top-k weights to sum 1
        /// Per-layer SwiGLU clamps, `swiglu_clamp_exp` (routed) / `swiglu_clamp_shexp` (shared).  Measured:
        /// 43 entries of 10.0 each; the engine applies the layer's own value.
        std::vector<float> swiglu_clamp_exp;
        std::vector<float> swiglu_clamp_shexp;
        /// Per layer: 0 = sliding window only; 4 = the OVERLAPPING compressor variant, and the only layers
        /// that carry an indexer; 128 = the plain compressor.  Measured on the artifact: 2 + 21 + 20 = 43,
        /// and the array itself is 46 entries long - the last three belong to no layer and are ignored.
        std::vector<int64_t> compress_ratios;
    } dsv4;

    int64_t hc_dim() const { return hc * n_embd; }
    /// `layer % qsa_interval == qsa_interval - 1` is full attention.  Derived, not a second list.
    int64_t n_qsa_layers() const { return n_layers / qsa_interval; }
    int64_t n_gdn_layers() const { return n_layers - n_qsa_layers(); }
};

/// True for the full-attention layers.  `docs/semantics.md` gives this twice over - `full_attention_interval
/// = 4` and an explicit `attention.compress_ratios` array - and this is the first of the two.
inline bool is_qsa_layer(const ModelGeometry& g, int64_t layer) {
    return layer % g.qsa_interval == g.qsa_interval - 1;
}

/// deepseek4's per-layer attention class comes from the artifact's own `compress_ratios` array, not from an
/// interval: the measured UD-IQ2_XXS artifact is 2 window layers, then 4 and 128 alternating.  A ratio of 4
/// is the overlapping compressor variant and the only class that carries an indexer (measured: all 21 r=4
/// layers have `indexer.*` and `indexer_compressor_*`; the 20 r=128 layers and the 2 r=0 layers have neither).
inline int64_t dsv4_ratio(const ModelGeometry& g, int64_t layer) {
    return layer < (int64_t) g.dsv4.compress_ratios.size() ? g.dsv4.compress_ratios[(size_t) layer] : -1;
}
inline bool dsv4_has_indexer(const ModelGeometry& g, int64_t layer) { return dsv4_ratio(g, layer) == 4; }
inline bool dsv4_is_hash_layer(const ModelGeometry& g, int64_t layer) { return layer < g.dsv4.hash_layers; }

/// One layer's tensors, resolved by NAME.  `get("attn_qkv.weight")` looks up `blk.<layer>.attn_qkv.weight`
/// and returns null if the pack does not have it - a null is information, because a GDN layer has no
/// `attn_q` and a QSA layer has no `attn_qkv`.
///
/// Holds a reference to the table; the table must outlive it.
class LayerView {
public:
    LayerView(const WeightTable& table, int64_t layer) : table_(&table), layer_(layer) {}

    int64_t layer() const { return layer_; }
    std::string name(const char* suffix) const;
    const WeightRef* get(const char* suffix) const { return table_->find(name(suffix)); }

private:
    const WeightTable* table_;
    int64_t layer_;
};

/// Every shape the kernels depend on, asserted for ONE layer.  Returns false and fills `err` with the first
/// mismatch, naming the tensor, what it has and what is required.
///
/// This is deliberately a separate function from `LayerView`: a caller that only wants the pointers should
/// not pay for the checks, and a caller that wants the checks should get ALL of them rather than the ones
/// its own call site happens to touch.
bool check_layer(const WeightTable& table, const ModelGeometry& g, int64_t layer, std::string& err);

/// `check_layer` over every layer, plus the cross-layer properties: exactly 12 QSA layers at the right
/// indices, every GDN layer having the GDN set and no QSA tensor, and vice versa.  Returns the first failure.
bool check_all(const WeightTable& table, const ModelGeometry& g, std::string& err);

}  // namespace strata::core
