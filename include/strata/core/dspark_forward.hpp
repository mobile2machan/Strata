// include/strata/core/dspark_forward.hpp - the DSpark drafter for DeepSeek-V4, docs/DSV4.md P4.
//
// The transcription of llama.cpp's `llama_model_dflash::graph_dsv4` (the DSpark decoder graph),
// dual-mode exactly like the reference:
//
//   * `inject`  - the encoder path.  The target's captured hidden states (the hc-mean of its last
//     three layer outputs, `target_layers`) are fused through `fc` + `enc.output_norm`, then each
//     stage's own `attn_kv` + `attn_kv_a_norm` + rope (trailing dims, plain theta - the r=0 rope
//     parameters) is written into that stage's sliding-window ring.  The drafter's ring is the only
//     state it keeps: no compressor, no indexer, no carries, so a speculative rollback needs
//     nothing from it - rejected positions are simply rewritten.
//
//   * `draft`   - the decoder path.  The block `[anchor, mask * (d-1)]` at positions
//     `pos..pos+d-1` walks the 3 r=0 dsv4 stages (the same parity-tested `dsv4_block_prefill`),
//     then `output_hc_*` collapse -> the PRE-NORM hidden (the confidence head's input) ->
//     `output_norm` -> the borrowed target head -> base logits.  The Markov head then chains
//     greedily from the anchor: `w1_prev = markov_w1[prev]`, `bias = markov_w2 @ w1_prev`,
//     `draft[i] = argmax(base[i] + bias)`, `prev = draft[i]`; the confidence head is
//     `sigmoid(conf_proj . [pre-norm hidden; w1_prev])`.
//
// The token embeddings and the output head are the TARGET's (`NativeEmbed` / `NativeHead` borrowed
// at init); the drafter pack carries neither.  The experts stream from the sidecar's `experts.bin`
// exactly as the target's do.
#pragma once

#include "strata/core/dsv4_block.hpp"
#include "strata/core/dsv4_state.hpp"
#include "strata/core/dsv4_weights.hpp"
#include "strata/core/native_head.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace strata::core {

class DsparkForward {
public:
    DsparkForward() = default;
    ~DsparkForward();
    DsparkForward(const DsparkForward&) = delete;
    DsparkForward& operator=(const DsparkForward&) = delete;

    /// `wt`/`dense` must be loaded for the SIDECAR pack; `embed`/`head` are the target's, already
    /// loaded.  `g` is the dflash geometry (n_layers = the drafter's stage count).
    bool init(const ModelGeometry& g, const std::string& pack_dir,
              const std::vector<std::string>& shards, WeightTable& wt, const NativeEmbed& embed,
              const NativeHead& head, std::string& err, int64_t max_seq = 64);

    /// Fuse the target features for positions `pos0..pos0+n-1` (n <= 8) into every stage's ring.
    /// `d_fused` is device [n][target_layers.size() * n_embd] f32, in `target_layers` order.
    bool inject(const float* d_fused, int64_t pos0, int64_t n, std::string& err);

    /// Draft up to `d` tokens (1..8) anchored at the committed token `anchor`, occupying positions
    /// `pos..pos+d-1`.  `tokens[i]` / `conf[i]` come back on the host.  Greedy, like the
    /// reference's in-graph chain.
    bool draft(int64_t anchor, int64_t pos, int64_t d, std::vector<int64_t>& tokens,
               std::vector<float>& conf, std::string& err);

    void reset() { state_.reset(); }

    int64_t block_size() const { return g_.dflash.block_size; }
    const std::vector<int64_t>& target_layers() const { return g_.dflash.target_layers; }
    int64_t markov_rank() const { return markov_rank_; }
    /// The base logits (before the Markov bias) of the last `draft` call, device [8][n_vocab] -
    /// the parity test reads them to check the chain against an independent float64 reference.
    const float* base_logits() const { return d_logits_; }
    const uint16_t* markov_w1() const { return markov_w1_; }
    const uint16_t* markov_w2() const { return markov_w2_; }
    const uint16_t* conf_proj() const { return conf_proj_; }
    const float* pre_norm_hidden() const { return d_y_; }
    /// A stage's ring row for an absolute position (the parity test reads the ring back).
    const uint16_t* ring_row(int64_t layer, int64_t pos) const {
        return state_.layer(layer).window + dsv4_window_slot(pos, g_.dsv4.sliding_window) * g_.head_dim;
    }

private:
    struct Stager : public Dsv4ExpertSource {
        DsparkForward* f;
        int64_t off = 0;
        explicit Stager(DsparkForward* ff) : f(ff) {}
        bool stage(const int32_t* ids, int64_t n, int64_t bytes, const uint8_t** out) override;
    };

    ModelGeometry g_;
    WeightTable* wt_ = nullptr;
    const NativeEmbed* embed_ = nullptr;
    const NativeHead* head_ = nullptr;
    Dsv4WeightResolver resolver_;
    Dsv4State state_;

    const float* fc_norm_ = nullptr;      // enc.output_norm
    int fc_type_ = -1;
    const uint8_t* fc_ = nullptr;
    int64_t markov_rank_ = 0;            // the markov head's rank, read from markov_w1's shape (measured 256)
    const uint16_t* markov_w1_ = nullptr;  // [rank][n_vocab] bf16
    const uint16_t* markov_w2_ = nullptr;  // [rank][n_vocab] bf16
    const uint16_t* conf_proj_ = nullptr;  // [n_embd + rank] bf16
    const float* hc_head_fn_ = nullptr;
    const float* hc_head_base_ = nullptr;
    const float* hc_head_scale_ = nullptr;
    const float* out_norm_ = nullptr;
    std::vector<uint16_t> h_conf_proj_;    // host copy for the (tiny) confidence dot
    std::vector<uint16_t> h_markov_row_;   // host copy of one w1 row

    float* d_pa_ = nullptr;    // [8][hc * n_embd] streams
    float* d_pb_ = nullptr;
    int64_t* d_ptok_ = nullptr;
    float* d_mixes_ = nullptr;  // [8][mix_hc]
    float* d_pre_ = nullptr;    // [8][hc]
    float* d_y_ = nullptr;      // [8][n_embd] collapsed (pre-norm = the conf input)
    float* d_normed_ = nullptr;  // [8][n_embd] after output_norm
    float* d_logits_ = nullptr;  // [8][n_vocab] base logits
    float* d_enc_ = nullptr;     // [8][n_embd] fc output
    float* d_kv_ = nullptr;      // [8][head_dim]
    uint16_t* d_kv_bf_ = nullptr;
    uint8_t* d_xq_enc_ = nullptr;   // q8_1 of an [n_embd * cap] activation
    uint8_t* d_xq_dim_ = nullptr;   // q8_1 of an [n_embd] activation
    uint16_t* d_w1row_ = nullptr;  // [rank] bf16
    float* d_bias_ = nullptr;      // [n_vocab]
    float* d_scratch_ = nullptr;

    std::vector<float> h_logits_;   // [8][n_vocab]
    std::vector<float> h_bias_;     // [n_vocab]
    std::vector<float> h_tembd_;    // [n_embd]
    std::vector<float> h_normed_;   // [n_embd]

    std::FILE* experts_file_ = nullptr;
    uint8_t* h_stage_ = nullptr;
    uint8_t* d_blobs_ = nullptr;
    std::vector<int64_t> layer_off_;
    std::vector<kernels::NativeExpertLayout> layer_layout_;
    Stager stager_{this};
    void* stream_ = nullptr;
};

}  // namespace strata::core
