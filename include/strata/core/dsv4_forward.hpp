// include/strata/core/dsv4_forward.hpp - drive one DeepSeek-V4 model, one decode token or one
// prefill chunk at a time, docs/DSV4.md P2 integration: embed -> 43 blocks -> the head's mHC
// collapse -> output_norm -> the native head.
//
// The transcription of `model.py::Model.decode` for B=1.  The embedding is replicated across
// the hc streams (`h.unsqueeze(2).repeat(1, 1, hc_mult, 1)`), each token walks every block in
// order, and the final collapse is `hc_head` - `hc_mixes` + `hc_head_pre` + `hc_pre_combine` -
// then `output_norm` and `output.weight` through `NativeHead`.
//
// The pack is loaded by the caller (WeightTable + NativeDense, as `generate` does); this owns
// only the run-time state: the embedding table, the head, the resolver's conversions, the
// `Dsv4State` pools, and a pinned staging buffer that streams each layer's expert region from
// `experts.bin` into VRAM just before that layer runs (78 GiB does not live in VRAM; 2.5 GB
// per layer does).
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

class Dsv4Forward {
public:
    Dsv4Forward() = default;
    ~Dsv4Forward();
    Dsv4Forward(const Dsv4Forward&) = delete;
    Dsv4Forward& operator=(const Dsv4Forward&) = delete;

    /// `wt` must be loaded (with the NativeDense skip set) and `dense` loaded before this call.
    bool init(const ModelGeometry& g, const std::string& pack_dir, const std::vector<std::string>& shards,
              WeightTable& wt, std::string& err, int64_t max_seq = 64);

    /// One token at absolute position `pos`; the n_vocab logits stay in `logits()`.
    bool decode(int64_t token, int64_t pos, std::string& err);

    /// The prompt tokens at positions 0..tokens.size()-1 in chunks of 8 (the native GEMV column
    /// limit); `logits()` ends up holding the last token's.  Call `reset()` first - the chunks
    /// assume the pools start empty.
    bool prefill(const std::vector<int64_t>& tokens, std::string& err);

    /// Zero every layer's pools (a fresh sequence on the same weights).
    void reset() { state_.reset(); }

    const float* logits() const { return d_logits_; }

private:
    /// Stages exactly the picked experts: reads each id's blob from `experts.bin` into its own
    /// pinned slot, then one H2D copy; the slots are contiguous, which is the pick order the
    /// FFN half expects.
    struct Stager : public Dsv4ExpertSource {
        Dsv4Forward* f;
        int64_t off = 0;  // the current layer's region start in experts.bin
        explicit Stager(Dsv4Forward* ff) : f(ff) {}
        bool stage(const int32_t* ids, int64_t n, int64_t bytes, const uint8_t** out) override;
    };
    ModelGeometry g_;
    WeightTable* wt_ = nullptr;
    NativeEmbed embed_;
    NativeHead head_;
    Dsv4WeightResolver resolver_;
    Dsv4State state_;
    const float* hc_head_fn_ = nullptr;
    const float* hc_head_base_ = nullptr;
    const float* hc_head_scale_ = nullptr;
    const float* out_norm_ = nullptr;
    float* d_a_ = nullptr;      // the two stream buffers the block hands back and forth
    float* d_b_ = nullptr;
    float* d_pa_ = nullptr;     // the same, [8][hc * n_embd], for prefill chunks
    float* d_pb_ = nullptr;
    int64_t* d_ptok_ = nullptr;  // [8] token ids for the hash router
    float* d_mixes_ = nullptr;  // [6*hc] for the layers' mixing, [hc] for the head's
    float* d_pre_ = nullptr;
    float* d_y_ = nullptr;      // the collapsed [n_embd] stream
    float* d_logits_ = nullptr;
    float* d_scratch_ = nullptr;
    std::FILE* experts_file_ = nullptr;
    uint8_t* h_stage_ = nullptr;   // pinned host staging for the k picked blobs
    uint8_t* d_blobs_ = nullptr;   // device copy of those blobs, pick order
    std::vector<int64_t> layer_off_;
    std::vector<kernels::NativeExpertLayout> layer_layout_;
    Stager stager_{this};
    void* stream_ = nullptr;
};

}  // namespace strata::core
