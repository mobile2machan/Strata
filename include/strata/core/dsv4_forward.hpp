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
#include <functional>
#include <list>
#include <string>
#include <unordered_map>
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

    /// Residency (docs/DSV4.md, "why the decode is slow"): mirror the front of `experts.bin` in host memory
    /// before `init`, so a picked blob is copied to the GPU instead of read out of the file once per token.
    /// The part the driver page-locks is copied by DMA; the rest of the mirror is copied by the CPU.  `bytes`
    /// 0 keeps the file path; `kExpertsRamWhatFits` mirrors as much as the free RAM allows.  Call before `init`.
    static constexpr uint64_t kExpertsRamWhatFits = ~0ull;
    void set_experts_ram(uint64_t bytes) { experts_ram_want_ = bytes; }
    uint64_t experts_ram_bytes() const { return experts_arena_bytes_; }       ///< mirrored, 0 = the file path
    uint64_t experts_locked_bytes() const { return experts_locked_bytes_; }   ///< of those, page-locked

    /// One token at absolute position `pos`; the n_vocab logits stay in `logits()`.
    bool decode(int64_t token, int64_t pos, std::string& err);

    /// The prompt tokens at positions 0..tokens.size()-1 in chunks of 8 (the native GEMV column
    /// limit); `logits()` ends up holding the last token's.  Call `reset()` first - the chunks
    /// assume the pools start empty.
    bool prefill(const std::vector<int64_t>& tokens, std::string& err);

    /// Zero every layer's pools (a fresh sequence on the same weights).
    void reset() { state_.reset(); stage_stats_ = StageStats{}; }

    const float* logits() const { return d_logits_; }

    /// P4 (docs/DSV4.md): capture the hidden states the DSpark drafter's `fc` consumes.  A layer id
    /// `l < n_layers` captures the stream ENTERING layer l (llama.cpp's `layer_inp`); `n_layers`
    /// captures the final stream before the head's collapse.  Each captured stream is the MEAN over
    /// the hc streams (`build_hc_mean`), and the sink receives a device [n][cap * n_embd] f32
    /// buffer, token-major, in `target_layers` order, after every prefill chunk and decode token.
    void set_hidden_sink(std::function<bool(const float* d_fused, int64_t pos0, int64_t n)> sink,
                         std::vector<int64_t> target_layers);

    /// One speculative verify window: `n` tokens (1..8) at positions `pos0..pos0+n-1` in one pass,
    /// the per-token logits in `logits_window()` (device [n][n_vocab]), the captured hiddens
    /// through the sink.  The pools advance as if these tokens were committed - a rejected tail is
    /// rewritten by the next window, and only the carries need `carry_restore`.
    bool verify_window(const std::vector<int64_t>& tokens, int64_t pos0, std::string& err);
    const float* logits_window() const { return d_logits_w_; }

    /// The compressor/indexer carries, for the verify window's rollback (see Dsv4State).
    int64_t carry_bytes() const { return state_.carry_bytes(); }
    void carry_save(void* dst) const { state_.carry_save(dst); }
    void carry_restore(const void* src) const { state_.carry_restore(src); }
    /// The verify window's per-position carry snapshots: `carry_snapshot(t)` is the state AFTER
    /// window token t - the source for `carry_restore` when the accept stops at token t.
    const float* carry_snapshot(int64_t t) const {
        return d_carry_snap_ + t * (state_.carry_bytes() / 4);
    }

    /// The drafter borrows these (docs/DSV4.md P4): it ships no embeddings and no output head.
    const NativeEmbed& embed() const { return embed_; }
    const NativeHead& head() const { return head_; }

    /// Expert staging accounting (docs/DSV4.md P4): what the picked-blob fetch path costs - the
    /// `experts.bin` reads, the H2D copy and the sync that guards the reused slots.  Counted only
    /// when STRATA_DSV4_STAGE_DEBUG is set; per sequence (`reset()` zeroes it).
    struct StageStats {
        double ms = 0.0;
        int64_t calls = 0;      // stage() calls: one per layer pass
        int64_t reads = 0;      // blobs actually read from experts.bin
        int64_t read_bytes = 0;
        int64_t copied = 0;     // bytes copied to the device
        int64_t resident = 0;   // blobs taken from the host mirror instead of the file (see set_experts_ram)
        int64_t resident_bytes = 0;
    };
    const StageStats& stage_stats() const { return stage_stats_; }

private:
    /// Stages exactly the picked experts: reads each id's blob from `experts.bin` into its own
    /// pinned slot, then one H2D copy; the slots are contiguous, which is the pick order the
    /// FFN half expects.
    struct Stager : public Dsv4ExpertSource {
        Dsv4Forward* f;
        int64_t off = 0;  // the current layer's region start in experts.bin
        explicit Stager(Dsv4Forward* ff) : f(ff) {}
        bool stage(const int32_t* ids, int64_t n, int64_t bytes, const uint8_t** out) override;
        std::vector<uint8_t> served;  ///< per slot: filled by a copy straight into `d_blobs_` (1) or via `h_stage_`
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
    uint8_t* h_cache_ = nullptr;   // pinned host LRU cache for recently staged expert blobs
    int64_t cache_slots_ = 0;
    int64_t cache_blob_bytes_ = 0;
    std::unordered_map<int64_t, int64_t> cache_slot_;
    std::list<int64_t> cache_lru_;
    int64_t reserve_cache(int64_t key);
    void touch_cache(int64_t key);
    // The host mirror of `experts.bin` (see set_experts_ram): a prefix of the file, cut at a blob boundary,
    // read once at startup.  [0, experts_locked_bytes_) is page-locked and reaches the GPU by DMA; the rest of
    // the mirror is reached by a CPU copy into `h_stage_`.  Both are cut at blob boundaries: a blob that starts
    // inside the locked range and runs past it is refused by cudaMemcpyAsync.
    uint64_t experts_ram_want_ = 0;
    uint8_t* experts_arena_ = nullptr;
    uint64_t experts_arena_bytes_ = 0;
    uint64_t experts_locked_bytes_ = 0;
    bool load_experts_arena(std::string& err);
    std::vector<int64_t> layer_off_;
    std::vector<kernels::NativeExpertLayout> layer_layout_;
    Stager stager_{this};
    void* stream_ = nullptr;
    StageStats stage_stats_;
    bool stage_dbg_ = false;

    /// P4 hidden capture (see set_hidden_sink).
    std::function<bool(const float* d_fused, int64_t pos0, int64_t n)> sink_;
    std::vector<int64_t> cap_slot_;  ///< per layer id up to n_layers, the slot in the fused row, -1 = none
    int64_t cap_ = 0;                ///< the fused row's slot count
    void capture(const float* in, int64_t n, int64_t pos0);
    float* d_hcap_ = nullptr;        ///< [8][cap * n_embd] the fused rows handed to the sink
    float* d_hcap_tmp_ = nullptr;    ///< [8][n_embd] one slot's hc-mean before interleaving
    float* d_logits_w_ = nullptr;    ///< [8][n_vocab] the verify window's per-token logits
    float* d_carry_snap_ = nullptr;  ///< [8][carry_bytes/4] the verify window's per-position carries
};

}  // namespace strata::core
