// include/strata/core/dsv4_state.hpp - per-layer DeepSeek-V4 attention state, docs/DSV4.md P2.
//
// One sequence's DSv4 attention state per layer, mirroring `attention.py`/`compress.py`:
//
//   * the sliding-window ring: `sliding_window` (128) rows of `head_dim` (512) bf16, already
//     rope-applied and quant-dequant-rounded (the pool contract).  Slot for absolute position p is
//     `p % window`; a decode read lists `[p-window+1, p]` clamped at 0, -1 past the start.
//   * the compressed pool: one row per completed compressor block, row = block index =
//     `p / ratio` (the arithmetic the reference's `full_loc // ratio` reduces to for a single
//     contiguous sequence).  Capacity `max_seq / ratio`.
//   * the indexer pool (r=4 layers only): same block addressing, `idx_key_dim` (128) wide.
//   * the compressor carry: `ks`/`ss` fp32, `coff*ratio` rows of `coff*head_dim` (`coff = 2` for
//     the r=4 overlap variant, 1 for r=128) - the rolling register `compressor_decode_step`
//     advances.  The paged state ring FreeToken keeps for cross-request carry-by-value is not
//     needed for one contiguous sequence: the register IS the carry.
//   * the indexer carry: same shape at `idx_key_dim`.
//
// Sizes come from the geometry; `max_seq` is the session's bound (the 1M context_length is not
// reservable - the pools scale with what the session actually allows, like the QSA pool plan).
// All addressing here is pure arithmetic and unit-tested without a GPU.
#pragma once

#include "strata/core/layout.hpp"

#include <cstdint>
#include <vector>

namespace strata::core {

/// Per-layer pool sizes for one sequence bound of `max_seq` tokens.
struct Dsv4PoolSizes {
    int64_t window_rows = 0, window_bytes = 0;
    int64_t cmp_rows = 0, cmp_bytes = 0;
    int64_t idx_bytes = 0;          // 0 unless the layer has an indexer
    int64_t carry_floats = 0;       // one compressor's ks+ss floats
    int64_t idx_carry_floats = 0;   // the indexer compressor's carry (r=4 only)
    int64_t layer_bytes = 0;
};
Dsv4PoolSizes dsv4_pool_sizes(const ModelGeometry& g, int64_t layer, int64_t max_seq);

/// Total for all layers.
int64_t dsv4_state_bytes(const ModelGeometry& g, int64_t max_seq);

/// One layer's live buffers (device pointers; null where the layer class has no such pool).
struct Dsv4LayerState {
    uint16_t* window = nullptr;  ///< [window][head_dim] bf16
    uint16_t* cmp = nullptr;     ///< [cmp_rows][head_dim] bf16
    uint16_t* idx = nullptr;     ///< [cmp_rows][idx_key_dim] bf16 (r=4 only)
    float* ks = nullptr;         ///< compressor carry rows
    float* ss = nullptr;         ///< (score rows already carry the APE, as the reference stores them)
    float* iks = nullptr;        ///< indexer carry (r=4 only)
    float* iss = nullptr;
};

/// Allocates one arena for every layer's state and points each layer into it.
class Dsv4State {
public:
    bool init(const ModelGeometry& g, int64_t max_seq, std::string& err);
    void free();
    /// Zero every pool back to its fresh `init` contents (same positions, same run twice).
    void reset();
    Dsv4LayerState& layer(int64_t l) { return layers_[(size_t) l]; }
    const Dsv4LayerState& layer(int64_t l) const { return layers_[(size_t) l]; }

    /// The compressor/indexer carry registers, all layers, as one flat buffer (docs/DSV4.md P4).
    /// A speculative verify advances them per token; the window ring and the compressed pool are
    /// position-keyed and self-heal when a rejected position is rewritten, but the carries are a
    /// rolling register - they are the only part of the state a rollback has to restore.
    int64_t carry_bytes() const { return carry_total_; }
    /// Where one layer's carries sit in the flat snapshot buffer (the same span `carry_save` copies).
    int64_t carry_offset(int64_t l) const { return carry_off_[(size_t) l]; }
    void carry_save(void* dst) const;
    void carry_restore(const void* src) const;

private:
    std::vector<Dsv4LayerState> layers_;
    void* arena_ = nullptr;
    size_t arena_bytes_ = 0;
    std::vector<int64_t> carry_off_;      ///< per layer, the byte offset of its carries in the flat buffer
    std::vector<int64_t> carry_len_;      ///< per layer, ks+ss (+iks+iss) bytes
    int64_t carry_total_ = 0;
};

// ---- addressing (pure arithmetic; the layer and its tests share these) ----

/// Window-ring slot for absolute position `p`.
inline int64_t dsv4_window_slot(int64_t p, int64_t window) { return p % window; }

/// Compressed-pool row for the block containing position `p`.
inline int64_t dsv4_cmp_row(int64_t p, int64_t ratio) { return p / ratio; }

/// Live compressed-block count after position `p` (the reference's `(pos+1)//ratio`).
inline int64_t dsv4_cmp_valid(int64_t p, int64_t ratio) { return (p + 1) / ratio; }

/// Fill `out[0..window)`: the decode window id list for position `p` - ring slots for
/// `[p-window+1, p]` in increasing position order, -1 for positions below 0.
inline void dsv4_window_ids(int64_t p, int64_t window, int32_t* out) {
    for (int64_t j = 0; j < window; ++j) {
        const int64_t q = p - window + 1 + j;
        out[j] = q < 0 ? -1 : (int32_t) (q % window);
    }
}

}  // namespace strata::core
