// include/strata/core/dsv4_block.hpp - one full DeepSeek-V4 decoder block, one decode token,
// docs/DSV4.md P2 integration: the attention half and the FFN half in the reference's order
// (`model.py::Block.decode_step`), sharing the hc=4 residual stream.
//
// The halves are separately parity-tested (`dsv4_attn_layer_parity`, `dsv4_ffn_layer_parity`);
// this file is the join.  It exists as a function rather than two call sites because the join has
// one real hazard: `hc_post_combine` reads the residual (`stream_in`) while writing its output, so
// the two halves cannot share one buffer - the block keeps an internal staging stream for the
// hand-off, and one scratch arena split between the halves.
#pragma once

#include "strata/core/dsv4_attn.hpp"
#include "strata/core/dsv4_ffn.hpp"
#include "strata/core/dsv4_state.hpp"

#include <cstdint>

namespace strata::core {

struct Dsv4BlockWeights {
    Dsv4AttnWeights attn;
    Dsv4FfnWeights ffn;
};

/// Workspace bytes for one decode token through the full block at `layer` with `n_stage` cmp columns.
int64_t dsv4_block_scratch_bytes(const ModelGeometry& g, int64_t layer, int64_t n_stage);

/// One decode token through the whole block.  `stream_in`/`stream_out` are [hc * n_embd] f32 and
/// must not alias each other (the block stages the hand-off itself); `pos` is the absolute
/// position, `token_id` selects the hash row on a hash layer, `st` is the layer's pool state.
/// `src`, when set, supplies the picked expert blobs (see `Dsv4ExpertSource`).
bool dsv4_block_decode_step(const ModelGeometry& g, int64_t layer, const Dsv4BlockWeights& w,
                            const Dsv4LayerState& st, const float* stream_in, float* stream_out,
                            int64_t pos, int64_t token_id, int64_t n_stage, float* scratch,
                            void* cu_stream, Dsv4ExpertSource* src = nullptr);

/// Workspace bytes for one prefill chunk of `n` tokens (1..8) at `layer` with `n_stage` cmp columns.
int64_t dsv4_block_prefill_scratch_bytes(const ModelGeometry& g, int64_t layer, int64_t n,
                                         int64_t n_stage);

/// `n` tokens (1..8) at positions `pos0..pos0+n-1` through the whole block in one pass.  Streams
/// are token-major [n][hc * n_embd] f32 and must not alias each other; `token_ids` selects the
/// hash row per token.  Same math as `n` decode steps at consecutive positions.  `carry_snap`
/// (see `dsv4_attn_prefill_step`) is the verify window's per-position carry rollback data.
bool dsv4_block_prefill(const ModelGeometry& g, int64_t layer, const Dsv4BlockWeights& w,
                        Dsv4LayerState& st, const float* stream_in, float* stream_out,
                        const int64_t* token_ids, int64_t pos0, int64_t n, int64_t n_stage,
                        float* scratch, void* cu_stream, Dsv4ExpertSource* src = nullptr,
                        float* carry_snap = nullptr, int64_t carry_stride = 0);

}  // namespace strata::core
