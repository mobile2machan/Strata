// include/strata/kernels/dsv4_indexer.hpp - the DeepSeek-V4 Lightning Indexer's logits, docs/DSV4.md P2.
//
// One shared compressed key per block (MQA), scored against `n_heads` index heads and reduced to ONE
// logit per block (FreeToken `kernel/triton/dsv4/indexer.py`, DeepSeek's `mqa_attn_return_logits`):
//
//     logits[t] = sum_h relu(q[h] . k_t) * weights[h]
//
// The RELU is part of the definition - dropping it lets anti-correlated heads cancel instead of
// contributing nothing - and `weights` already folds in `softmax_scale * n_heads ** -0.5`
// (`Indexer.forward`), so this kernel applies no extra scale.  Blocks past `valid` (the live count,
// `(pos+1)/ratio`) and ids of -1 (no row) come back -inf, which is what makes the downstream top-k
// pick only real blocks.
//
// `q` and the pool rows are bf16 - the index queries went through hadamard + an fp4 round-trip and the
// compressed keys likewise, but by the time they reach here they are just bf16 values.  Which blocks
// exist and where their rows live is the layer's job; this kernel scores a given id list.  Decode
// shape: one query.  Prefill (many queries, causal) is a separate kernel.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// `logits[0..n_stage)`: for `t < valid`, the head-reduced score of pool row `ids[t]` (-1 -> -inf);
/// for `t >= valid`, -inf.  `q` is `[n_heads][d]` bf16, `weights` `[n_heads]` f32, `idx_pool`
/// `[rows][d]` bf16.  One block per scored column.
void dsv4_indexer_logits(const uint16_t* q, const float* weights, const uint16_t* idx_pool,
                         const int32_t* ids, int64_t n_stage, int64_t valid, int64_t n_heads, int64_t d,
                         float* logits, void* stream);

/// Pick the `topk` best columns of `logits` as block indices, descending by score; ties go to the
/// lower index (torch's top-k tie order is unspecified; exact fp32 score ties are measure-zero and
/// only permute equal-score picks).  Columns whose score is -inf (past `valid`, or ids of -1) are
/// never picked: picks past the real ones come back -1, which is what the gather kernel skips.
/// CONSUMES `logits` (picked entries are set to -inf) - the caller keeps them in scratch.
void dsv4_indexer_select(float* logits, int64_t n_stage, int64_t topk, int32_t* out, void* stream);

}  // namespace strata::kernels
