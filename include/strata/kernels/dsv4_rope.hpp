// include/strata/kernels/dsv4_rope.hpp - DeepSeek-V4 rotary application, docs/DSV4.md P2.
//
// DSv4 rotates only the LAST `rd` (=64) dims of each row, in INTERLEAVED pairs
// (`unflatten(-1, (-1,2))` -> complex mul in `ops.py::apply_rotary_emb`) - not the neox
// half-split pairing the qwen path uses.  The frequencies are the reference's YaRN verbatim
// (`ops.py::precompute_freqs_cis`):
//
//   f[j] = theta^(-2j/rd)
//   if original_seq_len > 0:
//     low  = floor(rd*log(orig/(32*2pi)) / (2*ln theta))   clamped to [0, rd-1]
//     high = ceil (rd*log(orig/(1 *2pi)) / (2*ln theta))   clamped to [0, rd-1]
//     smooth[j] = 1 - clamp((j - low)/(high - low), 0, 1)          (j in [0, rd/2))
//     f[j] = f[j]/factor*(1 - smooth[j]) + f[j]*smooth[j]
//   angle = pos * f[j]
//
// The model runs TWO regimes (one table each, `attention.py::__init__`): the attention rope
// (q / kv / o) uses theta 10000 with YaRN OFF for r=0 layers (`original_seq_len = 0`) and ON
// (orig 65536, factor 16, beta 32/1) for compressed layers; the compressor/indexer rope uses
// theta 160000 with YaRN ON.  `inverse` conjugates (the o-path un-rotation).
//
// Rows are addressed as `x[row * row_dim + (row_dim - rd) + 2j]` - q is [heads][head_dim],
// kv/compressed one row, indexer q is [heads][128]; the same call serves all.  The compressor's
// APE is a learned checkpoint parameter, not computed here.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// Rotate the last `rd` dims of each of `n_rows` rows (stride `row_dim`) at absolute `pos`.
/// `theta`/`factor` are the regime's; `orig == 0` means no YaRN scaling.
void dsv4_rope_f32(float* x, int64_t n_rows, int64_t row_dim, int64_t rd, int64_t pos, double theta,
                   double factor, int64_t orig, int64_t beta_fast, int64_t beta_slow, bool inverse,
                   void* stream);
void dsv4_rope_bf16(uint16_t* x, int64_t n_rows, int64_t row_dim, int64_t rd, int64_t pos, double theta,
                    double factor, int64_t orig, int64_t beta_fast, int64_t beta_slow, bool inverse,
                    void* stream);

}  // namespace strata::kernels
