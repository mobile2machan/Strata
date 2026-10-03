// include/strata/kernels/dsv4_hc.hpp - DeepSeek-V4 manifold-constrained Hyper-Connections,
// docs/DSV4.md P2.
//
// The residual stream is `hc` parallel copies of the hidden state (hc=4, dim 4096). Twice per block
// (attention and FFN) the engine:
//
//   1. projects the flattened stream to a `mix_hc = (2+hc)*hc` mix vector (a plain bf16 GEMV - the
//      existing `bf16_gemv` serves it; the caller applies the stream's RMS `rsqrt` factor first,
//      `model.py::hc_pre`), then splits it (`model.py` / `sinkhorn.py::hc_split_sinkhorn`):
//         pre[h]  = sigmoid(mixes[h] * scale[0] + base[h]) + eps
//         post[h] = 2 * sigmoid(mixes[hc+h] * scale[1] + base[hc+h])
//         comb    = the [hc,hc] block: row-softmax, +eps, then Sinkhorn: an initial COLUMN
//                   normalization, then (iters-1) rounds of row-then-column normalization
//                   (each dividing by the sum + eps).  Iters and eps are model config
//                   (measured: 20 and 1e-6).
//   2. collapses the streams into the sublayer input:  y[d] = sum_h pre[h] * x[h][d]
//   3. re-expands the sublayer output `a` back over the streams:
//         y[q][d] = post[q] * a[d] + sum_p comb[p][q] * res[p][d]
//      - the reduction is over comb's FIRST axis, the pre-Sinkhorn row axis; swapping the axes keeps
//      every shape legal and breaks the mixing.
//
// `hc_split_sinkhorn` runs one thread per token with the whole hc x hc matrix in registers; the two
// combines are elementwise over dim.  All fp32 - the stream is fp32 between the bf16 GEMVs.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// `mixes` is `[n][mix_hc]` with `mix_hc = (2+hc)*hc`; `scale` is 3 floats (pre/post/comb), `base`
/// `mix_hc` floats.  Outputs `pre [n][hc]`, `post [n][hc]`, `comb [n][hc][hc]`.
void hc_split_sinkhorn(const float* mixes, const float* scale, const float* base, int64_t n, int64_t hc,
                       int64_t iters, float eps, float* pre, float* post, float* comb, void* stream);

/// `y[m][d] = sum_h pre[m][h] * x[m][h][d]`.
void hc_pre_combine(const float* x, const float* pre, float* y, int64_t m, int64_t hc, int64_t d,
                    void* stream);

/// `y[m][q][d] = post[m][q] * a[m][d] + sum_p comb[m][p][q] * res[m][p][d]`.
void hc_post_combine(const float* a, const float* res, const float* post, const float* comb, float* y,
                     int64_t m, int64_t hc, int64_t d, void* stream);

}  // namespace strata::kernels
