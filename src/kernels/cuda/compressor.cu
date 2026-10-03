// src/kernels/cuda/compressor.cu - the DeepSeek-V4 compressor's pooling + carry, docs/DSV4.md P2.
// See the header for the math and the register layout.  The addressing mirrors FreeToken
// `models/deepseek_v4/compress.py` exactly: `overlap_transform` is the (previous-half | current-half)
// window, `decode_step`'s scatter/roll is the register update.
#include "strata/kernels/compressor.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;

void sync_if_needed(void* stream, const char* what) {
    if (stream != nullptr) return;
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

bool check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

/// Resolve one effective-window row of block `b` to its (kv, score) values at column `c`.
///
/// The two readings the whole file turns on: the previous block contributes ONLY its first-half columns
/// (`c`, not `d + c`) and the current block ONLY its second half; and a row with no source (block 0 of a
/// from-scratch prefill) is kv=0 / score=-inf, so it drops out of the softmax rather than joining it with
/// weight 1.  `carry_*` rows are read as-is - the carry stores scores WITH the ape already added.
__device__ __forceinline__ void window_row(int64_t b, int64_t j, int64_t ratio, bool overlap, int64_t d,
                                           int64_t item, const float* __restrict__ kv,
                                           const float* __restrict__ score, const float* __restrict__ ape,
                                           const float* __restrict__ carry_kv,
                                           const float* __restrict__ carry_score, int64_t c, float& k,
                                           float& s) {
    if (overlap && j < ratio) {
        if (b == 0) {
            if (carry_score == nullptr) { k = 0.0f; s = -INFINITY; return; }
            k = carry_kv[j * item + c];
            s = carry_score[j * item + c];
            return;
        }
        const int64_t row = (b - 1) * ratio + j;
        k = kv[row * item + c];
        s = score[row * item + c] + ape[j * item + c];
        return;
    }
    const int64_t jj = overlap ? j - ratio : j;
    const int64_t row = b * ratio + jj;
    const int64_t col = overlap ? d + c : c;
    k = kv[row * item + col];
    s = score[row * item + col] + ape[jj * item + col];
}

__global__ void compressor_prefill_kernel(int64_t ratio, bool overlap, int64_t d, int64_t item,
                                          const float* __restrict__ kv, const float* __restrict__ score,
                                          const float* __restrict__ ape, const float* __restrict__ carry_kv,
                                          const float* __restrict__ carry_score, float* __restrict__ out) {
    const int64_t b = blockIdx.x;
    const int64_t c = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= d) return;
    // the per-block window reads are the same addressing with block `b` folded into the row indices
    const int64_t R = overlap ? 2 * ratio : ratio;
    float maxv = -INFINITY;
    for (int64_t j = 0; j < R; ++j) {
        float k, s;
        window_row(b, j, ratio, overlap, d, item, kv, score, ape, carry_kv, carry_score, c, k, s);
        maxv = fmaxf(maxv, s);
    }
    float acc = 0.0f, denom = 0.0f;
    for (int64_t j = 0; j < R; ++j) {
        float k, s;
        window_row(b, j, ratio, overlap, d, item, kv, score, ape, carry_kv, carry_score, c, k, s);
        const float w = (s == -INFINITY) ? 0.0f : expf(s - maxv);
        denom += w;
        acc += w * k;
    }
    out[b * d + c] = denom > 0.0f ? acc / denom : 0.0f;
}

__global__ void seed_carry_kernel(int64_t seqlen, int64_t ratio, bool overlap, int64_t item,
                                  const float* __restrict__ kv, const float* __restrict__ score,
                                  const float* __restrict__ ape, float* __restrict__ ks,
                                  float* __restrict__ ss) {
    const int64_t row = blockIdx.x;
    const int64_t col = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= item) return;
    const int64_t cutoff = seqlen - seqlen % ratio;
    const int64_t rem = seqlen % ratio;
    float k = 0.0f, s = -INFINITY;
    if (overlap) {
        if (row < ratio) {
            // the reference's `if overlap and cutoff >= ratio` guard: a chunk with no complete block
            // leaves the previous-half rows empty rather than reading negative rows
            if (cutoff >= ratio) {
                const int64_t src = cutoff - ratio + row;
                k = kv[src * item + col];
                s = score[src * item + col] + ape[row * item + col];
            }
        } else if (row - ratio < rem) {
            const int64_t idx = row - ratio;
            const int64_t src = cutoff + idx;
            k = kv[src * item + col];
            s = score[src * item + col] + ape[idx * item + col];
        }
    } else if (row < rem) {
        const int64_t src = cutoff + row;
        k = kv[src * item + col];
        s = score[src * item + col] + ape[row * item + col];
    }
    ks[row * item + col] = k;
    ss[row * item + col] = s;
}

__global__ void write_token_kernel(int64_t slot, int64_t item, const float* __restrict__ kv,
                                   const float* __restrict__ score, const float* __restrict__ ape_row,
                                   float* __restrict__ ks, float* __restrict__ ss) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= item) return;
    ks[slot * item + i] = kv[i];
    ss[slot * item + i] = score[i] + ape_row[i];
}

__global__ void decode_reduce_kernel(int64_t ratio, bool overlap, int64_t d, int64_t item,
                                     const float* __restrict__ ks, const float* __restrict__ ss,
                                     float* __restrict__ compressed) {
    const int64_t c = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= d) return;
    const int64_t R = overlap ? 2 * ratio : ratio;
    float maxv = -INFINITY;
    for (int64_t j = 0; j < R; ++j) {
        const int64_t col = (overlap && j >= ratio) ? d + c : c;
        maxv = fmaxf(maxv, ss[j * item + col]);
    }
    float acc = 0.0f, denom = 0.0f;
    for (int64_t j = 0; j < R; ++j) {
        const int64_t col = (overlap && j >= ratio) ? d + c : c;
        const float s = ss[j * item + col];
        const float w = (s == -INFINITY) ? 0.0f : expf(s - maxv);
        denom += w;
        acc += w * ks[j * item + col];
    }
    compressed[c] = denom > 0.0f ? acc / denom : 0.0f;
}

__global__ void roll_kernel(int64_t ratio, int64_t item, float* __restrict__ ks, float* __restrict__ ss) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = ratio * item;
    if (i >= n) return;
    const int64_t row = i / item, col = i % item;
    ks[row * item + col] = ks[(row + ratio) * item + col];
    ss[row * item + col] = ss[(row + ratio) * item + col];
}

}  // namespace

void compressor_prefill(int64_t seqlen, int64_t ratio, bool overlap, int64_t d, const float* kv,
                        const float* score, const float* ape, const float* carry_kv,
                        const float* carry_score, float* out, void* stream) {
    const int64_t nb = seqlen / ratio;
    if (nb == 0) return;
    const int64_t item = (overlap ? 2 : 1) * d;
    compressor_prefill_kernel<<<dim3((unsigned) nb, (unsigned) ((d + THREADS - 1) / THREADS)), THREADS, 0,
                                (cudaStream_t) stream>>>(ratio, overlap, d, item, kv, score, ape, carry_kv,
                                                         carry_score, out);
    check_launch("compressor_prefill");
    sync_if_needed(stream, "compressor_prefill");
}

void compressor_seed_carry(int64_t seqlen, int64_t ratio, bool overlap, int64_t item, const float* kv,
                           const float* score, const float* ape, float* ks, float* ss, void* stream) {
    const int64_t rows = (overlap ? 2 : 1) * ratio;
    seed_carry_kernel<<<dim3((unsigned) rows, (unsigned) ((item + THREADS - 1) / THREADS)), THREADS, 0,
                       (cudaStream_t) stream>>>(seqlen, ratio, overlap, item, kv, score, ape, ks, ss);
    check_launch("compressor_seed_carry");
    sync_if_needed(stream, "compressor_seed_carry");
}

void compressor_decode_step(int64_t pos, int64_t ratio, bool overlap, int64_t d, const float* kv,
                            const float* score, const float* ape, float* ks, float* ss, float* compressed,
                            void* stream) {
    const int64_t item = (overlap ? 2 : 1) * d;
    const int64_t idx = pos % ratio;
    const int64_t slot = overlap ? ratio + idx : idx;
    write_token_kernel<<<(unsigned) ((item + THREADS - 1) / THREADS), THREADS, 0, (cudaStream_t) stream>>>(
        slot, item, kv, score, ape + idx * item, ks, ss);
    if ((pos + 1) % ratio == 0) {
        decode_reduce_kernel<<<(unsigned) ((d + THREADS - 1) / THREADS), THREADS, 0,
                               (cudaStream_t) stream>>>(ratio, overlap, d, item, ks, ss, compressed);
        if (overlap) {
            roll_kernel<<<(unsigned) ((ratio * item + THREADS - 1) / THREADS), THREADS, 0,
                          (cudaStream_t) stream>>>(ratio, item, ks, ss);
        }
    }
    check_launch("compressor_decode_step");
    sync_if_needed(stream, "compressor_decode_step");
}

}  // namespace strata::kernels
