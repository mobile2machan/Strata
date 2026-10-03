// src/kernels/cuda/dsv4_indexer.cu - the DSv4 indexer's head-reduced logits, docs/DSV4.md P2.
// See the header.  One block per scored column: the query and the key row go through shared memory
// (both read many times), each thread owns one head's dot, and the head sum is a block reduction.
#include "strata/kernels/dsv4_indexer.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

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

__global__ void dsv4_indexer_logits_kernel(const uint16_t* __restrict__ q, const float* __restrict__ weights,
                                           const uint16_t* __restrict__ idx_pool, const int32_t* __restrict__ ids,
                                           int64_t valid, int64_t n_heads, int64_t d, float* __restrict__ logits) {
    const int64_t t = blockIdx.x;
    if (t >= valid || ids[t] < 0) {
        if (threadIdx.x == 0) logits[t] = -INFINITY;
        return;
    }
    extern __shared__ float smem[];
    float* qs = smem;                       // [n_heads][d]
    float* ks = smem + n_heads * d;         // [d]
    const uint16_t* qrow = q;
    const uint16_t* krow = idx_pool + (int64_t) ids[t] * d;
    for (int64_t i = threadIdx.x; i < n_heads * d; i += blockDim.x) qs[i] = f32_from_bf16(qrow[i]);
    for (int64_t i = threadIdx.x; i < d; i += blockDim.x) ks[i] = f32_from_bf16(krow[i]);
    __syncthreads();

    float dot = 0.0f;
    const float* qr = qs + (int64_t) threadIdx.x * d;
    for (int64_t c = 0; c < d; ++c) dot += qr[c] * ks[c];
    // the relu is the definition; a negative head contributes nothing, it does not cancel others
    float term = (dot > 0.0f ? dot : 0.0f) * weights[threadIdx.x];
    for (int off = 16; off > 0; off >>= 1) term += __shfl_down_sync(0xFFFFFFFFu, term, off);
    __shared__ float warp_sum[32];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) warp_sum[warp] = term;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int w = 0; w < (int) (blockDim.x + 31) / 32; ++w) total += warp_sum[w];
        logits[t] = total;
    }
}

}  // namespace

void dsv4_indexer_logits(const uint16_t* q, const float* weights, const uint16_t* idx_pool,
                         const int32_t* ids, int64_t n_stage, int64_t valid, int64_t n_heads, int64_t d,
                         float* logits, void* stream) {
    const size_t smem = (size_t) (n_heads * d + d) * 4;
    dsv4_indexer_logits_kernel<<<(unsigned) n_stage, (unsigned) n_heads, smem, (cudaStream_t) stream>>>(
        q, weights, idx_pool, ids, valid, n_heads, d, logits);
    check_launch("dsv4_indexer_logits");
    sync_if_needed(stream, "dsv4_indexer_logits");
}

// top-k over the logits, descending; ties to the lower index.  CONSUMES `logits` (picked entries
// are set to -inf), which is fine - the layer keeps them in a scratch buffer.  -inf is never
// picked, so picks past the real ones come back -1.
__global__ void indexer_select_kernel(float* __restrict__ logits, int64_t n_stage, int64_t topk,
                                      int32_t* __restrict__ out) {
    __shared__ float wv[8];
    __shared__ int wi[8];
    for (int64_t j = 0; j < topk; ++j) {
        float best = -INFINITY;
        int bidx = -1;
        for (int64_t t = threadIdx.x; t < n_stage; t += blockDim.x) {
            const float v = logits[t];
            if (v > best || (v == best && v != -INFINITY && (bidx < 0 || (int) t < bidx))) {
                best = v;
                bidx = (int) t;
            }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xFFFFFFFFu, best, off);
            const int oi = __shfl_down_sync(0xFFFFFFFFu, bidx, off);
            if (ov > best || (ov == best && ov != -INFINITY && (bidx < 0 || (oi >= 0 && oi < bidx)))) {
                best = ov;
                bidx = oi;
            }
        }
        const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
        if (lane == 0) { wv[warp] = best; wi[warp] = bidx; }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 1; w < (int) (blockDim.x + 31) / 32; ++w)
                if (wv[w] > best || (wv[w] == best && wv[w] != -INFINITY && (bidx < 0 || (wi[w] >= 0 && wi[w] < bidx)))) {
                    best = wv[w];
                    bidx = wi[w];
                }
            out[j] = (best == -INFINITY) ? -1 : bidx;
            if (bidx >= 0) logits[bidx] = -INFINITY;
        }
        __syncthreads();
    }
}

void dsv4_indexer_select(float* logits, int64_t n_stage, int64_t topk, int32_t* out, void* stream) {
    indexer_select_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(logits, n_stage, topk, out);
    check_launch("dsv4_indexer_select");
    sync_if_needed(stream, "dsv4_indexer_select");
}

}  // namespace strata::kernels
