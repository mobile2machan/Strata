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

}  // namespace strata::kernels
