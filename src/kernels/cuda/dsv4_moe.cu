// src/kernels/cuda/dsv4_moe.cu - the DSv4 router (sqrtsoftplus score path + hash table path) and the
// clamped SwiGLU, docs/DSV4.md P2.  See the header for the math.
#include "strata/kernels/dsv4_moe.hpp"

#include <cuda_runtime.h>

#include <algorithm>
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

__device__ __forceinline__ float sqrtsoftplus(float x) {
    const float sp = x > 20.0f ? x : log1pf(expf(x));
    return sqrtf(sp > 0.0f ? sp : 0.0f);
}

// One block per token.  smem holds the biased scores; the pre-bias scores are recomputed from `raw`
// when the top-k indices are gathered (cheap: topk reads only).
__global__ void router_score_kernel(const float* __restrict__ raw, const float* __restrict__ bias,
                                    float* __restrict__ weights, int32_t* __restrict__ indices,
                                    int64_t n_expert, int64_t topk, float route_scale) {
    extern __shared__ float biased[];
    const int64_t row = blockIdx.x;
    const float* r = raw + row * n_expert;
    const int ne = (int) n_expert;
    for (int i = threadIdx.x; i < ne; i += blockDim.x) biased[i] = sqrtsoftplus(r[i]) + bias[i];
    __syncthreads();
    if (threadIdx.x != 0) return;
    float sum = 0.0f;
    for (int64_t j = 0; j < topk; ++j) {
        int best = 0;
        float bv = biased[0];
        for (int i = 1; i < ne; ++i)
            if (biased[i] > bv) { bv = biased[i]; best = i; }
        biased[best] = -INFINITY;
        indices[row * topk + j] = best;
        weights[row * topk + j] = sqrtsoftplus(r[best]);
        sum += weights[row * topk + j];
    }
    const float inv = route_scale / sum;
    for (int64_t j = 0; j < topk; ++j) weights[row * topk + j] *= inv;
}

// One thread per token: the table gives the indices, the weights are the pre-bias scores at them.
__global__ void router_hash_kernel(const float* __restrict__ raw, const int32_t* __restrict__ tid2eid,
                                   const int64_t* __restrict__ tokens, float* __restrict__ weights,
                                   int32_t* __restrict__ indices, int64_t n, int64_t n_expert, int64_t topk,
                                   float route_scale) {
    const int64_t row = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (row >= n) return;
    const int64_t tok = tokens[row];
    float w[64];
    float sum = 0.0f;
    for (int64_t j = 0; j < topk; ++j) {
        const int32_t e = tid2eid[tok * topk + j];
        indices[row * topk + j] = e;
        w[j] = sqrtsoftplus(raw[row * n_expert + e]);
        sum += w[j];
    }
    const float inv = route_scale / sum;
    for (int64_t j = 0; j < topk; ++j) weights[row * topk + j] = w[j] * inv;
}

__global__ void swiglu_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                              float* __restrict__ out, int64_t n, float limit) {
    const bool clamped = limit > 0.0f;
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n;
         i += (int64_t) blockDim.x * gridDim.x) {
        float g = gate[i], u = up[i];
        if (clamped) {
            u = fminf(fmaxf(u, -limit), limit);
            g = fminf(g, limit);
        }
        out[i] = g / (1.0f + expf(-g)) * u;
    }
}

}  // namespace

void dsv4_router_score(const float* raw, const float* bias, float* weights, int32_t* indices, int64_t n,
                       int64_t n_expert, int64_t topk, float route_scale, void* stream) {
    router_score_kernel<<<(unsigned) n, 128, (size_t) n_expert * 4, (cudaStream_t) stream>>>(
        raw, bias, weights, indices, n_expert, topk, route_scale);
    check_launch("dsv4_router_score");
    sync_if_needed(stream, "dsv4_router_score");
}

void dsv4_router_hash(const float* raw, const int32_t* tid2eid, const int64_t* tokens, float* weights,
                      int32_t* indices, int64_t n, int64_t n_expert, int64_t topk, float route_scale,
                      void* stream) {
    router_hash_kernel<<<(unsigned) ((n + 127) / 128), 128, 0, (cudaStream_t) stream>>>(
        raw, tid2eid, tokens, weights, indices, n, n_expert, topk, route_scale);
    check_launch("dsv4_router_hash");
    sync_if_needed(stream, "dsv4_router_hash");
}

void dsv4_swiglu(const float* gate, const float* up, float* out, int64_t n, float limit, void* stream) {
    const int blocks = (int) std::min<int64_t>((n + 255) / 256, 4096);
    swiglu_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(gate, up, out, n, limit);
    check_launch("dsv4_swiglu");
    sync_if_needed(stream, "dsv4_swiglu");
}

}  // namespace strata::kernels
