// src/kernels/cuda/dsv4_attn.cu - the DSv4 tiered-pool gather, docs/DSV4.md P2.  See the header.
//
// One block per head; threads split the 512-wide row by column, so a row read is coalesced and the
// dot product is a block reduction.  The online softmax mirrors `sparse_attn.py`: max/sum over the real
// rows first, the sink joining the denominator once at the end with zero value.
#include "strata/kernels/dsv4_attn.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;
constexpr int MAXC = 4;  // d <= THREADS * MAXC

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

__device__ __forceinline__ float block_sum(float v, float* smem) {
    // warp reduce, then the 8 warp leaders through shared memory
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < THREADS / 32; ++w) total += smem[w];
    return total;
}

__global__ void dsv4_attn_decode_kernel(const float* __restrict__ q, const int32_t* __restrict__ win_ids,
                                        int64_t n_win, const int32_t* __restrict__ cmp_ids, int64_t n_cmp,
                                        const float* __restrict__ sinks, float scale,
                                        const uint16_t* __restrict__ window, const uint16_t* __restrict__ cmp,
                                        int64_t window_cap, int64_t d, float* __restrict__ o) {
    __shared__ float smem[THREADS / 32];
    const int64_t h = blockIdx.x;
    const int c0 = threadIdx.x;
    int nc = 0;
    float qc[MAXC], acc[MAXC];
    for (int k = 0; k < MAXC && c0 + k * THREADS < d; ++k) {
        qc[nc] = q[h * d + c0 + k * THREADS];
        acc[nc] = 0.0f;
        ++nc;
    }
    float m = -INFINITY, l = 0.0f;

    const int64_t total = n_win + n_cmp;
    for (int64_t i = 0; i < total; ++i) {
        const bool is_win = i < n_win;
        const int32_t id = is_win ? win_ids[i] : cmp_ids[i - n_win];
        if (id < 0) continue;  // a padded pick reads nothing
        const uint16_t* row = (is_win ? window + (int64_t) id * d : cmp + (int64_t) id * d);
        float partial = 0.0f;
        for (int k = 0; k < nc; ++k) partial += qc[k] * f32_from_bf16(row[c0 + k * THREADS]);
        const float s = block_sum(partial, smem) * scale;
        __syncthreads();  // reuse smem next row only after everyone read it
        const float m_new = fmaxf(m, s);
        const float corr = (m == -INFINITY) ? 0.0f : expf(m - m_new);
        const float p = expf(s - m_new);
        for (int k = 0; k < nc; ++k) acc[k] = acc[k] * corr + p * f32_from_bf16(row[c0 + k * THREADS]);
        l = l * corr + p;
        m = m_new;
    }

    // the sink: a null key with logit sinks[h] and zero value, once, after the real rows
    if (sinks != nullptr && m != -INFINITY) l += expf(sinks[h] - m);
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    for (int k = 0; k < nc; ++k) o[h * d + c0 + k * THREADS] = acc[k] * inv;
}

}  // namespace

void dsv4_attn_decode(const float* q, const int32_t* win_ids, int64_t n_win, const int32_t* cmp_ids,
                      int64_t n_cmp, const float* sinks, float scale, const Dsv4AttnPools& pools,
                      int64_t window_cap, int64_t n_heads, int64_t d, float* o, void* stream) {
    (void) window_cap;
    dsv4_attn_decode_kernel<<<(unsigned) n_heads, THREADS, 0, (cudaStream_t) stream>>>(
        q, win_ids, n_win, cmp_ids, n_cmp, sinks, scale, pools.window, pools.cmp, window_cap, d, o);
    check_launch("dsv4_attn_decode");
    sync_if_needed(stream, "dsv4_attn_decode");
}

}  // namespace strata::kernels
