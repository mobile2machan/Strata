// src/kernels/cuda/dsv4_quant.cu - the DSv4 round-trips + WHT, docs/DSV4.md P2.  See the header.
//
// The scale bit-trick mirrors `_log2_ceil` in fp8_linear.py exactly: ceil(log2(v)) read off the IEEE
// exponent, +1 only when the mantissa is nonzero - so an exact power of two keeps its own exponent and
// the round-trip of an already-on-grid value is the identity.
#include "strata/kernels/dsv4_quant.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_fp8.h>
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

__device__ __forceinline__ float pow2_scale(float amax, float inv_max) {
    const float v = amax * inv_max;
    const uint32_t b = __float_as_uint(v);
    const int e = (int) ((b >> 23) & 0xFFu) - 127 + (((b & 0x7FFFFFu) != 0u) ? 1 : 0);
    return ldexpf(1.0f, e);
}

__device__ __forceinline__ float group_amax(const float* vals, int n, float* smem) {
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) amax = fmaxf(amax, fabsf(vals[i]));
    // block is 32 or 64 threads: one or two warps
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
    if (blockDim.x > 32) {
        const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
        if (lane == 0) smem[warp] = amax;
        __syncthreads();
        amax = fmaxf(smem[0], smem[1]);
    }
    return amax;
}

__global__ void roundtrip_fp8_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int block) {
    __shared__ float smem[2];
    const int64_t base = (int64_t) blockIdx.x * block;
    float v[1];
    v[0] = x[base + threadIdx.x];
    float amax = group_amax(v, 1, smem);
    amax = fmaxf(amax, 1e-4f);
    const float s = pow2_scale(amax, 1.0f / 448.0f);
    const float q = fminf(fmaxf(v[0] / s, -448.0f), 448.0f);
    const __nv_fp8_storage_t c = __nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
    const float back = __half2float(__nv_cvt_fp8_to_halfraw(c, __NV_E4M3));
    y[base + threadIdx.x] = bf16_from_f32(back * s);
}

/// The hardware e2m1 cast's tie table, verbatim from `_round_fp4`: odd-magnitude midpoints (0.75,
/// 1.75, 3.5) round UP to the even grid point, even-magnitude midpoints (0.25, 1.25, 2.5, 5.0) DOWN.
__device__ __forceinline__ float round_e2m1(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float a = fabsf(x);
    float r;
    if (a <= 0.25f) r = 0.0f;
    else if (a < 0.75f) r = 0.5f;
    else if (a <= 1.25f) r = 1.0f;
    else if (a < 1.75f) r = 1.5f;
    else if (a <= 2.5f) r = 2.0f;
    else if (a < 3.5f) r = 3.0f;
    else if (a <= 5.0f) r = 4.0f;
    else r = 6.0f;
    return sign * r;
}

__global__ void roundtrip_fp4_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int block) {
    __shared__ float smem[2];
    const int64_t base = (int64_t) blockIdx.x * block;
    float v[1];
    v[0] = x[base + threadIdx.x];
    float amax = group_amax(v, 1, smem);
    amax = fmaxf(amax, 6.0f * ldexpf(1.0f, -126));
    const float s = pow2_scale(amax, 1.0f / 6.0f);
    const float q = fminf(fmaxf(v[0] / s, -6.0f), 6.0f);
    y[base + threadIdx.x] = bf16_from_f32(round_e2m1(q) * s);
}

__global__ void hadamard_kernel(const float* __restrict__ x, float* __restrict__ y, int d, float scale) {
    extern __shared__ float v[];
    for (int i = threadIdx.x; i < d; i += blockDim.x) v[i] = x[(int64_t) blockIdx.x * d + i];
    __syncthreads();
    for (int h = 1, lg = 0; h < d; h <<= 1, ++lg) {
        for (int i = threadIdx.x; i < d / 2; i += blockDim.x) {
            const int lo = ((i >> lg) << (lg + 1)) | (i & (h - 1));
            const int hi = lo | h;
            const float a = v[lo], b = v[hi];
            v[lo] = a + b;
            v[hi] = a - b;
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < d; i += blockDim.x) y[(int64_t) blockIdx.x * d + i] = v[i] * scale;
}

}  // namespace

void roundtrip_fp8_e4m3(const float* x, uint16_t* y, int64_t n, int64_t block, void* stream) {
    roundtrip_fp8_kernel<<<(unsigned) (n / block), (unsigned) block, 0, (cudaStream_t) stream>>>(x, y,
                                                                                                 (int) block);
    check_launch("roundtrip_fp8_e4m3");
    sync_if_needed(stream, "roundtrip_fp8_e4m3");
}

void roundtrip_fp4_e2m1(const float* x, uint16_t* y, int64_t n, int64_t block, void* stream) {
    roundtrip_fp4_kernel<<<(unsigned) (n / block), (unsigned) block, 0, (cudaStream_t) stream>>>(x, y,
                                                                                                (int) block);
    check_launch("roundtrip_fp4_e2m1");
    sync_if_needed(stream, "roundtrip_fp4_e2m1");
}

void hadamard(const float* x, float* y, int64_t rows, int64_t d, void* stream) {
    int lg = 0;
    while ((1LL << lg) < d) ++lg;
    const float scale = 1.0f / sqrtf((float) d);
    hadamard_kernel<<<(unsigned) rows, (unsigned) d, (size_t) d * 4, (cudaStream_t) stream>>>(x, y, (int) d,
                                                                                             scale);
    check_launch("hadamard");
    sync_if_needed(stream, "hadamard");
}

}  // namespace strata::kernels
