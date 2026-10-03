// src/kernels/cuda/dsv4_rope.cu - the DSv4 interleaved rope with the reference's YaRN,
// docs/DSV4.md P2.  See the header.  One thread per (row, pair); the YaRN blend is computed
// inline per pair (rd/2 = 32 threads of transcendental math per row - negligible next to the
// GEMVs around it, and it keeps the two regimes table-free).
#include "strata/kernels/dsv4_rope.hpp"
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

constexpr double PI = 3.14159265358979323846;

// ops.py::precompute_freqs_cis, per pair j.  The reference builds its table in fp32, but the
// angle `pos * f` reaches 1e5 rad at long context, where fp32 argument reduction is ~1e-2 off;
// the frequency and the angle are kept in double so the rotation matches the math, not the
// reference's own table rounding.
__device__ __forceinline__ double yarn_freq(int j, int rd, double theta, double factor, int64_t orig,
                                            int beta_fast, int beta_slow) {
    double f = pow(theta, -(double)(2 * j) / (double)rd);
    if (orig > 0) {
        const double dim = (double)rd, base = theta, msl = (double)orig;
        auto corr = [&](double rot) {
            return dim * log(msl / (rot * 2.0 * PI)) / (2.0 * log(base));
        };
        int low = (int)floor(corr((double)beta_fast));
        int high = (int)ceil(corr((double)beta_slow));
        low = low < 0 ? 0 : low;
        high = high > rd - 1 ? rd - 1 : high;
        double smooth;
        if (low == high) {
            smooth = (double)j <= (double)low ? 1.0 : 0.0;  // mx += 0.001: the ramp is 0 at low, 1 above
        } else {
            double t = ((double)j - (double)low) / ((double)high - (double)low);
            smooth = 1.0 - (t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t));
        }
        f = f / factor * (1.0 - smooth) + f * smooth;
    }
    return f;
}

template <typename T>
__device__ __forceinline__ float ld_elem(const T* p);
template <>
__device__ __forceinline__ float ld_elem<float>(const float* p) { return *p; }
template <>
__device__ __forceinline__ float ld_elem<uint16_t>(const uint16_t* p) { return f32_from_bf16(*p); }

template <typename T>
__device__ __forceinline__ void st_elem(T* p, float v);
template <>
__device__ __forceinline__ void st_elem<float>(float* p, float v) { *p = v; }
template <>
__device__ __forceinline__ void st_elem<uint16_t>(uint16_t* p, float v) { *p = bf16_from_f32(v); }

template <typename T>
__global__ void rope_kernel(T* __restrict__ x, int64_t row_dim, int rd, int64_t pos, double theta,
                            double factor, int64_t orig, int beta_fast, int beta_slow, int inverse) {
    const int64_t row = blockIdx.x;
    const int j = threadIdx.x;  // rd/2 threads
    if (j * 2 + 1 >= rd) return;
    T* p = x + row * row_dim + (row_dim - rd) + 2 * j;
    const double f = yarn_freq(j, rd, theta, factor, orig, beta_fast, beta_slow);
    double s, c;
    sincos((double)pos * f, &s, &c);
    if (inverse) s = -s;
    const float x0 = ld_elem<T>(p);
    const float x1 = ld_elem<T>(p + 1);
    st_elem<T>(p, (float)(x0 * c - x1 * s));
    st_elem<T>(p + 1, (float)(x0 * s + x1 * c));
}

}  // namespace

void dsv4_rope_f32(float* x, int64_t n_rows, int64_t row_dim, int64_t rd, int64_t pos, double theta,
                   double factor, int64_t orig, int64_t beta_fast, int64_t beta_slow, bool inverse,
                   void* stream) {
    rope_kernel<float><<<(unsigned) n_rows, (unsigned) ((rd + 1) / 2), 0, (cudaStream_t) stream>>>(
        x, row_dim, (int) rd, pos, (float) theta, (float) factor, orig, (int) beta_fast,
        (int) beta_slow, inverse ? 1 : 0);
    check_launch("dsv4_rope_f32");
    sync_if_needed(stream, "dsv4_rope_f32");
}

void dsv4_rope_bf16(uint16_t* x, int64_t n_rows, int64_t row_dim, int64_t rd, int64_t pos, double theta,
                    double factor, int64_t orig, int64_t beta_fast, int64_t beta_slow, bool inverse,
                    void* stream) {
    rope_kernel<uint16_t><<<(unsigned) n_rows, (unsigned) ((rd + 1) / 2), 0, (cudaStream_t) stream>>>(
        x, row_dim, (int) rd, pos, (float) theta, (float) factor, orig, (int) beta_fast,
        (int) beta_slow, inverse ? 1 : 0);
    check_launch("dsv4_rope_bf16");
    sync_if_needed(stream, "dsv4_rope_bf16");
}

}  // namespace strata::kernels
