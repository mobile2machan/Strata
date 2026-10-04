// src/kernels/cuda/dsv4_hc.cu - the DSv4 mHC split/Sinkhorn and stream mixing, docs/DSV4.md P2.
// See the header.  The operation ORDER of the Sinkhorn loop is the reference's exactly: softmax rows,
// +eps, ONE column normalization, then (iters-1) rounds of row-then-column.
#include "strata/kernels/dsv4_hc.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int MAX_HC = 8;  // hc is 4 in the real model; the register matrix is hc x hc

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

__global__ void hc_split_sinkhorn_kernel(const float* __restrict__ mixes, const float* __restrict__ scale,
                                         const float* __restrict__ base, int64_t hc, int64_t iters, float eps,
                                         float* __restrict__ pre, float* __restrict__ post,
                                         float* __restrict__ comb) {
    const int64_t row = blockIdx.x;
    const float* m = mixes + row * (2 + hc) * hc;
    const int h0 = (int) hc;

    for (int h = 0; h < h0; ++h)
        pre[row * h0 + h] = 1.0f / (1.0f + expf(-(m[h] * scale[0] + base[h]))) + eps;
    for (int h = 0; h < h0; ++h)
        post[row * h0 + h] = 2.0f / (1.0f + expf(-(m[h0 + h] * scale[1] + base[h0 + h])));

    float c[MAX_HC][MAX_HC];
    const int off = 2 * h0;
    for (int p = 0; p < h0; ++p) {
        float mx = -INFINITY;
        for (int q = 0; q < h0; ++q) mx = fmaxf(mx, m[off + p * h0 + q] * scale[2] + base[off + p * h0 + q]);
        float s = 0.0f;
        for (int q = 0; q < h0; ++q) {
            c[p][q] = expf(m[off + p * h0 + q] * scale[2] + base[off + p * h0 + q] - mx);
            s += c[p][q];
        }
        for (int q = 0; q < h0; ++q) c[p][q] = c[p][q] / s + eps;
    }
    // the initial column normalization, then (iters-1) rounds of row-then-column
    for (int q = 0; q < h0; ++q) {
        float s = eps;  // the reference adds eps to the sum itself
        for (int p = 0; p < h0; ++p) s += c[p][q];
        for (int p = 0; p < h0; ++p) c[p][q] /= s;
    }
    for (int64_t it = 1; it < iters; ++it) {
        for (int p = 0; p < h0; ++p) {
            float s = eps;
            for (int q = 0; q < h0; ++q) s += c[p][q];
            for (int q = 0; q < h0; ++q) c[p][q] /= s;
        }
        for (int q = 0; q < h0; ++q) {
            float s = eps;
            for (int p = 0; p < h0; ++p) s += c[p][q];
            for (int p = 0; p < h0; ++p) c[p][q] /= s;
        }
    }
    for (int p = 0; p < h0; ++p)
        for (int q = 0; q < h0; ++q) comb[row * h0 * h0 + p * h0 + q] = c[p][q];
}

__global__ void hc_pre_combine_kernel(const float* __restrict__ x, const float* __restrict__ pre,
                                      float* __restrict__ y, int64_t hc, int64_t d) {
    const int64_t m = blockIdx.x;
    const int64_t c = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= d) return;
    float acc = 0.0f;
    for (int64_t h = 0; h < hc; ++h) acc += pre[m * hc + h] * x[(m * hc + h) * d + c];
    y[m * d + c] = acc;
}

__global__ void hc_post_combine_kernel(const float* __restrict__ a, const float* __restrict__ res,
                                       const float* __restrict__ post, const float* __restrict__ comb,
                                       float* __restrict__ y, int64_t hc, int64_t d) {
    const int64_t m = blockIdx.x;
    const int64_t c = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= d) return;
    for (int64_t q = 0; q < hc; ++q) {
        float acc = post[m * hc + q] * a[m * d + c];
        for (int64_t p = 0; p < hc; ++p) acc += comb[(m * hc + p) * hc + q] * res[(m * hc + p) * d + c];
        y[(m * hc + q) * d + c] = acc;
    }
}

// model.py::hc_pre mixes: F.linear(x, fn) * rsqrt(mean(x^2) + eps).  One block per (token, output
// row); the block reads x twice (dot + sumsq) - n_out is 24, the redundancy is noise next to the
// GEMVs around it.
__global__ void hc_mixes_kernel(const float* __restrict__ x, const float* __restrict__ fn,
                                float* __restrict__ mixes, int64_t n_in, int64_t n_out, float eps) {
    const int64_t m = blockIdx.x;
    const int o = blockIdx.y;
    const float* xr = x + m * n_in;
    const float* wr = fn + o * n_in;
    float dot = 0.0f, sq = 0.0f;
    for (int64_t i = threadIdx.x; i < n_in; i += blockDim.x) {
        dot += xr[i] * wr[i];
        sq += xr[i] * xr[i];
    }
    __shared__ float red[16];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (int off = 16; off > 0; off >>= 1) {
        dot += __shfl_down_sync(0xFFFFFFFFu, dot, off);
        sq += __shfl_down_sync(0xFFFFFFFFu, sq, off);
    }
    if (lane == 0) { red[warp] = dot; red[8 + warp] = sq; }
    __syncthreads();
    if (threadIdx.x == 0) {
        const int nw = (int) blockDim.x / 32;
        for (int w = 1; w < nw; ++w) { dot += red[w]; sq += red[8 + w]; }
        mixes[m * n_out + o] = dot * rsqrtf(sq / (float) n_in + eps);
    }
}

}  // namespace

void hc_split_sinkhorn(const float* mixes, const float* scale, const float* base, int64_t n, int64_t hc,
                      int64_t iters, float eps, float* pre, float* post, float* comb, void* stream) {
    hc_split_sinkhorn_kernel<<<(unsigned) n, 1, 0, (cudaStream_t) stream>>>(mixes, scale, base, (int) hc,
                                                                            (int) iters, eps, pre, post, comb);
    check_launch("hc_split_sinkhorn");
    sync_if_needed(stream, "hc_split_sinkhorn");
}

void hc_mixes(const float* x, const float* fn, float* mixes, int64_t n, int64_t n_in, int64_t n_out,
              float eps, void* stream) {
    hc_mixes_kernel<<<dim3((unsigned) n, (unsigned) n_out), 256, 0, (cudaStream_t) stream>>>(
        x, fn, mixes, n_in, n_out, eps);
    check_launch("hc_mixes");
    sync_if_needed(stream, "hc_mixes");
}

void hc_pre_combine(const float* x, const float* pre, float* y, int64_t m, int64_t hc, int64_t d,
                    void* stream) {
    hc_pre_combine_kernel<<<dim3((unsigned) m, (unsigned) ((d + 255) / 256)), 256, 0, (cudaStream_t) stream>>>(
        x, pre, y, hc, d);
    check_launch("hc_pre_combine");
    sync_if_needed(stream, "hc_pre_combine");
}

void hc_post_combine(const float* a, const float* res, const float* post, const float* comb, float* y,
                     int64_t m, int64_t hc, int64_t d, void* stream) {
    hc_post_combine_kernel<<<dim3((unsigned) m, (unsigned) ((d + 255) / 256)), 256, 0, (cudaStream_t) stream>>>(
        a, res, post, comb, y, hc, d);
    check_launch("hc_post_combine");
    sync_if_needed(stream, "hc_post_combine");
}

// The head's collapse (model.py::Model.hc_head): pre = sigmoid(mixes * scale + base) + eps -
// the layer hc_pre's mixing without the Sinkhorn, one scalar scale.
__global__ void hc_head_pre_kernel(const float* __restrict__ mixes, const float* __restrict__ scale,
                                   const float* __restrict__ base, int64_t hc, float eps,
                                   float* __restrict__ pre) {
    const int64_t m = blockIdx.x;
    const int h = threadIdx.x;
    if (h >= hc) return;
    const float s = 1.0f / (1.0f + __expf(-(mixes[m * hc + h] * scale[0] + base[h])));
    pre[m * hc + h] = s + eps;
}

void hc_head_pre(const float* mixes, const float* scale, const float* base, float* pre, int64_t n,
                 int64_t hc, float eps, void* stream) {
    hc_head_pre_kernel<<<(unsigned) n, (unsigned) hc, 0, (cudaStream_t) stream>>>(mixes, scale, base,
                                                                                  hc, eps, pre);
    check_launch("hc_head_pre");
    sync_if_needed(stream, "hc_head_pre");
}

// The DSpark drafter's view of the target: llama.cpp's deepseek4 graph exposes each captured layer's
// hidden state as `build_hc_mean(inpL)` - the mean over the hc streams (docs/DSV4.md P4).  The
// drafter's fc consumes exactly those, so this is the capture the target's forward must offer.
__global__ void hc_mean_kernel(const float* __restrict__ x, int64_t hc, int64_t d,
                               float* __restrict__ out) {
    const int64_t m = blockIdx.x;
    for (int64_t j = blockIdx.y * blockDim.x + threadIdx.x; j < d; j += (int64_t) blockDim.x * gridDim.y) {
        float s = 0.0f;
        for (int64_t h = 0; h < hc; ++h) s += x[(m * hc + h) * d + j];
        out[m * d + j] = s / (float) hc;
    }
}

void hc_mean(const float* x, int64_t m, int64_t hc, int64_t d, float* out, void* stream) {
    hc_mean_kernel<<<dim3((unsigned) m, 4), 256, 0, (cudaStream_t) stream>>>(x, hc, d, out);
    check_launch("hc_mean");
    sync_if_needed(stream, "hc_mean");
}

}  // namespace strata::kernels
