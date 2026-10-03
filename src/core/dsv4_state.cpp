// src/core/dsv4_state.cpp - the DSv4 per-layer pool sizing and arena, docs/DSV4.md P2.
// See the header for what each buffer is.
#include "strata/core/dsv4_state.hpp"

#include <cuda_runtime.h>

#include <cstdio>

namespace strata::core {
namespace {

int64_t align64(int64_t n) { return (n + 63) / 64 * 64; }

}  // namespace

Dsv4PoolSizes dsv4_pool_sizes(const ModelGeometry& g, int64_t layer, int64_t max_seq) {
    Dsv4PoolSizes s;
    const int64_t ratio = g.dsv4.compress_ratios[(size_t) layer];
    s.window_rows = g.dsv4.sliding_window;
    s.window_bytes = align64(s.window_rows * g.head_dim * 2);
    if (ratio > 0) {
        s.cmp_rows = max_seq / ratio;
        s.cmp_bytes = align64(s.cmp_rows * g.head_dim * 2);
        const int64_t coff = ratio == 4 ? 2 : 1;
        s.carry_floats = coff * ratio * coff * g.head_dim * 2;  // ks and ss, item = coff*head_dim
        if (ratio == 4) {
            s.idx_bytes = align64(s.cmp_rows * g.idx_key_dim * 2);
            s.idx_carry_floats = coff * ratio * (coff * g.idx_key_dim) * 2;
        }
    }
    s.layer_bytes = s.window_bytes + s.cmp_bytes + s.idx_bytes + align64(s.carry_floats * 4) +
                    align64(s.idx_carry_floats * 4);
    return s;
}

int64_t dsv4_state_bytes(const ModelGeometry& g, int64_t max_seq) {
    int64_t total = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) total += dsv4_pool_sizes(g, l, max_seq).layer_bytes;
    return total;
}

bool Dsv4State::init(const ModelGeometry& g, int64_t max_seq, std::string& err) {
    const int64_t total = dsv4_state_bytes(g, max_seq);
    if (total == 0) { err = "dsv4 state: no layers"; return false; }
    const cudaError_t e = cudaMalloc(&arena_, (size_t) total);
    if (e != cudaSuccess) {
        err = std::string("dsv4 state: cudaMalloc ") + std::to_string(total) + " bytes: " +
              cudaGetErrorString(e);
        return false;
    }
    cudaMemset(arena_, 0, (size_t) total);
    arena_bytes_ = (size_t) total;
    layers_.assign((size_t) g.n_layers, Dsv4LayerState{});
    char* at = (char*) arena_;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const int64_t ratio = g.dsv4.compress_ratios[(size_t) l];
        const Dsv4PoolSizes s = dsv4_pool_sizes(g, l, max_seq);
        Dsv4LayerState& st = layers_[(size_t) l];
        st.window = (uint16_t*) at;
        at += s.window_bytes;
        if (ratio == 0) continue;
        st.cmp = (uint16_t*) at;
        at += s.cmp_bytes;
        const int64_t coff = ratio == 4 ? 2 : 1;
        const int64_t item = coff * g.head_dim;
        st.ks = (float*) at;
        st.ss = st.ks + coff * ratio * item;
        at += align64(s.carry_floats * 4);
        if (ratio != 4) continue;
        st.idx = (uint16_t*) at;
        at += s.idx_bytes;
        const int64_t iitem = 2 * g.idx_key_dim;  // the indexer compressor's item is coff*idx_key_dim... 
        // (coff=2 for r=4, so item = 2*idx_key_dim)
        st.iks = (float*) at;
        st.iss = st.iks + coff * ratio * iitem;
        at += align64(coff * ratio * iitem * 2 * 4);
    }
    return true;
}

void Dsv4State::free() {
    if (arena_ != nullptr) cudaFree(arena_);
    arena_ = nullptr;
    layers_.clear();
}

void Dsv4State::reset() {
    if (arena_ != nullptr) cudaMemset(arena_, 0, arena_bytes_);
}

}  // namespace strata::core
