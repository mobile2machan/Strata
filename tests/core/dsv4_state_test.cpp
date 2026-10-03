// tests/core/dsv4_state_test.cpp - docs/DSV4.md P2: the DSv4 per-layer pool sizing, arena and
// addressing arithmetic.  The sizing is checked against hand-computed numbers for the three layer
// classes (r=0 / r=4 / r=128), the arena init against the pointer contract, and the window/compressed
// addressing against the reference's `(pos+1)//ratio` / ring-slot arithmetic.
#include "strata/core/dsv4_state.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace strata::core;

int failures = 0;
void expect(bool ok, const char* what) {
    std::printf("    %-46s %s\n", what, ok ? "ok" : "*** FAIL ***");
    if (!ok) ++failures;
}

ModelGeometry synth() {
    ModelGeometry g;
    g.n_layers = 3;
    g.n_embd = 4096;
    g.head_dim = 512;
    g.idx_key_dim = 128;
    g.dsv4.sliding_window = 128;
    g.dsv4.compress_ratios = {0, 4, 128};
    return g;
}

int64_t al(int64_t n) { return (n + 63) / 64 * 64; }

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) != "--selftest") {
        std::fprintf(stderr, "usage: %s --selftest\n", argv[0]);
        return 2;
    }
    std::printf("dsv4_state_test: pool sizing, arena, addressing\n");
    const ModelGeometry g = synth();
    const int64_t max_seq = 8192;

    std::printf("  sizing per layer class\n");
    {
        const Dsv4PoolSizes s0 = dsv4_pool_sizes(g, 0, max_seq);
        expect(s0.window_rows == 128 && s0.window_bytes == al(128 * 512 * 2) && s0.cmp_rows == 0 &&
                   s0.idx_bytes == 0 && s0.carry_floats == 0,
               "r=0: window only");
        const Dsv4PoolSizes s4 = dsv4_pool_sizes(g, 1, max_seq);
        const int64_t cmp_rows = max_seq / 4;
        expect(s4.cmp_rows == cmp_rows && s4.cmp_bytes == al(cmp_rows * 512 * 2) &&
                   s4.idx_bytes == al(cmp_rows * 128 * 2) &&
                   s4.carry_floats == 2 * 4 * 2 * 512 * 2 && s4.idx_carry_floats == 2 * 4 * 256 * 2,
               "r=4: window + cmp + idx + both carries");
        const Dsv4PoolSizes s128 = dsv4_pool_sizes(g, 2, max_seq);
        expect(s128.cmp_rows == max_seq / 128 && s128.idx_bytes == 0 &&
                   s128.carry_floats == 1 * 128 * 512 * 2 && s128.idx_carry_floats == 0,
               "r=128: window + cmp + one carry");
        int64_t want = 0;
        for (int64_t l = 0; l < 3; ++l) want += dsv4_pool_sizes(g, l, max_seq).layer_bytes;
        expect(dsv4_state_bytes(g, max_seq) == want, "total = sum of layers");
    }

    std::printf("  arena init\n");
    {
        Dsv4State st;
        std::string err;
        expect(st.init(g, max_seq, err), "init allocates");
        expect(st.layer(0).window != nullptr && st.layer(0).cmp == nullptr && st.layer(0).idx == nullptr &&
                   st.layer(0).ks == nullptr,
               "r=0 layer: window only, others null");
        expect(st.layer(1).window && st.layer(1).cmp && st.layer(1).idx && st.layer(1).ks &&
                   st.layer(1).ss && st.layer(1).iks && st.layer(1).iss,
               "r=4 layer: all buffers");
        expect(st.layer(2).window && st.layer(2).cmp && st.layer(2).idx == nullptr && st.layer(2).ks &&
                   st.layer(2).ss && st.layer(2).iks == nullptr && st.layer(2).iss == nullptr,
               "r=128 layer: no idx pool/carry");
        // the carry is zeroed by init; the compressor's first decode step reads it
        float probe[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        cudaMemcpy(probe, st.layer(1).ks, 32, cudaMemcpyDeviceToHost);
        bool zero = true;
        for (float v : probe)
            if (v != 0.0f) zero = false;
        expect(zero, "carry starts zeroed");
        st.free();
    }

    std::printf("  addressing\n");
    {
        expect(dsv4_window_slot(257, 128) == 1 && dsv4_cmp_row(259, 4) == 64 &&
                   dsv4_cmp_valid(259, 4) == 65,
               "ring slot / cmp row / valid count");
        std::vector<int32_t> ids(128);
        dsv4_window_ids(5, 128, ids.data());  // pos 5: only 6 live positions
        bool ok = true;
        for (int64_t j = 0; j < 128; ++j) {
            const int64_t q = 5 - 128 + 1 + j;
            const int32_t want = q < 0 ? -1 : (int32_t) q;
            if (ids[(size_t) j] != want) ok = false;
        }
        expect(ok, "window ids: -1 before position 0, ring order after");
        dsv4_window_ids(200, 128, ids.data());
        ok = true;
        for (int64_t j = 0; j < 128; ++j) {
            const int64_t q = 200 - 128 + 1 + j;
            if (ids[(size_t) j] != (int32_t) (q % 128)) ok = false;
        }
        expect(ok, "window ids: full ring, increasing position order");
    }

    std::printf("dsv4_state_test: %s\n", failures == 0 ? "ok" : "*** FAIL ***");
    return failures == 0 ? 0 : 1;
}
