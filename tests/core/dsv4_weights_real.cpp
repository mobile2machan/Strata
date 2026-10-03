// tests/core/dsv4_weights_real.cpp - the resolver against the real pack (docs/DSV4.md P2).
//
// `dsv4_weights_real --real <pack-dir> <shard-1.gguf>`:
//   1. loads the pack (with the NativeDense skip set) and the native blobs, as generate does;
//   2. resolves EVERY layer into a Dsv4BlockWeights and checks every pointer the engine reads
//      is non-null and every native GEMV type is one `native_mmvq` implements;
//   3. runs one decode token through the full block at a hash r=0 layer and an indexed r=4
//      layer with the pack's real weights (experts copied from experts.bin) and requires a
//      finite stream out.  Correctness of the arithmetic is the parity tests' job; this test
//      is the proof the pack's bytes actually drive the engine.
#include "strata/artifact/dsv4_geometry.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/gguf_split.hpp"
#include "strata/core/dsv4_block.hpp"
#include "strata/core/dsv4_weights.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace strata;

namespace {

int failures = 0;
void expect(bool ok, const char* what) {
    std::printf("    %-58s %s\n", what, ok ? "ok" : "*** FAIL ***");
    if (!ok) ++failures;
}

struct LayerExperts { int gu = -1, d = -1; long long blob_bytes = 0; };

// native_experts.txt v4: `layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard]`
bool parse_native_experts(const std::string& path, std::vector<LayerExperts>& out, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open " + path; return false; }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        int layer, gu, d;
        long long offset, blob;
        if (!(ss >> layer >> gu >> d >> offset >> blob)) continue;
        if ((size_t) layer >= out.size()) out.resize((size_t) layer + 1);
        out[(size_t) layer] = {gu, d, blob};
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4 || std::strcmp(argv[1], "--real") != 0) {
        std::printf("dsv4_weights_real: run with --real <pack> <shard1> (skipping)\n");
        return 77;
    }
    const std::string pack = argv[2], shard1 = argv[3];
    std::string err;

    core::ModelGeometry g;
    {
        GgufFile f(shard1);
        if (!artifact::deepseek4_geometry(f, g, err)) { std::fprintf(stderr, "geometry: %s\n", err.c_str()); return 1; }
    }
    const std::vector<std::string> shards = gguf_split_paths(shard1);

    // ---- skip set: the index rows the pack serves natively (as dsv4_layout_test builds it).
    std::set<std::string> skip;
    {
        std::ifstream idx(pack + "/index.txt");
        std::string line;
        while (std::getline(idx, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            std::string name, file, kind;
            long long src_off, src_bytes, dst_off, dst_bytes, ne0, ne1, bits, bias, grp;
            ss >> name >> file >> kind >> src_off >> src_bytes >> dst_off >> dst_bytes >> ne0 >> ne1 >>
                bits >> bias >> grp;
            if (kind == "0" && bits != 0 && dst_bytes == 0) skip.insert(name);
        }
    }
    uint64_t pool = 0;
    if (!core::WeightTable::pool_bytes(pack, pool, err, &skip)) { std::fprintf(stderr, "pool: %s\n", err.c_str()); return 1; }
    void* arena = nullptr;
    if (cudaMalloc(&arena, (size_t) pool) != cudaSuccess) { std::fprintf(stderr, "cudaMalloc arena\n"); return 1; }
    core::WeightTable wt;
    if (!wt.load(pack, arena, pool, err, &skip)) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }
    if (!core::check_all(wt, g, err)) { std::fprintf(stderr, "layout: %s\n", err.c_str()); return 1; }
    core::NativeDense dense;
    if (!dense.load(shards, wt, err, false)) { std::fprintf(stderr, "native dense: %s\n", err.c_str()); return 1; }

    std::vector<LayerExperts> le;
    if (!parse_native_experts(pack + "/native_experts.txt", le, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    if (le.size() != (size_t) g.n_layers) { std::fprintf(stderr, "native_experts.txt has %zu layers\n", le.size()); return 1; }

    core::Dsv4WeightResolver resolver;
    if (!resolver.init(g, wt, shards, err)) { std::fprintf(stderr, "resolver init: %s\n", err.c_str()); return 1; }

    // ---- 1+2: every layer resolves, every pointer present, every native type supported.
    std::printf("dsv4_weights_real: %lld layers resolved\n", (long long) g.n_layers);
    bool all_ok = true;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const kernels::NativeExpertLayout lay =
            kernels::native_expert_layout(le[(size_t) l].gu, le[(size_t) l].d, g.n_embd, g.n_ff);
        if (lay.bytes != (size_t) le[(size_t) l].blob_bytes) {
            std::fprintf(stderr, "layer %lld: native_expert_layout bytes %zu != pack blob_bytes %lld\n",
                         (long long) l, lay.bytes, le[(size_t) l].blob_bytes);
            return 1;
        }
        core::Dsv4BlockWeights w;
        if (!resolver.resolve(g, wt, l, lay, nullptr, w, err)) {
            std::fprintf(stderr, "resolve layer %lld: %s\n", (long long) l, err.c_str());
            return 1;
        }
        const core::Dsv4AttnWeights& a = w.attn;
        bool ok = a.q_a && a.q_b && a.kv && a.out_a && a.out_b && a.norm && a.q_a_norm && a.kv_norm &&
                  a.sinks && a.hc_fn && a.hc_base && a.hc_scale && w.ffn.gate && w.ffn.norm &&
                  w.ffn.hc_fn && w.ffn.hc_base && w.ffn.hc_scale && w.ffn.sh_gate && w.ffn.sh_up &&
                  w.ffn.sh_down;
        for (int t : {a.q_a_type, a.q_b_type, a.kv_type, a.out_a_type, a.out_b_type, w.ffn.sh_gate_type,
                      w.ffn.sh_up_type, w.ffn.sh_down_type})
            ok = ok && t >= 0 && kernels::native_mmvq_supported(t);
        const int64_t r = core::dsv4_ratio(g, l);
        if (r > 0) ok = ok && a.comp_kv && a.comp_gate && a.ape && a.comp_norm &&
                        kernels::native_mmvq_supported(a.comp_kv_type) &&
                        kernels::native_mmvq_supported(a.comp_gate_type);
        if (r == 4) ok = ok && a.idx_qb && a.idx_proj && a.idx_comp_kv && a.idx_comp_gate && a.idx_ape &&
                         a.idx_comp_norm && kernels::native_mmvq_supported(a.idx_qb_type);
        if (core::dsv4_is_hash_layer(g, l)) ok = ok && w.ffn.tid2eid != nullptr;
        else ok = ok && w.ffn.probs_b != nullptr;
        if (!ok) { std::fprintf(stderr, "layer %lld: a required weight is missing or unsupported\n", (long long) l); all_ok = false; }
    }
    expect(all_ok, "all 43 layers resolve with supported native types");

    // ---- 3: one decode token through the real block, at a hash r=0 layer and an indexed r=4 one.
    core::Dsv4State state;
    if (!state.init(g, 64, err)) { std::fprintf(stderr, "state init: %s\n", err.c_str()); return 1; }
    std::mt19937 rng(7);
    std::normal_distribution<float> nrm(0.f, 0.05f);
    int64_t hash_layer = -1, indexed_layer = -1;
    for (int64_t l = 0; l < g.n_layers && (hash_layer < 0 || indexed_layer < 0); ++l) {
        if (hash_layer < 0 && core::dsv4_is_hash_layer(g, l) && core::dsv4_ratio(g, l) == 0) hash_layer = l;
        if (indexed_layer < 0 && core::dsv4_ratio(g, l) == 4) indexed_layer = l;
    }
    for (int64_t layer : {hash_layer, indexed_layer}) {
        if (layer < 0) continue;
        // the layer's expert region in experts.bin: 256 blobs, cumulative over earlier layers.
        long long off = 0;
        for (int64_t l = 0; l < layer; ++l) off += le[(size_t) l].blob_bytes * g.n_expert;
        const long long bytes = le[(size_t) layer].blob_bytes * g.n_expert;
        std::vector<uint8_t> host((size_t) bytes);
        {
            std::ifstream ef(pack + "/experts.bin", std::ios::binary);
            ef.seekg(off);
            ef.read((char*) host.data(), bytes);
            if (!ef) { std::fprintf(stderr, "reading experts.bin at %lld\n", off); return 1; }
        }
        uint8_t* dblobs = nullptr;
        if (cudaMalloc(&dblobs, (size_t) bytes) != cudaSuccess) { std::fprintf(stderr, "cudaMalloc blobs\n"); return 1; }
        cudaMemcpy(dblobs, host.data(), (size_t) bytes, cudaMemcpyHostToDevice);
        host.clear();
        host.shrink_to_fit();

        const kernels::NativeExpertLayout lay =
            kernels::native_expert_layout(le[(size_t) layer].gu, le[(size_t) layer].d, g.n_embd, g.n_ff);
        core::Dsv4BlockWeights w;
        if (!resolver.resolve(g, wt, layer, lay, dblobs, w, err)) {
            std::fprintf(stderr, "resolve layer %lld: %s\n", (long long) layer, err.c_str());
            return 1;
        }
        const int64_t dim = g.n_embd, hc = g.hc;
        std::vector<float> h_in((size_t) hc * dim);
        for (auto& v : h_in) v = nrm(rng);
        float *d_in = nullptr, *d_out = nullptr;
        cudaMalloc(&d_in, (size_t) hc * dim * 4);
        cudaMalloc(&d_out, (size_t) hc * dim * 4);
        cudaMemcpy(d_in, h_in.data(), (size_t) hc * dim * 4, cudaMemcpyHostToDevice);
        const int64_t scratch_bytes = core::dsv4_block_scratch_bytes(g, layer, 1);
        float* d_scratch = nullptr;
        if (cudaMalloc(&d_scratch, (size_t) scratch_bytes) != cudaSuccess) { std::fprintf(stderr, "cudaMalloc scratch\n"); return 1; }
        static cudaStream_t stream = [] { cudaStream_t s; cudaStreamCreate(&s); return s; }();
        core::dsv4_block_decode_step(g, layer, w, state.layer(layer), d_in, d_out, /*pos=*/0,
                                     /*token_id=*/(int64_t)(rng() % 1000), /*n_stage=*/1,
                                     d_scratch, (void*) stream);
        std::vector<float> h_out((size_t) hc * dim);
        cudaMemcpy(h_out.data(), d_out, (size_t) hc * dim * 4, cudaMemcpyDeviceToHost);
        size_t bad = 0;
        for (float v : h_out) if (!std::isfinite(v)) ++bad;
        std::printf("  layer %lld (r=%lld): stream out finite %zu/%zu\n", (long long) layer,
                    (long long) core::dsv4_ratio(g, layer), h_out.size() - bad, h_out.size());
        expect(bad == 0, "real weights drive a finite block output");
        cudaFree(d_in); cudaFree(d_out); cudaFree(dblobs); cudaFree(d_scratch);
    }

    std::printf("dsv4_weights_real: %s\n", failures == 0 ? "ok" : "*** FAIL ***");
    return failures == 0 ? 0 : 1;
}
