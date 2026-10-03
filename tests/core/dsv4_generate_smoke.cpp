// tests/core/dsv4_generate_smoke.cpp - the first tokens: the full DSv4 model driven over real
// weights, docs/DSV4.md P2. `dsv4_generate_smoke --real <pack> <shard1>`:
//
//   loads the pack as generate does, then greedily decodes eight positions of a fixed prompt.
//   Every logit row must be finite and the argmax in range, and two fresh runs must sample
//   identically (same weights, same kernels, same ids).  Text quality is P3's (the tokenizer
//   is not here); this test's claim is that the model computes, end to end.
#include "strata/artifact/dsv4_geometry.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/gguf_split.hpp"
#include "strata/core/dsv4_forward.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/weights.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace strata;

int main(int argc, char** argv) {
    if (argc < 4 || std::strcmp(argv[1], "--real") != 0) {
        std::printf("dsv4_generate_smoke: run with --real <pack> <shard1> (skipping)\n");
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

    std::set<std::string> skip;
    {
        std::ifstream idx(pack + "/index.txt");
        std::string line;
        while (std::getline(idx, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            std::string name, file, kind;
            long long src_off, src_bytes, dst_off, dst_bytes, ne0, ne1, bits;
            ss >> name >> file >> kind >> src_off >> src_bytes >> dst_off >> dst_bytes >> ne0 >> ne1 >> bits;
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

    const std::vector<int64_t> prompt = {0, 151, 4023, 917, 2046, 88, 12345, 60231};
    int failures = 0;
    std::vector<int64_t> sampled;
    core::Dsv4Forward fwd;
    if (!fwd.init(g, pack, shards, wt, err)) { std::fprintf(stderr, "forward init: %s\n", err.c_str()); return 1; }
    for (int pass = 0; pass < 2; ++pass) {
        fwd.reset();
        std::vector<int64_t> got;
        for (size_t pos = 0; pos < prompt.size(); ++pos) {
            if (!fwd.decode(prompt[pos], (int64_t) pos, err)) {
                std::fprintf(stderr, "decode pos %zu: %s\n", pos, err.c_str());
                return 1;
            }
            std::vector<float> logits((size_t) g.dsv4.n_vocab);
            cudaMemcpy(logits.data(), fwd.logits(), (size_t) g.dsv4.n_vocab * 4, cudaMemcpyDeviceToHost);
            size_t bad = 0;
            for (float v : logits) if (!std::isfinite(v)) ++bad;
            int64_t best = 0;
            float bv = -1e30f;
            for (int64_t v = 0; v < g.dsv4.n_vocab; ++v)
                if (logits[(size_t) v] > bv) { bv = logits[(size_t) v]; best = v; }
            std::printf("  pass %d pos %zu: token %lld -> argmax %lld (logit %.3f, nonfinite %zu)\n",
                        pass, pos, (long long) prompt[pos], (long long) best, bv, bad);
            if (bad != 0) ++failures;
            got.push_back(best);
        }
        if (pass == 0) sampled = got;
        else if (got != sampled) { std::fprintf(stderr, "the two passes sampled differently\n"); ++failures; }
    }
    std::printf("dsv4_generate_smoke: %s\n", failures == 0 ? "ok" : "*** FAIL ***");
    return failures == 0 ? 0 : 1;
}
