// tests/core/dflash_layout_test.cpp - the dflash (DSpark drafter) geometry reader and its shape checks
// (docs/DSV4.md P4, first step).
//
//   1. A synthetic dflash GGUF (small dimensions, the sidecar's OWN key names) reads through
//      `dflash_geometry` into a ModelGeometry; a wrong arch, a missing key, a nonzero compress_ratio, a
//      nonzero hash_layer_count, and a missing block_size / target_layers are each refused by name.
//   2. A synthetic pack (index.txt + dense.bin, the row format iq_pack writes for deepseek4 - the same
//      forms the sidecar uses) covers the three r=0 layers AND the top-level head tensors, and `check_all`
//      passes.
//   3. The negatives that motivated the cross-checks: a conf_proj whose width disagrees with the Markov
//      rank, a markov_w2 that disagrees with markov_w1, a missing fc, and an fc whose width disagrees with
//      the geometry's target_layers count.
//   4. --real PACK_DIR SIDECAR_GGUF: the same checks against the real sidecar.
// Needs a CUDA device for the arena (exits 77 without one), like dsv4_layout_test.
#include "gguf_fixture.hpp"

#include "strata/artifact/dflash_geometry.hpp"
#include "strata/core/layout.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using strata::core::ModelGeometry;
using strata::core::WeightTable;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-84s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
void check_err(const std::string& err, const std::string& names, const std::string& what) {
    bool ok = !err.empty() && err.find(names) != std::string::npos;
    if (!ok) std::printf("    (err was: %s)\n", err.c_str());
    check(ok, what);
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-dflash-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

// ---- the fixture geometry: small dimensions, the sidecar's key names.  3 layers, all r=0; the Markov
// rank and vocab are fixture numbers standing in for the measured 256 / 129280.
constexpr int64_t NL = 3, H = 64, NE = 4, USED = 2, FF = 64, HC = 4;
constexpr int64_t NH = 2, NKV = 1, HD = 8, QL = 16, OL = 16, OG = 2;
constexpr int64_t MRK = 8, VOCAB = 32, BLOCK = 5;
const std::vector<uint64_t> TARGETS = {1, 2, 3};

std::vector<fixture::Kv> dflash_meta() {
    std::vector<fixture::Kv> kv = {
        fixture::str("general.architecture", "dflash"),
        fixture::u32("dflash.block_count", NL),
        fixture::u32("dflash.embedding_length", H),
        fixture::u32("dflash.expert_count", NE),
        fixture::u32("dflash.expert_used_count", USED),
        fixture::u32("dflash.expert_feed_forward_length", FF),
        fixture::u32("dflash.expert_shared_count", 1),
        fixture::u32("dflash.hash_layer_count", 0),
        fixture::u32("dflash.hyper_connection.count", HC),
        fixture::u32("dflash.hyper_connection.sinkhorn_iterations", 20),
        fixture::u32("dflash.attention.head_count", NH),
        fixture::u32("dflash.attention.head_count_kv", NKV),
        fixture::u32("dflash.attention.key_length", HD),
        fixture::u32("dflash.attention.value_length", HD),
        fixture::u32("dflash.attention.q_lora_rank", QL),
        fixture::u32("dflash.attention.output_lora_rank", OL),
        fixture::u32("dflash.attention.output_group_count", OG),
        fixture::u32("dflash.attention.sliding_window", 128),
        fixture::f32("dflash.attention.layer_norm_rms_epsilon", 1e-6f),
        fixture::f32("dflash.rope.freq_base", 10000.0f),
        fixture::u32("dflash.rope.dimension_count", 4),
        fixture::f32("dflash.rope.scaling.factor", 16.0f),
        fixture::u32("dflash.rope.scaling.original_context_length", 65536),
        fixture::f32("dflash.rope.scaling.yarn_beta_fast", 32.0f),
        fixture::f32("dflash.rope.scaling.yarn_beta_slow", 1.0f),
        fixture::f32("dflash.attention.compress_rope_freq_base", 160000.0f),
        fixture::u32("dflash.expert_gating_func", 4),
        fixture::f32("dflash.expert_weights_scale", 1.5f),
        fixture::boolean("dflash.expert_weights_norm", true),
        fixture::f32("dflash.hyper_connection.epsilon", 1e-6f),
        fixture::u32("dflash.block_size", BLOCK),
        fixture::u32("tokenizer.ggml.mask_token_id", 128799),
    };
    std::vector<uint64_t> ratios;
    for (uint64_t l = 0; l < NL; ++l) ratios.push_back(0);
    kv.push_back(fixture::u32arr("dflash.attention.compress_ratios", ratios));
    kv.push_back(fixture::u32arr("dflash.target_layers", TARGETS));
    std::vector<float> clamps;
    for (uint64_t l = 0; l < NL; ++l) clamps.push_back(10.0f);
    kv.push_back(fixture::f32arr("dflash.swiglu_clamp_exp", clamps));
    kv.push_back(fixture::f32arr("dflash.swiglu_clamp_shexp", clamps));
    return kv;
}

// ---- the pack: rows in the format iq_pack writes for deepseek4 (the sidecar's forms are the same).
struct PackBuilder {
    std::string idx = "# align 256 pool 0 tensors 0\n";
    std::vector<char> dense;
    uint64_t pool = 0;
    size_t rows = 0;
    std::set<std::string> native;

    void align256() { pool = (pool + 255) / 256 * 256; }

    void add_float(const std::string& name, int kind, int64_t ne0, int64_t ne1) {
        align256();
        const uint64_t per = kind == 4 ? 2 : 4;
        const uint64_t elems = (uint64_t) ne0 * (uint64_t) ne1;
        const uint64_t bytes = elems * per;
        const uint64_t src = dense.size();
        for (uint64_t i = 0; i < bytes; ++i) dense.push_back((char) fixture::pattern(3, i));
        idx += name + " 0 " + std::to_string(kind) + " " + std::to_string(src) + " " + std::to_string(bytes) +
               " " + std::to_string(pool) + " " + std::to_string(bytes) + " " + std::to_string(ne0) + " " +
               std::to_string(ne1) + " 0 0 0 0 0 0 0 0 0 0\n";
        pool += bytes;
        ++rows;
    }
    void add_native(const std::string& name, int64_t ne0, int64_t ne1) {
        idx += name + " 0 0 0 0 0 0 " + std::to_string(ne0) + " " + std::to_string(ne1) +
               " 8 0 32 0 0 0 0 0 0 0\n";
        native.insert(name);
        ++rows;
    }
    void write(const fs::path& dir) {
        auto nl = idx.find('\n');
        idx.replace(0, nl, "# align 256 pool " + std::to_string(pool) + " tensors " + std::to_string(rows));
        fs::create_directories(dir);
        std::ofstream(dir / "index.txt", std::ios::binary) << idx;
        std::ofstream(dir / "dense.bin", std::ios::binary).write(dense.data(), (std::streamsize) dense.size());
    }
};

// Every tensor one r=0 layer must have - the same table check_one_dsv4 asserts, minus the compressor and
// indexer classes the drafter does not carry.
void add_layer(PackBuilder& p, int64_t l) {
    const std::string b = "blk." + std::to_string(l) + ".";
    p.add_native(b + "attn_q_a.weight", H, QL);
    p.add_native(b + "attn_q_b.weight", QL, NH * HD);
    p.add_native(b + "attn_kv.weight", H, NKV * HD);
    p.add_native(b + "attn_output_a.weight", H, OG * OL);
    p.add_native(b + "attn_output_b.weight", OG * OL, H);
    p.add_float(b + "hc_attn_fn.weight", 2, H * HC, 6 * HC);
    p.add_float(b + "hc_ffn_fn.weight", 2, H * HC, 6 * HC);
    p.add_float(b + "hc_attn_base.weight", 2, 6 * HC, 1);
    p.add_float(b + "hc_attn_scale.weight", 2, HC - 1, 1);
    p.add_float(b + "hc_ffn_base.weight", 2, 6 * HC, 1);
    p.add_float(b + "hc_ffn_scale.weight", 2, HC - 1, 1);
    p.add_float(b + "attn_norm.weight", 2, H, 1);
    p.add_float(b + "ffn_norm.weight", 2, H, 1);
    p.add_float(b + "attn_q_a_norm.weight", 2, QL, 1);
    p.add_float(b + "attn_kv_a_norm.weight", 2, HD, 1);
    p.add_float(b + "attn_sinks.weight", 2, NH, 1);
    p.add_float(b + "ffn_gate_inp.weight", 4, H, NE);
    p.add_native(b + "ffn_gate_shexp.weight", H, FF);
    p.add_native(b + "ffn_up_shexp.weight", H, FF);
    p.add_native(b + "ffn_down_shexp.weight", FF, H);
    p.add_float(b + "exp_probs_b.bias", 2, NE, 1);
}

// The top-level tensors: the fusion, the Markov head, the confidence head, the head's mHC collapse.
void add_extras(PackBuilder& p) {
    p.add_native("fc.weight", H * (int64_t) TARGETS.size(), H);
    p.add_float("enc.output_norm.weight", 2, H, 1);
    p.add_float("markov_w1.weight", 4, MRK, VOCAB);
    p.add_float("markov_w2.weight", 4, MRK, VOCAB);
    p.add_float("conf_proj.weight", 4, H + MRK, 1);
    p.add_float("output_hc_base.weight", 2, HC, 1);
    p.add_float("output_hc_fn.weight", 2, H * HC, HC);
    p.add_float("output_hc_scale.weight", 2, 1, 1);
    p.add_float("output_norm.weight", 2, H, 1);
}

ModelGeometry fixture_geometry() {
    ModelGeometry g;
    g.arch = strata::core::ModelArch::DFlash;
    g.n_layers = NL; g.n_embd = H; g.n_expert = NE; g.n_ff = FF; g.hc = HC;
    g.n_head = NH; g.n_head_kv = NKV; g.head_dim = HD;
    g.dsv4.q_lora = QL; g.dsv4.o_lora = OL; g.dsv4.o_groups = OG;
    g.dsv4.sliding_window = 128; g.dsv4.n_expert_used = USED;
    g.dsv4.hash_layers = 0; g.dsv4.sinkhorn_iters = 20;
    for (int64_t l = 0; l < NL; ++l) g.dsv4.compress_ratios.push_back(0);
    g.dflash.block_size = BLOCK;
    for (uint64_t t : TARGETS) g.dflash.target_layers.push_back((int64_t) t);
    return g;
}

struct Loaded {
    bool ok = false;
    std::string err;
    WeightTable wt;
    void* arena = nullptr;
    ~Loaded() { if (arena) cudaFree(arena); }
};

void load_pack(const fs::path& pack, const std::set<std::string>& skip, Loaded& l) {
    uint64_t pool = 0;
    if (!WeightTable::pool_bytes(pack.string(), pool, l.err, &skip)) return;
    if (cudaMalloc(&l.arena, (size_t) pool) != cudaSuccess) { l.err = "cudaMalloc"; return; }
    l.ok = l.wt.load(pack.string(), l.arena, pool, l.err, &skip);
}

std::string read_text(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

void mutate_row(const fs::path& pack, const std::string& name, const std::string& replacement) {
    std::string text = read_text(pack / "index.txt");
    auto at = text.find(name + " ");
    auto eol = text.find('\n', at);
    text.replace(at, eol - at, replacement);
    std::ofstream(pack / "index.txt", std::ios::binary) << text;
}

void drop_row(const fs::path& pack, const std::string& name) {
    std::string text = read_text(pack / "index.txt");
    auto at = text.find(name + " ");
    auto eol = text.find('\n', at);
    text.erase(at, eol - at + 1);
    std::ofstream(pack / "index.txt", std::ios::binary) << text;
}

void build_pack(PackBuilder& p) {
    for (int64_t l = 0; l < NL; ++l) add_layer(p, l);
    add_extras(p);
}
}  // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("dflash_layout_test: no CUDA device, skipped\n");
        return 77;
    }
    TempDir tmp;

    if (argc >= 4 && std::strcmp(argv[1], "--real") == 0) {
        const fs::path pack = argv[2], sidecar = argv[3];
        std::printf("real: %s\n", sidecar.string().c_str());
        std::string err;
        ModelGeometry g;
        {
            strata::GgufFile f(sidecar.string());
            check(strata::artifact::dflash_geometry(f, g, err), "geometry reads from the real sidecar");
            if (g.arch != strata::core::ModelArch::DFlash) {
                std::printf("dflash_layout_test: real geometry failed\n");
                return 1;
            }
            std::printf("    %lld layers, n_embd %lld, %lld experts, block_size %lld, target_layers:",
                        (long long) g.n_layers, (long long) g.n_embd, (long long) g.n_expert,
                        (long long) g.dflash.block_size);
            for (int64_t t : g.dflash.target_layers) std::printf(" %lld", (long long) t);
            std::printf("\n");
        }
        std::set<std::string> skip;
        {
            std::ifstream idx(pack / "index.txt");
            std::string line;
            while (std::getline(idx, line)) {
                if (line.empty() || line[0] == '#') continue;
                char name[256] = {0};
                int file = 0, kind = 0, bits = 0;
                unsigned long long dummy = 0, dst_bytes = 0;
                long long ne = 0;
                if (std::sscanf(line.c_str(), "%255s %d %d %llu %llu %llu %llu %lld %lld %d", name, &file, &kind,
                                &dummy, &dummy, &dummy, &dst_bytes, &ne, &ne, &bits) != 10)
                    continue;
                if (kind == 0 && bits != 0 && dst_bytes == 0) skip.insert(name);
            }
        }
        std::printf("    native rows found in index.txt: %zu\n", skip.size());
        Loaded ld;
        uint64_t pool = 0;
        if (WeightTable::pool_bytes(pack.string(), pool, err, &skip)) {
            if (cudaMalloc(&ld.arena, (size_t) pool) == cudaSuccess)
                ld.ok = ld.wt.load(pack.string(), ld.arena, pool, err, &skip);
        }
        if (!ld.ok) std::printf("    (load err: %s)\n", err.c_str());
        check(ld.ok, "the real sidecar's pack loads (dense only; natives skipped)");
        check(strata::core::check_all(ld.wt, g, err), "check_all passes over every real drafter layer and head tensor");
        if (!err.empty()) std::printf("    (err was: %s)\n", err.c_str());
        std::printf(g_fail ? "dflash_layout_test: %d FAILED\n" : "dflash_layout_test: all passed\n", g_fail);
        return g_fail ? 1 : 0;
    }

    std::printf("1. the geometry reader (synthetic dflash GGUF)\n");
    {
        auto gguf = tmp.path / "dflash-meta.gguf";
        fixture::write(gguf, dflash_meta(), {});
        strata::GgufFile f(gguf.string());
        ModelGeometry g;
        std::string err;
        check(strata::artifact::dflash_geometry(f, g, err), "reads the fixture");
        check(g.arch == strata::core::ModelArch::DFlash && g.n_layers == NL && g.n_embd == H, "arch / n_layers / n_embd");
        check(g.head_dim == HD && g.n_head == NH && g.n_head_kv == NKV, "head_dim / heads");
        check(g.dsv4.compress_ratios == std::vector<int64_t>(3, 0) && g.dsv4.hash_layers == 0,
              "compress_ratios all zero / hash layers zero");
        check(g.dflash.block_size == BLOCK && g.dflash.target_layers == std::vector<int64_t>(TARGETS.begin(), TARGETS.end()),
              "block_size / target_layers");
        check(g.dflash.mask_token_id == 128799 && g.dflash.sample_from_anchor,
              "mask_token_id / sample_from_anchor default");
        check(g.dsv4.gating_func == 4 && g.dsv4.route_scale == 1.5f && g.dsv4.weights_norm,
              "gating func / route scale / weights norm");
        check(g.dsv4.sliding_window == 128 && g.dsv4.q_lora == QL && g.dsv4.o_groups == OG,
              "sliding window / lora ranks");

        auto bad = dflash_meta();
        bad[0] = fixture::str("general.architecture", "deepseek4");
        auto gg2 = tmp.path / "dflash-wrongarch.gguf";
        fixture::write(gg2, bad, {});
        strata::GgufFile f2(gg2.string());
        ModelGeometry g2;
        check(!strata::artifact::dflash_geometry(f2, g2, err) && err.find("general.architecture") != std::string::npos,
              "refuses a non-dflash arch by name");

        auto missing = dflash_meta();
        missing.erase(std::find_if(missing.begin(), missing.end(),
                                   [](const fixture::Kv& k) { return k.key == "dflash.attention.q_lora_rank"; }));
        auto gg3 = tmp.path / "dflash-missing.gguf";
        fixture::write(gg3, missing, {});
        strata::GgufFile f3(gg3.string());
        ModelGeometry g3;
        check(!strata::artifact::dflash_geometry(f3, g3, err) && err.find("q_lora_rank") != std::string::npos,
              "refuses a missing key by name");

        auto nonzero = dflash_meta();
        for (auto& k : nonzero)
            if (k.key == "dflash.attention.compress_ratios") k.arr = {0, 4, 0};
        auto gg4 = tmp.path / "dflash-ratio.gguf";
        fixture::write(gg4, nonzero, {});
        strata::GgufFile f4(gg4.string());
        ModelGeometry g4;
        check(!strata::artifact::dflash_geometry(f4, g4, err) && err.find("compress_ratio") != std::string::npos,
              "refuses a nonzero compress_ratio (a class the drafter has no tensors for)");

        auto hash = dflash_meta();
        for (auto& k : hash)
            if (k.key == "dflash.hash_layer_count") k.u = 1;
        auto gg5 = tmp.path / "dflash-hash.gguf";
        fixture::write(gg5, hash, {});
        strata::GgufFile f5(gg5.string());
        ModelGeometry g5;
        check(!strata::artifact::dflash_geometry(f5, g5, err) && err.find("hash_layer_count") != std::string::npos,
              "refuses a nonzero hash_layer_count (no tid2eid in a drafter)");

        auto noblock = dflash_meta();
        noblock.erase(std::find_if(noblock.begin(), noblock.end(),
                                   [](const fixture::Kv& k) { return k.key == "dflash.block_size"; }));
        auto gg6 = tmp.path / "dflash-noblock.gguf";
        fixture::write(gg6, noblock, {});
        strata::GgufFile f6(gg6.string());
        ModelGeometry g6;
        check(!strata::artifact::dflash_geometry(f6, g6, err) && err.find("block_size") != std::string::npos,
              "refuses a missing block_size");

        auto notargets = dflash_meta();
        notargets.erase(std::find_if(notargets.begin(), notargets.end(),
                                     [](const fixture::Kv& k) { return k.key == "dflash.target_layers"; }));
        auto gg7 = tmp.path / "dflash-notargets.gguf";
        fixture::write(gg7, notargets, {});
        strata::GgufFile f7(gg7.string());
        ModelGeometry g7;
        check(!strata::artifact::dflash_geometry(f7, g7, err) && err.find("target_layers") != std::string::npos,
              "refuses a missing target_layers");

        auto nomask = dflash_meta();
        nomask.erase(std::find_if(nomask.begin(), nomask.end(),
                                  [](const fixture::Kv& k) { return k.key == "tokenizer.ggml.mask_token_id"; }));
        auto gg8 = tmp.path / "dflash-nomask.gguf";
        fixture::write(gg8, nomask, {});
        strata::GgufFile f8(gg8.string());
        ModelGeometry g8;
        check(!strata::artifact::dflash_geometry(f8, g8, err) && err.find("mask_token_id") != std::string::npos,
              "refuses a missing mask_token_id (the block is [anchor, mask...])");
    }

    std::printf("2. check_all over a full drafter pack\n");
    {
        PackBuilder p;
        build_pack(p);
        auto pack = tmp.path / "pack";
        p.write(pack);
        Loaded ld;
        load_pack(pack, p.native, ld);
        if (!ld.ok) std::printf("    (load err: %s)\n", ld.err.c_str());
        check(ld.ok, "the fixture pack loads");
        std::string err;
        check(strata::core::check_all(ld.wt, fixture_geometry(), err), "check_all passes on the layers and the head");
        if (!err.empty()) std::printf("    (err was: %s)\n", err.c_str());

        // 3. the negatives
        auto pack2 = tmp.path / "pack-conf";
        PackBuilder p2;
        build_pack(p2);
        p2.write(pack2);
        mutate_row(pack2, "conf_proj.weight",
                   "conf_proj.weight 0 4 0 146 0 146 73 1 0 0 0 0 0 0 0 0 0 0");
        Loaded ld2;
        load_pack(pack2, p2.native, ld2);
        check(ld2.ok, "the broken pack still loads (the row is metadata to the loader)");
        check(!strata::core::check_all(ld2.wt, fixture_geometry(), err), "a conf_proj wider than n_embd + markov rank is refused");
        check_err(err, "conf_proj", "naming the tensor");

        auto pack3 = tmp.path / "pack-markov";
        PackBuilder p3;
        build_pack(p3);
        p3.write(pack3);
        mutate_row(pack3, "markov_w2.weight",
                   "markov_w2.weight 0 4 0 512 0 512 8 31 0 0 0 0 0 0 0 0 0 0");
        Loaded ld3;
        load_pack(pack3, p3.native, ld3);
        check(!strata::core::check_all(ld3.wt, fixture_geometry(), err), "a markov_w2 that disagrees with markov_w1 is refused");
        check_err(err, "markov_w2", "naming the tensor");

        auto pack4 = tmp.path / "pack-nofc";
        PackBuilder p4;
        build_pack(p4);
        p4.write(pack4);
        drop_row(pack4, "fc.weight");
        Loaded ld4;
        load_pack(pack4, p4.native, ld4);
        check(!strata::core::check_all(ld4.wt, fixture_geometry(), err), "a missing fc is refused");
        check_err(err, "fc.weight", "naming the fusion");

        auto pack5 = tmp.path / "pack-fcwidth";
        PackBuilder p5;
        build_pack(p5);
        p5.write(pack5);
        // fc as [H * 2, H]: the geometry names three target layers, the fusion is two wide.
        mutate_row(pack5, "fc.weight", "fc.weight 0 0 0 0 0 0 128 64 8 0 32 0 0 0 0 0 0 0");
        Loaded ld5;
        load_pack(pack5, p5.native, ld5);
        check(!strata::core::check_all(ld5.wt, fixture_geometry(), err), "an fc whose width disagrees with target_layers is refused");
        check_err(err, "fc.weight", "naming the fusion");
    }

    std::printf(g_fail ? "dflash_layout_test: %d FAILED\n" : "dflash_layout_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
