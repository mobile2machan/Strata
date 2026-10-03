// tests/core/dsv4_layout_test.cpp - the deepseek4 geometry reader and the deepseek4 shape checks
// (docs/DSV4.md P2, first step).
//
//   1. A synthetic deepseek4 GGUF (small dimensions, the artifact's OWN key names) reads through
//      `deepseek4_geometry` into a ModelGeometry; a wrong arch, a missing key, and a short
//      compress_ratios array are each refused by name.
//   2. A synthetic pack (index.txt + dense.bin, the row format `tools/iq_pack.py` writes for deepseek4:
//      floats as stored kind 2, the router as raw BF16 kind 4, quantized projections and the I32 hash
//      table as native shape-only rows) covers all four layer classes of the fixture - window, r=4 with
//      indexer, r=128, and the hash layer - and `check_all` passes.
//   3. The negatives that motivated the checks: a wrong ne1, a hash layer missing its tid2eid, a routed
//      layer missing exp_probs_b, and a router row in the wrong engine form.
// Needs a CUDA device for the arena (exits 77 without one), like native_dense_ple_key_test.
#include "gguf_fixture.hpp"

#include "strata/artifact/dsv4_geometry.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/weights.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
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
               ("strata-dsv4-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

// ---- the fixture geometry: small dimensions, the artifact's key names and class structure.
// 4 layers: window (also the hash layer), r=4 with indexer, r=128, r=4.
constexpr int64_t NL = 4, H = 64, NE = 4, USED = 2, FF = 64, HC = 4, VOCAB = 32;
constexpr int64_t NH = 2, NKV = 1, HD = 8, QL = 16, OL = 16, OG = 2;
constexpr int64_t IDX_H = 2, IDX_K = 4, HASH = 1;
const std::vector<uint64_t> RATIOS = {0, 4, 128, 4};

std::vector<fixture::Kv> dsv4_meta() {
    std::vector<fixture::Kv> kv = {
        fixture::str("general.architecture", "deepseek4"),
        fixture::u32("deepseek4.block_count", NL),
        fixture::u32("deepseek4.embedding_length", H),
        fixture::u32("deepseek4.expert_count", NE),
        fixture::u32("deepseek4.expert_used_count", USED),
        fixture::u32("deepseek4.expert_feed_forward_length", FF),
        fixture::u32("deepseek4.expert_shared_count", 1),
        fixture::u32("deepseek4.hash_layer_count", HASH),
        fixture::u32("deepseek4.hyper_connection.count", HC),
        fixture::u32("deepseek4.hyper_connection.sinkhorn_iterations", 20),
        fixture::u32("deepseek4.attention.head_count", NH),
        fixture::u32("deepseek4.attention.head_count_kv", NKV),
        fixture::u32("deepseek4.attention.key_length", HD),
        fixture::u32("deepseek4.attention.value_length", HD),
        fixture::u32("deepseek4.attention.q_lora_rank", QL),
        fixture::u32("deepseek4.attention.output_lora_rank", OL),
        fixture::u32("deepseek4.attention.output_group_count", OG),
        fixture::u32("deepseek4.attention.sliding_window", 128),
        fixture::u32("deepseek4.attention.indexer.head_count", IDX_H),
        fixture::u32("deepseek4.attention.indexer.key_length", IDX_K),
        fixture::u32("deepseek4.attention.indexer.top_k", 512),
    };
    kv.push_back(fixture::u32arr("deepseek4.attention.compress_ratios", RATIOS));
    std::vector<uint64_t> toks;
    for (uint64_t i = 0; i < VOCAB; ++i) toks.push_back(i);
    kv.push_back(fixture::u32arr("tokenizer.ggml.tokens", toks));
    return kv;
}

// ---- the pack: rows in the format iq_pack writes for deepseek4 (see tools/test_dsv4_pack.py).
struct PackBuilder {
    std::string idx = "# align 256 pool 0 tensors 0\n";
    std::vector<char> dense;
    uint64_t pool = 0;
    size_t rows = 0;
    std::set<std::string> native;   // the skip set generate.cpp builds from NativeDense::served_names

    void align256() { pool = (pool + 255) / 256 * 256; }

    /// kind 2 (F32) / kind 4 (raw BF16): real bytes in dense.bin.
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
    /// kind 0 with code_bits 8: the native shape-only row (the GGUF serves the bytes).
    void add_native(const std::string& name, int64_t ne0, int64_t ne1) {
        idx += name + " 0 0 0 0 0 0 " + std::to_string(ne0) + " " + std::to_string(ne1) +
               " 8 0 32 0 0 0 0 0 0 0\n";
        native.insert(name);
        ++rows;
    }
    void write(const fs::path& dir) {
        // rewrite the header now that pool and the row count are known
        auto nl = idx.find('\n');
        idx.replace(0, nl, "# align 256 pool " + std::to_string(pool) + " tensors " + std::to_string(rows));
        fs::create_directories(dir);
        std::ofstream(dir / "index.txt", std::ios::binary) << idx;
        std::ofstream(dir / "dense.bin", std::ios::binary).write(dense.data(), (std::streamsize) dense.size());
    }
};

// Every tensor one layer must have, per its class - the same table check_one_dsv4 asserts, written from
// the measured artifact's names.
void add_layer(PackBuilder& p, int64_t l) {
    const int64_t r = RATIOS[(size_t) l];
    const bool indexed = r == 4, compressed = r > 0, hash = l < HASH;
    const int64_t comp_w = indexed ? 2 * HD : HD;
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
    p.add_float(b + "ffn_gate_inp.weight", 4, H, NE);          // raw BF16, as the packer writes it
    p.add_native(b + "ffn_gate_shexp.weight", H, FF);
    p.add_native(b + "ffn_up_shexp.weight", H, FF);
    p.add_native(b + "ffn_down_shexp.weight", FF, H);
    if (compressed) {
        p.add_native(b + "attn_compressor_gate.weight", H, comp_w);
        p.add_native(b + "attn_compressor_kv.weight", H, comp_w);
        p.add_float(b + "attn_compressor_ape.weight", 2, comp_w, r);
        p.add_float(b + "attn_compressor_norm.weight", 2, HD, 1);
    }
    if (indexed) {
        p.add_native(b + "indexer.attn_q_b.weight", QL, IDX_H * IDX_K);
        p.add_native(b + "indexer.proj.weight", H, IDX_H);
        p.add_native(b + "indexer_compressor_gate.weight", H, 2 * IDX_K);
        p.add_native(b + "indexer_compressor_kv.weight", H, 2 * IDX_K);
        p.add_float(b + "indexer_compressor_ape.weight", 2, 2 * IDX_K, 4);
        p.add_float(b + "indexer_compressor_norm.weight", 2, IDX_K, 1);
    }
    if (hash) p.add_native(b + "ffn_gate_tid2eid.weight", USED, VOCAB);
    if (!hash) p.add_float(b + "exp_probs_b.bias", 2, NE, 1);
}

ModelGeometry fixture_geometry() {
    ModelGeometry g;
    g.arch = strata::core::ModelArch::DeepSeek4;
    g.n_layers = NL; g.n_embd = H; g.n_expert = NE; g.n_ff = FF; g.hc = HC;
    g.n_head = NH; g.n_head_kv = NKV; g.head_dim = HD;
    g.idx_q_heads = IDX_H; g.idx_key_dim = IDX_K;
    g.dsv4.n_vocab = VOCAB; g.dsv4.q_lora = QL; g.dsv4.o_lora = OL; g.dsv4.o_groups = OG;
    g.dsv4.sliding_window = 128; g.dsv4.idx_topk = 512; g.dsv4.n_expert_used = USED;
    g.dsv4.hash_layers = HASH; g.dsv4.sinkhorn_iters = 20;
    for (uint64_t r : RATIOS) g.dsv4.compress_ratios.push_back((int64_t) r);
    return g;
}

struct Loaded {
    bool ok = false;
    std::string err;
    WeightTable wt;
    void* arena = nullptr;
    ~Loaded() { if (arena) cudaFree(arena); }
};

// generate.cpp's order for a native pack: pool_bytes WITH the skip set, then load with it - the native
// rows keep their metadata (which is all check_layer reads) and get no arena bytes.
void load_pack(const fs::path& pack, const std::set<std::string>& skip, Loaded& l) {
    uint64_t pool = 0;
    if (!WeightTable::pool_bytes(pack.string(), pool, l.err, &skip)) return;
    if (cudaMalloc(&l.arena, (size_t) pool) != cudaSuccess) { l.err = "cudaMalloc"; return; }
    l.ok = l.wt.load(pack.string(), l.arena, pool, l.err, &skip);
}

/// index.txt with one row's fields rewritten (a crude but honest way to break exactly one thing).
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
}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("dsv4_layout_test: no CUDA device, skipped\n");
        return 77;
    }
    TempDir tmp;

    std::printf("1. the geometry reader (synthetic deepseek4 GGUF)\n");
    {
        auto gguf = tmp.path / "dsv4-meta.gguf";
        fixture::write(gguf, dsv4_meta(), {});
        strata::GgufFile f(gguf.string());
        ModelGeometry g;
        std::string err;
        check(strata::artifact::deepseek4_geometry(f, g, err), "reads the fixture");
        check(g.arch == strata::core::ModelArch::DeepSeek4 && g.n_layers == NL && g.n_embd == H, "arch / n_layers / n_embd");
        check(g.head_dim == HD && g.n_head == NH && g.idx_key_dim == IDX_K, "head_dim / heads / indexer key");
        std::vector<int64_t> want_ratios(RATIOS.begin(), RATIOS.end());
        check(g.dsv4.compress_ratios == want_ratios && g.dsv4.n_vocab == VOCAB, "compress_ratios / vocab");
        check(g.dsv4.hash_layers == HASH && g.dsv4.n_expert_used == USED, "hash layers / experts used");

        auto bad = dsv4_meta();
        bad[0] = fixture::str("general.architecture", "qwen4exp");
        auto gg2 = tmp.path / "dsv4-wrongarch.gguf";
        fixture::write(gg2, bad, {});
        strata::GgufFile f2(gg2.string());
        ModelGeometry g2;
        check(!strata::artifact::deepseek4_geometry(f2, g2, err) && err.find("general.architecture") != std::string::npos,
              "refuses a non-deepseek4 arch by name");

        auto missing = dsv4_meta();
        missing.erase(std::find_if(missing.begin(), missing.end(),
                                   [](const fixture::Kv& k) { return k.key == "deepseek4.attention.q_lora_rank"; }));
        auto gg3 = tmp.path / "dsv4-missing.gguf";
        fixture::write(gg3, missing, {});
        strata::GgufFile f3(gg3.string());
        ModelGeometry g3;
        check(!strata::artifact::deepseek4_geometry(f3, g3, err) && err.find("q_lora_rank") != std::string::npos,
              "refuses a missing key by name");

        auto short_ratios = dsv4_meta();
        for (auto& k : short_ratios)
            if (k.key == "deepseek4.attention.compress_ratios") k.arr = {0, 4};
        auto gg4 = tmp.path / "dsv4-short.gguf";
        fixture::write(gg4, short_ratios, {});
        strata::GgufFile f4(gg4.string());
        ModelGeometry g4;
        check(!strata::artifact::deepseek4_geometry(f4, g4, err) &&
              err.find("compress_ratios") != std::string::npos, "refuses a compress_ratios array shorter than block_count");
    }

    std::printf("2. check_all over a full four-class pack\n");
    {
        PackBuilder p;
        for (int64_t l = 0; l < NL; ++l) add_layer(p, l);
        auto pack = tmp.path / "pack";
        p.write(pack);
        Loaded ld;
        load_pack(pack, p.native, ld);
        if (!ld.ok) std::printf("    (load err: %s)\n", ld.err.c_str());
        check(ld.ok, "the fixture pack loads");
        std::string err;
        check(strata::core::check_all(ld.wt, fixture_geometry(), err), "check_all passes on all four classes");
        if (!err.empty()) std::printf("    (err was: %s)\n", err.c_str());

        // 3. the negatives
        auto pack2 = tmp.path / "pack-bad-ne1";
        PackBuilder p2;
        for (int64_t l = 0; l < NL; ++l) add_layer(p2, l);
        p2.write(pack2);
        mutate_row(pack2, "blk.1.attn_q_b.weight",
                   "blk.1.attn_q_b.weight 0 0 0 0 0 0 16 15 8 0 32 0 0 0 0 0 0 0");
        Loaded ld2;
        load_pack(pack2, p2.native, ld2);
        check(ld2.ok, "the broken pack still loads (the row is metadata to the loader)");
        check(!strata::core::check_all(ld2.wt, fixture_geometry(), err), "a wrong ne1 is refused");
        check_err(err, "attn_q_b", "naming the tensor");
        check_err(err, "ne1", "naming the field");

        auto pack3 = tmp.path / "pack-no-tid2eid";
        PackBuilder p3;
        for (int64_t l = 0; l < NL; ++l) add_layer(p3, l);
        p3.write(pack3);
        drop_row(pack3, "blk.0.ffn_gate_tid2eid.weight");
        Loaded ld3;
        load_pack(pack3, p3.native, ld3);
        check(!strata::core::check_all(ld3.wt, fixture_geometry(), err), "a hash layer without tid2eid is refused");
        check_err(err, "ffn_gate_tid2eid", "naming the table");

        auto pack4 = tmp.path / "pack-no-bias";
        PackBuilder p4;
        for (int64_t l = 0; l < NL; ++l) add_layer(p4, l);
        p4.write(pack4);
        drop_row(pack4, "blk.1.exp_probs_b.bias");
        Loaded ld4;
        load_pack(pack4, p4.native, ld4);
        check(!strata::core::check_all(ld4.wt, fixture_geometry(), err), "a routed layer without exp_probs_b is refused");
        check_err(err, "exp_probs_b", "naming the bias");

        auto pack5 = tmp.path / "pack-router-form";
        PackBuilder p5;
        for (int64_t l = 0; l < NL; ++l) add_layer(p5, l);
        p5.write(pack5);
        // the router row as kind 2 (F32): same name, same shape, the WRONG engine form - 2x the bytes.
        mutate_row(pack5, "blk.1.ffn_gate_inp.weight",
                   "blk.1.ffn_gate_inp.weight 0 2 0 1024 0 1024 64 4 0 0 0 0 0 0 0 0 0 0");
        Loaded ld5;
        load_pack(pack5, p5.native, ld5);
        check(ld5.ok, "the F32-router pack loads");
        check(!strata::core::check_all(ld5.wt, fixture_geometry(), err), "a router in the wrong engine form is refused");
        check_err(err, "engine form", "naming the form");
    }

    std::printf(g_fail ? "dsv4_layout_test: %d FAILED\n" : "dsv4_layout_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
