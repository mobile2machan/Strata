// src/program/dsv4_serve.cpp - the DeepSeek-V4 serve loop, docs/DSV4.md P3.
//
// `strata --serve --pack <dsv4 pack> --dsv4 <shard1.gguf> --max-context N` speaks the same
// stdin/stdout protocol as the Qwen serve path (GEN / T / DONE / STOP / QUIT, READY first), so
// serve/server.py drives it unchanged.  It is deliberately the small sibling of that loop: the
// model's own forward (core::Dsv4Forward, P2) over the pack, greedy decoding, and nothing else -
// no expert cache (the forward stages picked experts itself), no spec/drafter (P4), no
// conversation cache, no sampling keys yet (a request's k=v keys are read and ignored).  One
// sequence at a time, each request from an empty state (`reset`).
//
// The geometry comes from the artifact's own metadata (shard 1, the 5 MB metadata-only file the
// packer was pointed at); the weights from the pack, with the same skip set the P2 smoke test
// uses.  The end-of-turn id comes from the same file's tokenizer metadata.
#include "strata/artifact/dflash_geometry.hpp"
#include "strata/artifact/dsv4_geometry.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/gguf_split.hpp"
#include "strata/core/dspark_forward.hpp"
#include "strata/core/dsv4_forward.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/weights.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace strata::program {

namespace {

/// The pack's skip set: rows the pack serves natively from the GGUF, not into the GPU arena
/// (the same reading as tests/core/dsv4_generate_smoke.cpp).
std::set<std::string> skip_rows(const std::string& pack) {
    std::set<std::string> skip;
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
    return skip;
}

int64_t argmax(const float* v, int64_t n) {
    int64_t best = 0;
    float bv = -1e30f;
    for (int64_t i = 0; i < n; ++i)
        if (v[i] > bv) { bv = v[i]; best = i; }
    return best;
}

}  // namespace

int dsv4_serve_main(const std::string& pack, const std::string& shard1, int64_t max_context, bool serve,
                    const std::string& dspark_gguf, const std::string& dspark_pack,
                    int64_t spec_depth, double spec_conf, const std::string& experts_ram) {
    if (!serve) {
        std::fprintf(stderr, "strata dsv4: the DeepSeek-V4 path serves only (--serve); one-shot generation "
                             "is the Qwen path's\n");
        return 2;
    }
    if (max_context <= 0) {
        std::fprintf(stderr, "strata dsv4: --max-context N is required (the pools scale with it)\n");
        return 2;
    }
    std::string err;
    core::ModelGeometry g;
    int64_t eos_id = 1;
    {
        GgufFile f(shard1);
        if (!artifact::deepseek4_geometry(f, g, err)) {
            std::fprintf(stderr, "strata dsv4: %s: %s\n", shard1.c_str(), err.c_str());
            return 1;
        }
        if (const MetaValue* v = f.get("tokenizer.ggml.eos_token_id")) eos_id = (int64_t) v->u;
    }
    const std::vector<std::string> shards = gguf_split_paths(shard1);
    std::fprintf(stderr, "strata dsv4: geometry %lld layers, hidden %lld, vocab %lld; shards %zu\n",
                 (long long) g.n_layers, (long long) g.n_embd, (long long) g.dsv4.n_vocab, shards.size());

    const std::set<std::string> skip = skip_rows(pack);
    uint64_t pool = 0;
    if (!core::WeightTable::pool_bytes(pack, pool, err, &skip)) {
        std::fprintf(stderr, "strata dsv4: pool: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "strata dsv4: loading the pack (%.2f GiB of weights)...\n", (double) pool / 1073741824.0);
    void* arena = nullptr;
    if (cudaMalloc(&arena, (size_t) pool) != cudaSuccess) {
        std::fprintf(stderr, "strata dsv4: the GPU has no room for the pack's weights (%.2f GiB)\n",
                     (double) pool / 1073741824.0);
        return 1;
    }
    core::WeightTable wt;
    if (!wt.load(pack, arena, pool, err, &skip)) {
        std::fprintf(stderr, "strata dsv4: load: %s\n", err.c_str());
        return 1;
    }
    if (!core::check_all(wt, g, err)) {
        std::fprintf(stderr, "strata dsv4: layout: %s\n", err.c_str());
        return 1;
    }
    core::NativeDense dense;
    if (!dense.load(shards, wt, err, false)) {
        std::fprintf(stderr, "strata dsv4: native dense: %s\n", err.c_str());
        return 1;
    }
    core::Dsv4Forward fwd;
    // --dsv4-experts-ram: keep the experts in host memory instead of reading them per token (docs/DSV4.md).
    // "auto" mirrors as much as the free RAM takes; a number is honoured as asked.
    if (!experts_ram.empty()) {
        const bool fits = experts_ram == "auto";
        const double gib = fits ? 0.0 : std::atof(experts_ram.c_str());
        if (!fits && !(gib > 0.0)) {
            std::fprintf(stderr, "strata dsv4: --dsv4-experts-ram wants a GiB number or \"auto\"\n");
            return 2;
        }
        fwd.set_experts_ram(fits ? core::Dsv4Forward::kExpertsRamWhatFits
                                 : (uint64_t) (gib * 1073741824.0));
    }
    if (!fwd.init(g, pack, shards, wt, err, max_context)) {
        std::fprintf(stderr, "strata dsv4: forward init: %s\n", err.c_str());
        return 1;
    }

    // P4: the DSpark drafter.  Its geometry comes from the sidecar's own metadata; its vocab is
    // the TARGET's (the drafter borrows the embeddings and the head), and its `target_layers` are
    // checked against the target's layer count here, where both are known.
    std::unique_ptr<core::DsparkForward> drafter;
    core::ModelGeometry gd;
    if (!dspark_gguf.empty()) {
        if (dspark_pack.empty()) {
            std::fprintf(stderr, "strata dsv4: --dspark needs --dspark-pack (the sidecar's packed weights)\n");
            return 2;
        }
        {
            GgufFile fs(dspark_gguf);
            if (!artifact::dflash_geometry(fs, gd, err)) {
                std::fprintf(stderr, "strata dsv4: %s: %s\n", dspark_gguf.c_str(), err.c_str());
                return 1;
            }
        }
        gd.dsv4.n_vocab = g.dsv4.n_vocab;
        for (int64_t l : gd.dflash.target_layers)
            if (l > g.n_layers) {
                std::fprintf(stderr, "strata dsv4: dspark target_layer %lld is past the target's %lld layers\n",
                             (long long) l, (long long) g.n_layers);
                return 1;
            }
        const std::vector<std::string> dshards = gguf_split_paths(dspark_gguf);
        const std::set<std::string> dskip = skip_rows(dspark_pack);
        uint64_t dpool = 0;
        if (!core::WeightTable::pool_bytes(dspark_pack, dpool, err, &dskip)) {
            std::fprintf(stderr, "strata dsv4: drafter pool: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata dsv4: loading the drafter (%.2f GiB)...\n", (double) dpool / 1073741824.0);
        void* darena = nullptr;
        if (cudaMalloc(&darena, (size_t) dpool) != cudaSuccess) {
            std::fprintf(stderr, "strata dsv4: no GPU room for the drafter (%.2f GiB)\n",
                         (double) dpool / 1073741824.0);
            return 1;
        }
        static core::WeightTable dwt;  // outlives the loop; the drafter's refs point into it
        if (!dwt.load(dspark_pack, darena, dpool, err, &dskip)) {
            std::fprintf(stderr, "strata dsv4: drafter load: %s\n", err.c_str());
            return 1;
        }
        if (!core::check_all(dwt, gd, err)) {
            std::fprintf(stderr, "strata dsv4: drafter layout: %s\n", err.c_str());
            return 1;
        }
        static core::NativeDense ddense;
        if (!ddense.load(dshards, dwt, err, false)) {
            std::fprintf(stderr, "strata dsv4: drafter native dense: %s\n", err.c_str());
            return 1;
        }
        drafter = std::make_unique<core::DsparkForward>();
        if (!drafter->init(gd, dspark_pack, dshards, dwt, fwd.embed(), fwd.head(), err, max_context)) {
            std::fprintf(stderr, "strata dsv4: drafter init: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata dsv4: drafter ready (%lld stages, block_size %lld, target_layers:",
                     (long long) gd.n_layers, (long long) gd.dflash.block_size);
        for (int64_t l : gd.dflash.target_layers) std::fprintf(stderr, " %lld", (long long) l);
        std::fprintf(stderr, ")\n");
        if (spec_depth < 1) spec_depth = 1;
        // Every hidden the target captures goes straight into the drafter's ring.
        fwd.set_hidden_sink(
            [&](const float* d_fused, int64_t p0, int64_t n) {
                static std::string serr;
                if (drafter->inject(d_fused, p0, n, serr)) return true;
                std::fprintf(stderr, "strata dsv4: drafter inject at %lld: %s\n", (long long) p0,
                             serr.c_str());
                return false;
            },
            gd.dflash.target_layers);
    }
    std::fprintf(stderr, "strata dsv4: ready (max context %lld)\n", (long long) max_context);

    // stdin on its own thread, STOP reaching a running request: the same reading as the Qwen
    // serve loop (read(2)/_read, not std::cin - see the note there).
    std::atomic<bool> stop_req{false};
    std::atomic<bool> quit_req{false};
    std::mutex in_mu;
    std::condition_variable in_cv;
    std::deque<std::string> in_lines;
    bool in_eof = false;
    std::thread([&] {
        std::string l, buf;
        char chunk[4096];
        auto getline_fd = [&](std::string& out) -> bool {
            for (;;) {
                const size_t nlpos = buf.find('\n');
                if (nlpos != std::string::npos) {
                    out.assign(buf, 0, nlpos);
                    buf.erase(0, nlpos + 1);
                    return true;
                }
#if defined(_WIN32)
                const int n = _read(0, chunk, (unsigned) sizeof chunk);
#else
                const ssize_t n = ::read(0, chunk, sizeof chunk);
                if (n < 0 && errno == EINTR) continue;
#endif
                if (n <= 0) {
                    if (buf.empty()) return false;
                    out.swap(buf);
                    buf.clear();
                    return true;
                }
                buf.append(chunk, (size_t) n);
            }
        };
        while (getline_fd(l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l == "STOP") { stop_req.store(true); continue; }
            if (l == "QUIT") { quit_req.store(true); }
            std::lock_guard<std::mutex> lk(in_mu);
            in_lines.push_back(l);
            in_cv.notify_one();
        }
        std::lock_guard<std::mutex> lk(in_mu);
        in_eof = true;
        in_cv.notify_one();
    }).detach();

    std::printf("INFO engine=" STRATA_VERSION " dsv4=1 context=%lld\n", (long long) max_context);
    std::printf("READY %lld stop\n", (long long) max_context);
    std::fflush(stdout);

    for (;;) {
        std::string line;
        {
            std::unique_lock<std::mutex> lk(in_mu);
            in_cv.wait(lk, [&] { return !in_lines.empty() || in_eof; });
            if (in_lines.empty()) return 0;  // stdin closed: the server is gone
            line = std::move(in_lines.front());
            in_lines.pop_front();
        }
        if (line == "QUIT") return 0;
        std::istringstream ss(line);
        std::string verb;
        ss >> verb;
        if (verb != "GEN" && verb != "GENI") {
            std::printf("ERR unknown serve command: %s\n", line.c_str());
            std::fflush(stdout);
            continue;
        }
        long long max_new = 0;
        ss >> max_new;
        if (max_new <= 0) max_new = 1;
        std::vector<std::string> rest;
        std::string tok;
        while (ss >> tok) rest.push_back(tok);
        if (rest.empty()) {
            std::printf("ERR GEN without tokens\n");
            std::fflush(stdout);
            continue;
        }
        // the request's k=v sampling keys are read and ignored (greedy); the ids are the last field
        std::vector<int64_t> ids;
        {
            std::istringstream ids_ss(rest.back());
            std::string one;
            while (std::getline(ids_ss, one, ',')) {
                if (one.empty()) continue;
                ids.push_back(std::atoll(one.c_str()));
            }
        }
        if (ids.empty() || ids.size() + (size_t) max_new > (size_t) max_context) {
            std::printf("ERR prompt (%zu tokens) + max_new does not fit the context (%lld)\n", ids.size(),
                        (long long) max_context);
            std::fflush(stdout);
            continue;
        }
        stop_req.store(false);

        const auto t0 = std::chrono::steady_clock::now();
        fwd.reset();
        if (drafter) drafter->reset();
        std::string ferr;
        if (!fwd.prefill(ids, ferr)) {
            std::printf("ERR prefill: %s\n", ferr.c_str());
            std::fflush(stdout);
            continue;
        }
        const auto t_prompt = std::chrono::steady_clock::now();
        std::printf("PP %zu %zu 0 0\n", ids.size(), ids.size());  // one chunk: the whole prompt
        std::fflush(stdout);

        int64_t generated = 0;
        std::string finish = "length";
        int64_t pos = (int64_t) ids.size();
        std::vector<float> logits((size_t) g.dsv4.n_vocab);
        cudaMemcpy(logits.data(), fwd.logits(), (size_t) g.dsv4.n_vocab * 4, cudaMemcpyDeviceToHost);

        // P4 speculative decoding (docs/DSV4.md): draft a block, verify it in one window pass,
        // accept the greedy-matching prefix, restore the carries to the last accepted position.
        // The window ring and the compressed pool are position-keyed and self-heal; the carries are
        // the only state the rollback touches, and `verify_window` snapshotted them per position.
        int64_t spec_drafts = 0, spec_accepted = 0;
        double draft_ms = 0.0, verify_ms = 0.0;
        while (generated < max_new) {
            if (stop_req.load()) { finish = "cancel"; break; }
            const int64_t t = argmax(logits.data(), g.dsv4.n_vocab);
            std::printf("T %lld\n", (long long) t);
            std::fflush(stdout);
            ++generated;
            if (t == eos_id) { finish = "stop"; break; }
            if (generated >= max_new) break;

            if (!drafter) {
                if (!fwd.decode(t, pos++, ferr)) {
                    std::printf("ERR decode: %s\n", ferr.c_str());
                    std::fflush(stdout);
                    finish = "stop";
                    break;
                }
                cudaMemcpy(logits.data(), fwd.logits(), (size_t) g.dsv4.n_vocab * 4,
                           cudaMemcpyDeviceToHost);
                continue;
            }

            int64_t d = std::min<int64_t>(spec_depth, drafter->block_size());
            d = std::min(d, max_new - generated);
            d = std::min(d, max_context - pos - 1);
            if (d < 1) d = 1;
            std::vector<int64_t> drafts;
            std::vector<float> conf;
            const auto td0 = std::chrono::steady_clock::now();
            if (!drafter->draft(t, pos, d, drafts, conf, ferr)) {
                std::printf("ERR draft: %s\n", ferr.c_str());
                std::fflush(stdout);
                finish = "stop";
                break;
            }
            draft_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count();
            if (spec_conf > 0.0)
                for (int64_t i = 0; i < d; ++i)
                    if (conf[(size_t) i] < spec_conf) { d = i; break; }
            if (d < 1) d = 1;

            std::vector<int64_t> window;
            window.push_back(t);
            for (int64_t i = 0; i < d; ++i) window.push_back(drafts[(size_t) i]);
            const auto tv0 = std::chrono::steady_clock::now();
            if (!fwd.verify_window(window, pos, ferr)) {
                std::printf("ERR verify: %s\n", ferr.c_str());
                std::fflush(stdout);
                finish = "stop";
                break;
            }
            verify_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tv0).count();
            ++spec_drafts;
            std::vector<float> wl((size_t) g.dsv4.n_vocab);
            int64_t accepted = 0;
            bool rejected = false;
            int64_t expected = -1;
            for (int64_t i = 0; i < d; ++i) {
                cudaMemcpy(wl.data(), fwd.logits_window() + i * g.dsv4.n_vocab,
                           (size_t) g.dsv4.n_vocab * 4, cudaMemcpyDeviceToHost);
                const int64_t e = argmax(wl.data(), g.dsv4.n_vocab);
                if (e != drafts[(size_t) i]) { rejected = true; expected = e; break; }
                std::printf("T %lld\n", (long long) drafts[(size_t) i]);
                std::fflush(stdout);
                ++generated;
                ++accepted;
                ++spec_accepted;
                if (drafts[(size_t) i] == eos_id) { finish = "stop"; break; }
                if (generated >= max_new) break;
            }
            if (finish == "stop" || generated >= max_new) {
                fwd.carry_restore(fwd.carry_snapshot(accepted));
                break;
            }
            if (rejected) {
                // The carries back to just past the last accepted token; `expected` (the target's
                // own pick at the rejection point) is then committed through a normal decode, which
                // also re-injects its hidden into the drafter's ring.
                fwd.carry_restore(fwd.carry_snapshot(accepted));
                pos += accepted + 1;
                std::printf("T %lld\n", (long long) expected);
                std::fflush(stdout);
                ++generated;
                if (expected == eos_id) { finish = "stop"; break; }
                if (!fwd.decode(expected, pos++, ferr)) {
                    std::printf("ERR decode: %s\n", ferr.c_str());
                    std::fflush(stdout);
                    finish = "stop";
                    break;
                }
                cudaMemcpy(logits.data(), fwd.logits(), (size_t) g.dsv4.n_vocab * 4,
                           cudaMemcpyDeviceToHost);
                continue;
            }
            // Fully accepted: the target already processed the whole window.  Window slot i holds
            // the distribution for position pos+i+1, so the next position (pos+d+1) is slot d - the
            // last window token's own output.
            pos += d + 1;
            cudaMemcpy(logits.data(), fwd.logits_window() + d * g.dsv4.n_vocab,
                       (size_t) g.dsv4.n_vocab * 4, cudaMemcpyDeviceToHost);
        }
        const auto t_end = std::chrono::steady_clock::now();
        const double prompt_ms = std::chrono::duration<double, std::milli>(t_prompt - t0).count();
        const double decode_ms = std::chrono::duration<double, std::milli>(t_end - t_prompt).count();
        const core::Dsv4Forward::StageStats st = fwd.stage_stats();
        if (st.calls > 0)
            std::fprintf(stderr, "strata dsv4: expert staging %.1f ms in %lld calls, %lld blob reads "
                                 "(%.2f GiB from experts.bin), %lld from the host mirror (%.2f GiB), "
                                 "%.2f GiB copied to the GPU\n",
                         st.ms, (long long) st.calls, (long long) st.reads,
                         (double) st.read_bytes / 1073741824.0, (long long) st.resident,
                         (double) st.resident_bytes / 1073741824.0, (double) st.copied / 1073741824.0);
        if (drafter)
            std::fprintf(stderr, "strata dsv4: spec windows %lld, drafts accepted %lld, draft %.1f ms, verify %.1f ms\n",
                         (long long) spec_drafts, (long long) spec_accepted, draft_ms, verify_ms);
        std::printf("DONE %lld %zu %.1f %.1f %s\n", (long long) generated, ids.size(), prompt_ms, decode_ms,
                    finish.c_str());
        std::fflush(stdout);
        if (quit_req.load()) return 0;
    }
}

}  // namespace strata::program
