// src/core/dsv4_forward.cpp - see include/strata/core/dsv4_forward.hpp.
#include "strata/core/dsv4_forward.hpp"

#include "strata/kernels/dsv4_hc.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/platform/memory.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace strata::core {
namespace {

bool parse_layer_experts(const std::string& path, int64_t n_layers, int64_t n_expert, int64_t dim,
                         int64_t ff, std::vector<int64_t>& off, int64_t& max_blob,
                         std::vector<kernels::NativeExpertLayout>& lay, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open " + path; return false; }
    std::vector<int64_t> blob((size_t) n_layers, 0);
    std::vector<int> gu((size_t) n_layers, -1), d((size_t) n_layers, -1);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        int layer, g, dd;
        long long offset, bl;
        if (!(ss >> layer >> g >> dd >> offset >> bl)) continue;
        if (layer < 0 || layer >= n_layers) continue;
        gu[(size_t) layer] = g;
        d[(size_t) layer] = dd;
        blob[(size_t) layer] = bl;
    }
    int64_t cur = 0;
    max_blob = 0;
    for (int64_t l = 0; l < n_layers; ++l) {
        if (gu[(size_t) l] < 0) { err = "native_experts.txt has no row for layer " + std::to_string(l); return false; }
        const kernels::NativeExpertLayout nl = kernels::native_expert_layout(gu[(size_t) l], d[(size_t) l], dim, ff);
        if (nl.bytes != (size_t) blob[(size_t) l]) {
            err = "layer " + std::to_string(l) + ": native_expert_layout bytes != pack blob_bytes";
            return false;
        }
        off.push_back(cur);
        max_blob = std::max(max_blob, (int64_t) nl.bytes);
        lay.push_back(nl);
        cur += blob[(size_t) l] * n_expert;
    }
    return true;
}

}  // namespace

Dsv4Forward::~Dsv4Forward() {
    cudaFree(d_a_); cudaFree(d_b_); cudaFree(d_mixes_); cudaFree(d_pre_); cudaFree(d_y_);
    cudaFree(d_logits_); cudaFree(d_scratch_); cudaFree(d_blobs_);
    cudaFree(d_pa_); cudaFree(d_pb_); cudaFree(d_ptok_);
    cudaFree(d_hcap_); cudaFree(d_hcap_tmp_); cudaFree(d_logits_w_); cudaFree(d_carry_snap_);
    if (h_stage_ != nullptr) {
        cudaHostUnregister(h_stage_);
        std::free(h_stage_);
    }
    if (h_cache_ != nullptr) {
        std::free(h_cache_);
    }
    if (experts_arena_ != nullptr) {
        if (experts_locked_bytes_ != 0) cudaHostUnregister(experts_arena_);
        std::free(experts_arena_);
    }
    if (experts_file_ != nullptr) std::fclose(experts_file_);
    if (stream_ != nullptr) cudaStreamDestroy((cudaStream_t) stream_);
}

bool Dsv4Forward::init(const ModelGeometry& g, const std::string& pack_dir,
                       const std::vector<std::string>& shards, WeightTable& wt, std::string& err,
                       int64_t max_seq) {
    g_ = g;
    wt_ = &wt;
    if (!embed_.load(shards, g.n_embd, g.dsv4.n_vocab, err)) return false;
    if (!head_.load(shards, g.n_embd, g.dsv4.n_vocab, err)) return false;
    if (!resolver_.init(g, wt, shards, err)) return false;
    if (!state_.init(g, max_seq, err)) return false;

    auto resident = [&](const char* name, const float*& out) -> bool {
        const WeightRef* ref = wt.find(std::string(name));
        if (!ref || ref->kind != WeightKind::F32 || ref->data == nullptr) {
            err = std::string(name) + " is not a resident F32 tensor";
            return false;
        }
        out = (const float*) ref->data;
        return true;
    };
    if (!resident("output_hc_fn.weight", hc_head_fn_)) return false;
    if (!resident("output_hc_base.weight", hc_head_base_)) return false;
    if (!resident("output_hc_scale.weight", hc_head_scale_)) return false;
    if (!resident("output_norm.weight", out_norm_)) return false;

    int64_t max_blob = 0;
    if (!parse_layer_experts(pack_dir + "/native_experts.txt", g.n_layers, g.n_expert, g.n_embd,
                             g.n_ff, layer_off_, max_blob, layer_layout_, err))
        return false;
    experts_file_ = std::fopen((pack_dir + "/experts.bin").c_str(), "rb");
    if (experts_file_ == nullptr) { err = "cannot open " + pack_dir + "/experts.bin"; return false; }
    // 8 tokens per prefill chunk, k picks each - one staging buffer serves decode and prefill.
    const int64_t stage_bytes = max_blob * g.dsv4.n_expert_used * 8;
    h_stage_ = (uint8_t*) std::malloc((size_t) stage_bytes);
    if (h_stage_ == nullptr) { err = "staging malloc"; return false; }
    if (cudaHostRegister(h_stage_, (size_t) stage_bytes, cudaHostRegisterMapped) != cudaSuccess)
        (void) cudaGetLastError();  // pageable staging is legal, just slower
    if (cudaMalloc((void**) &d_blobs_, (size_t) stage_bytes) != cudaSuccess) { err = "cudaMalloc blobs"; return false; }
    stage_dbg_ = std::getenv("STRATA_DSV4_STAGE_DEBUG") != nullptr;
    int64_t cache_blobs = 0;
    if (const char* e = std::getenv("STRATA_DSV4_EXPERT_CACHE_BLOBS")) cache_blobs = std::atoll(e);
    if (cache_blobs < 0) cache_blobs = 0;
    if (cache_blobs > 256) cache_blobs = 256;
    if (cache_blobs > 0) {
        const int64_t cache_bytes = max_blob * cache_blobs;
        h_cache_ = (uint8_t*) std::malloc((size_t) cache_bytes);
        if (h_cache_ != nullptr) {
            cache_slots_ = cache_blobs;
            cache_blob_bytes_ = max_blob;
        }
    }

    const int64_t dim = g.n_embd, hc = g.hc;
    cudaMalloc((void**) &d_a_, (size_t) hc * dim * 4);
    cudaMalloc((void**) &d_b_, (size_t) hc * dim * 4);
    cudaMalloc((void**) &d_mixes_, (size_t) 6 * hc * 4);
    cudaMalloc((void**) &d_pre_, (size_t) hc * 4);
    cudaMalloc((void**) &d_y_, (size_t) dim * 4);
    cudaMalloc((void**) &d_logits_, (size_t) g.dsv4.n_vocab * 4);
    const int64_t max_stage = max_seq / 4 + 2;
    int64_t scratch = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        scratch = std::max(scratch, dsv4_block_scratch_bytes(g, l, max_stage));
        scratch = std::max(scratch, dsv4_block_prefill_scratch_bytes(g, l, 8, max_stage));
    }
    cudaMalloc((void**) &d_scratch_, (size_t) scratch);
    cudaMalloc((void**) &d_pa_, (size_t) 8 * hc * dim * 4);
    cudaMalloc((void**) &d_pb_, (size_t) 8 * hc * dim * 4);
    cudaMalloc((void**) &d_ptok_, (size_t) 8 * sizeof(int64_t));
    cudaStream_t s = nullptr;
    if (cudaStreamCreate(&s) != cudaSuccess) { err = "cudaStreamCreate"; return false; }
    stream_ = (void*) s;

    // Last, after every small page-locked buffer is already allocated: registering a large mirror first has
    // left the driver unable to page-lock the small ones afterwards.  A mirror that cannot be made is not an
    // error - the file path stays, and the reason is printed.
    if (experts_ram_want_ != 0) {
        std::string arena_err;
        if (!load_experts_arena(arena_err))
            std::fprintf(stderr, "strata dsv4: the experts stay in the file: %s\n", arena_err.c_str());
    }
    return true;
}

void Dsv4Forward::set_hidden_sink(std::function<bool(const float*, int64_t, int64_t)> sink,
                                  std::vector<int64_t> target_layers) {
    sink_ = std::move(sink);
    const int64_t dim = g_.n_embd;
    cap_slot_.assign((size_t) g_.n_layers + 1, -1);
    for (size_t s = 0; s < target_layers.size(); ++s)
        if (target_layers[s] >= 0 && target_layers[s] <= g_.n_layers)
            cap_slot_[(size_t) target_layers[s]] = (int64_t) s;
    cudaFree(d_hcap_);
    cudaFree(d_hcap_tmp_);
    d_hcap_ = d_hcap_tmp_ = nullptr;
    if (target_layers.empty()) { cap_ = 0; return; }
    cap_ = (int64_t) target_layers.size();
    cudaMalloc((void**) &d_hcap_, (size_t) 8 * cap_ * dim * 4);
    cudaMalloc((void**) &d_hcap_tmp_, (size_t) 8 * dim * 4);
    cudaMalloc((void**) &d_logits_w_, (size_t) 8 * g_.dsv4.n_vocab * 4);
}

// The captured stream is the mean over the hc streams (llama.cpp's `build_hc_mean`); the fused row
// the drafter's `fc` reads is token-major [n][cap * n_embd], so each slot's [n][n_embd] block is
// interleaved into it with a strided copy.
void Dsv4Forward::capture(const float* in, int64_t n, int64_t pos0) {
    if (d_hcap_ == nullptr) return;
    const int64_t dim = g_.n_embd, hc = g_.hc;
    for (size_t l = 0; l < cap_slot_.size(); ++l) {
        const int64_t slot = cap_slot_[l];
        if (slot < 0) continue;
        kernels::hc_mean(in, n, hc, dim, d_hcap_tmp_, stream_);
        cudaMemcpy2DAsync(d_hcap_ + slot * dim, (size_t) cap_ * dim * 4, d_hcap_tmp_,
                          (size_t) dim * 4, (size_t) dim * 4, (size_t) n, cudaMemcpyDeviceToDevice,
                          (cudaStream_t) stream_);
    }
    (void) pos0;
}

bool Dsv4Forward::verify_window(const std::vector<int64_t>& tokens, int64_t pos0, std::string& err) {
    const int64_t n = (int64_t) tokens.size();
    if (n < 1 || n > 8) { err = "verify window: n must be 1..8"; return false; }
    const int64_t dim = g_.n_embd, hc = g_.hc;
    for (int64_t t = 0; t < n; ++t)
        for (int64_t h = 0; h < hc; ++h)
            embed_.gather_one(tokens[(size_t) t], d_pa_ + (t * hc + h) * dim, stream_);
    cudaMemcpyAsync(d_ptok_, tokens.data(), (size_t) n * sizeof(int64_t), cudaMemcpyHostToDevice,
                    (cudaStream_t) stream_);
    float* in = d_pa_;
    float* out = d_pb_;
    const int64_t n_stage = (pos0 + n - 1) / 4 + 2;
    const int64_t carry_floats = state_.carry_bytes() / 4;
    if (carry_floats > 0 && d_carry_snap_ == nullptr)
        cudaMalloc((void**) &d_carry_snap_, (size_t) 8 * carry_floats * 4);
    for (int64_t l = 0; l < g_.n_layers; ++l) {
        capture(in, n, pos0);
        stager_.off = layer_off_[(size_t) l];
        Dsv4BlockWeights w;
        if (!resolver_.resolve(g_, *wt_, l, layer_layout_[(size_t) l], nullptr, w, err)) return false;
        float* snap = d_carry_snap_ != nullptr
                          ? d_carry_snap_ + state_.carry_offset(l) / 4
                          : nullptr;
        if (!dsv4_block_prefill(g_, l, w, state_.layer(l), in, out, d_ptok_, pos0, n, n_stage,
                                d_scratch_, stream_, &stager_, snap, carry_floats)) {
            err = "verify window failed at layer " + std::to_string(l);
            return false;
        }
        std::swap(in, out);
    }
    capture(in, n, pos0);
    if (sink_ && !sink_(d_hcap_, pos0, n)) { err = "hidden sink rejected the window"; return false; }
    // Every token's logits: the accept loop compares each position against the target's own pick.
    for (int64_t t = 0; t < n; ++t) {
        const float* row = in + t * hc * dim;
        kernels::hc_mixes(row, hc_head_fn_, d_mixes_, 1, hc * dim, hc, g_.dsv4.hc_eps, stream_);
        kernels::hc_head_pre(d_mixes_, hc_head_scale_, hc_head_base_, d_pre_, 1, hc,
                             g_.dsv4.hc_eps, stream_);
        kernels::hc_pre_combine(row, d_pre_, d_y_, 1, hc, dim, stream_);
        kernels::rms_norm_weighted(d_y_, out_norm_, 1, dim, g_.dsv4.norm_eps, stream_);
        if (!head_.run(d_y_, d_logits_w_ + t * g_.dsv4.n_vocab, stream_, err)) return false;
    }
    return true;
}

bool Dsv4Forward::decode(int64_t token, int64_t pos, std::string& err) {
    const bool dbg = std::getenv("DSV4_DBG") != nullptr;
    auto nonfinite = [&](const float* d, int64_t n) -> std::pair<int64_t, float> {
        std::vector<float> h((size_t) n);
        cudaMemcpy(h.data(), d, (size_t) n * 4, cudaMemcpyDeviceToHost);
        int64_t bad = 0;
        float mx = 0;
        for (float v : h) { if (!std::isfinite(v)) ++bad; else mx = std::max(mx, std::fabs(v)); }
        return {bad, mx};
    };
    const int64_t dim = g_.n_embd, hc = g_.hc;
    for (int64_t h = 0; h < hc; ++h) embed_.gather_one(token, d_a_ + h * dim, stream_);
    if (dbg) {
        const auto [bad, mx] = nonfinite(d_a_, hc * dim);
        std::printf("[dbg] after embed: max %.3e nonfinite %lld\n", mx, (long long) bad);
    }
    float* in = d_a_;
    float* out = d_b_;
    const int64_t n_stage = pos / 4 + 2;
    for (int64_t l = 0; l < g_.n_layers; ++l) {
        capture(in, 1, pos);
        stager_.off = layer_off_[(size_t) l];
        Dsv4BlockWeights w;
        if (!resolver_.resolve(g_, *wt_, l, layer_layout_[(size_t) l], nullptr, w, err)) return false;
        if (!dsv4_block_decode_step(g_, l, w, state_.layer(l), in, out, pos, token, n_stage,
                                    d_scratch_, stream_, &stager_)) {
            err = "block step failed at layer " + std::to_string(l);
            return false;
        }
        std::swap(in, out);
        if (dbg) {
            const auto [bad, mx] = nonfinite(in, hc * dim);
            std::printf("[dbg] after layer %lld: max %.3e nonfinite %lld\n", (long long) l, mx, (long long) bad);
            if (bad != 0) break;
        }
    }
    capture(in, 1, pos);
    if (sink_ && !sink_(d_hcap_, pos, 1)) { err = "hidden sink rejected the token"; return false; }
    // ---- the head's collapse: hc_mixes, sigmoid pre, combine, output_norm, the native head.
    kernels::hc_mixes(in, hc_head_fn_, d_mixes_, 1, hc * dim, hc, g_.dsv4.hc_eps, stream_);
    kernels::hc_head_pre(d_mixes_, hc_head_scale_, hc_head_base_, d_pre_, 1, hc, g_.dsv4.hc_eps, stream_);
    kernels::hc_pre_combine(in, d_pre_, d_y_, 1, hc, dim, stream_);
    kernels::rms_norm_weighted(d_y_, out_norm_, 1, dim, g_.dsv4.norm_eps, stream_);
    if (dbg) std::printf("[dbg] y: max %.3e nonfinite %lld\n", (double) nonfinite(d_y_, dim).second, (long long) nonfinite(d_y_, dim).first);
    if (!head_.run(d_y_, d_logits_, stream_, err)) return false;
    if (dbg) std::printf("[dbg] logits nonfinite %lld\n", (long long) nonfinite(d_logits_, g_.dsv4.n_vocab).first);
    return true;
}

bool Dsv4Forward::prefill(const std::vector<int64_t>& tokens, std::string& err) {
    const int64_t dim = g_.n_embd, hc = g_.hc;
    for (size_t c0 = 0; c0 < tokens.size(); c0 += 8) {
        const int64_t n = std::min<int64_t>(8, (int64_t) tokens.size() - (int64_t) c0);
        const int64_t pos0 = (int64_t) c0;
        for (int64_t t = 0; t < n; ++t)
            for (int64_t h = 0; h < hc; ++h)
                embed_.gather_one(tokens[c0 + t], d_pa_ + (t * hc + h) * dim, stream_);
        cudaMemcpyAsync(d_ptok_, tokens.data() + c0, (size_t) n * sizeof(int64_t),
                        cudaMemcpyHostToDevice, (cudaStream_t) stream_);
        float* in = d_pa_;
        float* out = d_pb_;
        const int64_t n_stage = (pos0 + n - 1) / 4 + 2;
        for (int64_t l = 0; l < g_.n_layers; ++l) {
            capture(in, n, pos0);
            stager_.off = layer_off_[(size_t) l];
            Dsv4BlockWeights w;
            if (!resolver_.resolve(g_, *wt_, l, layer_layout_[(size_t) l], nullptr, w, err))
                return false;
            if (!dsv4_block_prefill(g_, l, w, state_.layer(l), in, out, d_ptok_, pos0, n, n_stage,
                                    d_scratch_, stream_, &stager_)) {
                err = "prefill failed at layer " + std::to_string(l);
                return false;
            }
            std::swap(in, out);
        }
        capture(in, n, pos0);
        if (sink_ && !sink_(d_hcap_, pos0, n)) { err = "hidden sink rejected the chunk"; return false; }
        // the head's collapse on the last token of the whole prompt - that is the one sampled.
        if (c0 + (size_t) n == tokens.size()) {
            const float* last = in + (n - 1) * hc * dim;
            kernels::hc_mixes(last, hc_head_fn_, d_mixes_, 1, hc * dim, hc, g_.dsv4.hc_eps, stream_);
            kernels::hc_head_pre(d_mixes_, hc_head_scale_, hc_head_base_, d_pre_, 1, hc,
                                 g_.dsv4.hc_eps, stream_);
            kernels::hc_pre_combine(last, d_pre_, d_y_, 1, hc, dim, stream_);
            kernels::rms_norm_weighted(d_y_, out_norm_, 1, dim, g_.dsv4.norm_eps, stream_);
            if (!head_.run(d_y_, d_logits_, stream_, err)) return false;
        }
    }
    return true;
}

int64_t Dsv4Forward::reserve_cache(int64_t key) {
    if (cache_slots_ <= 0 || h_cache_ == nullptr) return -1;
    auto it = cache_slot_.find(key);
    if (it != cache_slot_.end()) return it->second;
    int64_t slot = -1;
    if ((int64_t) cache_slot_.size() < cache_slots_) {
        slot = (int64_t) cache_slot_.size();
    } else {
        const int64_t victim = cache_lru_.back();
        auto vit = cache_slot_.find(victim);
        slot = vit == cache_slot_.end() ? 0 : vit->second;
        cache_lru_.pop_back();
        cache_slot_.erase(victim);
    }
    cache_slot_[key] = slot;
    cache_lru_.push_front(key);
    return slot;
}

void Dsv4Forward::touch_cache(int64_t key) {
    auto it = std::find(cache_lru_.begin(), cache_lru_.end(), key);
    if (it == cache_lru_.end()) return;
    cache_lru_.erase(it);
    cache_lru_.push_front(key);
}

// The host mirror of experts.bin (see Dsv4Forward::set_experts_ram).  A picked blob is 7.2-9.4 MiB read out of
// the file at 2.7 GiB/s; the same bytes from page-locked RAM reach the GPU at 22 GiB/s (docs/DSV4.md, "why the
// decode is slow").  The driver's page-lock limit is smaller than the file here (it refused 64 GiB and took 48
// GiB of the 78.11 GiB), so only the prefix it takes is copied by DMA and the rest is copied by the CPU into
// the staging buffer.
bool Dsv4Forward::load_experts_arena(std::string& err) {
    int64_t file_bytes = 0;
#ifdef _WIN32
    if (_fseeki64(experts_file_, 0, SEEK_END) != 0) { err = "cannot size experts.bin"; return false; }
    file_bytes = (int64_t) _ftelli64(experts_file_);
#else
    if (fseeko(experts_file_, 0, SEEK_END) != 0) { err = "cannot size experts.bin"; return false; }
    file_bytes = (int64_t) ftello(experts_file_);
#endif
    if (file_bytes <= 0) { err = "experts.bin is empty"; return false; }

    // The largest offset at or below `w` that no expert blob straddles: a blob that starts inside the
    // page-locked range and runs past it is taken as page-locked by cudaMemcpyAsync and refused.
    auto boundary = [&](uint64_t w) {
        for (size_t l = 0; l < layer_off_.size(); ++l) {
            const int64_t a = layer_off_[l];
            const int64_t b = l + 1 < layer_off_.size() ? layer_off_[l + 1] : file_bytes;
            if ((int64_t) w < b) {
                const int64_t blob = (b - a) / g_.n_expert;
                return (uint64_t) (a + ((int64_t) w - a) / blob * blob);
            }
        }
        return w;
    };

    // An explicit request is honoured as asked; "what fits" is the free RAM minus the same 8 GiB headroom the
    // Qwen path's resident mode keeps.  Either way the mirror is a prefix of the file, cut at a blob boundary.
    const uint64_t avail = platform::available_physical_memory();
    uint64_t want = experts_ram_want_ == kExpertsRamWhatFits
                        ? (avail > (8ull << 30) ? avail - (8ull << 30) : 0)
                        : experts_ram_want_;
    want = boundary(std::min<uint64_t>(want, (uint64_t) file_bytes));
    if (want == 0) { err = "no room in the free RAM for a mirror of experts.bin"; return false; }
    if (experts_ram_want_ == kExpertsRamWhatFits && want < (uint64_t) file_bytes)
        std::fprintf(stderr, "strata dsv4: the free RAM covers %.2f of the file's %.2f GiB; the rest is read "
                             "from the file\n", (double) want / 1073741824.0, (double) file_bytes / 1073741824.0);
    const uint64_t asked = want;

    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        experts_arena_ = (uint8_t*) std::malloc((size_t) want);
        if (experts_arena_ != nullptr) break;
        if (want > (8ull << 30)) {  // an explicit request degrades in 8 GiB steps before giving up
            const uint64_t less = boundary(want - (8ull << 30));
            if (less == 0 || less >= want) break;
            want = less;
            continue;
        }
        err = "no memory for a mirror of experts.bin";
        return false;
    }
    if (want < asked)
        std::fprintf(stderr, "strata dsv4: the machine would not commit %.2f GiB; the mirror is %.2f GiB\n",
                     (double) asked / 1073741824.0, (double) want / 1073741824.0);
    auto release_arena = [&]() {
        std::free(experts_arena_);
        experts_arena_ = nullptr;
        experts_arena_bytes_ = 0;
        experts_locked_bytes_ = 0;
    };
    experts_arena_bytes_ = want;
    std::fprintf(stderr, "strata dsv4: reading %.2f GiB of the experts into RAM...\n",
                 (double) want / 1073741824.0);
    std::fflush(stderr);

    // One sequential pass over the file (1.3 GiB/s here; the per-token path's random 7 MiB reads are the slow
    // way to read this file).  Rewind first: the size probe above left the position at the end.  experts_file_
    // is left wherever this stopped: every stage() call seeks first.
#ifdef _WIN32
    if (_fseeki64(experts_file_, 0, SEEK_SET) != 0) { err = "cannot rewind experts.bin"; release_arena(); return false; }
#else
    if (fseeko(experts_file_, 0, SEEK_SET) != 0) { err = "cannot rewind experts.bin"; release_arena(); return false; }
#endif
    std::vector<uint8_t> chunk_buf(1u << 22);
    for (uint64_t got = 0; got < want;) {
        const uint64_t n = std::min<uint64_t>(chunk_buf.size(), want - got);
        if (std::fread(chunk_buf.data(), 1, (size_t) n, experts_file_) != (size_t) n) {
            err = "reading experts.bin into the mirror";
            release_arena();
            return false;
        }
        std::memcpy(experts_arena_ + got, chunk_buf.data(), (size_t) n);
        got += n;
        if (got % (8ull << 30) < n || got == want) {
            const double el = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                                  .count();
            std::fprintf(stderr, "strata dsv4:   %.2f of %.2f GiB (%.2f GiB/s)\n", (double) got / 1073741824.0,
                         (double) want / 1073741824.0, (double) got / 1073741824.0 / (el / 1000.0));
            std::fflush(stderr);
        }
    }

    // Page-lock as much of the mirror as the driver takes: the whole thing if it will, otherwise 2 GiB steps
    // down, each cut at a blob boundary.
    if (cudaHostRegister(experts_arena_, (size_t) want, cudaHostRegisterMapped | cudaHostRegisterPortable) ==
        cudaSuccess) {
        experts_locked_bytes_ = want;
    } else {
        (void) cudaGetLastError();
        const uint64_t step = 2ull << 30;
        for (uint64_t ask = want > step ? want - step : 0; ask >= step; ask -= step) {
            const uint64_t w = boundary(ask);
            if (w == 0) break;
            if (cudaHostRegister(experts_arena_, (size_t) w, cudaHostRegisterMapped | cudaHostRegisterPortable) ==
                cudaSuccess) {
                experts_locked_bytes_ = w;
                break;
            }
            (void) cudaGetLastError();
            if (ask < step) break;
        }
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "strata dsv4: experts in RAM: %.2f GiB mirrored in %.1f s, %.2f GiB page-locked "
                         "(the rest copied by the CPU)\n",
                 (double) experts_arena_bytes_ / 1073741824.0, ms / 1000.0,
                 (double) experts_locked_bytes_ / 1073741824.0);
    std::fflush(stderr);
    return true;
}

bool Dsv4Forward::Stager::stage(const int32_t* ids, int64_t n, int64_t bytes, const uint8_t** out) {
    const auto t0 = std::chrono::steady_clock::now();
    int64_t file_reads = 0, from_mirror = 0;
    const bool allow_dedup = std::getenv("STRATA_DSV4_STAGE_DEDUP") == nullptr;
    const bool allow_cache = f->h_cache_ != nullptr && std::getenv("STRATA_DSV4_EXPERT_CACHE") == nullptr;
    const bool dbg_cache = allow_cache && std::getenv("STRATA_DSV4_CACHE_DEBUG") != nullptr;
    if (dbg_cache)
        std::fprintf(stderr, "[cache] stage n=%lld bytes=%lld slots=%lld blob=%lld\n", (long long) n,
                     (long long) bytes, (long long) f->cache_slots_, (long long) f->cache_blob_bytes_);
    // Residency (see Dsv4Forward::set_experts_ram): a blob inside the host mirror never touches the file.  The
    // page-locked part is copied straight into its device slot by DMA; the rest is copied into the host staging
    // buffer first, because the copy engine cannot read an unregistered region.  Host-filled slots are moved in
    // runs, and a run ends where a direct copy is issued, so the stream order matches the slot order.  With no
    // mirror the only run is [0, n) - the single bulk copy this path has always done.
    const uint8_t* const mirror = f->experts_arena_;
    const uint64_t mirrored = f->experts_arena_bytes_;
    const uint64_t locked = f->experts_locked_bytes_;
    if (served.size() < (size_t) n) served.assign((size_t) n, 0);
    int64_t run = -1;  // the host-filled slots [run, i) not yet copied
    auto flush = [&](int64_t upto) {
        if (run < 0 || upto <= run) return;
        cudaMemcpyAsync(f->d_blobs_ + (size_t) run * bytes, f->h_stage_ + (size_t) run * bytes,
                        (size_t) (upto - run) * bytes, cudaMemcpyHostToDevice, (cudaStream_t) f->stream_);
        run = -1;
    };
    for (int64_t i = 0; i < n; ++i) {
        served[i] = 0;
        int64_t dup = -1;
        if (allow_dedup)
            for (int64_t j = 0; j < i; ++j)
                if (ids[j] == ids[i]) { dup = j; break; }
        if (dup >= 0) {
            if (served[(size_t) dup]) {  // that blob is already in a device slot: copy it there
                flush(i);
                cudaMemcpyAsync(f->d_blobs_ + (size_t) i * bytes, f->d_blobs_ + (size_t) dup * bytes,
                                (size_t) bytes, cudaMemcpyDeviceToDevice, (cudaStream_t) f->stream_);
                served[i] = 1;
            } else {
                std::memcpy(f->h_stage_ + (size_t) i * bytes, f->h_stage_ + (size_t) dup * bytes,
                            (size_t) bytes);
                if (run < 0) run = i;
            }
            continue;
        }
        const int64_t at = off + (int64_t) ids[i] * bytes;
        if ((uint64_t) at + (uint64_t) bytes <= locked) {
            flush(i);
            cudaMemcpyAsync(f->d_blobs_ + (size_t) i * bytes, mirror + at, (size_t) bytes,
                            cudaMemcpyHostToDevice, (cudaStream_t) f->stream_);
            served[i] = 1;
            ++from_mirror;
            continue;
        }
        if ((uint64_t) at + (uint64_t) bytes <= mirrored) {
            std::memcpy(f->h_stage_ + (size_t) i * bytes, mirror + at, (size_t) bytes);
            if (run < 0) run = i;
            ++from_mirror;
            continue;
        }
        bool cached = false;
        if (allow_cache) {
            auto it = f->cache_slot_.find(at);
            if (it != f->cache_slot_.end()) {
                if (dbg_cache)
                    std::fprintf(stderr, "[cache] hit key=%lld slot=%lld\n", (long long) at,
                                 (long long) it->second);
                std::memcpy(f->h_stage_ + (size_t) i * bytes,
                            f->h_cache_ + (size_t) it->second * f->cache_blob_bytes_, (size_t) bytes);
                f->touch_cache(at);
                cached = true;
            }
        }
        if (!cached) {
            ++file_reads;
#ifdef _WIN32
            if (_fseeki64(f->experts_file_, (__int64) at, SEEK_SET) != 0) return false;
#else
            if (fseeko(f->experts_file_, (off_t) at, SEEK_SET) != 0) return false;
#endif
            if (std::fread(f->h_stage_ + (size_t) i * bytes, 1, (size_t) bytes, f->experts_file_) !=
                (size_t) bytes)
                return false;
            if (allow_cache) {
                const int64_t slot = f->reserve_cache(at);
                if (dbg_cache)
                    std::fprintf(stderr, "[cache] miss key=%lld slot=%lld\n", (long long) at,
                                 (long long) slot);
                if (slot >= 0)
                    std::memcpy(f->h_cache_ + (size_t) slot * f->cache_blob_bytes_,
                                f->h_stage_ + (size_t) i * bytes, (size_t) bytes);
            }
        }
        if (run < 0) run = i;  // this slot was filled in h_stage_ (cache hit or file read)
    }
    flush(n);
    cudaStreamSynchronize((cudaStream_t) f->stream_);  // the slots are reused by the next call
    *out = f->d_blobs_;
    if (f->stage_dbg_) {
        f->stage_stats_.ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ++f->stage_stats_.calls;
        f->stage_stats_.reads += file_reads;
        f->stage_stats_.read_bytes += file_reads * bytes;
        f->stage_stats_.resident += from_mirror;
        f->stage_stats_.resident_bytes += from_mirror * bytes;
        f->stage_stats_.copied += n * bytes;
    }
    return true;
}

}  // namespace strata::core
