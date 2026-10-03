// include/strata/core/dsv4_weights.hpp - the production resolver that fills `Dsv4BlockWeights`
// from a loaded pack, docs/DSV4.md P2 integration.
//
// Most of a DSv4 layer's weights are already device pointers: the resident ones (norms, mHC,
// the raw-BF16 router) sit in the `WeightTable` arena, and `NativeDense` has attached the
// quantized GEMV blobs (Q5_K/Q8_0/Q6_K) from the GGUF shards onto the same refs.  Two things
// are NOT, and this resolver owns them:
//
//   * `indexer.proj.weight` - the packer left it in dense.bin as F32 (the one GEMV the GGUF
//     does not serve quantized); the kernels want bf16, so it is converted once at load.
//   * `ffn_gate_tid2eid.weight` - an I32 lookup table the pack skips and `NativeDense` does
//     not serve (it is not a GEMV); it is read from the GGUF shard and copied to the device.
//
// The routed experts are not the resolver's either: 78 GiB does not live in VRAM, so the
// caller (the session's expert source) provides `expert_blobs` + the layout per call, exactly
// as `moe_layer` gets its experts today.
#pragma once

#include "strata/core/dsv4_block.hpp"
#include "strata/core/weights.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace strata::core {

class Dsv4WeightResolver {
public:
    Dsv4WeightResolver() = default;
    ~Dsv4WeightResolver();
    Dsv4WeightResolver(const Dsv4WeightResolver&) = delete;
    Dsv4WeightResolver& operator=(const Dsv4WeightResolver&) = delete;

    /// Convert every indexed layer's `indexer.proj` to bf16 and load every hash layer's
    /// `tid2eid` from the model's GGUF shards.  `shards` are the split paths of any shard.
    bool init(const ModelGeometry& g, const WeightTable& tables, const std::vector<std::string>& shards,
              std::string& err);

    /// Fill `w` for `layer`.  `expert_blobs` (n_expert contiguous blobs, `experts` layout) is
    /// caller-owned; pass null only for a resolve that will not run the FFN.
    bool resolve(const ModelGeometry& g, const WeightTable& tables, int64_t layer,
                 const kernels::NativeExpertLayout& experts, const uint8_t* expert_blobs,
                 Dsv4BlockWeights& w, std::string& err) const;

private:
    std::map<int64_t, uint16_t*> idx_proj_;   // device, resolver-owned
    std::map<int64_t, int32_t*> tid2eid_;     // device, resolver-owned
};

}  // namespace strata::core
