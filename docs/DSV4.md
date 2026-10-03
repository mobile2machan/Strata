# DeepSeek-V4-Flash: support plan

Status: **P0 done** (2026-10-03, artifact inspection), **P1 done** (the real UD-IQ2_XXS pack is built -
78.11 GiB of experts - and verified end to end) and **P2 loading done** (the geometry reader, the shape
checks, the native dense path and the expert kernels all pass against the real artifact: `dsv4_layout_real`
attaches 489 served matrices, 5.71 GiB, and accepts all 43 layers' expert pairs). The engine can now read
every weight of a deepseek4 pack; what it cannot do yet is compute a token - none of the deepseek4 math
exists.
Everything marked *measured* below came from the artifact
itself; speed figures are *estimates* with their reasoning stated, per the docs rule.

Why this model: it is the first candidate whose shape matches this engine - linear-window attention with a
Lightning indexer (the QSA family), a 4-stream residual (the gated-residual plumbing), a 256-expert MoE
(the expert streaming), and an official drafter for speculative decoding (the MTP/spec controller). It is
also the model no engine on the target machines can actually run: FreeToken needs the FP8/NVFP4 checkpoints
(284 GB / ~170 GB), and llama.cpp loads the GGUFs but offloads almost all of a 91-104 GB model past 16-24 GB
of VRAM.

## The model (measured from GGUF metadata, arch tag `deepseek4`)

| | |
|---|---|
| Layers | 43, hidden 4096, 64 heads, one latent KV (`head_count_kv = 1`, `key_length = 512`, rope 64 dims) |
| Attention | per-layer `compress_ratios` over the 43 layers: 2 window-only (layers 0, 1), 21 at 4:1 (with a Lightning indexer), 20 at 128:1; `sliding_window = 128`; `attn_sinks` (64 per layer) are attention sinks; output projection is grouped low-rank (`output_group_count = 8`, `output_lora_rank = 1024`) |
| Compressor | learned gated pooling of KV over `ratio` tokens with per-position APE; the 4:1 compressor is *overlapping* (its APE is `(1024, 4)` = 2x head_dim wide, matching `compress.py`'s `overlap = ratio == 4`); the indexer selects among compressed entries, `indexer.top_k = 512`, 64 heads x 128 |
| Residual | mHC: `hyper_connection.count = 4`, `sinkhorn_iterations = 20`, per-layer `hc_attn_*`/`hc_ffn_*` tensors, plus `output_hc_*` at the head |
| MoE | 256 experts, top-6 + 1 shared, `expert_feed_forward_length = 2048`, `expert_gating_func = 4` (sqrtsoftplus), `expert_weights_scale = 1.5`, `exp_probs_b` correction bias, `swiglu_clamp_exp/shexp = 10.0` on all 43 layers |
| Hash layers | `hash_layer_count = 3`: layers 0-2 additionally carry `ffn_gate_tid2eid` (I32, `(6, 129280)`) - a static token-id -> 6-expert-ids table. Measured: those layers keep their router tensors too; which path the engine uses is a P2 decision |
| RoPE | YaRN (`factor = 16`, `original_context_length = 65536`, betas 32/1), `compress_rope_freq_base = 160000` for compressed entries |
| Context | 1,048,576 |
| Tokenizer | `tokenizer.ggml.model = gpt2` (byte-level BPE, same machinery as Qwen here), `tokenizer.ggml.pre = joyai-llm` (a new pre-tokenizer regex + special tokens only) |

The `compress_ratios` array has 46 entries, not 43: indices 43-45 are three extra zeros. Their purpose is
unknown (the drafter is a separate file); the packer must carry the array verbatim.

## The artifact (measured)

`general.architecture = deepseek4`. Unsloth's Shard_Rewrite layout: **shard 1 is metadata-only (5 MB, zero
tensors)**; tensors live in shards 2+. Each shard's tensor-info section lists only its own tensors, so the
name map below (shard 2, layers 0-23) is the llama.cpp naming contract; later shards repeat it.

UD-IQ2_XXS quant split (measured across all three shards - the UD recipe **mixes types per layer**):

| tensors | type |
|---|---|
| `blk.N.ffn_gate_exps/up_exps` (4096x2048x256) | IQ2_XXS or IQ2_S, per layer |
| `blk.N.ffn_down_exps` | IQ3_XXS or **MXFP4**, per layer - three (gu, d) combinations measured: (IQ2_XXS, IQ3_XXS), (IQ2_XXS, MXFP4), (IQ2_S, MXFP4); expert blobs are 7.5-9.8 MB |
| `blk.N.ffn_gate_shexp` (Q5_K), `ffn_up/down_shexp` (Q5_K/Q6_K) | shared expert, K-quants |
| `blk.N.ffn_gate_inp` | BF16 router (all 43 layers); `exp_probs_b` F32 on the 40 routed layers; `ffn_gate_tid2eid` I32 on the 3 hash layers |
| `attn_q_a` Q5_K, `attn_q_b/attn_kv/attn_output_a/attn_output_b` Q8_0, norms/sinks/ape F32 | attention |
| `attn_compressor_gate/kv`, `indexer_compressor_gate/kv` Q8_0; `*_ape` F32 | compressors |
| `hc_attn/ffn_base/fn/scale`, `output_hc_*` | F32 |
| `token_embd`, `output` | Q4_K (kept high) |

Measured file sizes: **UD-IQ2_XXS = 90.9 GB** (3 shards; layers 0-23 in shard 2, 24-42 in shard 3),
**UD-IQ3_XXS = 104.2 GB** (4 shards), UD-Q4_K_XL = 155.2 GB (5 shards). The built pack's `experts.bin` is
**83.9 GB (78.11 GiB)** - 92% of the file - which is exactly the part this engine keeps in pinned RAM and
streams.

## The drafter: DSpark, not MTP (measured)

The 0731 checkpoint has **no MTP head** - asking llama.cpp for one logs `MTP requested but this GGUF has no
MTP head`. Speculative decoding is a separate sidecar: `dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf` (10.90 GB,
arch `dflash`, 81 tensors; a BF16 build is 11.31 GB). Measured from its header: one block, same hidden size,
its own 64-head attention with `sliding_window = 128` and `sliding_window_pattern = "SSSSSSL"`, its own
256-expert MoE with **MXFP4 experts** (native FP4, never requantized) + Q8_0 shared, mHC tensors of the same
shape as the target, `eh_proj` (4096x8192) fusing target hidden state + token embedding, and a **Markov head
+ confidence head** (64-dim) for drafting and early stopping. It ships no embeddings or output head and
borrows the target's. Trained block size is 5; llama.cpp's own measurements (unsloth's, 1x B200) give
**1.91x at draft depth 3** (0.764 acceptance), 1.37x at depth 1.

For this engine the drafter matters more than it does for llama.cpp: on a CPU-GPU split run the target's
bottleneck is RAM bandwidth, and a drafter that accepts k of n tokens removes whole target forwards - i.e.
whole passes over the expert bytes. The drafter's own experts (MXFP4, ~8 GB) should go through the same
streaming path rather than sit in VRAM.

## Target machines (the two this fork runs)

| machine | default quant | fit |
|---|---|---|
| Windows, 137 GB RAM, 24 GB NVIDIA | UD-IQ3_XXS (104.2 GB) | experts ~101 GB pinned + KV + OS ~ 115-120 GB; ~17 GB headroom |
| Linux, 128 GB RAM, RX 9070 XT (gfx1201), Ryzen 7 5700X | UD-IQ2_XXS (90.9 GB) | experts 83.9 GB (measured `experts.bin`) pinned ~ 95-100 GB; ~28 GB headroom. gfx1201 is a validated Strata backend (`docs/AMD_HIP.md`) |

Speed estimates (not measurements - to be replaced by real numbers at P4): the 5700X box has DDR4-3200
(~42 GB/s effective), and at ~10% expert-cache hit the CPU GEMV path moves ~1.7 GB of expert bytes per
token at IQ2_XXS -> ~15-22 tok/s, ~25-35 with the drafter at depth 3. The 24 GB NVIDIA box (DDR5) should
land ~25-50 tok/s at IQ3_XXS before the drafter. The same boxes under llama.cpp offload most of the model
and are expected to sit at 3-8 tok/s. These are the claim to beat, not a claim.

## Implementation phases

- **P1 - pack.** *Done.* `tools/iq_pack.py` reads `general.architecture = deepseek4` and packs it:
  experts stay native (the per-layer `gu_type/d_type` columns of `native_experts.txt` already carry mixed
  types), dense floats are written as stored (no FORM conversions), quantized projections and the I32 hash
  tables are served natively, and `model.json` records the geometry for P2. `tools/test_dsv4_pack.py` builds
  a two-shard deepseek4 fixture (metadata-only shard 1, a hash layer beside a routed layer, IQ2_XXS/IQ3_XXS
  experts) and checks the arena is the source bytes relaid per expert. The real UD-IQ2_XXS pack is built
  (43 layers, `experts.bin` 78.11 GiB, 1199 index rows of which 494 are served natively) and verified
  end to end by `dsv4_layout_real` below.
- **P2 - engine.** *First step done:* the geometry reader and the shape checks. `include/strata/artifact/
  dsv4_geometry.hpp` fills a `ModelGeometry` (`arch = DeepSeek4`, a `Dsv4` block for the numbers Qwen does
  not have) from the artifact's own `deepseek4.*` metadata, refusing a missing or implausible key by name;
  `check_layer`/`check_all` in `src/core/layout.cpp` gained the deepseek4 tables - every shape measured from
  the UD-IQ2_XXS tensor directory, including the two compressor widths (2x key for r=4, key for r=128), the
  mHC widths (6*hc, hc-1), the per-class presence rules (indexer only on r=4 layers) and the hash layers'
  `tid2eid` beside a missing `exp_probs_b`. `tests/core/dsv4_layout_test.cpp` reads a synthetic GGUF and
  checks a synthetic pack covering all four fixture classes, plus the five named negatives; its `--real
  PACK SHARD1` mode (`dsv4_layout_real`, registered when `STRATA_DSV4_PACK`/`STRATA_DSV4_SHARD1` point at
   one) reads the real metadata shard - 43 layers, 2 window / 21 r=4 / 20 r=128, 3 hash - and passes
   `check_all` over the real pack's own index.txt. *Load wiring done:* `check_architecture` now accepts
   `deepseek4` (presence check only - the reader above is the real gate), `block_geometry` gained MXFP4
   (32 elements / 17 bytes, confirmed from the artifact's own blob arithmetic: IQ2_S gate/up + MXFP4 down =
   9,830,400 B/expert) and the raw integer types (the I32 `tid2eid`), and `NativeDense::eligible()` serves
   the deepseek4 projections. `dsv4_layout_real` then runs the real path end to end: `served_names` covers
   489 of the pack's 494 native rows - the five left out are exactly the three `tid2eid` tables and the
    head/embedding (NativeHead's) - and `NativeDense::load` attaches all 489 (5.71 GiB). *Expert kernels
    done:* `STRATA_D_FMTS` gained IQ3_XXS and MXFP4 - the real file's three (gate/up, down) combinations are
    (IQ2_XXS, IQ3_XXS) on 41 layers, (IQ2_XXS, MXFP4) and (IQ2_S, MXFP4) on two each, and a format missing
    from the down list refused 41 of 43 layers. MXFP4's dot (`vec_dot_mxfp4_q8_1`) reads the e2m1 nibbles
    through `get_int_from_table_16` and the DOUBLED `kvalues_mxfp4` table, so the dp4a chain is exact integer
    arithmetic and the halving folds into the E8M0 scale (`2^(e-128)`, equal to ggml-cpu's
    `GGML_E8M0_TO_FP32_HALF`); `dq_mxfp4` dequantizes bit-exact against ggml's `to_float`. The CPU oracle
    needed nothing: the vendored ggml-cpu already has `ggml_vec_dot_mxfp4_q8_0`. Three new
    `native_expert_parity` pairs cover the real combinations (the IQ3_XXS pair runs at FF=512 - its 256-block
    does not divide the Qwen fixture's 640), and `dsv4_layout_real` asserts `native_expert_supported` for all
     43 real layers. *Compressor math started:* `include/strata/kernels/compressor.hpp` + `
     src/kernels/cuda/compressor.cu` pool a window by a softmax over the ROW axis per column (FreeToken's
     `gated_pool`), with the r=4 overlap window = previous block's first-half columns + current block's
     second half, the APE added at every carry write, and the register roll on completion - prefill, carry
     seed and decode step. `compressor_parity` checks all three real shapes (r=4 d=512, r=128 d=512,
     indexer r=4 d=128) against a float64 reference of `compress.py` (prefill rel ~7e-8, carry and decode
     exact) and asserts the three wrong readings - row-axis softmax, swapped overlap halves, -inf rows
      zeroed instead of masked - are observably apart. *Attention gather done:* `dsv4_attn.hpp/.cu` is the
      decode-side read over the tiered pools - one block per head, gather window rows then compressed rows
      (a row is BOTH key and value; -1 ids are skipped, not read), online softmax, and the attention SINK
      as a null key: logit `sinks[h]`, zero value, joining the denominator once after the real rows
      (`sparse_attn.py`'s own semantics). Pool rows are bf16 already rope-applied and quant-dequant-rounded
      (`act_quant_fp8_inplace` is a round-trip, so the pool stores e4m3-grid values in bf16 - packing them
      is a later memory win, not a correctness need). `dsv4_attn_parity` (full window+cmp, window-only, and
      sink-null cases) matches a float64 reference at ~3e-7 and asserts no-sink, -1-reads-row-0 and
      dropped-scale observably apart. *Indexer scoring done:* `dsv4_indexer.hpp/.cu` computes the
      Lightning-Indexer logit per block - `sum_h relu(q_h . k_t) * weights[h]`, one shared compressed
      key across the 64 index heads, the RELU part of the definition (anti-correlated heads drop out,
      they do not cancel), the `softmax_scale * n_heads ** -0.5` fold living in `weights`, and -inf past
      the live block count so the top-k only sees real keys. `dsv4_indexer_parity` (real shape 64x128,
      staged and fully-live cases) matches the `indexer.py` equivalence at ~1e-7 and asserts no-relu,
      flat-weights and dropped-valid observably apart (the last only where the fixture can see it).
      *Pool round-trips done:* `dsv4_quant.hpp/.cu` - the pools store plain bf16 whose VALUES sit on a
      reduced grid, because `act_quant_fp8_inplace` / `fp4_act_quant_inplace` are quant+dequant round-trips
      at a power-of-two scale (`s = 2**ceil(log2(amax/max))`, the reference's float bit-trick, floors
      1e-4 / 6*2**-126): e4m3 at block 64 for window/compressed rows, e2m1 at block 32 for indexer q and
      keys with the hardware tie table (0.75->1, 1.25->1, 1.75->2, 2.5->2, 3.5->4, 5.0->4); plus the
      indexer's normalized Sylvester Hadamard (`WHT * d**-0.5`) that spreads the energy before the fp4
      grid. `dsv4_quant_parity` asserts the round-trips BIT-EXACT against an enumerated e4m3 grid and the
      `_round_fp4` chain (grid-times-pow2 values are exact in bf16, so any difference is real), checks the
      tie points directly, and flags linear-scale and unnormalized-WHT. *mHC mixing done:* `dsv4_hc.hpp/.cu`
      - the stream is `hc`=4 parallel copies of the hidden state; per sublayer `hc_split_sinkhorn` splits
      the mix vector (`(2+hc)*hc`) into `pre` (sigmoid+eps), `post` (2*sigmoid) and a doubly-stochastic
      `comb` (row-softmax, +eps, one column normalization, then (iters-1)=19 rounds of row-then-column,
      each dividing by the sum+eps - the measured config is iters=20 eps=1e-6); `hc_pre_combine` collapses
      the streams (`sum_h pre[h]*x[h]`) and `hc_post_combine` re-expands them
      (`post[q]*a + sum_p comb[p][q]*res[p]` - the reduction is over comb's FIRST axis). The mixes
      projection itself is the existing `bf16_gemv` plus the stream RMS factor. `dsv4_hc_parity` matches
      `sinkhorn.py`/`hc.py` in float64 at ~1e-7 and asserts comb-axis-swap, iters=1, half-post and
      uniform-pre observably apart - and records that at the real iters=20 the FIRST normalization pass
      (row- vs column-first) is numerically irrelevant (~1e-14, converged either way), so only the
      iteration count matters. *Router and SwiGLU done:* the metadata the router needs is now read by
      `dsv4_geometry` (`expert_gating_func` must be 4 = sqrtsoftplus, `expert_weights_scale` 1.5,
      `expert_weights_norm` true, per-layer `swiglu_clamp_exp`/`shexp` arrays of 10.0,
      `hyper_connection.epsilon` 1e-6), and `dsv4_moe.hpp/.cu` computes the rest:
      `dsv4_router_score` (sqrtsoftplus scores; the top-k weights are gathered from the PRE-bias scores,
      the top-k is over scores+`exp_probs_b`, then renormalize and *1.5 - the bias order is the reference's),
      `dsv4_router_hash` (layers 0-2 read their six indices from the `tid2eid` I32 table, contiguous per
      token in GGML [used, vocab] order, and gather the same pre-bias scores), and `dsv4_swiglu`
      (`silu(min(gate,limit)) * clamp(up,-limit,limit)` in fp32, the layer's own limit). The gate GEMV
      itself is the existing `bf16_gemv`. `dsv4_moe_parity` matches `moe.py`/`swiglu.py` in float64 at
      ~1e-6 with exact index agreement, and asserts dropping the bias, skipping the renormalization,
      softmax-instead-of-sqrtsoftplus, gathering from the post-bias scores, and an unclamped swiglu all
      observably apart. *Rope and pool state done:* `dsv4_rope.hpp/.cu` rotates the LAST 64 dims in
      INTERLEAVED pairs (not the qwen path's neox pairing) with the reference's YaRN verbatim - the
      frequency and the angle are kept in double because `pos * f` reaches 1e5 rad at long context,
      where fp32 argument reduction is ~1e-2 off; `dsv4_rope_parity` matches `ops.py` in float64 at
      ~2e-8 (bf16 rows at bf16 rounding) across both measured regimes (theta 1e4 YaRN-on, theta 1.6e5
      YaRN-on), the r=0 no-YaRN regime, and the inverse o-path rotation, and asserts neox pairing,
      skipping the YaRN blend, and a wrong theta observably apart. `dsv4_state.hpp/.cpp` sizes and
      allocates the per-layer pools for one sequence bound: the 128-row window ring, the compressed
      pool (row = block index = `p / ratio`), the indexer pool, and the compressor/indexer carry
      registers (the paged carry ring FreeToken keeps for cross-request carry-by-value is not needed
      for one contiguous sequence - the register IS the carry); `dsv4_state_test` checks the sizing
      arithmetic per layer class, the arena's pointer contract, and the ring/`valid` addressing.
      `dsv4_indexer_select` picks the top-k blocks by score (ties to the lower index, -inf never
      picked, -1 padding) and its picks match a sort reference exactly. *Attention-half wiring done:*
      `hc_mixes` (the mHC pre's GEMV + RMS factor) plus `core/dsv4_attn.cpp` wire one decode token
      end to end through an r=4 layer - mHC pre, attn_norm, the LoRA-split Q with per-head norms,
      the single kv head into the window ring, the compressor decode into the cmp pool, the indexer
      (q rope + Hadamard + e2m1, weights with the folded scale, its own compressor into the idx
      pool, logits, top-k), the tiered gather with the sink, the inverse rope, the grouped `wo_a`
      einsum as one GEMV per group over the flattened weight, `wo_b`, mHC post. The rope/norm
      constants (both thetas, YaRN, `rope.dimension_count`, `layer_norm_rms_epsilon`) are now read
      from the artifact instead of being hardcoded. `dsv4_attn_layer_parity` runs eight consecutive
      tokens - crossing a compressor block boundary at token 3 and reaching two scored indexer rows
      at token 7 - at the real dimensions, against a float64 transcription of the reference decode
      path; the quantization round-trips and the Hadamard are the only parts the reference borrows
      from the GPU kernels (they are parity-tested on their own), so what is under test is the
      wiring. Streams match at ~2e-7 (1e-4 at the token where a quantization boundary rounds
      differently), the three pools match, and no-sink / window-only / no-top-k come out 0.25-0.4
      apart. *FFN-half wiring done:* `core/dsv4_ffn.cpp` wires the other half of the block - mHC
      pre(ffn), ffn_norm, the router (bf16 GEMV then `dsv4_router_score`, or `dsv4_router_hash` on
      a hash layer), the shared expert (bf16 GEMVs + clamped SwiGLU), and the top-k routed experts
      as the pack's own blobs: q8_1 activation, `iq_mmvq` over the `NativeExpertLayout` blob
      (gate, up), clamped SwiGLU, q8_1 again, down, each scaled by its router weight - then mHC
      post. `dsv4_ffn_layer_parity` runs a score-routed layer and a hash layer against a float64
      transcription with the real artifact's expert pair (IQ2_XXS gate/up, IQ3_XXS down) at the
      real 4096/2048, the blobs ggml-quantized and dequantized through the same kernels the expert
      parity tests vouch for; streams match at ~3e-4 to 8e-4 (the q8_1 activation rounding feeds
      on the engine's own fp32, which is the pipeline, not the wiring), and dropping the route
      scale, gathering weights from the post-bias scores, or dropping the SwiGLU clamp come out
      observably apart. One layout fact the test caught: the CUDA `block_q8_1` is `half2 ds
       (d, sum) + qs[32]` - 36 bytes, not the CPU's 34. *The join done:* `core/dsv4_block.cpp`
       puts the two halves in the reference's order with one hazard handled - `hc_post_combine`
       reads its residual while writing, so the block stages the hand-off in its own buffer and
       splits one arena between the halves. `dsv4_block_layer_parity` runs four tokens through a
       full r=4 layer and a full r=0+hash layer: the block is BIT-identical to running the two
       parity-tested halves in sequence with separate workspaces, its pools match the sequence's,
       and the r=4 layer's compressed and indexer pools actually fill. *The weight contract is
       native, not bf16:* the real pack does not serve the attention GEMVs or the shared expert as
       bf16 - the GGUF keeps them quantized (attn q_a and the shared gate/up are Q5_K, q_b/kv/
       output_a/output_b and the shared down Q8_0/Q6_K) and `NativeDense` attaches those blobs
       straight from the shards, so `Dsv4AttnWeights`/`Dsv4FfnWeights` carry a `type` + raw-block
       pointer per GEMV and both halves run them through `quantize_q8_1_rows` + `native_mmvq`
       (the same path the Qwen dense layers use; `native_mmvq` needs a real CUDA stream and
       covers Q3_K..Q6_K/Q8_0 - the IQ2/IQ3 routed experts stay on `iq_mmvq`, and `ffn_gate_inp`
       stays raw BF16). Two layout facts the switch caught: a quantized row is not `n_in` bytes
       wide, so the wo_a group offset must be counted in bytes (`native_mmvq_weight_bytes`), and
       a blob must be quantized with rows = n_out, cols = n_in, because the GEMV reads each
       output row as one contiguous run. With the reference dequantizing the same blobs, streams
       match at ~2e-2 and the pools at ~5e-3 - the q8_1 activation rounding, which measures 5e-3
       per GEMV standalone; the three wrong wirings still come out 0.25-0.4 apart. What still blocks a run:
       the session/generate path that feeds the block the pack's weights (the resolver, plus a
       loader for the I32 `tid2eid` tables, which the pack skips and `NativeDense` does not
       serve), and
       prefill attention.
    The new math, in order of risk: the tiered KV pool buffers (window 128 ring + compressed rows +
    per-page carry state) and the layer wiring that drives the compressor, indexer-score and gather kernels above.
    (The MXFP4 expert GEMV, the attention sinks, the mHC Sinkhorn mixing, the sqrtsoftplus/`tid2eid`
    router and the swiglu clamp this list used to carry are done - see above.)
- **P3 - around it.** Tokenizer pre-tokenizer `joyai-llm` + special tokens (the BPE core is already
  shared); chat template and the three thinking modes in `serve/`; a `dsv4` family in `setup.py` with the
  per-machine defaults above.
- **P4 - drafter + bench.** DSpark under the spec controller (depth 3 default, confidence-head early stop),
  parity tests per component, then measured numbers to replace the estimates above.
- **P5 - GLM-5.3-Flash.** Same mHC, same indexer family, same streaming; adds KDA (close to the GDN kernels)
  and MLA/kpool. Roughly a third the cost once P1-P3 exist.

## Reproducing the P0 dump

```
# metadata (shard 1 is 5 MB, metadata-only)
curl -L -o dsv4_shard1.gguf "https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/resolve/main/UD-IQ2_XXS/DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00001-of-00003.gguf"
# tensor names: the first 4 MB of shard 2 (tensor-info section sits at the head of each shard)
curl -L -r 0-4194303 -o shard2_head.bin "https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/resolve/main/UD-IQ2_XXS/DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00002-of-00003.gguf"
python -c "import sys; sys.path.insert(0,'tools'); from gguf_reader import GGUFFile; g=GGUFFile('shard2_head.bin'); print(g.by_type())"
```

The reference implementations for the semantics live in FreeToken (`python/freetoken/models/deepseek_v4/`:
`attention.py`, `compress.py`, `moe.py`) and in transformers `modeling_deepseek_v4`; the GGUF names above are
llama.cpp's.
