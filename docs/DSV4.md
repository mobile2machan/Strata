# DeepSeek-V4-Flash: support plan

Status: **P0 done** (2026-10-03, artifact inspection), **P1 done** (the real UD-IQ2_XXS pack is built -
78.11 GiB of experts - and verified end to end), **P2 computing done** (the engine drives the
whole 43-layer model over the real pack and produces its first finite, deterministic logits -
`dsv4_generate_smoke`; ~2-3 s/token decode, an 8-token prefill chunk in 4.3 s with picked-expert
staging), **P3 done** (2026-10-04: the `joyai-llm` pre-tokenizer, the model's chat template
and its thinking modes in `serve/`, and `strata --serve --dsv4` - the same stdin/stdout protocol as the
Qwen path - answer real chat-completion requests end to end; measured 0.8-1.2 s/token decode and
~1 s/token prefill on the Windows target (RTX PRO 4000 Blackwell, 137 GB RAM) - every picked expert
blob is read from `experts.bin` per token, the P2 forward has no expert cache; and a `dsv4` family
in `setup.py`, `tools/test_setup_dsv4.py`), **P4 done** (2026-10-04: the `dflash` geometry reader and
layout rules, `DsparkForward`, the speculative controller in `dsv4_serve.cpp`, `dspark_parity`, and measured
DSpark bench numbers on the Windows target replacing the old estimates), and **P4 follow-up done** (2026-10-04:
expert grouping and staging dedup in the verify path, with exact parity and measured depth-3/5 speedups; a
cross-window expert cache was measured and left default-off because it had 0 hits on this workload).
 The decode cost is now measured end to end (2026-10-04): 83% of it is fetching that token's picked expert
 blobs out of `experts.bin`, 5% is the GPU's own work - see the *Why the decode is slow* section under P4.
 `--dsv4-experts-ram` now keeps the experts in host RAM instead of reading them per token: the same depth-3
 bench runs 2.4x faster (67.0 s → 27.9 s for 128 tokens) with byte-identical output - same section.
Everything marked *measured* below came from the artifact or from runs on the target; the old speed estimates
are superseded by the measured P4 numbers.

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
MTP head`. Speculative decoding is a separate sidecar: `dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf` (10.90 GB;
a BF16 build is 11.31 GB). Measured from the sidecar's own metadata and tensor directory:
`dflash.block_count = 3`, hidden 4096, 64 heads with one KV head (`key_length = value_length = 512`, rope 64
dims), `sliding_window = 128`, `compress_ratios = [0, 0, 0]`, `hash_layer_count = 0`, a 256-expert MoE
(top-6 + 1 shared, `expert_feed_forward_length = 2048`, sqrtsoftplus, route scale 1.5, weights norm true,
swiglu clamps 10.0), mHC count 4 / sinkhorn 20 / eps 1e-6, `q_lora_rank = 1024`, `output_lora_rank = 1024`,
`output_group_count = 8`, trained `block_size = 5`, `target_layers = [41, 42, 43]`, and
`tokenizer.ggml.mask_token_id = 128799`. It ships no token embeddings or output head and borrows the target's.
The fusion tensor is `fc.weight` Q8_0 `[12288, 4096]` (three captured target hiddens x 4096), not an
`eh_proj`; the Markov tensors are `markov_w1/w2.weight` BF16 `[256, 129280]`, and the confidence head is
`conf_proj.weight` BF16 `[4352, 1]` (`4096 + 256`). llama.cpp's own measurements (unsloth's, 1x B200) give
**1.91x at draft depth 3** (0.764 acceptance), 1.37x at depth 1; those are not this machine's numbers.

For this engine the drafter matters more than it does for llama.cpp: on a CPU-GPU split run the target's
bottleneck is RAM bandwidth, and a drafter that accepts k of n tokens removes whole target forwards - i.e.
whole passes over the expert bytes. The drafter's own experts (MXFP4, ~8 GB) should go through the same
streaming path rather than sit in VRAM.

## Target machines (the two this fork runs)

| machine | default quant | fit |
|---|---|---|
| Windows, 137 GB RAM, 24 GB NVIDIA | UD-IQ3_XXS (104.2 GB) | experts ~101 GB pinned + KV + OS ~ 115-120 GB; ~17 GB headroom |
| Linux, 128 GB RAM, RX 9070 XT (gfx1201), Ryzen 7 5700X | UD-IQ2_XXS (90.9 GB) | experts 83.9 GB (measured `experts.bin`) pinned ~ 95-100 GB; ~28 GB headroom. gfx1201 is a validated Strata backend (`docs/AMD_HIP.md`) |

Speed estimates (written before any of this ran; superseded by the P4 measurements below): the 5700X box has
DDR4-3200 (~42 GB/s effective), and at ~10% expert-cache hit the CPU GEMV path moves ~1.7 GB of expert bytes
per token at IQ2_XXS -> ~15-22 tok/s, ~25-35 with the drafter at depth 3. The 24 GB NVIDIA box (DDR5) should
land ~25-50 tok/s at IQ3_XXS before the drafter. The same boxes under llama.cpp offload most of the model and
are expected to sit at 3-8 tok/s. These were the claim to beat; the measured numbers are in **P4** and they
are far below it, for a reason the estimates got wrong: the serve path streams every token's picked experts
from disk, and nothing about a drafter or a verify window removes those bytes.

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
       per GEMV standalone; the three wrong wirings still come out 0.25-0.4 apart. *The resolver
       done:* `core/dsv4_weights.cpp` fills a `Dsv4BlockWeights` for any layer from the
       `WeightTable` + `NativeDense` (resident tensors are already device pointers; the GEMVs come
       off the native refs with a shape + type check). It owns only the two things the pack cannot
       serve: `indexer.proj` (the one GEMV the packer left F32 in dense.bin - converted to bf16
       once at load) and the I32 `ffn_gate_tid2eid` tables (skipped by the pack, not a GEMV, so
       `NativeDense` does not serve them - read from the GGUF shard that holds them). The routed
       experts stay the caller's: 78.11 GiB does not live in VRAM, so the session's expert source
       supplies `expert_blobs` per call, as `moe_layer` does today. `dsv4_weights_real` loads the
       real pack, resolves all 43 layers (every pointer present, every native type one
       `native_mmvq` implements, `native_expert_layout` sizes matching `native_experts.txt`), and
        runs one decode token through the full block at layer 0 (hash, r=0) and layer 2 (hash +
        indexed, r=4) with the pack's own bytes - experts copied from `experts.bin` by the
        per-layer cumulative offsets - to a fully finite 16384-element stream out. *The first
        tokens done:* `core/dsv4_forward.cpp` drives the whole model per token - `NativeEmbed`
        replicated across the four hc streams, all 43 blocks in order, the head's mHC collapse
        (`hc_mixes` + the new `hc_head_pre` sigmoid + `hc_pre_combine`), `output_norm`, and
        `NativeHead` on `output.weight`. `dsv4_generate_smoke` decodes eight positions of a
        fixed prompt over the real pack: every logit row is finite, the argmaxes are in range,
        and a second run from a zeroed state samples identically. Two things that took a debug
        pass to find: `iq_mmvq`'s dispatch lacked MXFP4 (type 39) even though the format was
        implemented - layers 26 and 42 have FP4 down matrices - and a 32-bit `fseek` into
        `experts.bin` silently wrapped past 4 GiB, which read the wrong layer's experts from
        layer 3 on. *Picked-expert staging done:* the router's picks are known only after its
        GEMV, and the FFN half already syncs them to host before touching the blobs - so
        `dsv4_ffn_decode_step`/`dsv4_block_decode_step` take an optional `Dsv4ExpertSource`
        called at exactly that point, and `Dsv4Forward` stages the six picked blobs (45 MB)
        instead of the layer's 1.8 GiB region. Measured: the eight-position smoke run went from
        ~10 minutes to ~60 seconds on the 3090 (a decode step is now ~2-3 s, still dominated
        by cold random reads); the staged and full-region paths give bit-identical logits. *The
        prefill gather done:* `dsv4_attn_prefill` (same file as the decode gather) runs `n`
        queries in one launch, each with its own window/compressed id lists - the causal window
        and the per-query indexer picks are the caller's, the kernel just gathers, one block per
        (query, head). Parity: every prefill row matches the decode kernel's reference given the
        same lists (rel 1.3e-7), and a query given its neighbour's list is 0.66 away. *The
        batched prefill engine done:* `dsv4_ffn_prefill_step` / `dsv4_attn_prefill_step` /
        `dsv4_block_prefill` run a chunk of up to 8 tokens (the native GEMV's column limit) -
        every quantized GEMV goes through `native_mmvq`/`quantize_q8_1_rows` with `n` columns
        (multi-exact makes that bitwise the same as one column), the stateful parts (rope, ring
        writes, compressor steps, indexer selection, the strided `wo_a` group slices) run per
        position exactly as the decode step, and the gather is the batched kernel. Parity: the
        FFN half matches the decode loop bit for bit; the attention half matches the float64
        reference at the same ~2e-2 as the decode path; and end to end, `Dsv4Forward::prefill`
        over the real pack gives logits **bit-identical** to eight decode steps (rel 0.000e+00)
        in 4.3 s instead of ~16 s. What still blocks a usable run: the tokenizer.
    The new math, in order of risk: the tiered KV pool buffers (window 128 ring + compressed rows +
    per-page carry state) and the layer wiring that drives the compressor, indexer-score and gather kernels above.
    (The MXFP4 expert GEMV, the attention sinks, the mHC Sinkhorn mixing, the sqrtsoftplus/`tid2eid`
    router and the swiglu clamp this list used to carry are done - see above.)
- **P3 - around it.** *Done.* `tools/strata_tokenizer.py` reads the
  `joyai-llm` pre-tokenizer (its pattern set transcribed from llama.cpp, shipped in the pack's
  `tokenizer.json`) and the special ids; `serve/frontend.py` reads the template's own variables and
  picks its dialect (`thinking`/`bos_token` = DeepSeek-V4's, `enable_thinking` = Qwen's), mapping the
  six client effort spellings onto the template's `thinking` + `reasoning_effort high|max`;
  `src/program/dsv4_serve.cpp` is the small serve loop (GEN/T/DONE/STOP/QUIT, greedy, one sequence,
  the P2 forward over the pack) that `serve/server.py` drives unchanged - verified end to end with
  `strata-dsv4.json` on the Windows target, both thinking modes answering real requests.
  *Setup family done (2026-10-04):* `setup.py` gained a `dsv4` family - UD-IQ2_XXS (3 shards,
  90.9 GB) and UD-IQ3_XXS (4 shards, 104.2 GB) from Unsloth's repository pinned to one commit, every
  shard's size and SHA-256 pinned (the IQ2 hashes computed from the local files, the IQ3 ones are the
  repository's LFS oids, the same hash as verified for the IQ2 files); the pack is always built with
  `--experts-bin`; the start config carries `--pack`, `--dsv4 <shard 1>`, `--max-context`, and
  `--dsv4-experts-ram auto` when the installed engine knows that option -
  no PLE table, no expert profile, no draft layer, no KV or rope flags; NVIDIA only, one card, no
  images, and it asks the engine binary whether it knows `--dsv4` (its own help) before the 91 GB
  download - the published ready-made engine does not carry the serve path yet, so setup says to
  compile with `--build` until a release does. `tools/test_setup_dsv4.py` covers it mocked; the
  name check (`gguf_unsupported`) now reads which model a GGUF's name says, so a Qwen-named
  UD-IQ3_XXS is still refused even though that size exists for DeepSeek-V4.
- **P4 - drafter + bench.** *Done.* The drafter is now a first-class artifact and a measured serve path.
  `include/strata/artifact/dflash_geometry.hpp` reads `dflash.*` metadata into `ModelArch::DFlash` /
  `ModelGeometry::Dflash`, refusing a non-dflash arch, a missing key, a nonzero `compress_ratio`, a nonzero
  `hash_layer_count`, a missing `block_size`, a missing `target_layers`, or a missing
  `tokenizer.ggml.mask_token_id` by name. `src/core/layout.cpp` routes `DFlash` through the r=0 dsv4 layer
  checks and adds `check_dflash_extras`: `fc.weight` must be `[n_embd * target_layers.size(), n_embd]`,
  `markov_w1` and `markov_w2` must agree, and `conf_proj.ne0` must be `n_embd + markov_rank`.
  `tests/core/dflash_layout_test.cpp` covers the synthetic fixture and those negatives; its `--real` mode
  (`dflash_layout_real`, registered when `STRATA_DFLASH_PACK`/`STRATA_DFLASH_GGUF` point at one) reads the
  real sidecar pack and passes `check_all` over every drafter layer and head tensor.

  `Dsv4Forward` now captures the target hidden the drafter needs: at each captured target layer it writes
  `hc_mean` over the four mHC streams into a token-major fused buffer, and `DsparkForward::inject()` runs the
  encoder side - `fc` fusion of target layers `[41, 42, 43]`, `enc.output_norm`, each stage's `attn_kv` GEMV,
  `attn_kv_a_norm`, rope, e4m3 round-trip, and ring write. `DsparkForward::draft()` runs the decoder side:
  target embeddings replicated across the drafter's four streams, its three r=0 blocks, `output_hc_*` collapse,
  `output_norm`, the borrowed target head, the Markov chain, and the confidence head. `tests/core/
  dspark_parity.cpp` checks `hc_mean` against a float64 mean (worst abs err `3.909e-08`), the injected ring
  rows against a float64 encoder reference (`rel_l1` `4.102e-03` to `5.760e-03`, the expected q8_1 rounding),
  and the draft path against a float64 Markov/confidence chain (every draft token matches the reference argmax;
  worst confidence diff `1.813e-08`).

  `src/program/dsv4_serve.cpp` now runs the speculative loop when `--dspark` and `--dspark-pack` are supplied:
  `--spec-depth` defaults to 3 and is clamped to the sidecar's `block_size`; `--spec-conf` cuts the draft at
  the first confidence below the threshold. The controller's invariant is that the target logits at loop top are
  for position `pos`, the drafter block is `[t, mask * (d-1)]`, the target verify window is `[t, d0..d_{d-1}]`,
  and `logits_window[i]` is the target distribution for position `pos+i+1`. A full acceptance therefore advances
  to `pos+d+1` and reads `logits_window[d]`; a rejection restores the compressor/indexer carries to the last
  accepted position and commits the target's own pick through a normal decode. The window ring and compressed
  pools are position-keyed and self-heal, so only the carry registers need snapshot/restore. A 16-token greedy
  equivalence run on the Windows target produced identical token sequences with and without the drafter.

  Measured bench (Windows target, RTX PRO 4000 Blackwell 24 GB, 137 GB RAM, pack `dsv4-ud-iq2_xxs`, prompt
  `0,151,4023,917,2046,88,12345,60231`, 128 generated tokens, decode time only):

  | mode | DONE decode | tok/s | spec windows | accepted drafts | draft ms | verify ms |
  |---|---:|---:|---:|---:|---:|---:|
  | no drafter | 66945.1 ms | 1.91 | - | - | - | - |
  | `--spec-depth 1` | 69640.6 ms | 1.84 | 64 | 63 | 4421.5 | 64490.7 |
  | `--spec-depth 3` | 69760.6 ms | 1.83 | 33 | 94 | 6306.4 | 62703.8 |
  | `--spec-depth 5` | 69195.5 ms | 1.85 | 22 | 105 | 7153.0 | 61323.6 |
  | `--spec-depth 3 --spec-conf 0.5` | 68689.9 ms | 1.86 | 34 | 93 | 6292.3 | 61727.7 |
  | `--spec-depth 3 --spec-conf 0.95` | 74230.4 ms | 1.72 | 54 | 72 | 10519.8 | 62994.4 |

  The drafter is correct and accepts a high fraction of its drafts on this synthetic repetitive prompt
  (`63/64` at depth 1, `94/99` at depth 3, `105/110` at depth 5). In the original ungrouped verify path it did
  not speed this configuration up. The reason was visible in the split: verify was 85-93% of the time, and the
  target's MoE FFN streamed each token's picked expert blobs from `experts.bin`. A verify window amortized the
  dense GEMVs across the window, but it did not remove the per-token expert reads. Confidence early stop worked
  mechanically - at `0.95` the controller cut drafts and accepted fewer tokens - but it did not help throughput
  because the draft work it saved was smaller than the unchanged per-token verify cost. The old 15-50 tok/s
  estimates were wrong for that serve path; the measured ungrouped path was ~1.8-1.9 tok/s before and after the
  drafter.

  *P4 follow-up (2026-10-04): expert grouping and staging dedup.* The default verify path now groups repeated
  expert ids inside one verify/prefill window. `Dsv4Forward::Stager::stage` copies a repeated blob from the
  already-staged slot instead of reading the same `(layer, expert)` bytes from `experts.bin` again.
  `dsv4_ffn_prefill_step` then gathers the window tokens' q8_1 activations for one expert blob, runs gate/up and
  down through `iq_mmvq` with multiple columns, applies the same DSV4 clamped Swiglu and q8_1 quantization,
  copies the per-entry outputs back in original pick order, and combines them in the original order. The path is
  enabled by default; `STRATA_DSV4_GROUPED=0` disables expert grouping and `STRATA_DSV4_STAGE_DEDUP=0` disables
  staging dedup. Parity stayed exact: `dsv4_ffn_layer_parity` reports prefill-vs-decode `rel_l1 0.000e+00`,
  `dsv4_block_layer_parity` is bit-identical, `dspark_parity` passes, and a 16-token greedy run with and without
  the drafter produces identical token sequences.

  Measured after the follow-up (same Windows target, same pack, same prompt, 128 generated tokens, decode time
  only):

  | mode | DONE decode | tok/s | spec windows | accepted drafts | draft ms | verify ms |
  |---|---:|---:|---:|---:|---:|---:|
  | no drafter | 68122.7 ms | 1.88 | - | - | - | - |
  | `--spec-depth 1` | 69008.1 ms | 1.86 | 64 | 63 | 4548.7 | 63646.9 |
  | `--spec-depth 3` | 60671.4 ms | 2.11 | 33 | 94 | 6462.0 | 53506.9 |
  | `--spec-depth 5` | 57060.8 ms | 2.24 | 22 | 105 | 7067.5 | 49231.0 |
  | `--spec-depth 3 --spec-conf 0.5` | 59008.6 ms | 2.17 | 33 | 93 | 6628.4 | 51709.5 |
  | `--spec-depth 3 --spec-conf 0.95` | 69353.0 ms | 1.85 | 52 | 74 | 10195.3 | 58509.2 |

  The effective follow-up is within-window grouping/dedup, not a cross-window expert cache. On this prompt it
  makes depth 3 and depth 5 faster than the baseline: depth 3 improves from ~1.83 to ~2.11 tok/s, and depth 5
  reaches ~2.24 tok/s. Confidence early stop still does not help at `0.95` because cutting drafts increases the
  number of verify windows.

  A cross-window expert cache was also implemented and measured. `Dsv4Forward::Stager` can keep recently staged
  blobs in a host LRU cache (`STRATA_DSV4_EXPERT_CACHE_BLOBS=N`, disabled with `STRATA_DSV4_EXPERT_CACHE=0`),
  but the measured default is 0: with `STRATA_DSV4_EXPERT_CACHE_BLOBS=32`, the same depth-3 run recorded 0 cache
  hits and 20,741 misses and slowed to 73848.7 ms decode (verify 66414.2 ms). The reason is that these verify
  windows do not repeat the same `(layer, expert)` picks across windows; the only locality is inside a window,
  which the grouping path already exploits.

  *Why the decode is slow, measured (2026-10-04).* The time is in the expert bytes, not in the compute. One
  token needs 43 layers x 6 picked blobs = 1,874.6 MiB of expert weights (the blobs are 7.2-9.4 MiB per
  `native_experts.txt`), and `experts.bin` is 78.11 GiB - it fits neither 24 GB of VRAM nor the ~59 GB of page
  cache this box has free. `Dsv4Forward::Stager::stage` serves those bytes one blob at a time, in order:
  `_fseeki64` + `fread` into a single pinned staging buffer, then one `cudaMemcpyAsync` +
  `cudaStreamSynchronize` per layer, 43 times per token. A throwaway CUDA program that runs exactly that path
  against the real pack - same offsets, same sizes, same pinned buffer - measures it on its own: 1.63-1.75
  s/token for fresh random picks (reads only), 0.54-0.92 s/token once the picked pages are warm, and 79-83
  ms/token for the copy and the sync alone (23-25 GiB/s, so PCIe is not the wall). Six threads reading one
  layer's six blobs at once took 49-68 ms for 43 MiB - no better than reading them in order, so the drive sets
  the rate, not the thread count.

  The engine's own accounting for the same depth-3 bench (`Dsv4Forward::StageStats`, printed with the serve
  summary under `STRATA_DSV4_STAGE_DEBUG=1`; the Qwen server stopped first, so the page cache was as free as
  this box gets):

  | run | blob reads | read from `experts.bin` | staging | verify | decode |
  |---|---:|---:|---:|---:|---:|
  | grouped + dedup (default) | 20,741 | 147.07 GiB | 54,821.3 ms | 57,955.2 ms | 66,275.4 ms |
  | `STRATA_DSV4_GROUPED=0 STRATA_DSV4_STAGE_DEDUP=0` | 35,862 | 254.47 GiB | 71,872.9 ms | 75,514.1 ms | 83,645.0 ms |

  Both runs drafted and accepted the same 33 windows and 94 drafts, so the difference is the fetch alone:
  grouping removes 15,121 blob reads and 107.4 GiB, staging drops by 17.05 s and the decode by 17.37 s - 1:1.
  Staging is 82.7% of the default run's decode and 94.6% of its verify. What is left of verify - every dense
  GEMV over the 5.71 GiB of resident weights, the attention/compressor/indexer kernels, the mHC mixing - is
  3.13 s: 4.7% of the decode, ~24 ms/token. The drafter is 7.51 s (11.3%), and the copy volume is the same
  254.47 GiB in both runs - grouping cuts reads, not copies. These absolute times sit above the table above (a
  different page-cache state, `--max-context 2048`); the split is the point.

  So decode speed is (expert bytes per token) / (the rate they arrive at): 1.149 GiB/token at 2.68 GiB/s
  grouped, 1.988 GiB/token at 3.54 GiB/s ungrouped - between the probe's cold 1.1 GiB/s and warm 6.25 GiB/s,
  which is what a 78.11 GiB working set does against 59 GB of free RAM. The drafter, the verify window and the
  kernels are not where the time is, and nothing there can be worth more than the ~24 ms/token the GPU actually
  spends. What moves it is fewer bytes per token (IQ2_XXS is already the smallest quant in this family) or more
  of them resident: the page cache (78.11 GiB against this box's RAM) or a per-layer VRAM hot set (24 GB holds
  ~2.5 layers' full expert regions). The pick distribution settles the hot-set question and it is flat: only
  `blk.0/1/2` carry an `ffn_gate_tid2eid.weight` (`[6][129280]` I32, `include/strata/core/dsv4_ffn.hpp:35`),
  so layers 0-2 route by a token-id hash (`kernels::dsv4_router_hash`) while layers 3-42 route through
  `ffn_gate_inp` + `exp_probs_b` - sqrtsoftplus scores plus a learned bias, which is what keeps the load even
  over 256 experts. With 0 cross-window cache hits out of 20,741 reads there is no skew to exploit, so a
  resident prefix captures exactly its share of the file and nothing smarter does better.

  *Keeping the experts in host RAM: `--dsv4-experts-ram` (2026-10-04).* The split above says what to fix: 78.11
  GiB of expert bytes against 128 GB of system RAM. `Dsv4Forward::set_experts_ram` reads a prefix of
  `experts.bin` into one host buffer at load (`std::malloc`, cut at a blob boundary), page-locks as much of it as
  the driver takes, and `Stager::stage` then serves a picked blob from that mirror instead of the file: a
  page-locked blob is copied straight into its device slot by `cudaMemcpyAsync`, a mirrored-but-unregistered one
  is `memcpy`'d into the staging buffer and flushed in runs, and only a blob past the mirror still costs a seek
  and a read. `--dsv4-experts-ram 32` asks for 32 GiB; `auto` asks for what the free RAM leaves after the same
  8 GiB headroom the Qwen resident mode keeps. A mirror that cannot be allocated or page-locked is not an error:
  the engine prints why and keeps the file path. It is built last in `init()`, after every small pinned buffer,
  because registering a large arena first has previously left the driver unable to pin the small ones.

  The driver's page-lock ceiling, not the RAM, is the limit. A throwaway `cudaHostAlloc` probe refused 78.11 GiB
  and 64 GiB on this box and accepted 48 GiB; `cudaHostRegister` of the 78.11 GiB mirror took 50.11 GiB. The same
  probe measured the pinned copy path: 21.59 GiB/s for 258 per-pick `cudaMemcpyAsync` calls of 7.2-9.4 MiB per
  token and 21.9 GiB/s for one 6-blob bulk copy per layer, so copying blob by blob costs nothing over copying a
  whole layer at once - the 2.7 GiB/s file read was the entire gap.

  Same bench, same binary, same prompt, Qwen server stopped, `--max-context 2048`, depth 3, 128 tokens
  (`Dsv4Forward::StageStats` under `STRATA_DSV4_STAGE_DEBUG=1`):

  | arm | blob reads | from the mirror | staging | verify | decode |
  |---|---:|---:|---:|---:|---:|
  | file path (default) | 20,741 (147.07 GiB) | - | 57,148.1 ms | 58,981.9 ms | 67,003.1 ms |
  | `--dsv4-experts-ram 32` (32.00 GiB mirrored, all page-locked) | 12,216 (87.24 GiB) | 8,525 (59.84 GiB) | 42,163.0 ms | 40,898.7 ms | 48,041.2 ms |
  | `--dsv4-experts-ram auto` (78.11 GiB mirrored, 50.11 GiB page-locked) | 0 | 20,741 (147.07 GiB) | 15,871.3 ms | 19,533.5 ms | 27,882.7 ms |

  All three drafted and accepted the same 33 windows and 94 drafts, made the same 1,505 staging calls, copied the
  same 254.47 GiB to the GPU and printed byte-identical 128-token outputs (`diff` over the `T` streams: no
  difference), and `dsv4_ffn_layer_parity`, `dsv4_block_layer_parity`, `dsv4_attn_layer_parity` and
  `dspark_parity` pass with the change. The default row differs from the table above by 2.3 s of staging - the
  page-cache state of a second run, not a code difference; the three rows here were measured back to back.

  128 tokens in 67.0 s became 27.9 s: 523.5 → 217.8 ms/token, 1.91 → 4.59 tok/s, with staging 57.1 s → 15.9 s.
  The gain tracks coverage exactly - the 32 GiB prefix served 8,525 of 20,741 blobs (41.1%, against 32.00/78.11 =
  41.0% of the file) and cut the decode by 28%; the full mirror served all of them and cut it by 58%. What is
  left of the full mirror's staging is the 36% the driver would not pin: 147.07 GiB in 15.9 s is 9.3 GiB/s,
  between the pinned DMA rate (21.6 GiB/s) and the file rate (2.7 GiB/s), which is the CPU `memcpy` through the
  staging buffer. The rest of that decode is the drafter (8.1 s) and the ~24 ms/token of GPU work measured above;
  the file is no longer what sets the speed.

  The cost is startup time and held RAM: filling the mirror is one sequential read of the file - 78.11 GiB in
  70.3 s (1.11 GiB/s), 32.00 GiB in 46.0 s - and it stays allocated for the session. It is off by default and
  the file path is byte-for-byte unchanged when it is off.

  Measured with the Qwen 8080 server running (it pins 46.84 GiB; 58.5 GB free): `auto` sized itself to 42.04
  GiB, the driver page-locked all of it, and the same run took 11,169 of its 20,741 blobs from the mirror -
  staging 50,338.8 ms, decode 55,058.1 ms (2.32 tok/s), output identical again. That is 1.2x, not 2.4x: the
  68.68 GiB it still had to read came at 1.4 GiB/s against a page cache the Qwen model is holding, and its
  15.03 GiB GPU expert cache competes for the card. Both models at full size would need 46.84 + 78.11 GiB of
  experts on a 128 GB box, so on this machine the choice is which one is resident.

  This machine's start config (`strata-dsv4-ud-iq2_xxs.json`) carries the option and the drafter:
  `--dsv4-experts-ram auto`, `--dspark <the drafter's GGUF>`, `--dspark-pack packs/dspark-q8_0`,
  `--spec-depth 3`, and its `exe` points at `build/strata.exe`, because the installed `engine/strata.exe` is
  the 0.1.37 release - it has the `--dsv4` serve path but not the drafter or this option, and an engine
  refuses an argument it does not know. `setup.py` writes the residency flag into a dsv4 config only when the
  engine it installed answers `--help` with it (`engine_has_arg`); it has no drafter step yet, so re-running
  setup for this model rewrites the config back to the published engine, the file path, no drafter, and the
  card its own questions picked.

  The serve path runs on one card. `Dsv4Forward` and `dsv4_serve.cpp` never name a CUDA device, so every
  allocation lands on the device the process is given, and the Qwen path's `--peer-device` and
  `--expert-cache-device1..3` do not apply to it. The config pins the card with the `"gpu"` key, which
  `serve/server.py` turns into `CUDA_DEVICE_ORDER=PCI_BUS_ID` plus `CUDA_VISIBLE_DEVICES`; here `"gpu": 1`,
  the RTX 3090, which leaves the RTX PRO 4000 Blackwell free for the Qwen model.

  Measured on the 3090 with this config's arguments, same prompt and settings as the table above (128 tokens,
  `--max-context 2048`, `auto`): `--spec-depth 3` decodes in 55,632.5 ms (2.30 tok/s, 33 spec windows, 93
  drafts accepted) and without the drafter in 59,046.4 ms (2.17 tok/s). The same arm on the Blackwell was
  27,882.7 ms (4.59 tok/s), so the 3090 costs about 2x on this path and the drafter buys 5.8% there, not the
  2.4x the residency option bought. Through the server the same config answered a real 128-token chat request
  at 402.7 ms/token (2.5 tok/s); the card held 8,929 MiB for the whole model, the drafter adding 0.14 GiB of
  it (its own experts stream from its pack).

  `auto` sizes from the RAM free at the moment the model starts, and it says what it is doing. Measured with
  60.00 GiB held by another process: `the free RAM covers 42.30 of the file's 78.11 GiB; the rest is read from
  the file`, then 42.30 GiB mirrored in 56.0 s, all of it page-locked. An explicit `--dsv4-experts-ram 4` is
  honoured as asked (3.99 GiB after the blob-boundary cut) and prints nothing extra. The start order matters:
  the first start after the Qwen server had been stopped took 13.64 GiB - the size follows the OS's free-RAM
  reading, which was about 21.6 GiB at that moment - and the same start 20 minutes later took the whole
  78.11 GiB. Nothing fails in the smaller case; the rest of the file keeps the file path.
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
