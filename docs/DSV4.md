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
     zeroed instead of masked - are observably apart. What still blocks a run: the pool addressing and
     layer wiring around these kernels, and the rest of the math.
   The new math, in order of risk: the tiered KV pool (window 128 + compressed entries + sinks) and the
   layer wiring that drives the compressor kernels above; the indexer scoring
   compressed entries (the QSA stack is the base); mHC Sinkhorn mixing (the GR plumbing is the base);
   sqrtsoftplus router + `tid2eid` static routing; swiglu clamp; attention sinks. (The MXFP4 expert GEMV
   this list used to carry is done - see above.)
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
