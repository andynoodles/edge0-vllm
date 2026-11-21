# Edge0-145B on vLLM: design and contracts

Status: written 2026-11-19 by the `design` agent. This file is the **contract** between the converter agent
(`edge0/convert/`) and the model agent (`edge0/plugin/`, `edge0/docker/`, `edge0/serve.sh`). Change it only through
the coordinator; both agents re-read it after every change.

Inputs: `edge0/docs/MODEL-SPEC.md` (how Strata computes the model, file formats), `edge0/docs/VLLM-BASE.md` (what the
`nvcr.io/nvidia/vllm:26.11-py3` image provides), `edge0/convert/strata_formats.py` (numpy readers and a float32 MoE
oracle), and the TidalMoE plugin ([andynoodles/tidal-moe](https://github.com/andynoodles/tidal-moe) `plugin/moe_placement/__init__.py`). The vLLM facts below were re-checked in the
extracted image source at `5090:/home/andychen/work/edge0-vllm-ref/vllm` (paths below are relative to that directory).

The goal that drives every choice: **spend less time loading experts.** On this box the 44.3 GiB expert pool sits
behind PCIe Gen5 (about 50 GB/s) next to a 32 GB GPU. A cold expert byte is far more expensive than on TidalMoE's GB300,
and the hot set is far smaller, so every mechanism that moves fewer expert bytes per step, raises the VRAM hit rate, or
overlaps transfers with compute is worth more here, not less.

---

## 0. Decisions on one page

| # | Topic | Decision |
|---|---|---|
| D1 | Model class | `Edge0ForCausalLM` = vLLM's `Qwen4ExpForCausalLM` with **one change**: the MoE block of the 48 main layers becomes a new `Edge0LatentMoeBlock`. Everything else (HC, GDN, QSA + indexer, PLE, MTP, scheduler, paged KV, hybrid state, CUDA graphs, torch.compile) is vLLM's own code. |
| D2 | Checkpoint | HF-style directory with safetensors shards. `config.json` keeps `model_type: "qwen4_exp_text"` (so transformers' `Qwen4ExpTextConfig` and vLLM's MTP rewrite keep working) with `architectures: ["Edge0ForCausalLM"]` and an `edge0` sub-dict for the new keys. |
| D3 | Dense Q8_0 weights | Stored **as raw GGUF Q8_0 blocks** (`uint8 [N, K/32, 34]`, the vLLM ModelOpt `MIXED_PRECISION` `q8_0` contract). They are lossless and half the size of BF16. At runtime `EDGE0_DENSE=bf16` (bring-up default) dequantizes them while loading. `EDGE0_DENSE=q8` (tuning) serves them natively through vLLM's b12x block-quant kernels. Exceptions stored BF16: QSA `q/k/v` (fused with the BF16 indexer), `embed_tokens`, `lm_head`. |
| D4 | Routed experts | Stored in **ModelOpt NVFP4 layout, stacked per layer** (a lossless nibble/scale repack of `experts.bin`). They are served by vLLM's **Marlin W4A16 NVFP4** kernel (`fused_marlin_moe`), which is what vLLM auto-selects for `W4A16_NVFP4` on sm_120. There is no valid activation `input_scale`, so W4A4 is not used. The plugin owns the expert storage itself; vLLM's `FusedMoE` is **not** used for the main model's experts. |
| D5 | Latent MoE math | `xl = D_cat·x` (one Q8 GEMM, 2560 to 14336), then **route-as-token**: each of the T·10 routes becomes one row `xl[t, group(e)]` of width 1792, and one Marlin MoE call runs with `topk=1`. A deterministic Triton combine scatters the weighted outputs into `Z[T, 8·1792]`, then `U_cat·Z` (one Q8 GEMM, 14336 to 2560), plus the stock shared expert. |
| D6 | Placement | Per layer, a **hot pool in VRAM** and a **cold pool in pinned, mapped host memory**, both in Marlin layout. A physical slot table maps each logical expert to (hot or cold, index). The router is not permuted (exact routing). Hot and cold experts run as **two concurrent Marlin calls** (main stream and side stream), and the cold call reads host memory zero-copy. The default hot set is the top-H pairs of `profile.bin` (Strata's own ranking). |
| D7 | PLE | vLLM's `host_file_gather` backend reads rows in place from `ple-fp8.gguf` (hardlinked into the checkpoint). The plugin patches `discover_table_layout` to point it at the GGUF offset. The forward pass reads only fixed staging buffers, so it is CUDA-graph safe. |
| D8 | MTP | Phase 2. The stock `Qwen4ExpMTP` is used. Its Q2_0 experts are re-quantized to NVFP4 and served by the stock FusedMoE + Marlin path, fully VRAM-resident. |
| D9 | First-light fallbacks | F0 `EDGE0_EXPERT_VRAM_GIB=0`: every expert cold, one zero-copy Marlin call. F1 `EDGE0_MOE_IMPL=torch`: an eager pure-torch reference for the expert part. Both use the same loader and checkpoint. |

---

## 1. Converted checkpoint contract: `/home/andychen/models/Edge0-145B-vllm`

### 1.1 Directory

```
Edge0-145B-vllm/
  config.json                         # 1.2
  generation_config.json              # 1.3
  tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json chat_template.jinja   # 1.3
  model.safetensors.index.json        # 1.4
  model-00001-of-00050.safetensors    # dense: all non-expert, non-MTP tensors (~11.3 GB)
  model-00002-of-00050.safetensors    # routed experts of layer 0 (0.99 GB)
  ...                                 # model-{l+2:05d}: routed experts of layer l
  model-00049-of-00050.safetensors    # routed experts of layer 47
  model-00050-of-00050.safetensors    # every mtp.* tensor (~1.5 GB)
  ple-fp8.gguf                        # HARDLINK (ln, same filesystem) of Edge0-145B-NVFP4/ple-fp8.gguf; never a copy
  profile.bin                         # copy of Edge0-145B-NVFP4/profile.bin (196,632 B), default placement ranking
  draft_vocab.json                    # JSON list of the 106,299 ids in mtp/draft_vocab.bin, sorted ascending, unique
  edge0_convert.json                  # provenance + check results (1.7)
  README.md, LICENSE, NOTICE          # copied; license note: commercial serving needs a separate Edge0 license
```

Shard rule (unambiguous, the plugin relies on it only for logging): shard 1 holds every tensor that does not start
with `mtp.` and does not contain `.mlp.experts.`. Shard `l+2` holds exactly the six `model.layers.{l}.mlp.experts.*`
tensors of layer `l`. The last shard holds every `mtp.*` tensor. The total is `1 + num_hidden_layers + 1`. The index's
`metadata.total_size` is the sum of tensor bytes.

### 1.2 `config.json` (exact; the converter writes all keys)

```json
{
  "architectures": ["Edge0ForCausalLM"],
  "model_type": "qwen4_exp_text",
  "dtype": "bfloat16",
  "vocab_size": 248320,
  "hidden_size": 2560,
  "num_hidden_layers": 48,
  "layer_types": ["linear_attention","linear_attention","linear_attention","indexed_attention", "... x12, 48 entries; l%4==3 -> indexed_attention"],
  "num_attention_heads": 24,
  "num_key_value_heads": 2,
  "head_dim": 256,
  "hidden_act": "silu",
  "attention_bias": false,
  "rms_norm_eps": 1e-06,
  "max_position_embeddings": 262144,
  "rope_parameters": {"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25,
                      "mrope_section": [11, 11, 10], "mrope_interleaved": true},
  "linear_conv_kernel_dim": 4,
  "linear_key_head_dim": 128,
  "linear_value_head_dim": 128,
  "linear_num_key_heads": 16,
  "linear_num_value_heads": 48,
  "output_gate_type": "sigmoid",
  "num_experts": 512,
  "num_experts_per_tok": 10,
  "norm_topk_prob": true,
  "moe_intermediate_size": 640,
  "shared_expert_intermediate_size": 640,
  "hc_count": 4,
  "hc_lowrank": 320,
  "ple_layer_ids": [2],
  "ple_embed_dim": 2560,
  "ple_conv_kernel_size": 4,
  "ngram_size": 3,
  "heads_per_ngram": 8,
  "ngram_vocab_size_base": 20000000,
  "make_ngram_vocab_size_divisible_by": 128,
  "seed": 1234,
  "split_ngram_parts": 512,
  "ple_embedding_dtype": "float8_e4m3fn",
  "indexer_n_heads": 4,
  "indexer_kv_heads": 1,
  "indexer_head_dim": 128,
  "indexer_budget": 2048,
  "indexer_compress_ratio": 4,
  "mtp_num_hidden_layers": 1,
  "tie_word_embeddings": false,
  "bos_token_id": 248044,
  "pad_token_id": 248044,
  "eos_token_id": 248044,
  "quantization_config": {
    "quant_method": "edge0",
    "dense_format": "ggml_q8_0",
    "quantized_layers": {
      "model.layers.0.linear_attn.in_proj_qkv": {"quant_algo": "Q8_0", "group_size": 32, "block_payload_bytes": 34, "packing": "ggml"},
      "...": "one entry per Q8_0 module listed in 1.5 (370 entries), same recipe",
      "mtp.layers.0.mlp.experts": {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    }
  },
  "edge0": {
    "format": "edge0-vllm-v1",
    "latent_dim": 1792,
    "latent_groups": 8,
    "expert_intermediate_size": 640,
    "expert_format": "modelopt_nvfp4_stacked",
    "ple_file": "ple-fp8.gguf",
    "ple_data_offset": 352,
    "ple_rows": 320001536,
    "ple_row_bytes": 160,
    "ple_scale": 0.00019931793212890625,
    "profile_file": "profile.bin",
    "draft_vocab_file": "draft_vocab.json",
    "source_dir": "/home/andychen/models/Edge0-145B-NVFP4"
  }
}
```

Rules:
- `layer_types` is written out as the explicit 48-entry list (`"indexed_attention"` when `l % 4 == 3`, otherwise
  `"linear_attention"`). The placeholder string above is not part of the file.
- `quantized_layers` holds exactly these 370 Q8_0 entries, each with the recipe shown, plus the one MTP expert entry:
  - GDN layers: `model.layers.{l}.linear_attn.{in_proj_qkv,in_proj_z,out_proj}`
  - QSA layers: `model.layers.{l}.self_attn.o_proj`
  - every layer: `model.layers.{l}.mlp.shared_expert.{gate_proj,up_proj,down_proj}` and
    `model.layers.{l}.mlp.{latent_down,latent_up}`
  - MTP: `mtp.fc_embedding`, `mtp.fc_hidden`, `mtp.layers.0.self_attn.{q_proj,k_proj,v_proj,o_proj}`,
    `mtp.layers.0.self_attn.indexer.index_qk_proj` and `mtp.layers.0.mlp.shared_expert.{gate_proj,up_proj,down_proj}`
- `eos_token_id` **must be the integer 248044**. vLLM's PLE reads it as the n-gram reset id (MODEL-SPEC 3.1). The real
  stop ids go in `generation_config.json`.
- `quantization_config` must **not** have a top-level `quant_algo` key. If it did, ModelOpt's
  `override_quantization_method` would hijack it as `modelopt_mixed`. The plugin registers `edge0`, and `serve.sh` also
  passes `--quantization edge0`.
- `quantized_layers` keys use checkpoint module names (tensor name minus `.weight`). The MTP keys use `mtp.layers.0`,
  and vLLM's `_remap_quantized_layers` (`models/qwen4_exp/nvidia/mtp.py`) rewrites them to `mtp.layers.48`.
- `ple_file` is relative to the checkpoint directory, so the same config works on the host and in the container
  (`/models/Edge0-145B-vllm`).

### 1.3 Tokenizer and generation files

- Start from the HF tokenizer the coordinator downloaded at `5090:/home/andychen/work/edge0-bench/tokenizer-qwen38/`
  (`tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`). **Verify** that id→token is identical to
  Edge0's `tokenizer/vocab.json` for all 248,320 ids (including the specials in MODEL-SPEC 6 and the `[PADnnnnnn]`
  fillers) and that the merges are identical. If anything differs, build `tokenizer.json` from Edge0's own
  `vocab.json` and `merges.txt` with the `qwen35` pre-tokenizer regex instead.
- `tokenizer_config.json`: set `chat_template` to the contents of Edge0's `tokenizer/chat_template.jinja`, plus
  `eos_token "<|im_end|>"`, `pad_token "<|endoftext|>"`, `add_bos_token false`, and `tokenizer_class
  "Qwen2Tokenizer"`. Also copy the template file itself as `chat_template.jinja`.
- `generation_config.json`: `{"eos_token_id": [248046, 248044], "pad_token_id": 248044, "temperature": 1.0,
  "top_k": 20, "top_p": 0.95}`. The sampling values are the GGUF defaults; the bench overrides them with temperature 0.
- Check the rendered prompt with `enable_thinking=false`: it must end with
  `<|im_start|>assistant\n<think>\n\n</think>\n\n` (MODEL-SPEC 6).

### 1.4 Dtypes and safetensors encoding

Safetensors dtype strings: `BF16`, `F16`, `F32`, `I32`, `U8`, `F8_E4M3`. The converter may write files with its own
minimal writer (numpy has no bf16/fp8: write raw `uint16`/`uint8` bytes under those dtype names). Every file must load
with `safetensors.safe_open(path, framework="pt")` inside the vLLM image. That is the acceptance test (1.7).

### 1.5 Tensor catalogue (every tensor; `l` = layer 0..47; GDN layers are `l % 4 != 3`, QSA layers are `l % 4 == 3`)

Shapes are `[rows(out), ...]` in PyTorch order. "Q8" means a `U8 [N, K/32, 34]` tensor whose 34-byte blocks are GGUF
`block_q8_0` (`fp16 d` then `int8 qs[32]`, `value = d·q`), copied byte for byte from the source rows (after the row or
column-block permutations stated). "−1" means the stored F32 value is `gguf_value − 1.0` (vLLM's Gemma-style norms add
1 themselves; MODEL-SPEC 7).

**Global (shard 1)**

| Name | dtype | shape | Source / transform |
|---|---|---|---|
| `model.embed_tokens.weight` | BF16 | [248320, 2560] | `token_embd` Q8_0 → fp32 → bf16 (round to nearest even) |
| `lm_head.weight` | BF16 | [248320, 2560] | `output` Q8_0 → bf16 |
| `model.hyper_connection_mixer.hc_norm.weight` | F32 | [10240] | `output_hc_norm` −1 |
| `model.hyper_connection_mixer.input_mix_weight_down.weight` | BF16 | [320, 10240] | `output_hc_down` (GGUF [10240,320]: same bytes) |
| `model.hyper_connection_mixer.input_mix_weight_up.weight` | BF16 | [10240, 320] | `output_hc_up` |

**Every layer `l` (shard 1)**

| Name | dtype | shape | Source / transform |
|---|---|---|---|
| `model.layers.{l}.attn_hyper_connection.hc_norm.weight` | F32 | [10240] | `blk.l.hc_attn_norm` −1 |
| `model.layers.{l}.attn_hyper_connection.input_mix_weight_down.weight` | BF16 | [320, 10240] | `hc_attn_down` |
| `model.layers.{l}.attn_hyper_connection.input_mix_weight_up.weight` | BF16 | [10240, 320] | `hc_attn_up` |
| `model.layers.{l}.attn_hyper_connection.block_inject_weight.weight` | BF16 | [4, 10240] | `hc_attn_inject` |
| `model.layers.{l}.mlp_hyper_connection.{same four}` | same | same | `hc_ffn_*` (norm −1) |
| `model.layers.{l}.mlp.gate.weight` | BF16 | [512, 2560] | `ffn_gate_inp` (router) |
| `model.layers.{l}.mlp.shared_expert_gate.weight` | BF16 | [1, 2560] | `ffn_gate_inp_shexp` |
| `model.layers.{l}.mlp.shared_expert.gate_proj.weight` | Q8 U8 | [640, 80, 34] | `ffn_gate_shexp` |
| `model.layers.{l}.mlp.shared_expert.up_proj.weight` | Q8 U8 | [640, 80, 34] | `ffn_up_shexp` |
| `model.layers.{l}.mlp.shared_expert.down_proj.weight` | Q8 U8 | [2560, 20, 34] | `ffn_down_shexp` |
| `model.layers.{l}.mlp.latent_down.weight` | Q8 U8 | [14336, 80, 34] | `latent_dense_q8.bin` layer `l`, D rows 0..14335 (group g = rows 1792g..1792g+1791), bytes as-is |
| `model.layers.{l}.mlp.latent_up.weight` | Q8 U8 | [2560, 448, 34] | U_cat: row `r` = concat over g=0..7 of U row `2560g + r` (56 blocks each), bytes as-is |
| `model.layers.{l}.mlp.expert_group` | I32 | [512] | `labels.bin[l]` (values 0..7), indexed by **logical** expert id |

**GDN layers (36; shard 1).** Apply the v-head reorder `HF head i = 3k + r` ← `GGUF head 16r + k` (k < 16, r < 3) to
everything indexed by v-head (MODEL-SPEC 7).

| Name | dtype | shape | Source / transform |
|---|---|---|---|
| `model.layers.{l}.linear_attn.in_proj_qkv.weight` | Q8 U8 | [10240, 80, 34] | `attn_qkv`. Rows 0..4095 (q, k) as-is; v rows `4096+128i+d` ← `4096+128·src(i)+d` |
| `model.layers.{l}.linear_attn.in_proj_z.weight` | Q8 U8 | [6144, 80, 34] | `attn_gate`, row blocks of 128 reordered by v-head |
| `model.layers.{l}.linear_attn.in_proj_b.weight` | BF16 | [48, 2560] | `ssm_beta`, rows reordered |
| `model.layers.{l}.linear_attn.in_proj_a.weight` | BF16 | [48, 2560] | `ssm_alpha`, rows reordered |
| `model.layers.{l}.linear_attn.conv1d.weight` | F32 | [10240, 1, 4] | `ssm_conv1d` GGUF [4,10240] = numpy (10240,4) → reshape; channels 4096+ reordered like the v rows; tap 0 = oldest |
| `model.layers.{l}.linear_attn.A_log` | F32 | [48] | `log(-ssm_a)`, reordered |
| `model.layers.{l}.linear_attn.dt_bias` | F32 | [48] | `ssm_dt.bias`, reordered |
| `model.layers.{l}.linear_attn.norm.weight` | F32 | [128] | `ssm_norm` **as-is** (not folded) |
| `model.layers.{l}.linear_attn.out_proj.weight` | Q8 U8 | [2560, 192, 34] | `ssm_out`; in each row the 4-block group of v-head i (blocks 4i..4i+3) ← group of src(i). Lossless, since 128 columns = 4 whole blocks |

**QSA layers (12; shard 1).** The fused `qkv_proj` packs q, k, v and the BF16 indexer, and vLLM requires one quant
format per fused layer, so q/k/v are BF16.

| Name | dtype | shape | Source / transform |
|---|---|---|---|
| `model.layers.{l}.self_attn.q_proj.weight` | BF16 | [12288, 2560] | `attn_q` Q8_0 → bf16; per-head `[q 256 | gate 256]` halves kept |
| `model.layers.{l}.self_attn.k_proj.weight` | BF16 | [512, 2560] | `attn_k` → bf16 |
| `model.layers.{l}.self_attn.v_proj.weight` | BF16 | [512, 2560] | `attn_v` → bf16 |
| `model.layers.{l}.self_attn.o_proj.weight` | Q8 U8 | [2560, 192, 34] | `attn_output` |
| `model.layers.{l}.self_attn.q_norm.weight` | F32 | [256] | `attn_q_norm` −1 |
| `model.layers.{l}.self_attn.k_norm.weight` | F32 | [256] | `attn_k_norm` −1 |
| `model.layers.{l}.self_attn.indexer.index_qk_proj.weight` | BF16 | [640, 2560] | rows 0..511 `indexer.q_proj`, rows 512..639 `indexer.k_proj` |
| `model.layers.{l}.self_attn.indexer.q_layernorm.weight` | F32 | [128] | `indexer.q_norm` −1 |
| `model.layers.{l}.self_attn.indexer.k_layernorm.weight` | F32 | [128] | `indexer.k_norm` −1 |

**PLE (layer 1 only; shard 1).** HF `ple_layer_ids=[2]` is 1-based, and vLLM attaches it to `layer_idx == 1`.

| Name | dtype | shape | Source / transform |
|---|---|---|---|
| `model.layers.1.ple.key_proj.weight` | BF16 | [10240, 2560] | `ple_key` |
| `model.layers.1.ple.value_proj.weight` | BF16 | [2560, 2560] | `ple_value` |
| `model.layers.1.ple.norm_key.weight`, `norm_query.weight`, `norm_conv.weight` | F32 | [10240] | `ple_norm_*` −1 |
| `model.layers.1.ple.conv1d.weight` | F16 | [10240, 1, 4] | `ple_conv1d` GGUF [4,10240] → reshape like `ssm_conv1d` |
| `model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale` | F32 | [1] | `0.00019931793212890625` |

No `ngram_embedding.shard_*` tensors are written. The table stays in `ple-fp8.gguf` and is located by the plugin (2.7).
No `layer_multipliers` / `ngram_heads_*` buffers are written either; vLLM rebuilds them from `seed` and the config.

**Routed experts (shard `l+2`; ModelOpt NVFP4, stacked over 512 logical experts in logical order)**

| Name | dtype | shape | Meaning |
|---|---|---|---|
| `model.layers.{l}.mlp.experts.w13_weight` | U8 | [512, 1280, 896] | rows 0..639 = gate (w1), rows 640..1279 = up (w3); 1792 inputs per row; byte m = code(2m) low nibble, code(2m+1) high nibble |
| `model.layers.{l}.mlp.experts.w13_weight_scale` | F8_E4M3 | [512, 1280, 112] | one E4M3 scale per 16 inputs, linear (unswizzled) |
| `model.layers.{l}.mlp.experts.w13_weight_scale_2` | F32 | [512, 2] | `(s_gate, s_up)` from the record tail; must be equal (assert) |
| `model.layers.{l}.mlp.experts.w2_weight` | U8 | [512, 1792, 320] | down: 1792 output rows × 640 inputs |
| `model.layers.{l}.mlp.experts.w2_weight_scale` | F8_E4M3 | [512, 1792, 40] | |
| `model.layers.{l}.mlp.experts.w2_weight_scale_2` | F32 | [512] | `s_down` from the tail |

Dequantized value = `E2M1(code) × e4m3(scale) × scale_2`. That is exactly MODEL-SPEC 5.1's
`E2M1 × ue4m3(d) × s_proj`. **No `input_scale` is written** (W4A16).

Repack of one source row (lossless; source record layout in MODEL-SPEC 5.1). The source row is `K/64` ggml
`block_nvfp4` blocks of 36 B (`d[4]`, then `qs[32]`):

```
for block b, sub-block s in 0..3, j in 0..15:              # element k = 64b + 16s + j
    code[k] = (qs[8s + (j % 8)] >> (4 * (j // 8))) & 0xF     # E2M1 code incl. sign bit 3, same encoding in both formats
    scale[4b + s] = d[s]                                     # already an E4M3 bit pattern
out_row[m] = code[2m] | (code[2m+1] << 4)
```

Assert that no scale byte has bit 7 set and none equals `0x7F`. If either ever occurs, map it to `0x00` and count it
in `edge0_convert.json`.

**MTP (last shard).** Names come from `mtp/dense.txt` with the prefix `mtp.` and the layer index 0
(`mtp.layers.0.…` for everything under `self_attn.`, `mlp.`, `attn_hyper_connection.` and
`mlp_hyper_connection.`). vLLM's `_remap_mtp_weight_name` maps `mtp.` → `model.`.

| Name | dtype | shape | Source / transform |
|---|---|---|---|
| `mtp.fc_embedding.weight`, `mtp.fc_hidden.weight` | Q8 U8 | [2560, 80, 34] | dense.txt q8_0 rows as-is |
| `mtp.pre_fc_norm_embedding.weight` | F32 | [2560] | −1 (Strata stores 1+w) |
| `mtp.pre_fc_norm_hidden.weight` | F32 | [10240] | −1 |
| `mtp.hyper_connection_mixer.{hc_norm F32 [10240] −1, input_mix_weight_down BF16 [320,10240], input_mix_weight_up BF16 [10240,320]}.weight` | | | as-is; no `block_inject_weight` |
| `mtp.layers.0.{attn,mlp}_hyper_connection.{hc_norm (−1), input_mix_weight_down, input_mix_weight_up, block_inject_weight}.weight` | as in main layers | | |
| `mtp.layers.0.self_attn.q_proj.weight` | Q8 U8 | [12288, 80, 34] | |
| `mtp.layers.0.self_attn.{k_proj,v_proj}.weight` | Q8 U8 | [512, 80, 34] | |
| `mtp.layers.0.self_attn.o_proj.weight` | Q8 U8 | [2560, 192, 34] | |
| `mtp.layers.0.self_attn.indexer.index_qk_proj.weight` | Q8 U8 | [640, 80, 34] | (Q8_0 in the MTP, so the fused qkv_proj is uniformly Q8) |
| `mtp.layers.0.self_attn.{q_norm,k_norm}.weight` | F32 | [256] | −1 |
| `mtp.layers.0.self_attn.indexer.{q_layernorm,k_layernorm}.weight` | F32 | [128] | −1 |
| `mtp.layers.0.mlp.gate.weight` | BF16 | [512, 2560] | |
| `mtp.layers.0.mlp.shared_expert_gate.weight` | BF16 | [1, 2560] | |
| `mtp.layers.0.mlp.shared_expert.{gate_proj,up_proj}.weight` / `down_proj.weight` | Q8 U8 | [640, 80, 34] / [2560, 20, 34] | |
| `mtp.layers.0.mlp.experts.{e}.{gate_proj,up_proj}.weight` | U8 | [640, 1280] | NVFP4 re-quantized from Q2_0 (below), ModelOpt nibble order |
| `mtp.layers.0.mlp.experts.{e}.{gate_proj,up_proj}.weight_scale` | F8_E4M3 | [640, 160] | |
| `mtp.layers.0.mlp.experts.{e}.down_proj.weight` | U8 | [2560, 320] | |
| `mtp.layers.0.mlp.experts.{e}.down_proj.weight_scale` | F8_E4M3 | [2560, 40] | |
| `mtp.layers.0.mlp.experts.{e}.{gate_proj,up_proj,down_proj}.weight_scale_2` | F32 | [] (0-dim) | gate and up of one expert share one value |
| `mtp.layers.0.mlp.experts.{e}.{gate_proj,up_proj,down_proj}.input_scale` | F32 | [] | 1.0 (unused by W4A16; present so the stock loader finds it) |

MTP expert re-quantization: decode Q2_0 with `strata_formats.read_mtp_expert` to float32 values in `{−d, 0, d, 2d}`
(MODEL-SPEC 5.6). Then quantize to NVFP4 the standard ModelOpt way. The per-expert global scale is
`amax / (6·448)`, shared by gate and up. The per-16 E4M3 block scale is `amax16 / (6·global)`, and the codes are
round-to-nearest E2M1. Report the max and mean relative error. MTP numerics only change the draft acceptance rate,
never the output.

### 1.6 Contract notes

- `embed_tokens` and `lm_head` are BF16, not Q8, because `Qwen4ExpMTP` builds its own BF16 `embed_tokens` and
  `lm_head` from the same checkpoint names. Making the head Q8 later is a tuning lever worth 0.55 GiB (4.3 L5).
- The skeleton's F32 tensors are byte-identical in `dense.bin`; either source is fine. Never read
  `dense.bin`'s `ffn_*_exps.scale/input_scale`; they are stale (MODEL-SPEC 5.4).
- Expected sizes: shard 1 is about 11.3 GB (2.2 GDN Q8, 3.7 D/U Q8, 2.5 embedding and head BF16, 1.5 HC and other
  BF16, 0.9 QSA qkv BF16, 0.5 the rest), each expert shard is 990,910,464 B of tensor data, the MTP shard is about
  1.5 GB, and the whole checkpoint is about 60 GB plus the 51 GB hardlink (which takes no new space). The disk has
  262 GB free.

### 1.7 Converter acceptance checks (written to `edge0_convert.json`; a failure aborts)

1. `edge0/convert/check_ckpt.py --ckpt DIR` validates every name, shape and dtype in 1.5 against the index and the
   shard headers, using the shard rule in 1.1. The model agent reuses this script.
2. Lossless: Q8 blocks are byte-equal to the source rows for a sample of rows of every tensor kind. For layers 0, 23
   and 47 (all experts) plus 64 random (l, e) pairs, the NVFP4 decode of the converted tensors equals
   `EdgeModel.expert(l, e)` exactly (float32 equality).
3. GDN reorder: a numpy single-token GDN step on random input gives the same output with the GGUF tensors (tiled
   pairing `j % 16`) and with the converted tensors (grouped pairing `i // 3`).
4. Latent MoE: for 3 layers and 4 random x, a numpy re-implementation of D5 on the converted tensors matches
   `latent_moe_reference` (max relative error < 1e-5).
5. Tokenizer: id↔token equality (1.3). The chat template renders the thinking-off prompt, and `encode()` equals
   Strata's token ids for the prompts in `strata-ref.json` once that file exists.
6. Every shard opens with `safe_open(framework="pt")` in a container of the vLLM image (no `--gpus`).

### 1.8 Mini checkpoint for parallel bring-up: `/home/andychen/models/Edge0-145B-vllm-L4`

Same contract with `num_hidden_layers: 4` and `layer_types` = the first 4 entries (GDN, GDN, GDN, QSA; PLE on layer
1). Shards are `model-00001-of-00005` (dense) and `00002..00005` (experts of layers 0..3). There is no MTP and no
`mtp.*` keys in `quantized_layers`; the PLE hardlink and the tokenizer files are present. Its outputs are meaningless
text, but it exercises every code path in about 4.5 GB. **Deliver this first.**

---

## 2. Plugin package `edge0_vllm` (`edge0/plugin/`)

### 2.1 Layout

```
edge0/plugin/
  pyproject.toml          # name "edge0-vllm"; [project.entry-points."vllm.general_plugins"] edge0 = "edge0_vllm:register"
  edge0_vllm/
    __init__.py           # register(): idempotent; ModelRegistry.register_model("Edge0ForCausalLM",
                          #   "edge0_vllm.model:Edge0ForCausalLM") (lazy string form); import quant (registers "edge0");
                          #   ple.install_patch(); trace/adapt runner hooks when enabled. No CUDA init at import.
    config.py             # Edge0Settings: parses hf_text_config.edge0 + EDGE0_* env (2.9); logs the resolved values
    quant.py              # Edge0QuantConfig(ModelOptMixedPrecisionConfig), @register_quantization_config("edge0")
    q8.py                 # Q8_0 block <-> torch helpers (dequant on GPU, fp32 math, bf16 output)
    model.py              # Edge0ForCausalLM(Qwen4ExpForCausalLM)
    latent_moe.py         # Edge0LatentMoeBlock + the custom op vllm::edge0_latent_moe
    expert_pool.py        # ExpertPool: hot (VRAM) / cold (pinned mapped host) Marlin tensors, slot tables, swap()
    hostmem.py            # host allocators: host_register (default) | host_alloc | vmm_host_numa
    marlin_nvfp4.py       # per-expert Marlin repack + scale/global-scale processing with one factor per layer
    kernels.py            # Triton: route gather, deterministic latent combine, touch counter
    placement.py          # plan JSON / profile.bin top-H -> per-layer order + n_hot; budget math
    ple.py                # discover_table_layout patch (GGUF table); phase 2: parallel row reader
    trace.py              # routing trace + per-step hot/cold touch and byte counters (graph-safe GPU counters)
    adaptive.py           # phase 3: adaptive expert cache (port of TidalMoE _AdaptiveCache)
    stream.py             # phase 3: copy-engine staging ring for large-M (prefill) steps
  tests/                  # pytest; GPU tests skip without CUDA. test_q8, test_repack, test_moe_vs_oracle, test_placement
```

The package is pure Python plus Triton and builds no CUDA extension. It uses vLLM's compiled ops (`gptq_marlin_repack`,
`moe_wna16_marlin_gemm` through `fused_marlin_moe`, `topk_softmax`, b12x).

### 2.2 What is reused and what is new

| Part | Source |
|---|---|
| Decoder loop, embeddings, HC (`GatedResidual`, CuTe kernels), GDN (`QwenGatedDeltaNetAttention`, FlashInfer/FLA kernels), QSA attention + indexer + Triton kernels + paged KV, PLE layer + n-gram hashing + `Qwen4ExpModelState` + file-gather backend, logits, sampler, scheduler, hybrid KV/state manager, CUDA graphs, torch.compile | vLLM `models/qwen4_exp/nvidia/*`, unchanged |
| MTP drafter (phase 2) | vLLM `Qwen4ExpMTP` + `FusedMoE` (`ModelOptNvFp4FusedMoE`, W4A16 → `MarlinExperts`), unchanged |
| Shared expert | vLLM `Qwen3NextMLP(..., expert_gate=shared_expert_gate)`, same names as stock |
| Router | vLLM `GateLinear` (fp32 logits) + `topk_softmax` (softmax over 512, top-10, renormalize) |
| D / U projections | vLLM `ReplicatedLinear` (prefixes `mlp.latent_down`, `mlp.latent_up`) with the `edge0` quant config |
| Expert GEMMs | vLLM `fused_marlin_moe` (`model_executor/layers/fused_moe/experts/marlin_moe.py:226`) with `expert_map`, plus `ops.gptq_marlin_repack` and `marlin_utils_fp4` helpers |
| Q8 dense in q8 mode | vLLM `ModelOptBlockQuantLinearMethod("q8_0")` (b12x, `quantization/modelopt_block_quant.py`) |
| **New** | `Edge0ForCausalLM`, `Edge0LatentMoeBlock`, expert pools/placement/host memory, `Edge0QuantConfig` and the Q8→BF16 load path, the PLE layout patch, Triton gather/combine, traces, adaptive cache |

### 2.3 `Edge0ForCausalLM` (`model.py`)

```python
class Edge0ForCausalLM(Qwen4ExpForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        import vllm.models.qwen4_exp.nvidia.model as qm
        with swap_attr(qm, "Qwen4ExpSparseMoeBlock", Edge0LatentMoeBlock):   # Qwen4ExpDecoderLayer looks the class up
            super().__init__(vllm_config=vllm_config, prefix=prefix)          # in its module globals at construction
        self.set_moe_parameters(self.model.layers)   # re-run outside the swap -> 0 FusedMoE layers (no EPLB metadata)
```

- While the swap is active, `Qwen4ExpModel.__init__` and `set_moe_parameters` see the Edge0 block as the MoE class.
  The block therefore exposes `n_shared_experts=1`, `n_logical_experts = n_physical_experts =
  n_local_physical_experts = n_routed_experts = 512`, `n_redundant_experts=0`, `is_fused_shared_expert_enabled=False`
  and an `experts` attribute with a `moe_config`-like stub. `is_model_fused_shared_expert_compatible` must come out
  False, so the shared-expert names are not rewritten (`fused_moe/utils.py:133`).
- `load_weights(weights)`: names matching `model.layers.{l}.mlp.experts.*` go to
  `self.model.layers[l].mlp.load_expert_tensor(name, tensor)`. Every other name goes to `super().load_weights`, which
  already drops `mtp.*`. Return the union of loaded names. If the strict "weights not initialized" check fires, also
  add `model.layers.1.ple.ple_embedding.ngram_embedding.weight`, which the GGUF table serves.
- `process_weights_after_loading()` (the model-level hook, `model_loader/utils.py:209`, runs before memory profiling
  and graph capture): it finalizes the expert pools (global scales, slot tables, workspaces) and moves
  `model.embed_tokens.weight` to pinned mapped host memory (`EDGE0_EMBED_HOST=1`; gather-only, about 64 rows per decode
  step). It also starts the adaptive cache and the trace when enabled, then logs the VRAM and RAM breakdown.
  `Qwen4ExpForCausalLM` does not define this hook, so it is added here, not wrapped.

### 2.4 `Edge0QuantConfig` (`quant.py`)

- `get_name() == "edge0"`. `override_quantization_method` returns `"edge0"` when `quant_method == "edge0"`.
  `from_config` builds the parent `ModelOptMixedPrecisionConfig` state (sub-configs for `W4A16_NVFP4`, the
  `quantized_layers` dict) from `quantization_config.quantized_layers`.
- `get_quant_method(layer, prefix)`:
  - Q8_0 prefixes (`_resolve_quant_algo` resolves fused layers: `in_proj_qkvz` = [`in_proj_qkv`, `in_proj_z`],
    `gate_up_proj` = [`gate_proj`, `up_proj`], the MTP `qkv_proj` = [q, k, v, `indexer.index_qk_proj`]):
    - `EDGE0_DENSE=bf16`: `Edge0Q8DequantLinearMethod`. It is a subclass of `UnquantizedLinearMethod` that registers
      the normal BF16 `[N, K]` weight but wraps its `weight_loader`: an incoming `U8 [n, K/32, 34]` shard is
      dequantized on the GPU (fp32 math, bf16 result) and handed to the original loader. No uint8 copy stays
      resident. Because it is an `UnquantizedLinearMethod`, `enable_qwen4_exp_low_latency_gemm` treats it like any
      BF16 layer.
    - `EDGE0_DENSE=q8`: the parent's path, `ModelOptBlockQuantLinearMethod("q8_0")` (b12x, BF16 activations; needs
      the b12x linear backend on SM12x).
  - `W4A16_NVFP4` (the MTP experts): the parent's `ModelOptNvFp4FusedMoE` with the W4A16 sub-config, so FusedMoE picks
    Marlin.
  - Everything else (HC, router, `in_proj_ba`, PLE kv, QSA `qkv_proj`, embedding, LM head): unquantized BF16.
  - `Edge0LatentMoeBlock`'s expert pools never consult the quant config.
- The draft (MTP) gets the same config through `get_draft_quant_config`. `quantized_layers` must stay a public dict
  attribute so `_remap_quantized_layers` can rewrite `mtp.layers.0` → `mtp.layers.48`.

### 2.5 `Edge0LatentMoeBlock` (`latent_moe.py`)

Constructor signature `(vllm_config, prefix, reduce_results=True)`, the same as `Qwen4ExpSparseMoeBlock`. The layer
index comes from `extract_layer_index(prefix)`. Modules:
`gate = GateLinear(2560→512, fp32 out)`, `latent_down = ReplicatedLinear(2560→14336)`,
`latent_up = ReplicatedLinear(14336→2560)`, `shared_expert = Qwen3NextMLP(2560, 640, expert_gate=shared_expert_gate)`,
`shared_expert_gate`, plus an `ExpertPool` and the buffers `expert_group[512] int32` and `slot_of[512] int32`. The
block registers itself in `compilation_config.static_forward_context[prefix]`, as `FusedMoE` does.

```python
def forward(self, hidden_states, already_sequence_parallel=False):
    x = hidden_states.view(-1, 2560)
    logits = self.gate(x)                                    # [T,512] fp32
    xl = self.latent_down(x)                                 # [T,14336] = D_cat x (all 8 groups)
    z = torch.ops.vllm.edge0_latent_moe(xl, logits, self.prefix)   # [T,14336] weighted per-group sums
    return (self.latent_up(z) + self.shared_expert(x)).view_as(hidden_states)
```

Custom op `vllm::edge0_latent_moe` (registered with `vllm.utils.torch_utils.direct_register_custom_op`, plus a fake
impl that returns `empty_like(xl)`). Dynamo treats it as opaque. It is captured inside CUDA graphs for decode sizes.
The steps:

1. `ids, w = topk_softmax(logits, 10, renormalize=True)` (logical ids int32 [T,10], fp32 weights; the 2^-14 clamp of
   the renormalization is irrelevant in practice).
2. `slot = slot_of[ids]` and `grp = expert_group[ids]`. The router is not permuted, so routing is exactly the
   model's.
3. `x_routes[t*10+j] = xl.view(T,8,1792)[t, grp[t,j]]` (Triton gather, [T·10, 1792] bf16).
4. Hot/cold split. The side stream waits on the main stream. The **cold call** runs on the side stream:
   `fused_marlin_moe(x_routes, cold.w13, cold.w2, None, None, cold.s13, cold.s2, w.view(-1,1), slot.view(-1,1),
   float4_e2m1f.id, global_num_experts=512, expert_map=cold_map, global_scale1=cold.g13, global_scale2=cold.g2,
   workspace=cold_ws, ...)`. The **hot call** runs on the main stream with the hot pool, `hot_map` and its own
   workspace. Then the main stream joins the side stream. `hot_map[s] = s if s < n_hot else -1`;
   `cold_map[s] = s - n_hot if s >= n_hot else -1`. `moe_align_block_size(..., ignore_invalid_experts=True)` skips the
   other tier's routes. The route weight is applied in GEMM2 (`mul_topk_weights_in_gemm2=True`).
5. Combine (Triton, deterministic, no atomics): for each (t, g), `z[t,g,:] = Σ_{j: grp[t,j]=g, j ascending}
   Y[t*10+j]`, where `Y` comes from the hot output when `slot < n_hot` and from the cold output otherwise. It never
   relies on Marlin zero-filling skipped rows.
6. When `EDGE0_TRACE` is on: a GPU counter add of distinct hot/cold slots touched and a ring-buffer write of the
   logical ids (graph-safe, no host sync).

CUDA-graph rules: all workspaces, intermediate caches and outputs of both Marlin calls are preallocated per block
for `max_num_batched_tokens·10` routes and passed explicitly. The hot and cold calls never share a Marlin
`workspace`, because the split-K locks would race. There is no host synchronization and no data-dependent allocation
inside the op. A layer with `n_hot ∈ {0, 512}` makes a single call. Steps with more than `EDGE0_STREAM_MIN_TOKENS`
tokens (always larger than the largest captured graph size) may take the phase-3 streaming path in `stream.py`.

### 2.6 Expert pools and loading (`expert_pool.py`, `marlin_nvfp4.py`, `hostmem.py`)

Marlin layout per expert (`_repack_marlin_experts`, `marlin_utils_fp4.py:303`; no N padding is needed because K=1792
is a multiple of 128 and N=640 a multiple of 64):

| Tensor | per-expert shape | bytes |
|---|---|---|
| `w13` (gptq_marlin_repack, size_k=1792, size_n=1280) | int32 [112, 2560] | 1,146,880 |
| `w2` (size_k=640, size_n=1792) | int32 [40, 3584] | 573,440 |
| `s13` (marlin-permuted, S0E5M3 fp8) | [112, 1280] | 143,360 |
| `s2` | [40, 1792] | 71,680 |
| total = one **slot** | | **1,935,360 B (1.846 MiB)** |
| `g13`, `g2` (processed global scales) | fp32 [E] per pool | always in VRAM |

- Per layer, the pool has `hot` tensors `[n_hot, …]` in VRAM (allocated in the block constructor, so vLLM counts
  them as weights) and `cold` tensors `[512 − n_hot, …]` in host memory mapped for the device (UVA).
- `hostmem.host_register` (default) allocates each cold tensor as an anonymous `mmap` (`MAP_POPULATE`,
  `madvise(MADV_HUGEPAGE)`), then calls `cudaHostRegister(ptr, size, Mapped|Portable)`, and wraps the device
  pointer as a CUDA tensor via `__cuda_array_interface__`, with a CPU view for host writes. **Never use
  `torch.empty(pin_memory=True)`**: torch's caching host allocator rounds sizes up to a power of two, which would
  waste up to ~2x the RAM. The other backends are `host_alloc` (`cudaHostAlloc` Mapped) and `vmm_host_numa`
  (`cuMemCreate` HOST_NUMA, TidalMoE's `_alloc_segments`). `edge0/tools/gpu_host_probe.py` picks the default by
  measuring zero-copy read bandwidth for 1.85 MB records scattered over a 30 GB region.
- Loading. `load_expert_tensor` receives each full stacked tensor in logical order from vLLM's safetensors iterator.
  It processes the tensor in chunks of 64 experts: copy the chunk to the GPU, Marlin-repack each expert, and write it
  to `slot_of[e]` in the hot (VRAM) or cold (host, through the UVA view) tensor. For scales, the **layer-wide Marlin
  factor** `_nvfp4_compute_scale_factor` is computed over all 512 experts' `w13_weight_scale` (and over `w2`) when
  that tensor arrives. Every expert is processed with that one factor, so swaps between pools stay valid. The
  `*_scale_2` tensors are kept raw and finalized (`nvfp4_marlin_process_global_scale(...)/factor`) in the model-level
  hook. Peak transient memory is one stacked tensor in RAM (≤ 0.55 GiB) plus a 64-expert GPU chunk (≤ 120 MB).
- Note for reviewers: `nvfp4_marlin_process_scales` zeroes block scales below `2^-6/factor`. The converter reports,
  per layer, the fraction of E4M3 scales below `2^-6` (expected to be negligible).
- `ExpertPool.swap(layer, hot_idx, cold_idx)` exchanges the rows of `w13/w2/s13/s2/g13/g2` between the two pools and
  updates `slot_of` (and the `order` list kept on the host). Addresses never change, so CUDA graphs stay valid. A swap
  must run on the engine thread between steps.

### 2.7 PLE (`ple.py`)

- Run with `--engram-config '{"cpu_offload": false, "host_file_gather": true}'`. That selects
  `Qwen4ExpPLEFileGatherEmbedding` (`nvidia/ngram_embedding.py:467`). Each step, `prepare_inputs` computes the n-gram
  ids on the CPU (the same hash as Strata; constants rebuilt from `seed 1234`), reads each distinct row with
  `preadv` from the page cache after `posix_fadvise(WILLNEED)`, and copies the rows host-to-device into fixed staging
  buffers. The forward pass only reads staging, so the step is graph-safe and the table is never allocated.
- Patch: replace `vllm.models.qwen4_exp.nvidia.ngram_embedding.discover_table_layout` (imported there from
  `ple_pageable`, looked up at call time in `bind_storage_after_loading`). For `suffix == "weight"` it returns
  `TableLayout(num_rows=320001536, row_bytes=160, rows_per_shard=320001536,
  shards=(ShardLocation(<ckpt>/ple-fp8.gguf, 352, 320001536),))` and defers everything else to the original. The FP8
  method's `weight_scale` loads from the checkpoint tensor in 1.5.
- Phase 2, before the first benchmark: replace `_FilePlane.read` (serial `preadv`) with a thread-pool reader (16
  threads, `os.preadv` releases the GIL), optionally O_DIRECT, plus a pinned row LRU (`EDGE0_PLE_CACHE_MB`, default
  1024 ≈ 6.7M rows). Strata does the same (`--ple-io direct`, 24 workers, a 1M-row LRU).

### 2.8 MTP (phase 2)

`--speculative-config '{"method": "mtp", "num_speculative_tokens": 3, "draft_confidence_threshold": 0.5,
"draft_token_map": "/models/Edge0-145B-vllm/draft_vocab.json"}'` is Strata's `--spec 4 --spec-min-p 0.5` with
FR-Spec over the 106K draft vocabulary. `config/speculative.py` rewrites the `qwen4_exp_text` draft config to
`Qwen4ExpMTP`. Check that the draft **shares** the target's `embed_tokens` and `lm_head` (a private BF16 copy would
cost 2.4 GiB of VRAM); if it does not, drop the draft's copies after loading and alias them to the target's. The
experts take 1.32 GiB of VRAM as NVFP4, all resident.

### 2.9 Runtime knobs (all read once in `config.py` and logged)

| Env | Default | Meaning |
|---|---|---|
| `EDGE0_DENSE` | `bf16` | `bf16`: dequantize Q8 at load. `q8`: b12x native Q8_0 |
| `EDGE0_EXPERT_VRAM_GIB` | set by `serve.sh` | VRAM for hot expert slots: H = floor(GiB·2^30 / 1,935,360). `0` = fallback F0 |
| `EDGE0_PLACEMENT` | `profile` | `profile` (top-H of `profile.bin`), `plan:<json>`, or `identity` (control: logical order, H split evenly) |
| `EDGE0_COLD_ALLOC` | `host_register` | `host_register` / `host_alloc` / `vmm_host_numa` |
| `EDGE0_SPLIT` | `1` | hot and cold Marlin calls on two streams; `0` = sequential (A/B) |
| `EDGE0_MOE_IMPL` | `marlin` | `torch` = eager reference (F1) |
| `EDGE0_EMBED_HOST` | `1` | `embed_tokens` in mapped host memory |
| `EDGE0_TRACE` | `0` | `bw` (per-step hot/cold touches and bytes, logged every 10 s), `routes` (logical ids to `/root/.cache/edge0-vllm/traces/*.npy`) |
| `EDGE0_ADAPT`, `EDGE0_ADAPT_*` | `0` | phase 3: adaptive cache; knobs mirror TidalMoE's `MOE_ADAPT_*` (epoch 3 s, ≤64 swaps, half-life 10, z=3, `count=touch`, save/resume) |
| `EDGE0_STREAM_MIN_TOKENS` | `0` (off) | phase 3: steps with at least this many tokens stream the cold experts through a VRAM ring |
| `EDGE0_COLD_SCALES` | `host` | phase 3: `vram` keeps the cold experts' scales (11% of a slot) in VRAM |
| `EDGE0_PLE_IO`, `EDGE0_PLE_CACHE_MB` | `vllm`, 1024 | phase 2: `threads` reader + row LRU |
| `EDGE0_LOG_DIR` | `/root/.cache/edge0-vllm/logs` | plugin log mirror (TidalMoE-style file handler) |

### 2.10 Docker image and `serve.sh`

- `edge0/docker/Dockerfile`: `FROM nvcr.io/nvidia/vllm:26.11-py3`, `COPY plugin /opt/edge0/plugin`,
  `RUN pip install --no-deps /opt/edge0/plugin` (with `PIP_CONSTRAINT` respected; never touch torch, vllm or
  flashinfer). Tag `edge0-vllm:dev`. Build context: `/home/andychen/work/edge0-vllm`.
- `edge0/serve.sh`: refuses to start unless `/home/andychen/work/edge0-vllm-GPU_FREE` exists. It runs the container
  `edge0-vllm` in tmux session `edge0-vllm-serve` with the shared-context `docker run` flags, `DRY_RUN=1` prints the
  command, and extra arguments go to `vllm serve`. `EDGE0_DEV=1` adds `-e PYTHONPATH=/workspace/edge0/plugin`, so the
  bind-mounted source shadows the installed copy (no rebuild while iterating; the installed dist-info still provides
  the entry point). Bring-up command:

```
vllm serve /models/Edge0-145B-vllm --served-model-name edge0_145b --host 0.0.0.0 --port 8125 \
  --quantization edge0 --dtype bfloat16 --max-model-len 32768 --max-num-seqs 16 --max-num-batched-tokens 4096 \
  --kv-cache-memory-bytes $KV_BYTES --mamba-ssm-cache-dtype float32 --gpu-memory-utilization 0.96 \
  --engram-config '{"cpu_offload": false, "host_file_gather": true}' --generation-config auto
# env: EDGE0_DENSE=bf16 EDGE0_EXPERT_VRAM_GIB=9.5 EDGE0_PLACEMENT=profile ; first run adds --enforce-eager
```

The served name `edge0_145b` matches `edge0/bench/bench-openai.sh`. For vLLM the bench passes
`EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}'` and `BASE_URL=http://127.0.0.1:8125`. Its
`tokenizer-qwen38` is fine, since it has the same vocabulary.

---

## 3. Memory plan (32 GB VRAM, 60 GiB RAM)

### 3.1 VRAM (GiB; usable ≈ 31.1 after the CUDA context; computed from the tensor catalogue)

| Item | Bring-up (`bf16`, no MTP, 16 seqs) | Target (`q8`, MTP, 32 seqs) | Target (`q8`, MTP, 64 seqs) |
|---|---:|---:|---:|
| GDN `in_proj_qkvz` + `out_proj` (36 layers) | 3.87 | 2.05 | 2.05 |
| QSA `o_proj` / `qkv_proj` (BF16 incl. indexer) | 0.35 / 0.80 | 0.19 / 0.80 | 0.19 / 0.80 |
| Shared experts | 0.44 | 0.23 | 0.23 |
| D/U (`latent_down` + `latent_up`, 48 layers) | 6.56 | 3.49 | 3.49 |
| `lm_head` BF16 (`embed_tokens` on host) | 1.18 | 1.18 | 1.18 |
| BF16-native (HC 1.19, router 0.12, PLE kv, GDN a/b, F32) | 1.40 | 1.40 | 1.40 |
| **Dense total** | **14.60** | **9.34** | **9.34** |
| MTP (dense Q8 0.11 + NVFP4 experts 1.32) | 0 | 1.43 | 1.43 |
| Activations + Marlin workspaces + CUDA graphs + slack | 3.0 | 3.0 | 3.0 |
| `--kv-cache-memory-bytes`: GDN+conv+PLE state (116 MB/seq fp32, 59 MB/seq bf16) + QSA KV (24 KiB/token bf16, 12 KiB fp8) | 3.27 (fp32 state, 64K tok bf16) | 2.34 (bf16 state, 48K tok fp8) | 4.68 (bf16 state, 96K tok fp8) |
| **Hot experts (`EDGE0_EXPERT_VRAM_GIB`)** | **10.2 → H ≈ 5,680 (23%)** | **15.0 → H ≈ 8,320 (34%)** | **12.7 → H ≈ 7,020 (29%)** |

The numbers are planning estimates [inferred]. The reserve and KV rows must be calibrated from vLLM's own log at
first light (it prints the profiled peak and "GPU KV cache size … maximum concurrency"). Then set
`EDGE0_EXPERT_VRAM_GIB` to the leftover minus 0.5 GiB. Concurrency and hot experts trade against each other at about
**40 hot slots per extra sequence** (59 MB bf16 state plus 1.5K tokens of fp8 KV = 78 MB); lever L7 in 4.3 decides the split by measurement.

### 3.2 Host RAM (GiB; about 56 usable once Strata is stopped)

| Item | Bring-up | Target (32 seqs) |
|---|---:|---:|
| Cold expert pool (24,576 − H slots × 1.846 MiB), pinned and mapped | 34.1 | 29.3 |
| `embed_tokens` BF16, pinned and mapped | 1.2 | 1.2 |
| vLLM engine-core and API-server processes (python, torch, CUDA host state, tokenizer) | ~6 | ~6 |
| Loader transient (one stacked expert tensor) | 0.6 | 0.6 |
| PLE row LRU (phase 2) | 0 | 1.0 |
| OS, docker, idle services | ~2 | ~2 |
| **Committed** | **~44** | **~40** |
| Left for the page cache (PLE rows, safetensors during load) | ~12 | ~16 |

Pinned memory needs `--ulimit memlock=-1` (already in the run flags). The cold pool shrinks one-for-one as VRAM for
experts grows. Never keep a second host copy of an expert: the checkpoint is only page cache during loading.

### 3.3 Hot/cold placement

- **Default (`EDGE0_PLACEMENT=profile`)**: take the first H `(layer, expert)` pairs of `profile.bin` (best first;
  `strata_formats.read_profile`). Per layer, `n_hot[l]` = the count of chosen pairs in layer l. Slot order: the
  chosen experts of the layer by rank, then the rest by rank. This reproduces Strata's `--expert-cache auto` fill
  order, a fair baseline. The first 4096 pairs already spread 40 to 158 per layer.
- **Plan file** (`EDGE0_PLACEMENT=plan:<path>`, written by the phase-3 planner and by the adaptive cache's save):

```json
{"format": "edge0-placement-v1", "num_layers": 48, "num_experts": 512, "slot_bytes": 1935360,
 "source": "profile.bin top-6000 | trace:<file> B=<tokens/step> | adaptive epoch N",
 "layers": {"0": {"order": [512 logical ids; physical slot i holds order[i]], "n_hot": 117, "score": [optional]}, "...": {}}}
```

  Slots `< n_hot` live in VRAM. `slot_of = inverse(order)`. The plugin checks that `sum(n_hot)·slot_bytes` fits
  `EDGE0_EXPERT_VRAM_GIB`.
- No router permutation and no VMM single-range requirement. Expert permutation is done through `slot_of`. A swap
  updates `slot_of` and the pool rows (2.6).

### 3.4 First-light fallbacks

- **F0, all cold** (`EDGE0_EXPERT_VRAM_GIB=0`): the same loader writes all 512 experts of every layer into the host
  pool (44.3 GiB pinned). That fits only with the bring-up process sizes, so check `MemAvailable` first and set
  `EDGE0_EMBED_HOST=0` (the embedding fits in VRAM when no expert is hot). One cold Marlin
  call per layer. It is slow (about 0.9 GB of PCIe reads per decode token) but has the fewest moving parts. Use it for
  the first correctness run.
- **F1, reference** (`EDGE0_MOE_IMPL=torch`, `--enforce-eager`): the expert part runs per route in plain torch, from a
  ModelOpt-layout copy of the experts that the loader keeps for the first `EDGE0_REF_LAYERS` layers (default 4; the
  other layers stay on Marlin). This isolates repack or layout bugs from routing or latent bugs.
- vLLM's own `--offload-backend uva` is **not** a fallback for the routed experts (section 6). It remains available
  for experiments on dense parameters.

---

## 4. Tuning plan: less expert-load time per step under concurrency

### 4.1 Cost model (what we minimize)

Decode step time is roughly `max(hot compute, cold bytes / BW_zero_copy) + dense + attention` per layer, summed over
48 layers. Here `cold bytes = Σ_l (distinct cold slots touched in layer l) × 1.85 MB` (1.64 MB with VRAM-resident
scales). The cost unit is **distinct experts touched per step** (TidalMoE's `touch`), not selections. Under uniform
routing [inferred upper bound], the distinct experts per layer per step for B tokens are
`512·(1 − (1 − 10/512)^B)`: 10 (B=1), 76 (B=8), 238 (B=32), 366 (B=64), 470 (B=128). Real routing is skewed, so the
counts are lower. Consequences:
- At high concurrency, touch saturates, so **throughput grows almost linearly with tokens per step**: larger
  batches, and MTP verify tokens (nearly free once touch saturates).
- At conc 1, each token pays about 10 experts per layer, so the hit rate by touch is everything. MTP helps only if
  consecutive tokens share experts (measure it).
- Marlin's large-M path re-reads weights per M-block. Prefill should therefore stage cold experts in VRAM once
  rather than read them zero-copy several times (L6).

### 4.2 Instruments (build first, phase 1)

`EDGE0_TRACE=bw` logs per-step hot/cold touches, bytes, and achieved PCIe GB/s per layer.
`EDGE0_TRACE=routes` dumps logical routing per token (on ShareGPT) for offline planning. `gpu_host_probe.py`
measures the zero-copy SM read and copy-engine H2D bandwidth for 1.85 MB records. Nsight Systems covers one decode
step at conc 1, 8 and 32. All runs use the standard vllm-bench workload; interleave A/B runs (TidalMoE noise was
±2.5%).

### 4.3 Levers, in order (each one A/B-tested; keep it only if it wins)

| # | Lever | Mechanism | Expected effect |
|---|---|---|---|
| L1 | VRAM for experts | `EDGE0_DENSE=q8` (b12x; +5.3 GiB vs bf16), fp8 KV (`--kv-cache-dtype fp8_e4m3`), bf16 SSM state (check KL against fp32), embedding on host (done), `--kv-cache-memory-bytes` sized exactly | +2,900 to +4,000 hot slots |
| L2 | Touch-aware placement | Planner `edge0/tools/plan_placement.py`. From `routes` traces, estimate `P_touch(l,e | B)` at the target tokens per step and pick the top-H pairs (optimal for the additive objective). Then fit a per-layer cost model (cold count → layer time, accounting for hot/cold overlap) and re-balance per-layer budgets (TidalMoE `plan_convex.py`, re-fit for PCIe). Compare against the `profile` baseline. | fewer cold touches at the operating point |
| L3 | Adaptive expert cache | Port TidalMoE `_AdaptiveCache` (`plugin/moe_placement/__init__.py:694`): GPU `[48,512]` touch counters inside the op, epoch re-rank with EMA + Poisson z-test, ≤64 swaps per epoch between steps (≈80 µs per swap over PCIe), save/resume. Hook `GPUModelRunner.execute_model` (`v1/worker/gpu/model_runner.py:1785`) the way TidalMoE did. | follows the live workload |
| L4 | Overlap | Hot ∥ cold calls (built in). Then tune Marlin for zero-copy: block_size_m, CTA count, and a cold-call CTA limit so the cold call saturates PCIe while the hot call keeps the remaining SMs. Measure `EDGE0_SPLIT=0/1`. | layer time → max(hot, cold) |
| L5 | Fewer bytes per cold expert | `EDGE0_COLD_SCALES=vram` (−11% PCIe bytes per cold slot, costs 0.21 MB VRAM per cold expert; the planner decides how many); Q8 `lm_head` (+0.55 GiB of hot slots; needs the MTP head sharing solved). | −11% cold bytes |
| L6 | Prefill streaming | For steps ≥ `EDGE0_STREAM_MIN_TOKENS` (eager, outside graphs): copy-engine H2D of the touched cold experts (one device→host sync of the touch mask per layer) into a double-buffered VRAM ring of 2×32 slots, Marlin on the ring, next chunk copying meanwhile. Optionally prefetch the next layer's cold set during the current layer, as Strata's prefill does. | prefill no longer pays Marlin's re-reads; overlaps copy with compute |
| L7 | Tokens per step | Sweep `max_num_seqs` against hot slots (3.1 trade-off), `max_num_batched_tokens`, and MTP `num_speculative_tokens_per_batch_size` (more drafts at high batch where touch saturates, fewer at conc 1 unless traces show expert reuse across consecutive tokens) and `draft_confidence_threshold`. | higher tokens per byte moved |
| L8 | MTP expert placement | Keep the 1.32 GiB of MTP experts resident, or give the MTP layer its own Edge0 pool with hot/cold. Decide from L7 measurements. | up to +730 main hot slots |
| L9 | Host-aware fused latent MoE kernel (sm_120) | TidalMoE's persistent-kernel idea rewritten for sm_120 (CUTLASS sm120 block-scaled MMA, b12x, or Triton): one launch doing gather → experts → combine, with a few CTAs dedicated to streaming cold experts over PCIe and the rest on VRAM experts, using PDL. Only after L1 to L4 are measured and Nsight shows launch, tail or PCIe under-utilization. | removes two launches and the tail per layer |
| L10 | Predictive prefetch (research) | Predict layer l+1's experts from layer l's state (pre-gating), prefetch predicted cold experts into a VRAM victim ring on a side stream during layer l. Go only if offline traces show ≥ 70% recall at a useful prefetch size. | hides PCIe behind compute |

Rejected as tuning levers for now: lower-precision cold experts (changes the model) and CPU compute of cold experts
(section 6).

### 4.4 Benchmark protocol

`edge0/bench/bench-openai.sh <label> 1,8,32,64` against `:8125`, with the vLLM `EXTRA_BODY` above. Results go to
`/home/andychen/work/edge0-bench/results/vllm-<label>`. Every run records the env knobs, the plan file hash, and the
`bw` trace summary. Compare against `results/strata-*` at the same concurrencies.

---

## 5. Division of work

### 5.1 Converter agent: `edge0/convert/` (runs on 5090, CPU only)

Deliverables, in this order:
1. `convert_checkpoint.py --src /home/andychen/models/Edge0-145B-NVFP4 --out DIR [--num-layers N] [--no-mtp]`
   (numpy, uses `strata_formats.py`, writes with its own minimal safetensors writer, processes one layer at a time,
   `posix_fadvise(DONTNEED)` after each shard). It writes the config, tokenizer and generation files, the hardlink,
   `profile.bin`, `draft_vocab.json`, the index and `edge0_convert.json`.
2. `check_ckpt.py` (contract validation, 1.7 item 1), usable by the model agent.
3. **First the L4 mini checkpoint** (1.8), then the full checkpoint (1.1), including the MTP shard.
4. The 1.7 checks, with results in `edge0_convert.json` and a short section appended to `edge0/docs/CONVERT.md`
   (to be created) with timings, sizes and any assertion counts (scale bit 7, `s_gate != s_up`, small-scale fraction).
- Limits until GPU_FREE exists: `nice -n 19 ionice -c3`, at most 2 threads, RSS under 3 GB, reading layer slices
  only. Never modify `Edge0-145B-NVFP4`. Tensor names, shapes and dtypes are fixed by 1.5. If a fact in this
  contract turns out wrong (for example a norm that is not Gemma-style in vLLM), stop and report it to the
  coordinator rather than silently deviating.

### 5.2 Model agent: `edge0/plugin/`, `edge0/docker/`, `edge0/serve.sh`

Deliverables, in this order:
1. Package skeleton, registration, `Edge0QuantConfig` with the bf16 path, `Edge0ForCausalLM`, `Edge0LatentMoeBlock`
   with the pools and the Marlin path, `ple.py` patch, Dockerfile, `serve.sh`. Unit tests that run on CPU or meta
   where possible before GPU_FREE (config parse, quant dispatch per prefix, plan/budget math, Q8 dequant, NVFP4 repack
   index math against `strata_formats`).
2. After GPU_FREE:
   - Run `edge0/tools/gpu_host_probe.py` (CUDA 13.5 image on the 13.2 driver; VMM host pages; zero-copy and
     copy-engine bandwidth).
   - L4 checkpoint: load, block output against `latent_moe_reference` for layers 0..3, hot-only/cold-only/mixed
     placements, eager and graphs.
   - Full model: F0 eager → profile placement → CUDA graphs → teacher-forced comparison against `strata-ref.json`
     (top-1 agreement and logit KL; exact token parity is not expected, MODEL-SPEC 2.9) → first bench.
3. Phase 2: PLE reader, MTP, `EDGE0_DENSE=q8`, fp8 KV, `EDGE0_TRACE`.
- Documentation: append findings to `edge0/docs/PLUGIN.md` (to be created), with concrete numbers (load time, VRAM
  and RAM breakdown, per-step trace).

### 5.3 Interfaces and hand-offs

- Shared truth: this file (names, shapes, dtypes, config keys) and `strata_formats.py` (oracle).
- The model agent can start immediately with meta-device and CPU tests and with this catalogue. It needs the L4
  checkpoint for its first GPU run. It must never depend on files outside 1.1.
- The coordinator creates GPU_FREE, captures `strata-ref.json`, commits, and arbitrates contract changes. Tuning
  (section 4) starts once the model agent has a correct, benchmarked first light.
- Rsync: `rsync -a <WT>/edge0/ 5090:work/edge0-vllm/` (never `--delete`). In the container the tree appears at
  `/workspace/edge0`.

---

## 6. Rejected alternatives

- **Load the Strata/GGUF files directly in the plugin** (no conversion). It starts sooner, but needs a custom loader
  for everything, gives no HF tokenizer or config for the ecosystem, and leaves no checkpoint to share or validate.
  The conversion costs about 1 h of disk IO once.
- **vLLM's GGUF loader.** `gguf` is not installed in the image, and the qwen4exp, latent MoE, NVFP4 and PLE tensors
  are not mapped by it.
- **A new `model_type: "edge0"`.** vLLM keys the MTP draft rewrite and the Qwen4Exp config handling on
  `qwen4_exp(_text)`. The registered architecture name is enough to select the class.
- **All-BF16 dense checkpoint.** It is simpler, but loses 5.3 GiB of VRAM (about 2,900 hot experts) for good and is
  not what Strata computes. Raw Q8 with dequant-on-load gives the bf16 simplicity at bring-up without closing the
  door.
- **FP8 re-quantization of the dense weights.** Lossy, and still 1 B/param like Q8.
- **W4A4 FlashInfer CUTLASS (vLLM's auto choice for ModelOpt NVFP4).** It needs activation `input_scale`s, and Edge0
  has none (the `dense.bin` ones are stale). Calibration would add a numerics risk for little gain, since decode is
  memory-bound and prefill is PCIe-bound. It could be revisited after L6 for prefill compute.
- **b12x NVFP4 MoE.** Opt-in and newer. It is a tuning candidate (W4A8 or A16 modes) once Marlin works, but not the
  bring-up path.
- **vLLM `FusedMoE` + `--offload-backend uva` for the routed experts.** vLLM would own the allocation. Its post-load
  Marlin repack allocates whole layers on the GPU (0.88 GiB per layer, so OOM on 32 GB), the budget is filled whole
  params first come first served, and the latent groups would need masked virtual tokens. TidalMoE had to undo all of
  this after the fact. Owning the pools is simpler.
- **Latent MoE as 8 FusedMoE calls (one per group) or T×8 virtual tokens with masked slots.** Eight times the
  launches, or eight times the routing work, plus reliance on each backend's handling of invalid slots.
  Route-as-token is exact and does one call per tier.
- **U applied per route.** Ten times the work of one `U_cat` GEMM on the combined `Z`.
- **One CUDA VMM range per tensor mixing device and HOST_NUMA pages (TidalMoE's exact mechanism).** It is
  unverified on GeForce x86, and unnecessary because the split call takes two pointers anyway. `hostmem.vmm_host_numa`
  is kept as an allocator option, and a fused kernel (L9) can take a slot→tier table instead.
- **Permuting router rows (TidalMoE).** The `slot_of` remap after top-k keeps routing bit-exact, including ties, and
  makes a swap a table update.
- **CPU compute of cold experts (Strata's main decode path).** It breaks CUDA graphs (host callbacks inside the
  MoE), shares DRAM bandwidth with PCIe reads (a 9950X with dual-channel DDR5 gives about 70 GB/s in total), and is
  a large engineering item. Revisit only if `gpu_host_probe` shows zero-copy well under 40 GB/s.
- **PLE `cpu_offload` (51 GB pinned), device table, or `checkpoint_mapped`** (needs Grace host page tables). Also
  rejected: re-sharding the table into safetensors (a 51 GB copy).
- **CPU KV tier.** QSA reports `supports_kv_connector() = False`, and KV is small.
- **TidalMoE adaptive verification and uniform graphs.** DSpark-only; Edge0 uses MTP.

---

## 7. Risks and open items (owner)

1. CUDA 13.5 image on the 595.99 / 13.2 host driver (minor-version compatibility). Run `gpu_host_probe.py` first.
   (model)
2. Zero-copy bandwidth from `cudaHostRegister` memory with Marlin's access pattern is unmeasured. If it is low, try
   `vmm_host_numa` (2 MiB GPU mappings), `host_alloc`, or the copy-engine path. (model)
3. `fused_marlin_moe` with `expert_map` and `topk=1` has to be checked: skipped routes, the `moe_sum` path, and
   `global_num_experts=512` with smaller pools. The combine never reads skipped rows. (model)
4. `Edge0Q8DequantLinearMethod` relies on the fused loaders (`in_proj_qkvz`, `gate_up_proj`, MTP `qkv_proj`)
   slicing on dim 0 only; verify with the L4 checkpoint. (model)
5. The MTP draft might build private `embed_tokens` and `lm_head` (2.4 GiB) instead of sharing them. (model, phase 2)
6. The norm conventions (−1) and the GDN v-head reorder come from the llama.cpp converter source. They are confirmed
   only by the teacher-forced comparison. The converter's check 3 confirms the reorder algebra but not vLLM's
   convention. (converter + model)
7. QSA selection past 2051 cells may differ between vLLM and Strata (MODEL-SPEC 9.1). Measure KL at contexts above
   2K. (model)
8. PLE serial `preadv` may dominate prefill on a cold page cache. The phase-2 reader is a prerequisite for fair
   benchmarks. (model)
9. The `profile.bin` ranking was made by Strata on unknown traffic. It might be far from ShareGPT; L2 decides. (tuning)
10. License: commercial serving needs a separate Edge0 license (MODEL-SPEC 5.8). (coordinator)
