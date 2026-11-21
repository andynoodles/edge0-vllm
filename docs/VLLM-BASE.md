# VLLM-BASE: what `nvcr.io/nvidia/vllm:26.11-py3` gives the Edge0-145B port

Inspected 2026-11-19 on the 5090 box (no GPU used yet: GPU_FREE did not exist). Everything below comes from the image's
installed source, read with throwaway `--memory 3g --cpus 2` containers. Line numbers refer to that source.

## TL;DR

- **vLLM 0.33.0+121396e7** (NVIDIA build `nv26.11.81930723`), torch 2.15.0a0 (CUDA 13.5), FlashInfer 0.7.2.post1, Triton
  3.9.0, CUTLASS DSL 4.8.0, transformers 5.20.2. That is the same vLLM release TidalMoE ran (upstream `v0.33.0`).
- **The Qwen3.8-Flash-Next base is in the image**: `vllm/models/qwen4_exp/` (`Qwen4ExpForCausalLM`,
  `Qwen4ExpForConditionalGeneration`, `Qwen4ExpMTP`), with GDN linear attention, QSA sparse attention (indexer),
  4-stream hyper-connections, PLE n-gram embeddings (with host offload backends) and the MTP drafter, and **explicit
  sm_120 (RTX 5090) tuning tables**. No `vllm/vllm-openai:qwen38-flash-next` image or upstream source is needed.
- The Edge0 skeleton GGUF is exactly a qwen4exp model (`general.architecture = qwen4exp`, hidden 2560, 48 layers, QSA
  every 4th layer, HC 4 x 320, PLE on layer 1); vLLM's PLE hash parameters reproduce the GGUF's PLE multipliers, head
  vocab sizes and offsets **exactly** (checked, section 2.6).
- **Missing for Edge0**: (a) the **latent MoE** (D: 2560 -> 8 groups x 1792, experts at 1792 -> 640 -> 1792, U back to
  2560, expert -> group from `labels.bin`); vLLM only has a single-latent hook (`routed_input_transform` /
  `routed_output_transform`, used by NemotronH / Kimi-K3). (b) A loader for the GGUF / Strata pack files: the model
  must be presented as an HF safetensors checkpoint (the `gguf` Python package is not even installed). (c) A way to
  point PLE file-gather at `ple-fp8.gguf` in place (a two-function monkeypatch, section 2.6).
- **NVFP4 MoE on sm_120**: auto-selection picks **FlashInfer CUTLASS** (W4A4) for ModelOpt `NVFP4`, **Marlin** for
  `W4A16_NVFP4`; `--moe-backend b12x | flashinfer_b12x | flashinfer_cutile | cutlass | marlin | humming` are the
  sm_120-capable overrides. TRT-LLM-gen and CuTe-DSL MoE (TidalMoE's kernels) are sm_100-only.
- **b12x block codecs**: ModelOpt `MIXED_PRECISION` can carry GGML `q8_0` blocks (and `iq2_xs`, `iq2_xxs`) for
  Linear / embedding / LM head / MoE on sm_12x, so the skeleton's Q8_0 weights could be served as-is (half the VRAM of
  BF16 = more VRAM for hot experts).
- **TidalMoE hooks**: every vLLM symbol the plugin patches still exists, but the hot path it patches
  (`TrtLlmMxfp4ExpertsModular`, TRT-LLM MXFP4 tactics, CuTe tcgen05 persistent kernels, DSpark AV, DSV4.1 mHC) is not
  the sm_120 / NVFP4 / MTP path. The expert-load-time ideas (per-expert VMM hot/cold placement, adaptive cache, touch
  counting, hot/cold split, host-resident embedding, bandwidth trace) port with adaptation; section 6.
- **Risk to verify after GPU_FREE**: host driver 595.99 (CUDA 13.2) vs image CUDA 13.5. Forward compat is
  datacenter-only, so the container should fall back to "CUDA Minor Version Compatibility mode". Run
  `edge0/tools/gpu_host_probe.py` first (section 8).

## 0. How this was inspected (reusable artifacts)

| What | Where |
|---|---|
| Extracted `.py/.json/.cu/.h` source of vllm, flashinfer, transformers/qwen4_exp, qwen3_next, qwen3_5 | 5090: `/home/andychen/work/edge0-vllm-ref/{vllm,flashinfer,transformers}` (read-only reference, grep it instead of starting containers) |
| Probe: versions, kernel packages, registry, entry points, spec methods | `edge0/tools/vllm_base_probe.py`, output `/home/andychen/work/edge0-vllm/logs/vllm-base-probe.txt` |
| GGUF header dump (no tensor data read) | `edge0/tools/gguf_meta_dump.py`, outputs `logs/skeleton-gguf-meta.txt`, `logs/ple-gguf-meta.txt` |
| HF Qwen4Exp checkpoint names/shapes at Edge0 dims (meta device) | `edge0/tools/qwen4exp_weight_names.py`, output `logs/qwen4exp-hf-weight-names.txt` |
| GPU probe to run after GPU_FREE (CUDA compat, ops, VMM host pages, PCIe/UVA bandwidth) | `edge0/tools/gpu_host_probe.py` (not run yet) |

## 1. Versions and installation

| Component | Version | Notes |
|---|---|---|
| vLLM | `0.33.0+121396e7.nv26.11.81930723` (`vllm.__version__` = `0.33.0+121396e7.dev`) | git sha prefix `121396e7`; `_version.__commit_id__` is None. Image label `com.nvidia.build.ref=09b8e9bd…` is the NGC build repo, not vLLM |
| install path | `/usr/local/lib/python3.12/dist-packages/vllm` | wheel `file:///opt/transfer/vllm-0.33.0+121396e7.nv26.11.81930723-cp38-abi3-linux_x86_64.whl` (not editable) |
| Python | 3.12.3 | `PIP_CONSTRAINT=/etc/pip/constraint.txt`, `PIP_NO_BUILD_ISOLATION=1`, `PIP_BREAK_SYSTEM_PACKAGES=1` |
| torch | `2.15.0a0+fd23299d96.nv26.11` | `torch.version.cuda=13.5`, arch list `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120 compute_120` |
| CUDA toolkit | 13.5.0.025 (`CUDA_DRIVER_VERSION=620.70.08` compat libs in `/usr/local/cuda/compat`) | host driver is 595.99 / CUDA 13.2, see risk above |
| FlashInfer | `0.7.2.post1+c44a1936` (+ `flashinfer-cubin`, `flashinfer-jit-cache`, `flashinfer-jit-cache-sm120f`) | `flashinfer.fused_moe.cutlass_fused_moe`, `b12x_fused_moe`, `trtllm_fp4_block_scale_(routed_)moe` present |
| Triton | `3.9.0+gitaad2a60d.nv26.11`, `triton_kernels 1.0.0`, `tokenspeed-triton 3.9.10` | |
| CUTLASS | `nvidia-cutlass-dsl 4.8.0` (CuTe DSL, `import cutlass`), C++ CUTLASS compiled into `vllm/_C_stable_libtorch.abi3.so` / `_moe_C_stable_libtorch.abi3.so` | `CUTLASS_OPERATOR_VERSION=0.3.0` |
| transformers | 5.20.2 (has `models/qwen4_exp`) | |
| other | cuda-python/cuda-bindings 13.4.2, cuDNN 9.28, NCCL 2.32.6, nixl 1.5.0, safetensors 0.8.0, fastsafetensors 0.5.0, ModelOpt 0.47.1, `humming_kernels 0.1.19`, `b12x` (import resolves to `flashinfer/experimental/b12x`) | **not installed**: `gguf`, `lmcache`, standalone `deep_gemm` (vLLM vendors `vllm/third_party/deep_gemm`) |

Image: `nvcr.io/nvidia/vllm:26.11-py3`, id `cae1df3428f2`, 28.2 GB, created 2026-11-14. `CUDA_ARCH_LIST=7.5 8.0 8.6 9.0 10.0 12.0`.
Entrypoint scripts `/opt/nvidia/entrypoint.d/5{0,1,2}-*.sh` and `/etc/shinit_v2` decide forward-compat vs minor-version
compat at container start.

Default model runner: **Model Runner V2** (`vllm/v1/worker/gpu/model_runner.py`); `VllmConfig.use_v2_model_runner`
(`config/vllm.py:731`) falls back to V1 only for unsupported features (suffix/custom_class spec, heterogeneous draft
vocab, …). `VLLM_USE_V2_MODEL_RUNNER=0/1` forces it.

## 2. Qwen3.8-Flash-Next (qwen4exp) support

### 2.1 Files and classes

| File (under `vllm/`) | Contents |
|---|---|
| `models/qwen4_exp/__init__.py` | lazy `__getattr__`: CUDA -> `nvidia.model.Qwen4ExpForCausalLM`, `...ForConditionalGeneration`, `nvidia.mtp.Qwen4ExpMTP`; ROCm -> `amd/` |
| `models/qwen4_exp/nvidia/model.py` (1173 lines) | `Qwen4ExpSparseMoeBlock(Qwen3NextSparseMoeBlock)`, `Qwen4ExpDecoderLayer`, `Qwen4ExpMixtureOfExperts` (EPLB), `Qwen4ExpModel`, `Qwen4ExpForCausalLM` (`IsHybrid`, `HasInnerState`, `SupportsPP`, `SupportsLoRA`, `SupportsMRoPE`, `SupportsHCSP`), `Qwen4ExpForConditionalGeneration` (Qwen3-VL tower) |
| `models/qwen4_exp/nvidia/qsa.py`, `indexer_qsa.py`, `ops/qsa.py`, `ops/qsa_indexer.py`, `ops/qsa_prepare.py`, `common/qsa_cache.py` | QSA attention owner, indexer, Triton kernels, cache spec |
| `models/qwen4_exp/nvidia/hyperconnection.py`, `common/hyperconnection.py`, `ops/hc.py`, `ops/cute_dsl/hc_down_silu*.py` | `GatedResidual` hyper-connections (warp `mma.sync` CuTe kernel, works on sm_120) |
| `models/qwen4_exp/nvidia/ple_layer.py`, `ngram_embedding.py`, `ple_pageable.py`, `common/ngram_embedding.py`, `common/ple.py`, `common/ops/ple.py` | PLE layer (short conv as a Mamba-style state), n-gram hashing, storage backends |
| `models/qwen4_exp/nvidia/mtp.py` | `Qwen4ExpMultiTokenPredictor`, `Qwen4ExpMTP` |
| `models/qwen4_exp/nvidia/model_state.py` | `Qwen4ExpModelState` (V2 runner: builds `query_start_loc`, `ngram_context`, stages PLE rows) |
| `models/qwen4_exp/nvidia/low_latency_gemm.py` | skinny decode GEMM plans; `QWEN4_EXP_SM120_GEMM_PLANS` ("RTX 5090 / RTX PRO 6000 Blackwell (sm_120) plans, measured 2026-10-03") for the exact Edge0 shapes: LM head (248320, 2560), GDN QKVZ (16384, 2560), QSA QKV/gate (14336, 2560), out-proj (2560, 6144), shared expert (1280, 2560), indexer (640, 2560), HC (336, 10240), GDN B/A (96, 2560) |
| `model_executor/models/registry.py:119,622,737` | `"Qwen4ExpForCausalLM"`, `"Qwen4ExpForConditionalGeneration"`, `"Qwen4ExpMTP"` -> `vllm.models.qwen4_exp` |
| `transformers/models/qwen4_exp/configuration_qwen4_exp.py` | `Qwen4ExpTextConfig` (`model_type="qwen4_exp_text"`), `Qwen4ExpConfig` (`"qwen4_exp"`) |
| `config/engram.py` | `EngramConfig` (PLE storage: pinned / file-gather / mapped) |
| `transformers_utils/model_arch_config_convertor.py:713` | `Qwen4ExpMTPModelArchConfigConvertor` |

### 2.2 Config keys and the Edge0 values (from `model-skeleton.gguf` metadata)

| `Qwen4ExpTextConfig` key | Edge0 value | GGUF source |
|---|---|---|
| `hidden_size` | 2560 | `qwen4exp.embedding_length` |
| `num_hidden_layers` | 48 | `block_count` |
| `layer_types` | `[linear_attention x3, indexed_attention] x 12` (or `full_attention_interval=4`; `"full_attention"` is renamed to `"indexed_attention"`) | `attention.recurrent_layers`, `full_attention_interval=4` |
| `num_attention_heads` / `num_key_value_heads` / `head_dim` | 24 / 2 / 256 | `attention.head_count(_kv)`, `key_length` |
| `rope_parameters` | theta 1e7, `partial_rotary_factor=0.25` (64 of 256 dims), `mrope_section=[11,11,10]` | `rope.*` |
| `linear_num_key_heads` / `linear_num_value_heads` / `linear_key_head_dim` / `linear_value_head_dim` / `linear_conv_kernel_dim` | 16 / 48 / 128 / 128 / 4 | `ssm.group_count=16`, `ssm.time_step_rank=48`, `ssm.state_size=128`, `ssm.inner_size=6144`, `ssm.conv_kernel=4` |
| `num_experts` / `num_experts_per_tok` / `norm_topk_prob` | 512 / 10 / True | `expert_count`, `expert_used_count` |
| `moe_intermediate_size` / `shared_expert_intermediate_size` | 640 / 640 | `expert_feed_forward_length`, `expert_shared_feed_forward_length` |
| `hc_count` / `hc_lowrank` | 4 / 320 | `hyper_connection.count/low_rank` |
| `indexer_n_heads` / `indexer_kv_heads` / `indexer_head_dim` / `indexer_budget` / `indexer_compress_ratio` | 4 / 1 / 128 / 2048 / 4 | `attention.indexer.*`, `compress_ratios` |
| `ple_layer_ids` (1-indexed) / `ple_embed_dim` / `ngram_size` / `heads_per_ngram` / `ple_conv_kernel_size` / `ngram_vocab_size_base` / `seed` | `[1]` / 2560 (16 heads x 160) / 3 / 8 / 4 / 20000000 / 1234 | `ple.layers=[1]`, `embedding_length_per_layer_input=160`, `ple.*` |
| `eos_token_id` (PLE needs it) | 248044 (`ple.eos_token_id`; tokenizer eos is 248046) | |
| `vocab_size`, `max_position_embeddings` | 248320, 262144 | |
| `split_ngram_parts` | 512 by default; any value works if the PLE layout is patched (2.6) | |
| `ple_embedding_dtype` (optional) | `"float8_e4m3fn"` forces the FP8 PLE method regardless of `quantization_config` | read via `getattr(config, "ple_embedding_dtype")` in `nvidia/ngram_embedding.py` |

Edge0 also matches Qwen3.8-Flash-Next numerically: 512 x 48 x 3 x 2560 x 640 = 120.8 B routed parameters (the README's
"Qwen3.8-Flash-Next routed expert parameters 120.8 B"), so the stock model is the same skeleton with full-width experts.

### 2.3 Checkpoint tensor names vLLM expects (HF format)

Generated on the meta device at Edge0 dims (`logs/qwen4exp-hf-weight-names.txt`). `Qwen4ExpForCausalLM.hf_to_vllm_mapper`
also accepts the `model.language_model.` prefix (VL layout); `mtp.*` names are dropped by the target and loaded by
`Qwen4ExpMTP`.

```
model.embed_tokens.weight                                   [248320, 2560]
lm_head.weight                                              [248320, 2560]
model.hyper_connection_mixer.{hc_norm.weight [10240], input_mix_weight_down.weight [320,10240], input_mix_weight_up.weight [10240,320]}
# GDN layer (layers 0,1,2,4,...)
model.layers.0.linear_attn.{in_proj_qkv [10240,2560], in_proj_z [6144,2560], in_proj_b [48,2560], in_proj_a [48,2560],
                            conv1d.weight [10240,1,4], A_log [48], dt_bias [48], norm.weight [128], out_proj [2560,6144]}
# QSA layer (layers 3,7,...,47)
model.layers.3.self_attn.{q_proj [12288,2560] (q + output gate), k_proj [512,2560], v_proj [512,2560], o_proj [2560,6144],
                          q_norm [256], k_norm [256], indexer.index_qk_proj [640,2560] (4x128 q + 128 k),
                          indexer.q_layernorm [128], indexer.k_layernorm [128]}
# every layer
model.layers.N.mlp.gate.weight                              [512, 2560]   (router, BF16 in the GGUF)
model.layers.N.mlp.experts.{E}.{gate_proj,up_proj,down_proj}.weight   (per-expert names; FusedMoE ckpt_names default)
   HF transformers itself stores fused  mlp.experts.gate_up_proj [E,1280,2560] / mlp.experts.down_proj [E,2560,640]
model.layers.N.mlp.shared_expert.{gate_proj [640,2560], up_proj [640,2560], down_proj [2560,640]}
model.layers.N.mlp.shared_expert_gate.weight               [1, 2560]
model.layers.N.{attn,mlp}_hyper_connection.{hc_norm.weight [10240], input_mix_weight_down.weight [320,10240],
                                            input_mix_weight_up.weight [10240,320], block_inject_weight.weight [4,10240]}
# PLE layer (layer 0 only)
model.layers.0.ple.{key_proj [10240,2560], value_proj [2560,2560], norm_key [10240], norm_query [10240], norm_conv [10240],
                    conv1d.weight [10240,1,4]}
model.layers.0.ple.ple_embedding.ngram_embedding.weight     [total_rows, 160]  (HF runtime; checkpoint stores it as
   ngram_embedding.shard_<i>.weight, i < split_ngram_parts, + ngram_embedding.weight_scale (fp32) for FP8)
   buffers rebuilt in __init__ (skip in ckpt): ple_embedding.{layer_multipliers, ngram_heads_vocab_sizes, ngram_heads_offsets}
```

GGUF -> HF map for the skeleton (from `blk.0`/`blk.3`): `attn_qkv`->`linear_attn.in_proj_qkv`, `attn_gate`->`in_proj_z`,
`ssm_beta`->`in_proj_b`, `ssm_alpha`->`in_proj_a`, `ssm_conv1d`->`conv1d` (GGUF [4,10240]), `ssm_a`->`A_log` (check sign/log
convention), `ssm_dt.bias`->`dt_bias`, `ssm_norm`->`norm`, `ssm_out`->`out_proj`, `attn_q/k/v/output`->`q/k/v/o_proj`,
`attn_q_norm/attn_k_norm`, `indexer.q_proj`+`indexer.k_proj`->`indexer.index_qk_proj` (concat), `indexer.q_norm/k_norm`->
`q_layernorm/k_layernorm`, `ffn_gate_inp`->`mlp.gate`, `ffn_*_shexp`->`shared_expert.*`, `ffn_gate_inp_shexp`->
`shared_expert_gate`, `hc_attn_{norm,down,up,inject}`->`attn_hyper_connection.{hc_norm,input_mix_weight_down,
input_mix_weight_up,block_inject_weight}`, `hc_ffn_*`->`mlp_hyper_connection.*`, `output_hc_*`->`hyper_connection_mixer.*`,
`token_embd`->`embed_tokens`, `output`->`lm_head`. Note vLLM's loader merges `input_mix_weight_down` + `block_inject_weight`
into `input_mix_weight_down_block_inject` (`_EXTRA_WEIGHTS_MAPPER`) and `ple.key_proj`+`value_proj` into `ple.kv_proj`.
The converter agent owns the exact transposes; GGUF shapes are listed `[ne0, ne1]` (= `[in, out]`).

### 2.4 How the hybrid attention works in vLLM

- **GDN layers** (`linear_attention`): `QwenGatedDeltaNetAttention` (`model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`),
  Mamba-style constant state per request (`MambaSpec`, `GDN_ATTN` backend). Prefill backend resolver
  `_resolve_gdn_prefill_backend` picks **FlashInfer GDN prefill on sm_12x** (`head_k_dim==128`, CUDA runtime >= 13),
  else FLA Triton (`vllm/third_party/flash_linear_attention`). Override: `--additional-config '{"gdn_prefill_backend":"triton"}'`.
- **QSA layers** (`indexed_attention` with `indexer_n_heads` set): `Qwen4ExpQSAAttention` owns its KV cache directly
  (`FullAttentionSpec`), backend name `QWEN4_EXP_QSA_TRITON` (`Qwen4ExpQSAFlashAttentionBackend`, metadata from
  FlashAttention's builder, `AttentionCGSupport.UNIFORM_BATCH`). The **indexer** (`QSAIndexer`, `indexer_qsa.py`) keeps a
  compressed key cache (mean of every `indexer_compress_ratio=4` token keys) in a ring, scores blocks with 4 x 128-dim heads,
  writes per-token top-k token indices (budget 2048) into `topk_indices_buffer`; `ops/qsa.py:qsa_sparse_paged_attention`
  (Triton) then does sparse paged GQA over those indices. sm_120 has its own tuning table (`_select_sm120_config`).
  The fused QKV GEMM also carries the replicated indexer Q/K projection (`Qwen4ExpQSAQKVIndexerLinear`, excluded from
  ModelOpt FP4 by `without_modelopt_fp4`). KV cache dtypes: `auto/bfloat16/fp8/fp8_e4m3` (fp8 dequantized inside the
  Triton kernel; the backend comments call out sm120 explicitly). Requires `vllm_flash_attn` FA2 to be importable (gate only;
  `_vllm_fa2_C` ships in the image, built with arch list incl. 12.0). Block size must be a multiple of `lcm(16, ring capacity)`. `supports_kv_connector() -> False`.
- **Hyper-connections**: `GatedResidual` (`attn_hyper_connection`, `mlp_hyper_connection`, final `hyper_connection_mixer`);
  the residual stream is `[T, hc_count*hidden] = [T, 10240]`; combine of layer i is fused with the mix of layer i+1.
- **KV size is small**: only 12 of 48 layers have KV (2 KV heads x 256 + indexer 128 per token); GDN/PLE states are per
  request. About 25 KB/token in BF16 (12 x (2 x 2 x 256 x 2 B + compressed indexer keys)), so 32K tokens of context cost
  < 1 GB.

### 2.5 MoE block

`Qwen4ExpSparseMoeBlock(Qwen3NextSparseMoeBlock)` (`model_executor/models/qwen3_next.py:131`): `GateLinear` router
[512, 2560] -> softmax top-10, renormalized; `FusedMoEFactory(...)` returns a **`MoERunner`** (`fused_moe/runner/moe_runner.py`)
whose `.routed_experts` is a `RoutedExperts` (params `w13_weight`, `w2_weight`, scales) with `.quant_method.moe_kernel`
(`FusedMoEKernel`; `.impl.fused_experts` is the experts object, e.g. `FlashInferExperts`). Shared expert `Qwen3NextMLP` with
sigmoid gate `shared_expert_gate`; it may be fused into the routed kernel (`is_model_fused_shared_expert_compatible`).
The attribute path TidalMoE uses, `m.experts.routed_experts.quant_method.moe_kernel.impl.fused_experts`, is valid here.

**Latent MoE is not in Qwen4Exp.** What vLLM has: `FusedMoEFactory(routed_input_transform=..., routed_output_transform=...)`
(`fused_moe/layer.py:134`, applied in `MoERunner.apply_routed_input_transform/apply_routed_output_transform`,
`moe_runner.py:362-400`): the router and shared expert see the full hidden state, the routed experts see
`routed_input_transform(x)` and their summed output goes through `routed_output_transform`. NemotronH uses it with
`fc1_latent_proj`/`fc2_latent_proj` (`nemotron_h.py:195-249`); Kimi-K3 has a `latent_moe_runner.py`. This handles **one**
latent per token. Edge0 has **8 projection groups** (from file sizes: `latent_dense_q8.bin` = 48 x 77,987,840 B =
48 x (2 x 8 x 1792 x 2560) x 34/32 Q8_0 bytes; `labels.bin` = 48 x 512 uint32 group ids, layer 0 starts `1,0,1,4`), so each
token has 8 latents and expert e consumes latent `labels[e]`. A custom MoE block is needed, e.g. D -> [T, 8, 1792];
"virtual tokens" `t*8+g` with per-virtual-token top-k slots (slots of other groups marked invalid: expert id outside the
local range / `expert_map=-1`, weight 0); one FusedMoE call at hidden 1792; reshape [T, 8*1792] -> U. Alternatively 8 calls
with group-local expert ids. Check which sm_120 kernels skip invalid ids before choosing (model-port agent).

### 2.6 PLE / Engram n-gram embeddings and how they are offloaded

`config/engram.py:EngramConfig` (`--engram-config '{...}'`), selected in `Qwen4ExpNGramEmbedding.__init__`
(`nvidia/ngram_embedding.py:847`):

| Option | Backend class | What it does | For Edge0 on this box |
|---|---|---|---|
| `cpu_offload: true` (**default**) | `Qwen4ExpPLEPinnedHostEmbedding` (`common/ngram_embedding.py:634`) | whole table in **pinned** host RAM, Triton UVA gather on a side stream, prefetched one layer ahead (`start_prefetch` while the previous layer runs) | 51.2 GB pinned of 60 GiB RAM: **not feasible** next to cold experts |
| `host_file_gather: true` | `Qwen4ExpPLEFileGatherEmbedding` (`nvidia/ngram_embedding.py:467`) | no table allocation; in `prepare_inputs` (CPU) computes the step's n-gram ids, `posix_fadvise(WILLNEED)` + serial `os.preadv` of each distinct row from the checkpoint files (page cache, not O_DIRECT), H2D into fixed staging buffers; forward only reads staging -> CUDA-graph safe. ETP size 1 only | **the right one**: Strata does the same with `--ple-io direct` + 24 workers |
| `checkpoint_mapped: true` | `Qwen4ExpPLEPageableHostEmbedding` | GPU reads an mmap of the safetensors through host page tables | requires `PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES=1` (GB10/Grace); x86 + discrete GPU: no |
| none of the above | `Qwen4ExpPLEDeviceEmbedding` | table in VRAM | no |

Other knobs: `use_thp`, `dp_shared_memory`, `embedding_across_dp`. PLE FP8 format: `Qwen4ExpPLEFp8EmbeddingMethod` = e4m3
rows + **one fp32 per-tensor `weight_scale`** (NVFP4 and BF16 variants also exist).

Edge0 match, checked: `ple-fp8.gguf` is one tensor `per_layer_token_embd.weight`, raw e4m3 bytes, `[320001536 rows x 160]`,
data offset **352**, `strata.ple.scale = 0.00019931793212890625` (one global scale). vLLM's `_make_layer_multipliers(seed=1234,
layer 0, vocab 248320)` gives `[23703573157769, 20109073645365, 8052911324071]` and `_make_vocab_layout(base 20M, 16 heads)`
gives sizes `20000003 … 20000171`, offsets `0 … 300001275`, total 320001446 rows, padded to 128 -> 320001536 -> x 160 B =
51,200,245,760 B = file size - 352. All equal to the GGUF metadata. Row r is at byte `352 + 160*r`; head h's rows start at
`offsets[h]`, each row is that head's 160-dim slice of the 2560-dim PLE input. (Only the hash parameters were compared; the
per-token id formula in `Qwen4ExpNGramEmbedding` vs Strata `src/kernels/ngram.cpp` and the conv/norm/proj math still need a
numeric check by the correctness agent.)

To use it in place (no copy, no hardlink-into-safetensors possible because safetensors needs its header in the same file):
patch two module-level functions in `vllm.models.qwen4_exp.nvidia.ngram_embedding` from the plugin:
`resolve_checkpoint_files` (return the GGUF path) and `discover_table_layout` (return
`TableLayout(num_rows=320001536, row_bytes=160, rows_per_shard=320001536, shards=(ShardLocation(path, 352, 320001536),))`),
put only `...ngram_embedding.weight_scale` (fp32 0.000199317932…) in the safetensors, and run with
`--engram-config '{"cpu_offload": false, "host_file_gather": true}'`. Note `_FilePlane.read` is single-threaded `preadv`
on the engine's critical path; if PLE reads show up in step time, replace it with a thread pool / io_uring (Strata: O_DIRECT,
24 workers). The page cache competes with pinned cold experts for the ~56 GB of RAM.

### 2.7 MTP

`Qwen4ExpMTP` / `Qwen4ExpMultiTokenPredictor` (`nvidia/mtp.py`): one (or `mtp_num_hidden_layers`) Qwen4Exp decoder layer at
index `num_hidden_layers` (= 48) with **PLE forced off, HC kept, MoE with 512 experts**, inputs fused by
`fc_embedding(pre_fc_norm_embedding(embed))` + `fc_hidden(pre_fc_norm_hidden(multi-stream hidden))` ("residual_linear_shared"),
own `hyper_connection_mixer`, shares `embed_tokens`/`lm_head` with the target. The target keeps the pre-mixer multi-stream
hidden `[T, 4*2560]` in `_mtp_hidden_buffer` when `speculative_config.method == "mtp"`. Edge0's `mtp/dense.txt` tensor names
(`fc_embedding`, `fc_hidden`, `pre_fc_norm_*`, `hyper_connection_mixer.*`, `{attn,mlp}_hyper_connection.*`, `mlp.gate [512,2560]`,
`mlp.shared_expert*`, `self_attn.*` incl. `indexer`) match this module one to one (prefix them `mtp.` in the checkpoint).
Edge0 MTP experts are 1,382,400 B each (`mtp/experts.bin` / 512) = 2,457,600 params x 0.5625 B (NVFP4 + e4m3 scales, no
extra header) - either 3 x 1280 x 640 (latent 1280) or 3 x 2560 x 320; the model-port agent must decide from Strata source.

### 2.8 sm_120 backend summary for this model

GDN: FlashInfer GDN prefill + FLA Triton decode. QSA: Triton (sm_120 table) with FA2 present. HC: CuTe-DSL warp-MMA kernel
(`has_device_capability(90)`) or torch. Dense decode GEMMs: CuTe-DSL skinny GEMM with sm_120 plans (BF16 weights), else
cuBLAS/CUTLASS. MoE: section 3. KV: bf16 or fp8_e4m3. LM head: skinny GEMM plan (248320, 2560).

## 3. NVFP4 MoE on sm_120

### 3.1 Backends (`fused_moe/oracle/nvfp4.py`)

| `NvFp4MoeBackend` | Experts class | `_supports_current_device` | sm_120? | Scheme |
|---|---|---|---|---|
| `FLASHINFER_TRTLLM` | `TrtLlmNvFp4ExpertsMonolithic/Modular` | family 100 | no | W4A4 |
| `FLASHINFER_CUTEDSL(_BATCHED)` | `FlashInferCuteDSL(Batched)Experts` | family 100 | no | W4A4 |
| `FLASHINFER_CUTLASS` | `FlashInferExperts` (`experts/flashinfer_cutlass_moe.py`) | 90, family 100, **family 120** | **yes, auto pick for W4A4** | (kNvfp4Static, kNvfp4Dynamic) when cap >= 100 |
| `VLLM_CUTLASS` | `CutlassExpertsFp4` (`experts/cutlass_moe.py:701`) | families 100/110/120 | yes (auto 2nd) | W4A4; fuses `input_scale` into `w*_weight_scale_2` |
| `MARLIN` | `MarlinExperts` | cap >= 7.5 | yes, **auto pick for W4A16** | W4A16 (BF16 activations) |
| `HUMMING` | `HummingGroupedExperts` etc. | `humming.supports_device()` | yes | preferred over Marlin only on sm_90 (`prefers_humming`) |
| `EMULATION` | Triton dequant emulation | any | yes | last resort |
| `B12X` | `B12xExperts` (`fused_moe/b12x.py`) | family 120 + b12x | yes, **opt-in only** | modes: nvfp4 W4A4 (`w31`), W4A8 (`mxfp8` act), W4A16 (`w13`); `VLLM_B12X_MOE_FP4_FORCE_A16=1`, `VLLM_B12X_ACTIVATION_MODE_A16_M=<M>` (A16 for M <= cutoff) |
| `FLASHINFER_B12X` | `FlashInferB12xExperts` | family 120 | opt-in ("until the CUTLASS SM121 MMA op guard is resolved") | W4A4 |
| `FLASHINFER_CUTILE` | `FlashInferCuTileNvfp4Experts` | family 120 | opt-in | W4A4, can switch to W4A16 per forward (`nvfp4_moe_dynamic_max_tokens`); availability is `has_flashinfer_cutile_nvfp4()` (runner classes registered in `flashinfer.fused_moe.layer._BACKEND_RUNNERS`), not probed yet |

Auto order (`select_nvfp4_moe_backend`): TRTLLM, CUTEDSL, CUTEDSL_BATCHED, FLASHINFER_CUTLASS, VLLM_CUTLASS, MARLIN,
HUMMING, EMULATION. On sm_120: **ModelOpt `NVFP4` (W4A4) -> FLASHINFER_CUTLASS**; **`W4A16_NVFP4` -> MARLIN**.
A checkpoint whose `config_groups` all have `input_activations: null` is routed to W4A16 automatically
(`ModelOptNvFp4Config._from_config`). Switch: `--moe-backend {auto,flashinfer_cutlass,cutlass,marlin,humming,b12x,
flashinfer_b12x,flashinfer_cutile,triton,...}` (`KernelConfig.moe_backend`, `config/kernel.py:122/330`). Other env:
`VLLM_CSF_RESIDENT` (keep CSF-compressed NVFP4 scales compressed in VRAM, served by b12x; needs a CSF checkpoint;
scales are 11% of an Edge0 expert record, so this is a "fewer expert bytes" lever).

Edge0 numerics: the Strata pack has per-expert `ffn_{gate,up,down}_exps.scale` and `.input_scale` (`index.txt`), i.e.
ModelOpt-style weight global scales and activation scales exist, so the W4A4 FLASHINFER_CUTLASS path is usable; Strata
itself has both W4A4 and W4A8 prefill kernels (`src/prefill/mmq_nvfp4_w4a4.cu`, `mmq_nvfp4_w4a8.cu`). Pick the backend by
matching Strata's decode math (correctness agent).

### 3.2 ModelOpt NVFP4 checkpoint layout vLLM loads (`ModelOptNvFp4FusedMoE.create_weights`, `modelopt.py:912`)

Per layer, E experts, I = 640 (Edge0), H = expert input width (1792 for Edge0's latent experts):

| Param | Shape | dtype | Meaning |
|---|---|---|---|
| `w13_weight` | [E, 2I, H/2] | uint8 | rows 0..I-1 = **gate_proj (w1)**, rows I..2I-1 = **up_proj (w3)**; 2 e2m1 values per byte, **low nibble = even (first) element** (`break_fp4_bytes`); K (input) contiguous |
| `w13_weight_scale` | [E, 2I, H/16] | float8_e4m3fn | one scale per 16 inputs (group_size 16), linear (unswizzled) in the checkpoint |
| `w13_weight_scale_2` | [E, 2] | fp32 | per-expert global scale for gate and up; vLLM uses `[:,0]` and warns if they differ -> make them equal |
| `w13_input_scale` | [E, 2] | fp32 | activation global scale (amax / (6*448)) |
| `w2_weight` / `w2_weight_scale` | [E, H, I/2] / [E, H, I/16] | uint8 / e4m3 | down_proj |
| `w2_weight_scale_2` / `w2_input_scale` | [E] / [E] | fp32 | |

Dequant: `w = e2m1 * e4m3_block_scale * weight_scale_2` (`dequantize_to_dtype`; `quantize_nvfp4` in
`v1/worker/gpu/spec_decode/draft_vocab.py` shows the exact ModelOpt quantizer). Checkpoint names: per-expert
`...mlp.experts.{e}.{gate_proj,up_proj,down_proj}.{weight,weight_scale,weight_scale_2,input_scale}`; config
`quantization_config: {"quant_method":"modelopt","quant_algo":"NVFP4","group_size":16,"kv_cache_quant_algo":null,
"exclude_modules":[...]}` (or `MIXED_PRECISION` with `quantized_layers`, below).

What each backend does at `process_weights_after_loading` (`convert_to_nvfp4_moe_kernel_format`):

- FLASHINFER_CUTLASS / TRTLLM / FLASHINFER_B12X: **reorder [w1,w3] -> [w3,w1]** (`reorder_w1w3_to_w3w1`); input scales
  collapsed to one global max over experts (`amax_for_moe_activation_quant(...).repeat(E)`, "global SF").
- FLASHINFER_CUTLASS / VLLM_CUTLASS: block scales **swizzled** by `swizzle_blockscale` (`quantization/utils/nvfp4_utils.py:13`):
  pad rows to 128 and scale-columns to 4, reshape `[B, M/128, 4, 32, K/4, 4]`, permute `(0,1,4,3,2,5)` (the 128x4 CUTLASS
  SF layout); `is_scale_swizzled=True`; `g1_alphas = w13_weight_scale_2`, `a1_gscale = 1/a13_scale`.
- VLLM_CUTLASS: per-expert `a13_scale.max(dim=1)`; `CutlassExpertsFp4.process_weights_after_loading` multiplies
  `input_scale` into `weight_scale_2`.
- TRTLLM: pads hidden to 256 and intermediate to 64, shuffles weights (`prepare_static_weights_for_trtllm_fp4_moe`) - n/a on sm_120.
- MARLIN: `prepare_nvfp4_moe_layer_for_marlin` repacks to Marlin tiles, input scales dropped (A16).
- B12X: `prepare_nvfp4_moe_layer_for_b12x` (`quantization/utils/b12x_moe.py`), its own packing; W4A4 order `w31`, A16 `w13`.

All of these keep **expert as dim 0** of every per-expert tensor, so TidalMoE-style per-expert row permutation / VMM
placement over dim 0 still works after the conversion (true for TidalMoE's TRT-LLM layout too).

### 3.3 Q8_0 and other GGML blocks via b12x (sm_12x only)

`ModelOptMixedPrecisionConfig` (`modelopt.py:1574`, `quant_algo: "MIXED_PRECISION"`, per-prefix `quantized_layers`) accepts
`quant_algo` in `B12X_BLOCK_CODECS = {"iq2_xs": (256,74), "iq2_xxs": (256,66), "q8_0": (32,34)}` with
`{"group_size": 32, "block_payload_bytes": 34, "packing": "ggml"}` and dispatches to `ModelOptBlockQuantLinearMethod`
(Linear, VocabParallelEmbedding (Q8_0 only), ParallelLMHead) or `ModelOptBlockQuantMoEMethod` (routed experts) in
`quantization/modelopt_block_quant.py`; BF16 activations; requires b12x on SM12x and the b12x linear backend. So the
skeleton's Q8_0 attention / GDN / shared expert / embedding / LM head (and the Q8_0 latent D/U) can be loaded **bit-exact
without dequantizing to BF16**, at 1.0625 B/param instead of 2: roughly 4-5 GB saved on the skeleton alone (its Q8_0 tensors are most of the 5.99 GB file; HC weights are BF16 already), i.e. ~2,000-2,500 more hot experts (plus ~3.3 GB if the latent D/U stay Q8_0).
Mixed with `NVFP4` for experts and `FP8` elsewhere in the same `quantized_layers` map.

## 4. Out-of-tree model / plugin registration

- Entry-point group **`vllm.general_plugins`** (`vllm/plugins/__init__.py`, `DEFAULT_PLUGINS_GROUP`); `load_general_plugins()`
  runs every plugin's function once per process (API server, engine core, workers). `VLLM_PLUGINS=a,b` is an allowlist.
  Image ships two: `lora_filesystem_resolver`, `lora_hf_hub_resolver`. Other groups: `vllm.platform_plugins`,
  `vllm.io_processor_plugins`, `vllm.stat_logger_plugins`, `vllm.endpoint_plugins` (needs `VLLM_PLUGINS`).
- **`ModelRegistry.register_model(arch, "module:Class")`** (`model_executor/models/registry.py:1176`); use the lazy string
  form so CUDA is not initialized at import; re-registering an existing arch overwrites it (debug log). The arch name must
  equal `architectures[0]` in `config.json` (e.g. `Edge0LatentForCausalLM`, or override `Qwen4ExpForCausalLM` itself).
  For a new `model_type` also register the HF config (`AutoConfig.register`) inside the plugin, or reuse
  `model_type: "qwen4_exp_text"` and add Edge0 keys as extra attributes.
- **Model-level post-load hook**: `model_loader/utils.py:process_weights_after_loading` runs every module's
  `quant_method.process_weights_after_loading` (this is where MoE weights get reordered / swizzled and the MoE kernel is
  built), then deferred attention layers, then **`model.process_weights_after_loading()` if the model has it** (line 209).
  `Qwen4ExpForCausalLM` does **not** define it, so a plugin must add it (TidalMoE's `register()` only wraps a method already
  in `cls.__dict__`). This runs before memory profiling, KV sizing and CUDA-graph capture: the right place for expert placement.
- **Installing into a derived image** (`FROM nvcr.io/nvidia/vllm:26.11-py3`, tag `edge0-vllm:dev`): either
  `COPY edge0/plugin /opt/edge0/plugin` + `RUN pip install --no-deps /opt/edge0/plugin` (respects
  `PIP_CONSTRAINT=/etc/pip/constraint.txt`; never let pip touch torch/vllm/flashinfer), with
  `[project.entry-points."vllm.general_plugins"] edge0 = "edge0_vllm:register"`; or TidalMoE's zero-install trick: a
  directory on `PYTHONPATH` holding the package plus a `<name>-0.1.dist-info/{METADATA,entry_points.txt}`
  (`plugin/moe_placement-0.1.dist-info`), bind-mounted at run time (`-v ...:/moe/plugin:ro -e PYTHONPATH=/moe/plugin`). The
  second needs no image rebuild while iterating.

## 5. Offload facilities in 0.33.0

| Facility | CLI | Mechanism | CUDA graphs |
|---|---|---|---|
| UVA weight offload | `--offload-backend uva --cpu-offload-gb N [--cpu-offload-params experts.w13_weight experts.w2_weight]` (`config/offload.py:UVAOffloadConfig`) | `UVAOffloader` (`model_executor/offloader/uva.py`): copies params to pinned host and replaces `p.data` with a CUDA view (`get_accelerator_view_from_cpu_tensor`), sets `p._vllm_is_uva_offloaded=True`; budget is filled in module order (whole params, first come first served); segment-exact name matching | addresses fixed, graphs fine; kernels read host memory over PCIe on every replay. `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1` falls back to per-forward `functional_call` H2D copies |
| Prefetch offload | `--offload-backend prefetch` + `offload_group_size`, `offload_num_in_group`, `offload_prefetch_step`, `offload_params` | `PrefetchOffloader` (`offloader/prefetch.py`, from SGLang): whole layers offloaded to pinned host, H2D into a `StaticBufferPool` on a copy stream N layers ahead; custom ops `torch.ops.vllm.wait_prefetch/start_prefetch` | copy stream is forked/joined with events so the H2D copies are **captured into the graphs** (re-copied each replay). Layer granularity: an Edge0 layer's experts are 0.99 GB -> ~20 ms at PCIe Gen5, useless for decode |
| Triton MoE + host weights | automatic | `fused_moe.py:_weights_host_resident` disables the "naive block assignment" for UVA-offloaded experts so each expert is read once per call | n/a |
| Load-time staging | `--cpu-offload-gb` large | TidalMoE used it only to fit loading, then re-placed experts itself | |
| KV CPU offload | `--kv-offloading-size <GiB> [--kv-offloading-backend native|lmcache]` | `native` -> `OffloadingConnector` (`VLLM_USE_SIMPLE_KV_OFFLOAD=1` -> `SimpleCPUOffloadConnector`, what TidalMoE used via `--kv-transfer-config`); `lmcache` not installed | connector-based. **Qwen4Exp QSA backend returns `supports_kv_connector() = False`** (`qsa.py:221`), so expect it to be rejected for this model; KV is small anyway (2.4) |
| PLE / Engram | `--engram-config` | section 2.6 | file-gather staging is graph safe |
| EPLB / redundant experts | `--enable-eplb`, `num_redundant_experts` | multi-GPU expert balancing, `expert_map` (global -> local, -1 = not local) | n/a on one GPU, but `expert_map` is a usable indirection |
| Routing capture | `--enable-return-routed-experts` | returns per-token routed experts in responses (TidalMoE `tools/collect_routing.py` relies on it) | |

## 6. TidalMoE plugin: hook audit and portability to sm_120 + PCIe Gen5

Physical difference that matters: TidalMoE's cold experts sat behind NVLink-C2C (~378 GB/s measured) next to 288 GB HBM;
here they sit behind PCIe Gen5 x16 (~50-55 GB/s expected, to be measured) next to 32 GB GDDR7 (~1.8 TB/s). A cold byte is
~7x more expensive and the hot set is far smaller (estimate: 35-40% of the 47.6 GB of experts fit next to the skeleton, D/U, MTP and KV), so **every TidalMoE mechanism that
lowers the number of cold bytes moved per step, or overlaps them with compute, is worth more here, not less.**
Per decode token Edge0 touches 48 x 10 experts x 1,935,376 B = 929 MB of expert weights; at batch 32 roughly half of each
layer's 512 experts are touched per step, so "touch" (distinct experts per step) is the right cost unit, as TidalMoE found.

### 6.1 Hook audit (TidalMoE symbol -> status in this image)

| TidalMoE function | vLLM / FlashInfer symbol hooked | Exists here? | What changed / what to do |
|---|---|---|---|
| `register()` | `vllm.models.deepseek_v41.nvidia.{vl_model,model}.DeepseekV41ForCausalLM.process_weights_after_loading` | yes (`nvidia/vl_model.py:118/327`; `nvidia/model.py` class is now `DeepseekV41LLMForCausalLM:1609`) | target becomes the Edge0 / `Qwen4ExpForCausalLM` class; Qwen4Exp has no such method -> **add** it, do not wrap |
| `_moe_layers()` | `vllm.model_executor.models.utils.extract_layer_index`; modules with `.gate`, `.experts`, `.n_routed_experts` | yes (`utils.py:1066`); `Qwen3NextSparseMoeBlock` has all three | E = 512; the **MTP layer also has 512 experts** (TidalMoE filtered the 128-expert DSpark draft by count) -> filter by module prefix |
| `_walk_tensors()/_expert_groups()` | `m.experts.routed_experts...`, tensors with dim0 == E | yes | works on any backend layout that keeps expert on dim 0 (all of 3.2); gate rows `mlp.gate.weight [512,2560]` permute the same way; there is no `e_score_correction_bias` |
| `_alloc_segments()` (VMM) | CUDA driver `cuMemCreate` with `CU_MEM_LOCATION_TYPE_HOST_NUMA` + device pages on one `cuMemAddressReserve` range | enums present in cuda-bindings 13.4.2 (`HOST_NUMA`, `HOST`, `CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`, `HOST_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`) | **unverified on x86 + RTX 5090**: `gpu_host_probe.py` section 4 tests it. Fallback: hot tensor in VRAM + cold tensor as UVA view of pinned memory and the split call (6.2 #2) |
| `_place_layer()` sets `_parameters[k].data` and `_vllm_is_uva_offloaded=False` | `UVAOffloader` flag | yes (`offloader/uva.py:99/122`, `model_loader/utils.py:233-261`, `fused_moe.py:1546`) | same; note Triton MoE reads the flag |
| `_patch_split()` | `vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe.TrtLlmMxfp4ExpertsModular.apply` + `flashinfer.trtllm_fp4_block_scale_routed_moe`, `tm.fi_moe_largest_bucket`, `tm.RoutingMethodType`, `FusedMoEConfig.limit_deferred_moe_finalize/should_defer_moe_finalize` | all exist (`trtllm_mxfp4_moe.py:285`, `fused_moe/utils.py:566`, `config.py:1525/1534`) | **not on the sm_120 path** (TRT-LLM is sm_100). Re-implement on `FlashInferExperts.apply` (same modular signature, `modular_kernel.py:958`) with `cutlass_fused_moe`'s local-expert range, or on `CutlassExpertsFp4`/`MarlinExperts` |
| `_patch_adapt()` counting | `TrtLlmMxfp4ExpertsModular.apply` | exists, wrong class for sm_120 | wrap the selected experts class's `apply` (signature identical across `FusedMoEExpertsModular`) |
| `_patch_adapt()` epoch tick, `_StepTrace` | `vllm.v1.worker.gpu.model_runner.GPUModelRunner.execute_model(self, scheduler_output, intermediate_tensors=None, dummy_run=False, ...)` | yes (`model_runner.py:1785`, V2 is default) | portable as is |
| `_patch_tactics()` | `flashinfer.autotuner.AutoTuner.choose_one`, op `flashinfer::trtllm_fp4_block_scale_moe`, 384 experts / I 2304 | `choose_one` exists (`flashinfer/autotuner/autotuner.py:2165`) | CUTLASS fused MoE tunes via ops `"trtllm::fused_moe::gemm1"` / `"gemm2"` (`flashinfer/fused_moe/core.py:1038/1054`); retarget and re-derive bit-exact tactics |
| `_patch_mhc_mega()` | `vllm.models.deepseek_v41.nvidia.model.supports_mhc_overlap` | yes (`model.py:117`) | DSV4.1 only; n/a |
| `_patch_av()` | `vllm.v1.worker.gpu.spec_decode.adaptive_verification.AdaptiveVerificationManager.{set_initial_cost_curves,get_num_tokens,set_cost_curves,_batch_budget,_max_total_logits,num_bonus_tokens,cost_tables,_confidence_probs,num_speculative_steps,_batch_draft_capacity}`, `_assign_draft_token_budget_compiled` | all exist (lines 75-446) | AV is `method="dspark"` only; Edge0 has an MTP head -> n/a |
| `_patch_av_uniform_cg()` | `vllm.v1.worker.gpu.cudagraph_utils.CudaGraphManager._init_candidates/dispatch`, `BatchExecutionDescriptor`, `build_dynamic_sd_schedule_lookup`, fields `varlen_decode,_capture_descs,lora_capture_cases,ubatch_runner,decode_query_len,_candidates` | all exist (`cudagraph_utils.py:71/150/256/533`; `build_dynamic_sd_schedule_lookup` now imported from `vllm.v1.spec_decode.dynamic.utils`) | only meaningful with AV; n/a |
| `_patch_av()` gen counter | `vllm.v1.core.sched.scheduler.Scheduler.update_from_output` | yes (`scheduler.py:2259`) | portable (diagnostic) |
| `persist_moe.patch_apply/bind`, `_why_not` | `TrtLlmMxfp4ExpertsModular.apply`, `fused_moe.moe_output.UnfinalizedMoEOutput`, `fused_moe.activation.MoEActivation` | all exist | kernels are CuTe DSL **tcgen05/UMMA/TMEM (sm_100)** + MXFP8 activations + TRT-LLM MXFP4 layout (`persist_moe_cute.py`, from FlashInfer `low_latency_blockscaled_gemm.py`): **do not run on sm_120** |
| `persist_front.py` | Triton + `triton.language.extra.cuda.gdc_*` (PDL) | Triton 3.9 present | front/finalize logic portable; GEMMs need an sm_120 rewrite |
| `tools/collect_routing.py` | `--enable-return-routed-experts` | yes (`arg_utils.py:495`) | change L/E/K to 48/512/10 |
| `tools/analyze_routing.py`, `plan_convex.py`, `relayout.py` | offline | n/a | constants (L=40, E=384, K=6, 11.25+5.625 MiB/expert, G=8 groups, 64 us/cold expert) must be re-derived for 48/512/10, 1.85 MiB/expert, PCIe |

### 6.2 Mechanism classification for an sm_120 PCIe box (goal: less expert load time)

| # | TidalMoE mechanism | Class | Why / what to change |
|---|---|---|---|
| 1 | Per-expert hot/cold placement on one VA range (VMM device + host pages), experts permuted with router rows so routing is unchanged | **portable, needs-adaptation** | The core idea maps 1:1 (Strata's `--expert-profile profile.bin` ranking is the same idea). Adapt: model class hook, E=512, MTP filter, FlashInfer-CUTLASS layout (w13 = [w3,w1] after reorder, swizzled scales are per-expert rows too), plan from `profile.bin` + `collect_routing`. Verify HOST_NUMA VMM on this host. Expect a much smaller hot fraction; placement quality matters more |
| 2 | `MOE_SPLIT`: hot and cold experts as two concurrent kernel calls (hot on main stream, cold on side stream) | **needs-adaptation, high value** | On PCIe the cold call is bandwidth-bound and the hot call compute/VRAM-bound; running them concurrently hides the hot part. Re-implement on `FlashInferExperts`/`cutlass_fused_moe` (expert range = local experts) or two `MarlinExperts` calls. Also the natural fallback if VMM host pages are not supported (cold weights as a separate UVA tensor) |
| 3 | Adaptive expert cache: `[L,E]` GPU counters captured in graphs, epoch re-ranking (EMA, Poisson z-test), in-place row swaps HBM<->DRAM so graph addresses stay valid; save/resume | **portable, high value** | A swap costs 2 x 1.94 MB over PCIe (~80 us), each good swap saves a PCIe read on every step that touches it. Use `MOE_ADAPT_COUNT=touch` (cost model = distinct experts per step). Wrap the sm_120 experts class `apply`; MTP layer can have its own cache |
| 4 | Kernel-aware layout (cold slots in G=8 groups spread over the layer) + convex per-layer budgets (`plan_convex.py`) | **needs-adaptation** | The grouping was tuned for TRT-LLM CTA dispatch order and 2 MiB pages. For FlashInfer CUTLASS grouped GEMM on sm_120 re-measure per-layer cost vs cold count (Nsight), re-fit the convex model with PCIe numbers. Per-layer budgets still matter (routing skew differs per layer) |
| 5 | Forced MoE tactics (autotuner profiled on a different host/HBM mix) | **needs-adaptation** | Same failure mode exists (autotune sees all-VRAM weights). Retarget `AutoTuner.choose_one` to `trtllm::fused_moe::gemm1/gemm2`; only bit-exact tactics |
| 6 | `MOE_HOST_PARAMS`: token embedding to host (gather-only, ~64 rows per step) | **portable as is** | `embed_tokens` 248320 x 2560 is 1.27 GB BF16 / 0.68 GB Q8_0 -> 350-650 more hot experts. Never the LM head |
| 7 | KV sized exactly + 32 GiB CPU KV tier (`SimpleCPUOffloadConnector`) | KV sizing **portable**; CPU KV tier **not applicable** | KV is tiny here (2.4), so cap it explicitly (`--kv-cache-memory-bytes`) and give the rest to hot experts; QSA rejects KV connectors |
| 8 | Host-aware persistent MoE kernels (few CTAs stream cold experts, rest stream HBM, one launch, PDL) | **idea portable, code not portable** | tcgen05/TMEM CuTe kernels are sm_100-only. The insight transfers strongly: PCIe saturates with a handful of SMs, so dedicate few CTAs to cold tiles and keep the rest on VRAM tiles inside one launch. Needs a new sm_120 kernel (CUTLASS sm120 block-scaled MMA, b12x, or Triton `tl.dot_scaled`) - the biggest effort item; do after 1-3 are measured |
| 9 | Calibrated DSpark adaptive verification (cost curve incl. MoE) | **not applicable** (MTP, not DSpark) | Keep the lesson: vLLM's profiled step costs use padding tokens that route to no expert, so they omit expert loading; tune MTP `num_speculative_tokens` / `num_speculative_tokens_per_batch_size` / `draft_confidence_threshold` from measured step times |
| 10 | Uniform-decode CUDA graphs under AV | **not applicable** | AV only |
| 11 | Mega-mHC | **not applicable** | DSV4.1 only (Qwen4Exp has its own HC kernels with sm_120 plans) |
| 12 | Bandwidth trace (`MOE_BW_TRACE`), step trace, gen counter | **portable** | essential to measure cold reads per step on PCIe |
| 13 | Load-time UVA staging (`--cpu-offload-gb 60 --cpu-offload-params ...w13_weight ...w2_weight`) | **portable** | needed: 47.6 GB of experts cannot be staged in 32 GB VRAM; RAM budget (~56 GB) must hold the cold set + PLE page cache, so the placement must not keep a second host copy |

Additional sm_120/PCIe-specific levers found in this image (not in TidalMoE): Q8_0-native skeleton via b12x (3.3) and
CSF-compressed expert scales (`VLLM_CSF_RESIDENT`) both shrink bytes so more experts stay hot; copy-engine prefetch of
predicted experts (Strata's `--pcie-mode dma`) is the alternative to SM zero-copy reads, but cannot be data-dependent inside
a CUDA graph (needs a pre-graph host decision or a gather kernel). `gpu_host_probe.py` measures copy-engine vs SM gather
bandwidth for expert-sized records so the choice can be made on numbers.

## 7. Speculative decoding for the MTP head

- Methods in this build (`SpeculativeMethod`): `ngram`, `ngram_gpu`, `suffix` (needs arctic-inference, V1 only), `draft_model`,
  `eagle`, `eagle3`, `mtp` and per-model `*_mtp` aliases incl. **`qwen4_exp_mtp`**, `dflash`, `dspark`, `custom_class`,
  `extract_hidden_states`.
- For Qwen4Exp use `--speculative-config '{"method":"mtp","num_speculative_tokens":4,...}'`: `config/speculative.py:946`
  rewrites the draft hf_config `qwen4_exp(_text)` -> `qwen4_exp_mtp`, architecture `Qwen4ExpMTP`, `n_predict` from
  `mtp_num_hidden_layers` (or `num_nextn_predict_layers`), `hc_mult = hc_count`. The target exports its multi-stream
  hidden (`get_mtp_target_hidden_states`). V2 speculator: `v1/worker/gpu/spec_decode/mtp/speculator.py`.
- Useful knobs (all V2): `num_speculative_tokens_per_batch_size: [[lo,hi,k],...]` (dynamic SD schedule by batch size);
  `draft_confidence_threshold` (stop drafting below top-1 prob; Strata ships `--spec 4 --spec-min-p 0.5`, the same idea);
  `draft_token_map` (FR-Spec reduced draft vocab: path to a JSON list or SGLang `.pt` of target token ids; the image even ships
  `v1/worker/gpu/spec_decode/draft_vocab_lists/qwen38_flash_next_32k.json`, a 32,768-id frequency list for this tokenizer,
  12,602 of its ids >= 32768, max 248076) and `draft_token_map_dynamic_rows`
  / `draft_token_map_dynamic_rank` (low-rank rescue of rows outside the list, TP1); `index_share_for_mtp_iteration` (reuse the
  first MTP step's QSA top-k indices in later steps); `draft_sample_method`, `rejection_sample_method`.
  Edge0 ships `mtp/draft_vocab.bin` = 106,299 sorted uint32 token ids (425,196 B, starts 0,1,2,...): write it out as a JSON list
  for `draft_token_map`.
- `enable_adaptive_verification` is DSpark-only; `parallel_drafting` is for DFlash/DSpark; `use_heterogeneous_vocab` is
  `draft_model` only and forces the V1 runner.
- Every MTP step runs a 512-expert MoE: its experts (0.71 GB in `mtp/experts.bin`) should be fully VRAM-resident, otherwise
  drafting pays PCIe reads too.

## 8. Risks, blockers, next probes

1. **CUDA 13.5 image on a 13.2 driver** (GeForce, no forward compat): expected to run in Minor Version Compatibility mode
   (`52-gpu-driver-version-check.sh` prints a WARNING). Watch for PTX-JIT failures (anything shipping PTX newer than 13.2:
   Triton and FlashInfer JIT compile to cubin with the 13.5 `ptxas`/`nvcc`, so they should be fine). First thing after
   GPU_FREE: `edge0/tools/gpu_host_probe.py` (also checks vLLM custom ops, which cannot load without `libcuda`).
2. VMM HOST_NUMA backing on x86 + discrete GPU is unverified (probe section 4). Fallback is the split call (6.2 #2).
3. Latent MoE with 8 projection groups has no vLLM equivalent (2.5); FusedMoE handling of invalid / group-masked top-k slots
   must be checked per backend.
4. QSA `supports_kv_connector=False`: no CPU KV tier (not needed).
5. PLE file gather is single-threaded `preadv` in `prepare_inputs`; RAM is shared between pinned cold experts and PLE page cache.
6. No `gguf` package in the image: conversion runs outside vLLM (Strata's `third_party/llama.cpp/gguf-py` works when mounted,
   numpy is in the image).
7. `b12x` paths are sm_12x-specific and newer (FLASHINFER_B12X excluded from auto selection on purpose); validate numerics
   against Strata before relying on Q8_0-native layers.
