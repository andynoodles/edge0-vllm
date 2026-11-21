# edge0-vllm plugin: what was built, how it hooks into vLLM, what was measured

Status: 2026-11-19, `model` agent. It implements DESIGN.md sections 2 and 3 (phase 1) for `nvcr.io/nvidia/vllm:26.11-py3`
(vLLM 0.33.0). The full model has **not** been started on the GPU (no `vllm serve` yet). The GPU-only parts were
checked at unit level on the L4 mini checkpoint after GPU_FREE appeared: one layer's expert pools, the op, and one
complete `Edge0LatentMoeBlock`. All numbers below come from those runs.

Paths: the source lives in the worktree at `edge0/plugin`, `edge0/docker` and `edge0/serve.sh`. On the box the tree
is `~/work/edge0-vllm/`, mounted at `/workspace/edge0` in containers. The image is `edge0-vllm:dev`; the plugin is
installed at `/opt/edge0/plugin`.

---

## 1. Files

| File | What it does |
|---|---|
| `plugin/pyproject.toml` | Package `edge0-vllm`. Its entry point is `vllm.general_plugins: edge0 = edge0_vllm:register`. It has no dependencies, so pip never touches torch or vLLM. |
| `edge0_vllm/__init__.py` | `register()`: registers the architecture lazily, registers the `edge0` quantization, and patches vLLM's architecture tables (section 2). It does not initialize CUDA. |
| `config.py` | Geometry constants: one slot is 1,935,360 B; the Marlin per-expert shapes; the ModelOpt tensor table. It also has `Edge0Settings` (the `EDGE0_*` env, validated) and `Edge0CheckpointSpec` (the `config.json` `edge0` block). Pure Python. |
| `placement.py` | `profile.bin` reader, top-H plan from the ranking, identity plan, plan JSON (`edge0-placement-v1`) load and save, budget checks, `hot_map`/`cold_map`. Uses numpy only. |
| `runtime.py` | Per-process state: settings, spec, placement, logger mirrored to `EDGE0_LOG_DIR`, the trace, and the list of blocks. `Edge0ForCausalLM` creates it before the layers exist. |
| `quant.py` | `Edge0QuantConfig(ModelOptMixedPrecisionConfig)`, registered as `edge0`, plus the Q8_0→BF16 dequant-on-load linear method. |
| `q8.py` | Q8_0 block dequantization (fp32 math, bf16 RNE output, 4096-row chunks) and `wrap_weight_loader`, which keeps the `shard_id` tag. |
| `marlin_nvfp4.py` | ModelOpt→Marlin conversion per expert (`gptq_marlin_repack`), scale processing with a factor shared by the whole layer (computed chunk by chunk since round 2: no layer-sized fp32 transient at load), global scales. Also a pure-torch NVFP4 dequant and one-expert SwiGLU, used by F1 and the tests. |
| `hostmem.py` | Pinned, device-mapped host buffers with a CPU view and a UVA CUDA view. Backends: `host_register` (mmap + `MADV_HUGEPAGE` + `cudaHostRegister`), `host_alloc` (`cudaHostAlloc`), `torch_pinned` (vLLM's UVAOffloader mechanism). |
| `expert_pool.py` | `ExpertPool` for one layer: a hot `Tier` in VRAM and a cold `Tier` in host memory, both in Marlin layout. It loads chunk by chunk (64 experts), `finalize()` sets the global scales, and `swap()` / `swap_many()` (batched, graph-safe, used by `adaptive.py`) exchange hot and cold experts without moving addresses. Optional staging slots after the hot ones (`n_stage`, `prefetch.py`). |
| `kernels.py` | Triton kernels with torch fallbacks: route-as-token `gather_routes`, deterministic `combine_routes`, `touch_count` (TidalMoE's counter), `tier_count`, `route_topk_softmax` (vLLM `topk_softmax`, called without the padding mask), and `route_post` (`EDGE0_ROUTE_FUSE=1`: padding reroute + tier count of a decode call in one kernel, bit-identical to the six kernels it replaces). |
| `latent_moe.py` | `Edge0LatentMoeBlock`, the custom op `vllm::edge0_latent_moe`, the process-wide `_MoeWorkspace`, and the op bodies `latent_experts_marlin` and `latent_experts_torch` (F1). `marlin_tier_direct` runs a tier's Marlin call with a chosen M block / CTA cap (`EDGE0_COLD_BSM=64 EDGE0_BSM_MIN_TOKENS=64` in the tuned config: one host read per cold expert per prefill chunk, TUNING-LOG.md 2 and 4.4). |
| `model.py` | `Edge0ForCausalLM(Qwen4ExpForCausalLM)`: swaps the MoE class while the model is built, sends expert tensors to the pools in `load_weights`, runs the model-level `process_weights_after_loading`, and flushes the trace from `compute_logits`. |
| `ple.py` | Patch of `discover_table_layout` so the PLE table is read in place from `ple-fp8.gguf`, and the optional threaded row reader (`EDGE0_PLE_IO=threads`). |
| `tier.py` (round 3 additions) | `EDGE0_TIER_EARLY=1` (on in `PRESET=single`): `on_step` only snapshots the counts; a hook around `model_state.stage_checkpoint_rows` (`pre_sync` / `post_sync`) waits for the forward's snapshot at the next step's staging, decides, issues the copies (`EDGE0_TIER_COPY=batch`: one `cudaMemcpyBatchAsync` per direction, the H2D before the victims' bookkeeping) and flips on the copy stream; the next forward waits for the flip. `EDGE0_TIER_SHADOW` (default with EARLY): swapped-in experts keep their host slot, so evicting them needs no D2H; `EDGE0_TIER_SPARE` extra spare host slots per layer; `EDGE0_TIER_WB_COST`, `EDGE0_TIER_GPU_DECIDE` measured, off. `draft.install_dflash` (`EDGE0_DFLASH_SHARE_VOCAB=1`, off): the DFlash2 drafter built without its own vocab matrices. TUNING-LOG.md "Single-stream" R3. |
| `ple_defer.py`, `csrc/ple_reader.cpp`, `csrc/ple_wait.cu` | `EDGE0_PLE_DEFER=1` (on in `PRESET=single`): PLE rows off the critical path. `stage_checkpoint_rows` keeps its stream sync but no longer reads: a kernel posts the step's n-gram ids to mapped memory and rings a doorbell, a C++ reader thread (no GIL, no CUDA calls) reads the rows exactly as vLLM's `_FilePlane.read` does into mapped memory and publishes the step number; the graph is launched at once and the PLE embedding's forward (decoder layer 1) waits for that number in-graph, then copies the rows into vLLM's staging buffer. Separate extension `edge0_ple` (prebuilt by the Dockerfile). TUNING-LOG.md "Single-stream" R1. |
| `trace.py` | `EDGE0_TRACE=bw`: per-layer touch counters, safe inside CUDA graphs, flushed every 10 s. `EDGE0_TRACE=routes`: a GPU ring buffer of logical expert ids plus the MoE-call boundaries (`calls`), dumped to `.npz` for `tools/plan_placement.py`. `EDGE0_TRACE=predict` (`PredictTrace`): accuracy of next-layer pre-gating. (Tuning agent, see TUNING-LOG.md.) |
| `nvtx.py` | `EDGE0_NVTX=1`: NVTX ranges on the MoE op parts and on the host phases of each engine step (runner, graph replays, PLE staging and `preadv`, scheduler, speculative propose and reject, adaptive epochs); `EDGE0_NVTX=2` also adds per-module ranges for eager calibration. Off by default, and then nothing is patched. Used by `bench/nsys-capture.sh` and `tools/nsys_breakdown.py` (docs/PROFILE.md). |
| `adaptive.py` | `EDGE0_ADAPT=1`: TidalMoE's adaptive expert cache (graph-safe decode touch counters, EMA + Poisson z-test, <= 64 in-place swaps per 3 s epoch through `ExpertPool.swap_many`). Kept in the tuned config. |
| `prefetch.py` | `EDGE0_PREFETCH=S`: next-layer pre-gated prefetch into S sticky VRAM staging slots per layer (SM gather kernel on a low-priority stream). Opt-in experiment, rejected in TUNING-LOG.md 4.6. |
| `cpu_experts.py`, `csrc/` | `EDGE0_CPU_EXPERTS=1`: Strata-style CPU computation of part of each decode call's cold experts (docs/CPU-EXPERTS.md). C++/CUDA extension built by `torch.utils.cpp_extension` (prebuilt by the Dockerfile into `/opt/edge0/ext`): an AVX-512 kernel over the Marlin NVFP4 record (same roundings as Marlin's epilogue), a pinned busy-polling worker pool, and three graph-capturable kernels (dispatch, gather + doorbell, wait). `kernels.combine_routes` reads the CPU's rows from mapped memory. Off by default; see TUNING-LOG.md "CPU cold experts (user request)". |
| `tests/` | Pytest suite, see section 5. |
| `docker/Dockerfile`, `docker/Dockerfile.dockerignore` | `FROM nvcr.io/nvidia/vllm:26.11-py3`, copies `plugin/`, runs `pip install --no-deps`, and asserts the entry point. The BuildKit ignore file next to the Dockerfile limits the build context to `plugin/`. |
| `serve.sh` | Refuses to start until `GPU_FREE` exists. Runs the container in tmux session `edge0-vllm-serve` on port 8125. `DRY_RUN=1` prints the docker command. All switches are documented at the top of the script. |

Build: `cd ~/work/edge0-vllm && nice -n 19 docker build -f docker/Dockerfile -t edge0-vllm:dev .` (about 13 s).

## 2. How it attaches to vLLM

- **Registration.** `ModelRegistry.register_model("Edge0ForCausalLM", "edge0_vllm.model:Edge0ForCausalLM")`, plus
  `register_quantization_config("edge0")`. vLLM resolves the checkpoint directly: its log says "Resolved
  architecture: Edge0ForCausalLM" and `quantization == "edge0"`.
- **Architecture tables.** These were not in DESIGN.md but are required. vLLM keys Qwen4Exp behaviour on the
  architecture name in three places, and `register()` adds `Edge0ForCausalLM` to each:
  - `model_executor/models/config.py` `MODELS_CONFIG_MAP`: the Qwen4Exp hybrid-cache contract and config checks, the
    same `Qwen4ExpForCausalLMConfig` object.
  - `config/vllm.py` `DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES`: breakable CUDA graphs, which vLLM turns on
    automatically for Qwen4Exp. The `lru_cache` is cleared.
  - `config/engram.py` `_NGRAM_LAYER_FIELDS`: without this entry `model_has_engram_layers` is false and
    `--engram-config` does not apply.
- **The model.** `Edge0ForCausalLM.__init__` calls `runtime.setup` (placement), then `ple.install_patch`, then builds
  `Qwen4ExpForCausalLM` while `vllm.models.qwen4_exp.nvidia.model.Qwen4ExpSparseMoeBlock` is replaced by
  `Edge0LatentMoeBlock`. The decoder, the fused-shared-expert check and `set_moe_parameters` all look the class up at
  call time. `set_moe_parameters` is run again after the swap, so `num_moe_layers` is 0: there is no EPLB or
  routed-expert capture.
- **The block.** It has the same constructor signature and attribute surface as the stock block (`n_*_experts`,
  `is_fused_shared_expert_enabled=False`, `experts.moe_config` stub). Its modules are `gate` (`GateLinear`, fp32
  logits), `latent_down` and `latent_up` (`ReplicatedLinear`, quant config `edge0`), `shared_expert` (`Qwen2MoeMLP`
  with `expert_gate`), `shared_expert_gate`, and the buffer `expert_group[512]` (int32). The pool is a plain attribute,
  not a parameter, so vLLM never moves it or puts it in the state dict. The block registers itself in
  `static_forward_context[prefix]`, the way FusedMoE does.
- **The op `vllm::edge0_latent_moe(xl, router_logits, layer_name)`.** It is registered with
  `direct_register_custom_op` and a fake implementation. The layer name uses vLLM's `LayerName` opaque type
  (`_encode_layer_name`), as vLLM's own MoE op does, so the compiled graph does not recompile per layer. Its body:
  1. `topk_softmax`, top-10, renormalize;
  2. Triton gather into the shared `x_routes` buffer;
  3. the hot Marlin call on the main stream and the cold Marlin call on the side stream (`EDGE0_SPLIT=1`);
  4. Triton combine into `z`.
- **Loading.** `Edge0ForCausalLM.load_weights` matches
  `^(model.|model.language_model.)layers.N.mlp.experts.<name>$` and sends those tensors to
  `block.load_expert_tensor`. Everything else goes through vLLM's `Qwen4ExpForCausalLM.load_weights`, which applies
  the stacked-name mapper, AutoWeightsLoader, the `mtp.*` drop, and the PLE `weight_scale`.
- **Post-load hook.** `Edge0ForCausalLM.process_weights_after_loading()` is called by vLLM after the per-module quant
  finalization and before memory profiling and graph capture. It runs `pool.finalize()` (global scales, check of
  `expert_group` values) for every layer, moves `embed_tokens` to pinned host memory, creates the trace, and logs the
  VRAM and RAM breakdown.

## 3. Deviations from DESIGN.md (all deliberate; please review)

1. **The Q8 dequant method is an instance of the stock `UnquantizedLinearMethod`, not a subclass.** Only its
   `create_weights` is wrapped, so that the parameter's `weight_loader` dequantizes Q8 blocks. The reason is that
   `enable_qwen4_exp_low_latency_gemm` converts a layer only when `type(quant_method) is UnquantizedLinearMethod`. With
   the stock class, the dequantized GDN qkvz, out-proj, shared-expert and `latent_down` (14336×2560 happens to match a
   QSA plan shape) layers get the sm_120 skinny-GEMM plans. The meta build confirms that
   `Qwen4ExpLowLatencyLinearMethod` is in use.
2. **The slot table is folded into the expert maps.** The design's `slot = slot_of[ids]` gather is not needed:
   `hot_map[e]` and `cold_map[e]` map logical ids directly to tier slots, which is exactly the `expert_map` argument
   `fused_marlin_moe` takes. `ExpertPool.order` keeps slot→logical on the host. A swap updates two entries in each map.
3. **Workspaces are process-wide, not per block.** Layers run one after another, so there is a single `x_routes`, and
   per tier one `c13`, one `c2` and one Marlin lock buffer, plus one side stream. Size at max_num_batched_tokens 4096:
   x_routes 147 MB + 2 × (147 + 52) MB = 545 MB. `EDGE0_MOE_CHUNK_TOKENS` caps this, at the cost of re-reading cold
   experts once per chunk. The buffers are allocated on the first forward, which is vLLM's profile run, so vLLM
   counts them.
4. **"Fall back to vLLM's own offload"** (the task asked for a simple switch):
   - `EDGE0_COLD_ALLOC=torch_pinned` allocates the cold pools with vLLM's UVAOffloader mechanism
     (`torch.empty(pin_memory=True)` + `get_accelerator_view_from_cpu_tensor`).
   - `serve.sh VLLM_OFFLOAD_GB=N` adds `--offload-backend uva --cpu-offload-gb N` for dense parameters.
   - vLLM's FusedMoE + UVA for the routed experts stays rejected (DESIGN §6), and the plugin does not implement it.
   - F0 (`EDGE0_EXPERT_VRAM_GIB=0`) is the all-host fallback for the experts.
5. **`EDGE0_EXPERT_VRAM_GIB` has no default in the plugin** (`serve.sh` sets 9.5). An accidental all-cold run would
   pin 44.3 GiB.
6. **Phase-3 knobs raise `NotImplementedError` instead of being ignored.** They are `EDGE0_ADAPT`,
   `EDGE0_STREAM_MIN_TOKENS`, `EDGE0_COLD_SCALES=vram` and `EDGE0_COLD_ALLOC=vmm_host_numa`. `adaptive.py` and
   `stream.py` do not exist yet. `ExpertPool.swap()` and the touch counters, which the adaptive cache will use, are
   implemented and tested.
7. **The F1 reference copies stay in host RAM** as ModelOpt CPU tensors, about 1.6 GB per layer, for the first
   `EDGE0_REF_LAYERS` (default 4). The F1 path copies each touched expert to the GPU per call.

## 4. Numbers measured (RTX 5090, PCIe Gen5 x16, host driver 595.99, image CUDA 13.5)

One layer (L4 checkpoint, layer 0), Marlin W4A16 NVFP4, bf16 activations. This is the per-layer expert part of one
step, from route gather to combine:

| tokens in the step | distinct experts touched | all hot (VRAM) | all cold (zero-copy from pinned host) | effective cold read rate |
|---:|---:|---:|---:|---:|
| 1 | 10 | 0.085 ms | 0.435 ms | 44 GB/s |
| 8 | 75 | 0.149 ms | 2.78 ms | 52 GB/s |
| 32 | 234 | 0.342 ms | 8.51 ms | 53 GB/s |
| 64 | 374 | 0.508 ms | 13.56 ms | 53 GB/s |

(An earlier run, with the converter doing disk IO at the same time, gave 40 to 48 GB/s.)

What the table means for tuning (DESIGN 4.1, now calibrated):
- A cold expert costs about **36 µs per layer-step** (1.94 MB at 53 GB/s). The Marlin zero-copy path already
  saturates PCIe Gen5, so the remaining levers are fewer distinct cold experts per step (placement, adaptive cache,
  VRAM-resident scales) and overlap.
- A hot expert costs about 1 µs once the step has a few tokens.
- At concurrency 1, the expert part of a token costs about 48 × (0.085 + 10·(1−h)·0.036) ms, where h is the hot
  fraction of the token's 10 selections: about 21 ms at h=0 and about 12.5 ms at h=0.5. Strata does 121 tok/s solo
  (MTP and CPU expert compute), so at concurrency 1 the hit rate and MTP are what matter.
- At high concurrency the per-layer cost is set by distinct cold experts. At 64 tokens per step and h_touch=0.3,
  about 262 cold experts per layer → ~9.4 ms/layer → ~450 ms/step → ~140 tok/s, versus Strata's 81 tok/s at
  concurrency 32 with `--batch 8`.

Bring-up placement from `profile.bin`:

| EDGE0_EXPERT_VRAM_GIB | H | share of all experts | n_hot per layer (min–max) | cold pool, pinned |
|---:|---:|---:|---:|---:|
| 9.5 | 5,270 | 21.4% | 55 (L15) – 214 (L0) | 34.8 GiB |
| 12.7 | 7,045 | 28.7% | 87 (L15) – 240 (L0) | 31.6 GiB |
| 15.0 | 8,322 | 33.9% | 118 (L15) – 262 (L0) | 29.3 GiB |

## 5. Tests (`edge0/plugin/tests`, all passing)

Run on the box:

```
# CPU only (no --gpus): 28 pass, 16 GPU tests skip
docker run --rm --memory 6g --cpus 2 -v /home/andychen/models:/models:ro -v /home/andychen/work/edge0-vllm:/workspace/edge0 \
  -e PYTHONPATH=/workspace/edge0/plugin --entrypoint bash edge0-vllm:dev -c 'cd /workspace/edge0/plugin && python -m pytest -q tests'
# GPU (44 pass, ~17 s; needs the L4 checkpoint)
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 -v /home/andychen/models:/models:ro \
  -v /home/andychen/work/edge0-vllm:/workspace/edge0 -v /home/andychen/.cache/edge0-vllm:/root/.cache \
  -e PYTHONPATH=/workspace/edge0/plugin -e EDGE0_CKPT=/models/Edge0-145B-vllm-L4 --entrypoint bash edge0-vllm:dev \
  -c 'cd /workspace/edge0/plugin && python -m pytest -q -s tests'
```

| Test file | Checks |
|---|---|
| `test_config_placement.py` | Slot geometry and budget math. Settings parsing; phase-3 knobs are refused. Plans from a ranking and their maps; identity plans; plan JSON round trip; budget errors. On the real `profile.bin`: the top-5 pairs match MODEL-SPEC 5.7. |
| `test_q8_nvfp4.py` | Q8_0 dequant is bit-exact against `strata_formats.deq_q8_0` (fp32), and the bf16 output equals RNE. Real D rows from `latent_dense_q8.bin` match exactly. The loader wrapper keeps `shard_id` and extra arguments. ModelOpt nibble order. The DESIGN 1.5 repack decodes **bit-exactly** like Strata on synthetic rows and on real records (layer 0 expert 7, layer 47 expert 511), with no scale having bit 7 set and none equal to 0x7F. One expert's SwiGLU matches the oracle. |
| `test_latent_cpu.py` | Gather and combine against Python loops, including tier selection and never reading the other tier's rows (they are filled with NaN). Touch counter. Top-10 routing equals `strata_formats.route`. **The route-as-token + combine + U_cat formulation on real layer-0 weights equals Strata's per-group oracle `latent_moe_reference` (relative error < 1e-5, fp32).** |
| `test_ple_defer.py` | GPU: rows staged by the reader + in-graph wait equal a direct read of the file, for 12 steps enqueued without a host sync behind busy GPU work, duplicate ids, a captured CUDA graph replayed for new steps, the timeout path, and the reader's counters; both wake-ups (doorbell, event). |
| `test_tier.py` (round 3 additions) | GPU: `EDGE0_TIER_EARLY` (cudart and batch copies, with and without the CPU path): experts a step touched are hot in the very next replay, results equal the all-hot pool, host mirrors equal the device maps; shadow eviction without write-back (also served by the CPU path); shadow stealing when spare slots run out; the GPU decision equals the numpy one; batch copies move the same bytes both ways. |
| `test_route_post.py` | GPU (round 2): `kernels.route_post` (`EDGE0_ROUTE_FUSE`) equals the unfused routing tail (skip_pad's `where` x 2 + `zeros_like`, the tier's `full_like` + `where` + `tier_count`) for 1-100 tokens with and without padding rows, ids, weights and the counter row bit for bit; prints both inside a graph of 48 calls (5.1 vs 1.15 us per call). |
| `test_scale_factor.py` | Round 2: the chunked `marlin_nvfp4.scale_factor` equals vLLM's `_nvfp4_compute_scale_factor` on layer-sized random E4M3 scales, zeros, negative entries and the E4M3 maximum. |
| `test_ple.py` | vLLM's hash constants equal the GGUF constants (multipliers, 16 primes, offsets, 320,001,446 rows → 320,001,536 padded). **vLLM's `compute_ngram_ids` equals Strata's `ngram_rows` exactly** for prefill batches (EOS inside sequences, token 0, id 248319) and for decode continuation through `ngram_context`. The table-layout patch is checked, including that it falls through to vLLM's own function for other checkpoints. The real 51.2 GB file matches offset 352 + rows × 160. The threaded reader is byte-identical to the serial one. |
| `test_registration_quant.py` | The entry point loads through `load_general_plugins()`. The architecture is in the registry and in all three tables. `get_quantization_config("edge0")` works; ModelOpt does not hijack the config; a top-level `quant_algo` is rejected. Per-prefix resolution: fused qkvz and gate_up resolve to Q8_0; QSA qkv, in_proj_ba, PLE kv and HC resolve to BF16; the MTP experts resolve to W4A16_NVFP4. The dequant method has exactly the stock type. |
| `test_gpu_experts.py` (GPU) | Real layer-0 experts in pools for all-hot, all-cold and mixed placements (200 hot, random order), T = 1, 7, 64. **The Marlin path matches the fp32 reference with relative error 4.3e-3 to 4.5e-3.** Split and sequential runs are bit-identical; mixed equals all-hot (relative error < 1e-3). `swap()` keeps results and updates the maps. **CUDA-graph capture of the two-stream op replays bit-exactly.** Timing as in section 4. The trace counters are correct inside a captured graph, and the bw log and routes `.npz` are written. |
| `test_model_meta.py` (GPU) | EngineArgs resolves the L4 checkpoint to `Edge0ForCausalLM`, quantization `edge0`, hybrid. The whole model is built on `meta` with the real platform. All 4 MLPs are Edge0 blocks; `num_moe_layers == 0`; dequant layers are BF16. **Loading the checkpoint's tensors (shapes and dtypes from the headers) leaves only `ple_embedding.ngram_embedding.weight` unloaded** (expected: it is served from the GGUF). Every pool has its 6 tensors and every `expert_group` buffer is loaded. |
| `test_gpu_block.py` (GPU) | One real `Edge0LatentMoeBlock` (layer 0) on the GPU, loaded through vLLM's mapper and AutoWeightsLoader: Q8 dequant loader, fused `shared_expert.gate_up_proj` shards, pools, custom op. **The output matches Strata's float32 oracle, shared expert included, with relative error 0.56% to 0.66%, and the top-10 routing is identical** for 5 random tokens, with mixed and all-cold placements. The dequantized D rows equal the oracle's bf16-rounded values exactly. |

## 6. Bring-up recipe (for the next GPU step; not run yet)

```
cd ~/work/edge0-vllm
# 1. L4 mini checkpoint, eager, small budget. The output text is meaningless; this checks that the paths work.
MODEL_DIR=/home/andychen/models/Edge0-145B-vllm-L4 EAGER=1 EDGE0_EXPERT_VRAM_GIB=1 KV_BYTES=1000000000 ./serve.sh
# 2. Full model, F0 (all cold) eager, for the first correctness run: needs ~45 GiB pinned, embed stays in VRAM
EAGER=1 EDGE0_EXPERT_VRAM_GIB=0 EDGE0_EMBED_HOST=0 ./serve.sh
# 3. Profile placement, eager, then with CUDA graphs (drop EAGER), then EDGE0_TRACE=bw for the per-step cold bytes
EAGER=1 EDGE0_EXPERT_VRAM_GIB=9.5 ./serve.sh
```

After each start, calibrate `EDGE0_EXPERT_VRAM_GIB` and `KV_BYTES` from the startup log: the edge0 "after load" line
and vLLM's "GPU KV cache size … maximum concurrency" line (DESIGN 3.1). `EDGE0_DEV=1` runs the bind-mounted source
without a rebuild.

## 7. Open issues and risks

1. **Not run end to end.** Not yet exercised: the vLLM engine (profile run, KV sizing, torch.compile, breakable CUDA
   graphs) with Edge0 blocks inside, the PLE file-gather binding in a real load, and embed→host.
   `test_model_meta.py` shows the build and the name matching, and the GPU tests cover the op, but first light is the
   next step (section 6).
2. **MTP (phase 2).** Nothing Edge0-specific is registered: the design uses the stock `Qwen4ExpMTP`, and the draft gets
   the `edge0` quant config through `get_draft_quant_config`. That path, Q8 dequant of the draft's linears and Marlin
   for its W4A16 experts, is untested. Whether the draft shares `embed_tokens`/`lm_head` (DESIGN risk 5) also remains
   open; note that `embed_tokens` lives in host memory with `EDGE0_EMBED_HOST=1`.
3. **`EDGE0_DENSE=q8`** (b12x native Q8_0, worth +5.3 GiB of hot experts) is wired to the parent ModelOpt path but
   untested.
4. **Cold-call CTA cap (L4).** `fused_marlin_moe` does not expose Marlin's `blocks_per_sm`. Limiting the cold call's
   CTAs, so that the hot call keeps most SMs, needs a direct `_fused_marlin_moe` or `moe_wna16_marlin_gemm` call.
5. **Load time.** One layer's pool (512 repacks plus 4 copies per expert into VRAM or mapped host memory) takes
   about 1 s, so about 1 to 2 minutes for 48 layers. This is not yet measured on the full checkpoint.
6. **RAM at the bring-up budget.** The cold pools are 34.8 GiB pinned at 9.5 GiB of VRAM, plus about 6 GB of
   processes; F0 needs 44.3 GiB. Check `MemAvailable` before starting F0, and keep the converter and benchmarks off
   the box meanwhile.
7. **PLE.** The row reader is vLLM's serial `preadv` by default. `EDGE0_PLE_IO=threads` is implemented and tested for
   equality, but not yet measured. The pinned row LRU (`EDGE0_PLE_CACHE_MB`) is not implemented.
8. **The trace counts vLLM's dummy and profile runs** in the first flush after start-up; ignore that flush.
