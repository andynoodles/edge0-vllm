# Edge0-145B on vLLM: bring-up log

Status: 2026-11-19, `bring-up` agent. The full model (`/home/andychen/models/Edge0-145B-vllm`) serves on the RTX 5090
through the `edge0-vllm` plugin. Its output agrees closely with Strata (section 3). The healthy server is left running
in tmux session `edge0-vllm` on port 8125.

## 1. Serve command (what is running)

```
cd ~/work/edge0-vllm && KV_BYTES= ./serve.sh        # on the 5090; log in ~/work/edge0-vllm/logs/serve-*.log
docker stop edge0-vllm; tmux kill-session -t edge0-vllm   # stop (killing tmux alone leaves the container running)
```

`KV_BYTES=` (empty) lets vLLM profile the activation peak and size the KV cache from `GPU_UTIL=0.96`, instead of
using a fixed `--kv-cache-memory-bytes` (section 2.4). Everything else is the serve.sh default. The resulting docker
command:

```
docker run --rm --name edge0-vllm --gpus all --ipc=host --network host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/andychen/models:/models -v /home/andychen/work/edge0-vllm:/workspace/edge0 -v /home/andychen/.cache/edge0-vllm:/root/.cache \
  -e HF_HUB_OFFLINE=1 -e VLLM_ENGINE_READY_TIMEOUT_S=7200 -e TORCH_DIST_USE_NCCL2=0 -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128 \
  -e EDGE0_COLD_ALLOC=host_register -e EDGE0_DENSE=bf16 -e EDGE0_EMBED_HOST=1 -e EDGE0_EXPERT_VRAM_GIB=9.5 \
  -e EDGE0_MOE_IMPL=marlin -e EDGE0_PLACEMENT=profile -e EDGE0_PLE_IO=vllm -e EDGE0_SPLIT=1 -e EDGE0_TRACE=0 \
  edge0-vllm:dev vllm serve /models/Edge0-145B-vllm --served-model-name edge0_145b --host 0.0.0.0 --port 8125 \
  --quantization edge0 --dtype bfloat16 --max-model-len 32768 --max-num-seqs 16 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.96 --kv-cache-dtype auto --mamba-ssm-cache-dtype float32 \
  --engram-config '{"cpu_offload": false, "host_file_gather": true}' --generation-config auto
```

The MoE path is the design's Marlin W4A16 NVFP4 path (`fused_marlin_moe`, D4). It produced correct output on sm_120
from the first run, so no backend switch was needed. torch.compile and CUDA graphs are on: breakable graphs
(PIECEWISE for 7 sizes, FULL for 5 sizes) are captured in 2 s and take 0.08 GiB. A request with thinking off is
`{"chat_template_kwargs": {"enable_thinking": false}}`.

Images:
- `edge0-vllm:dev` was rebuilt at 15:35 from the worktree's plugin. It includes the `model.py` fixes below and the
  parallel agent's `loadcheck.py`.
- `edge0-vllm:bringup-1502` is the earlier build that the correctness and bench numbers in sections 3 and 5 were first
  measured on, kept as a fallback (`IMAGE=edge0-vllm:bringup-1502 ./serve.sh`).
- The final restart on the 15:35 image passed the same correctness check: 256/261, mean |Δlogprob| 0.034
  (`correctness/results/compare-final.txt`).

## 2. What broke and how it was fixed

### 2.1 Engine refused to start: "Free memory on device (27.44/31.36 GiB) ... less than desired GPU memory utilization"

The NGC image's torch 2.15 uses its new `nccl2` process-group backend by default. At distributed init it reserves
about 3.4 GiB of VRAM, even for `world_size=1`. vLLM's `gpu_worker.py` sets `TORCH_DIST_USE_NCCL2=0` only for
integrated GPUs ("WAR nvbug 6915005"). On the 5090 the worker therefore started with 3.9 GiB already in use: 0.5 GiB of
CUDA context plus 3.4 GiB for nccl2.

- **Fix:** `serve.sh` passes `-e TORCH_DIST_USE_NCCL2=0`, which selects the stock `ProcessGroupNCCL`. Free memory at
  worker start is now 30.66 / 31.36 GiB.
- **Measured** in the image (`torch.cuda.mem_get_info` around `init_process_group` and the first `barrier`): the stock
  backend creates its communicator lazily and uses 0.67 GiB at the first collective (0.49 GiB inside vLLM).
- **Net gain:** about 2.9 GiB of VRAM (3.4 GiB for nccl2 minus 0.49 GiB for stock NCCL), which can become about
  1,600 more hot expert slots. Directly in line with TidalMoE's "fewer cold expert loads".

### 2.2 Full model died after loading: `NCCL error ... Failed to CUDA calloc async 4 bytes`

The stock backend creates its NCCL communicator at the first collective. In vLLM that is the `barrier()` in
`base_loader._drop_checkpoint_page_cache`, which runs right after `load_weights`. By then the hot expert pools
(9.5 GiB), the dense weights and torch's cached load transients had filled the GPU.

- **Fix in `plugin/edge0_vllm/model.py`:**
  - `_init_nccl_early()` runs a `dist.barrier(device_ids=[cur])` at the start of `Edge0ForCausalLM.__init__`, before
    any pool is allocated. It logs "NCCL communicator created before weight allocation (0.49 GiB)".
  - `load_weights` ends with `torch.cuda.empty_cache()`, which drops the cached Q8 dequant and Marlin repack
    transients.

### 2.3 Engine died under load: `torch.OutOfMemoryError` in `qsa_select_paged_prefill`

The first `vllm-bench` sweep used `--kv-cache-memory-bytes 3.5e9`, the design default. Concurrency 1 was fine. At
concurrency 4, a prefill step of four requests asked for 512 MiB with only 476 MiB free.

- **The allocation:** `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py:623` allocates
  `logits_workspace = q.new_empty(max(VLLM_SPARSE_INDEXER_MAX_LOGITS_MB MiB, ...))`, 512 MiB by default, at its worst
  case, on every prefill call.
- **Why it was not reserved:** with `--kv-cache-memory-bytes` vLLM skips memory profiling ("skipped memory
  profiling"), so nothing kept room for this transient.
- **Fixes:**
  1. `serve.sh` passes `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128`. A row is `cdiv(32768, 4) = 8192` fp32 columns =
     32 KiB, so 128 MiB still holds all 4096 rows of a full `MAX_BATCHED` step in one chunk at `MAX_MODEL_LEN=32768`.
     Raise it together with longer contexts.
  2. The KV cache is sized by vLLM's profiler (`KV_BYTES=`, empty). vLLM measures the real activation peak (0.78 GiB,
     MoE workspace included) and keeps it free.
- **Result:** the 3-point sweep (84 requests) and the correctness runs since then show no errors.

### 2.4 Other observations (no fix needed)

- **Startup:** about 3.5 min in all. Model init takes 2 to 4 s. Loading the 50 shards takes 90 to 111 s: about
  1.4 s per expert shard for the Marlin repack and the copy into the hot and cold pools. Profiling, KV allocation,
  torch.compile and graph capture take 40 to 68 s; the compile cache in `/home/andychen/.cache/edge0-vllm` is warm
  after the first run.
- **Load warnings:** vLLM prints 449 `Attempted to load weight ... with dtype` warnings at load. Two classes are
  expected:
  - `uint8 -> bf16` on Q8_0 linears, which `q8.wrap_weight_loader` dequantizes;
  - `float32 -> bf16` on norms, GDN `dt_bias`/`conv1d` and similar, which stock vLLM itself declares bf16.

  A parallel agent added `plugin/edge0_vllm/loadcheck.py` (`EDGE0_VERIFY_LOAD`, `EDGE0_LOAD_WARNINGS`) to audit and
  summarize them. In the 15:35 image the 449 raw lines still appear in the serve log: the summarizing is not yet
  effective, which is cosmetic.
- **Logs:** the transformers warning "incorrect regex pattern ... fix_mistral_regex" is harmless; the converter proved
  encode parity. The Marlin NVFP4 scale factor is 1.0 for w13 and w2 in every layer.
- **Stopping the server:** `tmux kill-session -t edge0-vllm` does **not** stop the container, because the docker
  client is killed while the container keeps running. Use `docker stop edge0-vllm`.
- **L4 mini checkpoint:** a first eager run (`MODEL_DIR=.../Edge0-145B-vllm-L4 EAGER=1 EDGE0_EXPERT_VRAM_GIB=1
  KV_BYTES=1000000000`) started and generated, with meaningless text as expected, in about 80 s. It is a good smoke
  test after plugin changes.

## 3. Correctness against Strata

Script: `edge0/correctness/compare_strata.py`. It needs only the standard library and runs on the 5090 host:
`python3 -I compare_strata.py --out <json>`. Reference: `~/work/edge0-bench/strata-ref.json`, which holds 8 prompts,
greedy, thinking off, 48 tokens, and OpenAI logprobs with top 5.

Per prompt, the script does two things:

1. **Free-running greedy:** the same request Strata got. The script reports the common token prefix and, at the first
   divergence, both engines' top 5.
2. **Teacher-forced:** the chat prompt plus Strata's generated tokens are sent as one token-id prompt to
   `/v1/completions` with `prompt_logprobs=5`. At every position of Strata's output this gives vLLM's logprob of
   Strata's token and vLLM's own top 1. Tokens are matched by raw bytes through the checkpoint's byte-level
   `vocab.json`.

**Finding about the reference:** Strata's MTP-speculative "greedy" decoding sometimes emits a token that is **not**
the argmax of the logprobs it reports itself. This happened 9 times in its 261 tokens. For example, prompt 3,
position 19: Strata reports `' the'` at -0.451 as its top 1 but emits `' break'` at -1.924. In 4 of the 5
free-running divergences, vLLM emitted exactly Strata's own argmax. Teacher-forced agreement is therefore measured
against Strata's argmax.

Results, with CUDA graphs and torch.compile and the serve command above (also identical in eager mode). Full output:
`5090:~/work/edge0-vllm/results/compare-graph-9.5.{txt,json}`, copy in `edge0/correctness/results/`.

| # | prompt | common prefix (first 32) | vLLM top-1 == Strata argmax (teacher-forced) | Strata non-argmax picks | mean \|Δlogprob\| | KL(top-5) |
|---|---|---:|---:|---:|---:|---:|
| 0 | capital of France | 7/7 (identical, stop) | 7/7 | 0 | 0.0002 | 0.0000 |
| 1 | 不當得利 (zh-TW) | 19/31 | 29/31 | 1 | 0.069 | 0.010 |
| 2 | Fibonacci in Python | 23/32 | 48/48 | 2 | 0.013 | 0.001 |
| 3 | 37 * 43 | 19/32 | 48/48 | 2 | 0.027 | 0.003 |
| 4 | TCP vs UDP | 6/32 | 47/48 | 2 | 0.014 | 0.001 |
| 5 | translate to English | 19/19 (identical, stop) | 19/19 | 0 | 0.017 | 0.007 |
| 6 | why the sky is blue | 6/32 | 47/48 | 2 | 0.065 | 0.018 |
| 7 | translate to French | 12/12 (identical, stop) | 11/12 | 0 | 0.030 | 0.006 |
| **all** | | 111/197 | **256/261 (98.1%)** | 9 | **0.032** | **0.006** |

All 8 prompts pass. Every free-running divergence is explained:

| prompt | position | Strata's pick | top-1 minus top-2 margin | vLLM |
|---|---:|---|---:|---|
| 1 | 19 | 人 (its argmax) | 0.51 nat | picks 者 |
| 2 | 23 | `' using'` (not its argmax) | 0.03 nat | |
| 3 | 19 | `' break'` (not its argmax) | | `' the'` = Strata's argmax |
| 4 | 6 | `' **'` (not its argmax) | | `' TCP'` = Strata's argmax |
| 6 | 6 | `' sc'` (not its argmax) | | `' is'` = Strata's argmax |

Prompt counts match Strata's `usage.prompt_tokens` for all 8 prompts. The texts are fluent and correct in English,
Traditional Chinese and French. Remaining differences: Strata runs D/U and dense weights as Q8_0 (`du=q8`) where
vLLM uses bf16 dequant, norms are bf16 in vLLM, and the GEMM/attention kernels differ. They show up only as
|Δlogprob| ≈ 0.03 nat on average, with a worst position of 0.6 nat.

Phases checked:
- F1 (`EDGE0_MOE_IMPL=torch`) was not needed: the Marlin path was correct from the start.
- F0 (all cold) was not run on the full model: unit tests show that mixed placement equals all-hot within 1e-3.

## 4. Memory (calibrated at first light; DESIGN 3.1 planned it)

VRAM, as logged by the plugin and vLLM's profiler (GiB):

| item | GiB |
|---|---:|
| CUDA context + stock NCCL communicator | 0.70 + 0.49 |
| weights, torch-allocated after load (dense bf16 15.3 + 9.5 hot experts; embed_tokens moved to host) | 24.84 |
| consumed (weights + non-torch), vLLM's figure | 26.14 |
| peak activation (profiled at 4096 tokens; includes the 520 MB MoE workspace) | 0.78 |
| CUDA graphs | 0.08 |
| KV cache + GDN/PLE state (profiled): 80,956 tokens, 2.47x of 32K | 3.18 |
| total used (nvidia-smi) | 30.2 of 31.84 |

- **Hot experts:** 9.5 GiB = 5,270 slots = 21.4% of the 24,576 experts; per layer from 55 (L15) to 214 (L0),
  `profile.bin` placement.
- **Headroom:** vLLM's log says `--kv-cache-memory=3775133696` (3.52 GiB) would fully use the GPU. That is 0.34 GiB
  more than the current 3.18 GiB, enough for about 190 more hot slots at 554 slots per GiB. The 2.9 GiB won back from
  nccl2 is already inside the 30.66 GiB starting point: before the fix the same budget would not have started. This
  run did not raise `EDGE0_EXPERT_VRAM_GIB`; the tuning phase should, trading KV against hot slots (DESIGN 3.1,
  lever L7).

Host RAM (60 GiB box):
- EngineCore RSS is 40.1 GB: 34.8 GiB of pinned cold expert pools (`host_register`, 2 MiB THP; AnonHugePages
  39.6 GB), 1.18 GiB of pinned `embed_tokens`, and the rest is Python and torch.
- The API server takes 0.9 GB and vllm-bench 2.0 GB.
- MemAvailable is about 16 GB with the server up. The page cache is only 1 to 2 GB, so **the 51 GB PLE table is read
  mostly from NVMe**: vLLM's serial `preadv` file-gather, `EDGE0_PLE_IO=vllm`.

## 5. Quick throughput sanity (label `vllm-bringup`)

`vllm-bench` sharegpt, chat API, thinking off, `--sharegpt-output-len 256 --ignore-eos --temperature 0 --seed 0`,
4 prompts per concurrency level, 2 warm-ups. Run with `edge0/bench/bench-openai.sh vllm-bringup 1,4,16`
(`OUT_LEN=256`). Results are in `5090:~/work/edge0-bench/results/vllm-bringup/`, log in
`~/work/edge0-vllm/logs/bench-vllm-bringup.log`.

| concurrency | requests ok | output tok/s | mean TPOT ms | median TTFT ms | mean TTFT ms | P99 TTFT ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4/4 | 38.1 | 24.6 | 374 | 445 | 884 |
| 4 | 16/16 | 64.7 | 55.6 | 1,778 | 1,638 | 1,969 |
| 16 | 64/64 | 69.4 | 176.6 | 2,752 | 11,015 | 49,210 |

Strata, from the coordinator, used 1024 output tokens, and Strata ignores `ignore_eos`:
- `strata-solo`: 121 / 110 / 117 / 113 tok/s at concurrency 1 / 2 / 4 / 8. It uses MTP and CPU expert compute.
- `strata-batch8`: 85 / 32 / 45 / 67 / 77 / 81 tok/s at concurrency 1 / 2 / 4 / 8 / 16 / 32.

What the numbers say, for the tuning agents:
- **Concurrency 1.** TPOT of 24.6 ms with no MTP matches PLUGIN.md 4's model: 48 × (0.085 + 10 × (1 − h) × 0.036) ms
  gives h ≈ 0.45, about 4.5 of each token's 10 experts hot. MTP (phase 2) and a higher hit rate are the levers.
- **Concurrency 16.** TPOT is 177 ms per step. That is about 3.7 ms per layer, about 100 distinct cold experts per
  layer-step at 36 µs each. The step is dominated by **cold expert traffic over PCIe**, exactly TidalMoE's target:
  more hot slots, a better placement for batched traffic, an adaptive cache, VRAM-resident cold scales.
- **The P99 TTFT of 49 s at concurrency 16** comes from queueing. 64 ShareGPT prompts at 16 concurrent sequences fit
  in 80,956 KV tokens. Chunked prefill of long prompts while decoding 16 sequences, each prefill chunk touching up to
  512 cold experts per layer (about 18 ms per layer, 0.9 s per 4096-token chunk), makes new requests wait. The
  phase-3 prefill streaming path (`EDGE0_STREAM_MIN_TOKENS`) targets exactly this.

### 5.1 Expert traffic, measured (`EDGE0_TRACE=bw`)

**Run:** the same server with `EDGE0_TRACE=bw` (log `logs/serve-20261119-152535.log`) and the same workload, label
`vllm-bringup-trace`. Throughput was unchanged by the trace: 37.5 tok/s at concurrency 1 and 69.6 tok/s at
concurrency 16.

**How to read it:** counts are distinct experts per engine step, summed over the 48 layers. "Cold" means read
zero-copy from pinned host memory over PCIe, 1.85 MB per expert.

| phase | steps/s | hot / cold distinct experts per step | cold bytes per step | cold read rate | hot share of touched experts |
|---|---:|---:|---:|---:|---:|
| concurrency 1, decode | 37 to 39 | 168 to 242 / 270 to 322 (per layer about 4.9 hot / 5.7 cold of 10) | 0.52 to 0.62 GB | 20 to 23 GB/s | about 46% |
| concurrency 16, decode | 5.3 to 6.0 | 1,430 to 2,080 / 2,650 to 3,750 (per layer about 37 hot / 70 cold) | 5.1 to 7.3 GB | 34 to 38 GB/s | about 35% |
| concurrency 16, prefill-mixed steps | 3.6 to 3.7 | about 2,100 / 3,800 to 4,700 | 7.4 to 9.0 GB | 27 to 32 GB/s | about 33% |

What this means:
- **Concurrency 1.** About 10 ms of the 24.6 ms step is cold expert reads (0.52 GB at about 50 GB/s).
- **Concurrency 16.** The PCIe link is busy about 70% of the wall time: 36 GB/s average against about 50 GB/s peak.
  Cold expert loading is **the** bottleneck, as TidalMoE assumes.
- **Worst layers** are those with the fewest hot slots under the `profile.bin` plan: L15 (55 hot), L31, L39, L16,
  L40 and L23, with 90 to 118 cold experts per step at concurrency 16.
- **Levers, by expected gain:**
  1. more hot slots (VRAM headroom, `EDGE0_DENSE=q8` +5.3 GiB, bf16 SSM state);
  2. a placement planned from the vLLM routing trace of the real batched workload (`EDGE0_TRACE=routes` plus
     `tools/plan_convex.py`), instead of Strata's single-stream `profile.bin`;
  3. an adaptive cache (`ExpertPool.swap`, the touch counters are already there);
  4. VRAM-resident cold scales (11% of the bytes);
  5. MTP, which amortizes one cold read over several accepted tokens.

## 6. State left behind and follow-ups

- The server runs in tmux `edge0-vllm` (container `edge0-vllm`), started at 15:35 with the command in section 1, from
  image `edge0-vllm:dev` built at 15:35 from the worktree. Its log is `logs/serve-20261119-153459.log`.
- **After a plugin change,** rebuild with `cd ~/work/edge0-vllm && nice -n 19 docker build -f docker/Dockerfile -t
  edge0-vllm:dev .` (about 15 s), or run `EDGE0_DEV=1 ./serve.sh` to use the bind-mounted source.
- **The box is shared.** Another agent restarted `edge0-vllm` during this bring-up. Check `docker ps` before
  starting, and stop with `docker stop edge0-vllm`.
- **Correctness re-check after any plugin or serve change** (about 15 s):
  `cd ~/work/edge0-vllm/correctness && python3 -I compare_strata.py`. Exit status 0 means pass. The summary should
  keep `tf_top1_agree_strata_argmax` ≥ 250/261 and `tf_mean_abs_dlogprob` ≈ 0.03.
- **Open items:**
  - MTP (phase 2) is not enabled.
  - `EDGE0_EXPERT_VRAM_GIB` is not yet raised into the measured headroom.
  - `SSM_DTYPE=bfloat16` would halve the 116 MB per-sequence GDN state; check KL with `compare_strata.py` first.
  - The PLE reads go to NVMe because RAM is mostly pinned. Try `EDGE0_PLE_IO=threads`, and a row cache.
