# Edge0-145B on vLLM: where a decode step's time goes (Nsight Systems)

Agent `nsys`, 2026-11-20. Configuration: `./serve.sh` PRESET=tuned (TUNING-LOG.md section 5: q8 dense, 12.8 GiB hot
experts with `plan-s1-12.8.json`, adaptive swaps, 32 sequences, bf16 GDN state, cold M-block 64) with `EDGE0_DEV=1`
(current plugin source, so `EDGE0_SKIP_PAD=1` is on; it changes nothing here because c1 / c8 / c32 are CUDA-graph
sizes without padding). The workload is vllm-bench ShareGPT seed 0, chat API, thinking off, 512 output tokens and
ignore-EOS. Only steady decode windows are captured: every request of the wave is running and none is waiting.
Speculative decoding: DFlash2 k=3 at the calibration config (`MAX_SEQS=16 EDGE0_EXPERT_VRAM_GIB=11.0`,
`plan-s1-11.0.json`), compared with the same config without speculation.

## 1. Answer in short

| | c1 | c8 | c32 |
|---|---:|---:|---:|
| decode step, accurate (graph-level trace) | 16.4 ms | 67.0 ms | 165.2 ms |
| steady decode throughput implied | 61 tok/s | 119 tok/s | 194 tok/s |
| **cold-expert Marlin (PCIe-bound)**, share of the step | **24%** (3.8 ms) | **74%** (48 ms) | **86%** (149 ms) |
| dense weight streaming (attention projections, router, D, U, shared expert, HC GEMMs, lm_head) | **52%** (8.2 ms) | 14% (8.9 ms) | 6% (10.6 ms) |
| ...of which latent D + U | 18% (2.8 ms) | 4% | 2% |
| GPU idle (graph-level), almost all of it the host PLE row reads | 5.0% (0.8 ms) | 4.3% (2.9 ms) | 5.1% (8.5 ms) |
| PCIe RX, mean over the step (93% = link saturated, ~54 GB/s payload) | 13% | 64% | 77% |

1. **Cold experts read over PCIe are the bottleneck from c8 up.** They take 74-86% of every decode step. During
   that time the link runs at its 54 GB/s ceiling, so the step costs 36 us per distinct cold expert per layer
   (2.2 / 28 / 86 per layer at c1 / c8 / c32 = 0.20 / 2.6 / 8.0 GB per step). This is TidalMoE's "expert loading
   time". The hot-expert GEMMs are not a cost: they run inside the cold window, on SMs that the cold kernel leaves
   free (section 3.2).
2. **At c1, dense weight streaming dominates.** Each step reads about 10 GB of Q8/bf16 dense weights from VRAM
   (8.2 ms). The latent D/U projections alone cost 2.8 ms. Cold experts are second at 24%: the adaptive cache
   brings c1 down to about 2 cold experts per layer.
3. **The host stalls the GPU once per step for the PLE rows.** `prepare_inputs` makes a full stream sync, then
   reads the step's PLE rows with serial `preadv` (most of them miss the page cache and go to NVMe), and only then
   launches the graph. That costs 0.6 / 2.5-4.9 / 8 ms per step (4-7%) and is nearly all of the GPU idle time. The
   scheduler, Python, sampling and the graph launch (0.17-0.24 ms) are overlapped by async scheduling or are small.
4. **Speculative decoding (DFlash2 k=3).** The drafter costs only 1.7 / 2.1 ms per step (5% / 1.3%). Verify costs
   a lot because cold experts are hardly shared between draft tokens. A 4-token verify step costs 1.65x a 1-token
   step at c1 (2.1x the cold time). At c8 the 32-token verify costs 2.25x. Longer drafts (k=7, k=15) multiply this
   PCIe traffic (section 4).
5. **CPU computation of cold experts (Strata-style) is recommended, but only for c >= 8 and for speculative
   verify steps.** Measured on this box, the GPU's PCIe reads and the CPU's DDR5 reads share one DRAM. Combined they
   reach about 73 GB/s, against 54 GB/s for PCIe alone. A balanced CPU/GPU split of each layer's cold experts can
   therefore cut cold time by at most about 28%. That is an upper bound of +23% at c8 and +30% at c32 in throughput,
   and +13% for k=3 verify at c1. It gives at most +2% for plain c1 decode. It is a large piece of work: a CPU kernel
   for the Marlin record layout, because RAM cannot hold a second copy of the 31.5 GiB cold pool, plus
   graph-capturable host doorbells. Section 5 has the numbers and the plan.

## 2. Method

**Instrumentation.** These are the switches added for this work. All of them are off by default.

| what | where |
|---|---|
| `EDGE0_NVTX=1`: NVTX ranges on the MoE op parts (`edge0.gate/D/moe/route/gather/hot/cold/combine/U/shared`) and on the host phases that run every step outside the graphs: runner `prepare_inputs` / `execute_model` / `sample_tokens`, `graph.full` / `graph.pw` replays, PLE staging (`ple.stage` including its sync, `ple.read` = the preadv loop), adaptive-cache epochs, scheduler, `spec.propose` / `spec.reject`. `EDGE0_NVTX=2` adds one range per `nn.Module` forward, for eager calibration (vLLM's `--enable-layerwise-nvtx-tracing` is not wired into the V2 model runner). | `plugin/edge0_vllm/nvtx.py`, hooks in `latent_moe.py`, `model.py` |
| `NSYS="<nsys profile options>"` runs `vllm serve` under the host's Nsight Systems 2025.3.2, mounted read-only (the image has no nsys). | `serve.sh` |
| Captures under load: server under nsys with `--profiler-config '{"profiler":"cuda"}'`; per concurrency, start vllm-bench, wait for steady decode, `POST /start_profile {"max_iterations": N}`, collect the range's `.qdstrm` from the container and import it on the host, then take host-side GPU metrics (`sudo nsys --gpu-metrics-devices=0 --gpu-metrics-set=gb20x`, 10 kHz, 4 s). | `bench/nsys-capture.sh`, `bench/nsys-chain-s{1,2}.sh` |
| Per-step breakdown: step = one `execute_context_*` range (vLLM's CUDA-profiler step annotation), from the first GPU activity of step k to the first of step k+1. GPU time is attributed exclusively per category (cold branch over hot branch over the rest). Idle gaps are attributed to the engine thread's innermost host range. Also: kernel name listing and GPU-metric means. | `tools/nsys_breakdown.py` (`steps`, `names`, `gpum`), `tools/nsys_tables.py` |
| DRAM contention and CPU expert throughput microbenchmarks. | `bench/dram-contention.sh`, `tools/cpu_experts/{cpu_expert_bench.cpp,h2d_loop.py}` |

**Captures** (`5090:~/work/edge0-vllm/profiles/<set>/`, `.nsys-rep` + `.sqlite`; the JSON summaries are copied to
`results/profile/`):

| set | trace | what |
|---|---|---|
| `nospec` | `--trace=cuda,nvtx,osrt --cuda-graph-trace=node` | tuned preset: c1 x3 (150 steps each, 5 min apart), c8 (60), c32 (30); GPU metrics per concurrency |
| `graph` | the same with `--cuda-graph-trace=graph` | tuned preset, fresh server: accurate step period, idle and launch cost (node tracing adds ~1 ms of launch time per step) |
| `eager` | node, `EAGER=1 EDGE0_NVTX=2` | calibration: every kernel's module, which fixes the per-layer kernel order the analyzer uses to label graph replays |
| `dflash3`, `nospec11` | node | DFlash2 k=3 and the same config without speculation, c1 and c8 (+ GPU metrics for DFlash2) |

**How graph replays are labelled.** NVTX ranges do not exist inside CUDA-graph replays, and eager mode uses
different GEMM variants. In eager mode the b12x Q8 kernels have other grids and run 2-7x slower; U, for example,
takes 215 us eager and 28.5 us in the graph. So the eager capture is used only to learn the layer structure. In
the graph, each layer is HC (`_hc_combine_norm`, HC down/SiLU GEMM, mix GEMM, `_hc_gate_mix`), then the attention
block (GDN: in_proj_qkvz b12x, in_proj_ba skinny GEMM, conv1d, gated delta rule, norm, out_proj; QSA: qkv nvjet,
indexer, sparse attention, merge, o_proj), then HC, then the MoE block: router gate, D (b12x + split-K reduce),
topk + route kernels, gather, hot and cold branches, combine, U (b12x + reduce), shared expert.
`structural_labels()` applies this order to every replay.

**Hot and cold streams swap in graph replays.** In eager mode the cold Marlin call runs on the side stream. In a
graph replay the driver puts the cold branch on the launch stream and the hot branch on another stream. Three
checks show it. (1) Graph node ids: the cold branch is captured first, so its nodes have the lower ids. (2) Only
the launch-stream Marlin pair has the 2:1 GEMM1:GEMM2 duration ratio of the w13:w2 byte sizes. (3) That pair
runs at 51-54 GB/s implied by its bytes. `moe_tiers()` therefore labels the two branches per MoE op by node id,
not by stream.

**Caveats.** (a) The adaptive cache keeps learning the benchmark's prompts. Three c1 captures 5 minutes apart gave
19.9, 16.2 and 15.8 ms per step (cold time 6.0, 4.2 and 3.8 ms). The tables use the warmest. (b) Node-level
tracing inflates the CPU-side `cudaGraphLaunch` from 0.16-0.22 ms to 1.2-1.5 ms per step, which shows up as GPU
idle. Use section 3.3 for idle and launch. (c) These are steady decode steps with no prefill, so they run faster
than the bench averages, which include prefill and ramp-up (e.g. vllm-tuned c32: 142 tok/s, including prefill).

## 3. No speculation (tuned preset)

### 3.1 Breakdown per decode step (node-level trace, exclusive attribution)

| per decode step | c1 ms | % | c8 ms | % | c32 ms | % |
|---|---:|---:|---:|---:|---:|---:|
| Attention: GDN kernels (36 layers) | 0.23 | 1.5 | 0.66 | 1.0 | 1.64 | 0.9 |
| Attention: QSA indexer + sparse attention (12 layers) | 0.17 | 1.0 | 0.24 | 0.4 | 0.38 | 0.2 |
| Dense GEMMs: attention projections (GDN qkvz/ba/out, QSA qkv/o) | 2.31 | 14.6 | 2.43 | 3.7 | 2.81 | 1.6 |
| Dense: router gate (bf16, fp32 logits) | 0.17 | 1.1 | 0.22 | 0.3 | 0.25 | 0.1 |
| Latent D (Q8, 2560 -> 8x1792) | 1.40 | 8.9 | 1.46 | 2.2 | 1.97 | 1.1 |
| Latent U (Q8, 8x1792 -> 2560) | 1.37 | 8.7 | 1.42 | 2.2 | 1.48 | 0.9 |
| Shared expert (Q8 gate_up + down, sigmoid gate) | 0.82 | 5.2 | 0.93 | 1.4 | 1.24 | 0.7 |
| Hyper-connections + norms (incl. their bf16 GEMMs) | 1.37 | 8.7 | 1.61 | 2.5 | 1.92 | 1.1 |
| MoE routing (top-10, renormalize, touch counter) | 0.46 | 2.9 | 0.46 | 0.7 | 0.50 | 0.3 |
| MoE route gather + combine | 0.16 | 1.0 | 0.18 | 0.3 | 0.21 | 0.1 |
| Hot-expert Marlin (VRAM), part not hidden under the cold branch | 0.32 | 2.0 | 0.16 | 0.2 | 2.06 | 1.2 |
| **Cold-expert Marlin (zero-copy over PCIe)** | **3.79** | **23.9** | **48.09** | **73.5** | **148.93** | **85.6** |
| Final HC/norm + lm_head + sampler | 0.77 | 4.9 | 0.81 | 1.2 | 0.91 | 0.5 |
| Input prep / other GPU kernels / PLE GPU kernels | 0.16 | 1.0 | 0.30 | 0.5 | 0.19 | 0.1 |
| memcpy H2D + D2H + memset (PLE rows, ids) | 0.04 | 0.2 | 0.05 | 0.1 | 0.05 | 0.0 |
| GPU idle: PLE row reads (host preadv after a full stream sync) | 0.70 | 4.4 | 4.86 | 7.4 | 7.93 | 4.6 |
| GPU idle: graph launch (node tracing inflates it, see 3.3) | 1.23 | 7.8 | 1.27 | 1.9 | 1.28 | 0.7 |
| GPU idle: rest (input prep, Python, scheduler, sync) | 0.33 | 2.1 | 0.29 | 0.4 | 0.31 | 0.2 |
| **GPU idle total** | 2.27 | 14.4 | 6.42 | 9.8 | 9.52 | 5.5 |
| **step period (GPU, mean)** | **15.83** | 100 | **65.45** | 100 | **174.07** | 100 |

The forward graph replays about 2,400-2,500 kernels per step. "Cold-expert Marlin" is the whole cold branch: align,
sort, GEMM1, SiLU, GEMM2. Summed over the step, the hot-branch kernels run 2.9 / 32.4 / 84.8 ms, but all except the
row above overlaps the cold branch.

### 3.2 The MoE op inside the graph: the cold branch sets the pace, and the hot branch hides under it

Here is one c8 layer from the replay. Times are in µs from the route gather.

```
launch stream (= cold tier)                       other stream (= hot tier)
   1.4  align, sort                                  1.4  align, sort
   6.9  Marlin GEMM1 (w13)   530 us                  5.8  Marlin GEMM1      270 us  (starved: the cold grid
 537    SiLU                                       275    SiLU             227 us   holds 3 CTAs on every SM)
 540    Marlin GEMM2 (w2)    253 us                503    Marlin GEMM2       34 us
 793    combine (joins the hot stream)
```

- The cold GEMMs are PCIe-bound. GEMM2 alone reads about 21 x 0.645 MB in 253 us, which is 53.6 GB/s, and GEMM1
  reads about twice the bytes in about twice the time. The per-layer cold time at c8 (about 1.0 ms) equals
  28 distinct cold experts x 36 us.
- The cold Marlin launches a persistent grid of 510 CTAs (3 per SM x 170). It holds every SM while it waits on PCIe,
  so the hot GEMMs get only the leftover slots and are stretched several-fold. At c8 hot GEMM1 takes 270 us, and the
  hot SiLU, a 2 us kernel when it is alone, takes 227 us. They still finish inside the cold window.
  The only exposed hot time is 0.2-2 ms per step. Capping the cold call's CTAs (`EDGE0_COLD_BLOCKS_PER_SM`) could
  recover at most that much: 1.2% at c32.
- PCIe RX over the whole step (GPU metrics, 10 kHz): 12.6% / 63.8% / 77.3% at c1 / c8 / c32. Individual samples
  peak at 93%, which is the saturated link (about 54 GB/s payload). At c32 the link is saturated for 86% of the step.
  VRAM read bandwidth averages 44% / 13% / 7%. SMs-active averages 65% / 79% / 89%, but SM issue is only 2-6%:
  the GPU is waiting for PCIe, not computing.
- Cold bytes per step, from the cold time at 53.8 GB/s: 0.20 / 2.59 / 8.0 GB. That is 105 / 1,336 / 4,137 cold
  expert loads, or 2.2 / 28 / 86 per layer. Per generated token: 0.20 / 0.32 / 0.25 GB.

### 3.3 Accurate period, idle and launch (graph-level trace)

| | c1 | c8 | c32 |
|---|---:|---:|---:|
| step period, mean (median) | 16.38 (16.29) ms | 66.96 (67.18) ms | 165.20 (166.39) ms |
| forward graph on the GPU | 14.67 ms | 62.89 ms | 155.44 ms |
| GPU idle | 0.83 ms (5.0%) | 2.90 ms (4.3%) | 8.48 ms (5.1%) |
| ...waiting for PLE `preadv` | 0.59 | 2.54 | 8.04 |
| ...in `cudaGraphLaunch` | 0.17 | 0.18 | 0.24 |
| ...rest (prep, Python, sync) | 0.06 | 0.18 | 0.21 |
| `cudaGraphLaunch` CPU time | 0.16 ms | 0.17 ms | 0.22 ms |

**The host-side critical path in each step.** Async scheduling would let the CPU prepare step k+1 while the GPU runs
step k. In this model it does not, because of vLLM's PLE staging for `host_file_gather`.
`Qwen4ExpModelState.prepare_inputs` calls `stage_checkpoint_rows`, and that function computes the n-gram ids on the
GPU, copies them to the host and then calls `current_stream().synchronize()`. The sync waits for the whole previous
step (`sync` = 12-157 ms per step of CPU time). The function then reads every distinct PLE row with serial
`preadv`, and only then is the graph launched. The OS-runtime trace counts the reads that exceed nsys's 1 us
threshold: 3.1 / 11.7 / 34.6 per step, taking 0.56 / 4.19 / 5.93 ms. That is 180-360 us each, i.e. NVMe page-cache
misses (the 51.2 GB table cannot be cached next to the 31.5 GiB pinned cold pool). The time varies between runs
with the page-cache state: 2.5 or 4.9 ms at c8. The scheduler (`schedule` 0.05-0.4 ms, `update_from_output`
0.02-0.08 ms), sampling and the adaptive-cache epochs run while the GPU is busy and add almost no idle time.

## 4. Speculative decoding: DFlash2 k=3 vs no speculation (same config: 11 GiB experts, 16 sequences)

| per decode step | no-spec c1 ms | % | DFlash2 k=3 c1 ms | % | no-spec c8 ms | % | DFlash2 k=3 c8 ms | % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Attention (GDN + QSA kernels) | 0.40 | 2.0 | 0.64 | 1.9 | 0.90 | 1.2 | 1.41 | 0.9 |
| Dense GEMMs (attention proj + router + D + U + shared + HC) | 7.48 | 36.7 | 8.07 | 24.0 | 8.09 | 11.1 | 9.69 | 5.9 |
| ...of which latent D + U | 2.79 | 13.7 | 2.95 | 8.8 | 2.88 | 4.0 | 3.46 | 2.1 |
| MoE routing + gather + combine + exposed hot Marlin | 0.76 | 3.7 | 0.80 | 2.4 | 0.76 | 1.0 | 2.22 | 1.4 |
| **Cold-expert Marlin (PCIe)** | **8.77** | **43.0** | **18.57** | **55.1** | **57.56** | **79.2** | **134.93** | **82.4** |
| Final HC/norm + lm_head + sampler | 0.77 | 3.8 | 0.76 | 2.3 | 0.81 | 1.1 | 0.85 | 0.5 |
| **Drafter (DFlash2 forward, in `propose`)** | - | - | **1.73** | **5.1** | - | - | **2.06** | **1.3** |
| Rejection sampling | - | - | 0.03 | 0.1 | - | - | 0.05 | 0.0 |
| other GPU + memcpy | 0.22 | 1.1 | 0.39 | 1.2 | 0.51 | 0.7 | 0.69 | 0.4 |
| GPU idle (PLE read / launch / rest) | 1.99 | 9.8 | 2.68 | 8.0 | 4.05 | 5.6 | 11.86 | 7.2 |
| **step period** | **20.39** | 100 | **33.69** | 100 | **72.69** | 100 | **163.76** | 100 |
| tokens per step (acceptance length 2.65, dfcal-3) | 1 | | 2.65 | | 8 | | 21.2 | |
| ms per output token | 20.4 | | 12.7 | | 9.1 | | 7.7 | |

(GPU metrics during the DFlash2 runs: PCIe RX 54.7% at c1 against 12.6% for no-spec, and 74.1% at c8.)

- **The verify step is a batch of 4 tokens per request, and cold experts are what make it expensive.** At c1 the
  4-token verify reads 2.1x the cold bytes of a 1-token step: only part of a request's experts carry over from one
  position to the next (TUNING-LOG 3: 29% of a token's cold experts were used by the previous token). The step
  costs 1.65x and yields 2.65 tokens, which nets +60% at c1. At c8 the 32-token verify step is a c32-sized step
  (164 ms, against 174 ms for the tuned c32 step). It costs 2.25x for 2.65x the tokens, which nets +18%. At c32,
  speculation would push 128 tokens per step through the same saturated link.
- **The drafter is cheap**: 1.7-2.1 ms per step, run as its own CUDA graph inside `propose`. Rejection sampling is
  30-50 us. `propose` and `reject` together take 0.5 ms of host time.
- **Longer drafts (the user's k = 7 / 15) will lose on this box.** Each extra draft position adds about 3.3-3.9 ms
  per step at c1: (33.7 - 1.7 drafter - 20.4) / 3, nearly all of it cold experts. Measured per-position acceptance
  is 0.76 / 0.55 / 0.35, so positions 4-7 can add at most about 0.5 tokens in total. Even if cold traffic grows
  sublinearly (cold(n) ~ n^0.54 fits 1 -> 4 tokens), k=7 comes out at about 74 tok/s and k=15 at about 60 tok/s at
  c1, against about 79 tok/s for k=3. At c8 and above the extra verify tokens cost the full per-token rate. A
  cheaper verify (section 5, or fewer cold bytes per token) would make longer drafts worthwhile.

## 5. Would Strata-style CPU computation of cold experts help?

### 5.1 Measurements on this box (`bench/dram-contention.sh`, `results/profile/dram-contention.log`)

DDR5-5600, 2 x 32 GB (one DIMM per channel, dual rank), so the theoretical peak is 89.6 GB/s. The CPU is a Ryzen
9 9950X with 16 cores and AVX-512 VNNI. The GPU side is the image's torch, reading 4 GiB of pinned host memory in a
loop.

| test | CPU GB/s | GPU GB/s | total |
|---|---:|---:|---:|
| CPU streaming read alone, 1 / 4 / 8 / 16 / 32 threads | 61.7 / 59.8 / 65.3 / 61.2 / 62.2 | - | ~62 |
| GPU alone: SM zero-copy (UVA sum) / copy engine | - | 50.4 / 57.5 | - |
| GPU zero-copy + CPU read, 4 / 8 / 16 / 32 threads | 42.8 / 43.4 / 42.2 / 42.3 | 31.4 / 31.0 / 31.0 / 30.0 | **~73** |
| Strata `nvfp4_avx512` expert kernel alone, 16 threads, 1 / 2 / 4 tokens per expert | 62.1 / 61.0 / 58.2 (31.2 / 31.7 / 33.2 us per 1.935 MB expert) | - | |
| the same, 32 threads | 60.2 / 58.7 / 57.2 | - | |
| expert kernel (16 threads, 1 token) + GPU zero-copy | **41.5 (46.7 us/expert)** | **31.0** | **72.5** |
| expert kernel (32 threads) + GPU zero-copy | 2.6 (728 us/expert) | 47.9 | collapse |

The kernel is Strata's own `src/kernels/cpu/nvfp4_avx512.cpp`, compiled read-only from its tree into `/tmp`. Each
expert is an Edge0 latent expert: gate+up 1280 x 1792 and down 1792 x 640, NVFP4 with Q8_0 activations, 24 random
experts per call from an 8 GiB pool. Strata's docs report the same DRAM-bound behaviour: 58-74 GB/s for the pool on
other CPUs, flat up to 2 tokens and about 12% lower at 4.

What this means:
1. **The CPU and the GPU's PCIe reads draw from the same DRAM, which tops out at about 73 GB/s.** As soon as the
   CPU streams (4 threads are enough), the GPU's zero-copy rate drops from 50-54 to 31 GB/s (-38%). The pair then
   moves 1.35-1.45x the bytes of the GPU alone. That ratio, not the CPU's FLOPs, caps any CPU/GPU split.
2. Alone, the CPU computes an expert as fast as the PCIe path delivers one: 31 us against 36 us. Under contention
   it takes 46.7 us, and the GPU's share costs about 58.6 us per expert (53.8 x 31/50.4 GB/s).
3. The CPU pool must leave cores for the engine. With 32 OpenMP threads plus the GPU feeder thread, the barrier
   stalls and throughput collapses to 2.6 GB/s. 16 pinned workers (Strata's pool affinity) is the right size.

### 5.2 Estimate per concurrency (balanced split of each layer's cold experts)

Per layer with C cold experts, the CPU takes x of them so that both sides finish together: x / 41.5 = (C - x) /
33.1 GB/s, giving x = 0.56 C. The layer then takes C x 25.9 us instead of C x 36 us (-28%), plus a hand-off cost
h on the critical path. h covers the GPU -> host doorbell, the CPU wake from spinning, the D2H of the CPU's routes
(3.5 KB each), the H2D of their results and the GPU wait; it is estimated at 15-30 us per layer and must be
measured first. Splits are in whole experts.

| case | cold experts / layer | cold time / step now | with CPU share (h = 25 us) | step | throughput (upper bound) |
|---|---:|---:|---:|---:|---:|
| c1, no spec | 2.2 | 3.8 ms | ~3.4-3.8 ms (1 expert on the CPU: 25 + 47 us vs 1.2 x 59 us on the GPU) | 15.8 -> ~15.5-15.8 ms | **0 to +2%** (layers with 0-1 cold experts gain nothing) |
| c1, DFlash2 k=3 verify (11 GiB) | 10.7 | 18.6 ms | 13.4 + 1.2 ms | 33.7 -> 29.7 ms | **+13%** |
| c8 | 28 | 48.1 ms | 34.6 + 1.2 ms | 65.5 -> 53.2 ms | **+23%** |
| c32 | 86 | 148.9 ms | 107.2 + 1.2 ms | 174.1 -> 133.6 ms | **+30%** |

These are upper bounds. They assume the 73 GB/s combined rate holds with Marlin on the GPU, a perfect balance and
no other stall. A realistic expectation is about two thirds of them: +15% at c8 and +20% at c32.

### 5.3 What it takes inside vLLM (and why it is not cheap)

1. **Data layout.** The cold pool is pinned host memory in Marlin layout (31.5 GiB). The CPU must read those same
   bytes: with 60 GiB of RAM there is no room for a second Strata-layout copy. So the work needs an AVX-512 VNNI
   kernel for the Marlin NVFP4 record: the permuted 4-bit tiles, fp8 block scales with the layer-wide Marlin
   factor, and the global scale. Strata's `nvfp4_avx512.cpp` cannot be used as is. Its arithmetic (E2M1 magnitude
   table, `vpdpbusd`, Q8_0 activations) carries over, and the result must match the GPU path closely enough for the
   correctness gate (Strata itself uses Q8_0 activations).
2. **CUDA-graph hand-off.** The op runs inside FULL graphs, so every CPU interaction has to be a graph node.
   - A small kernel decides the split after top-k. The rule must be deterministic and independent of timing, for
     example the cold experts with the most routes go to the CPU up to a per-layer quota. The kernel writes the
     CPU's (expert, route) list, its activations and a sequence number to mapped host memory.
   - The GPU masks those experts out of this layer's cold `expert_map`. A per-step copy of the map is
     graph-safe, because it is data.
   - Before the combine, the GPU waits on a host-written flag: a `cuStreamWaitValue32` batch-memop node, or a
     1-CTA spin kernel, then copies the CPU results in.
   - 16 pinned CPU workers busy-poll the sequence number. A timeout fallback is required, because a missed
     doorbell would hang the graph.
3. **Scheduling interference.** The engine thread, the API server and the PLE readers need the SMT siblings. Long
   CPU stalls (page faults, preemption) stall the GPU directly, so the workers must never block.

**Decision: recommended, for c >= 8 and for speculative verify steps, and only after a gating prototype.** The
prototype is the doorbell plus spin-wait plus one CPU expert inside a captured graph. It must show h <= 30 us per
layer and the measured 31 + 41 GB/s split with the real Marlin cold call. Enable the path only when a layer has
>= ~8 cold experts (c >= 4, or any verify step). It brings nothing for single-stream non-speculative decode, where
dense weight streaming and the PLE stall dominate. The ceiling is set by DRAM, not by the CPU: at most about -28%
cold time.

## 6. Bottlenecks ranked, with the next levers

| # | bottleneck | evidence | lever (owner: next agents) | expected |
|---|---|---|---|---|
| 1 | cold experts over PCIe (c >= 8; 55% of a k=3 verify at c1) | 74-86% of the step, link at its 54 GB/s ceiling, 36 us per cold expert per layer | fewer cold bytes per step (more VRAM for experts vs KV, placement, adaptive cache), CPU co-compute (section 5), larger batches per step | the CPU split is worth at most +23-30% |
| 2 | dense weight streaming at c1 (10 GB per step, D+U = 2.8 ms) | 52% of the c1 step, flat across concurrency | amortize it: speculative decoding (k=3 already +60% at c1), batching | - |
| 3 | PLE row reads on the critical path | 0.6 / 2.5-4.9 / 8 ms idle per step; preadv calls of 180-360 us each (NVMe) | parallel reads (`EDGE0_PLE_IO=threads`, implemented but never A/B'd) or io_uring; a pinned LRU of hot PLE rows | -3% to -7% step time, cheap |
| 4 | speculative verify cost grows with draft length | +3.3-3.9 ms per extra draft position at c1, cold-dominated | keep k=3 (or adaptive k by batch size); do not go to k=7/15 without a cheaper verify | avoids a regression |
| 5 | hot experts starved by the cold grid | hidden, except 0.2-2 ms per step | cold CTA cap: at most 1.2% at c32 | small |
| 6 | graph launch, scheduler, Python | 0.17-0.24 ms launch, the rest overlapped | none needed | - |

## 7. Reproduce

```
# on the 5090 host, holding the GPU lock (README / coordinator protocol)
cd ~/work/edge0-vllm
bash bench/nsys-capture.sh nospec "EDGE0_DEV=1 EDGE0_NVTX=1" 1:150:6 8:60:10 32:30:15
GPUM=0 NSYS_OPTS="... --cuda-graph-trace=graph ..." bash bench/nsys-capture.sh graph "EDGE0_DEV=1 EDGE0_NVTX=1" ...
bash bench/nsys-chain-s2.sh      # DFlash2 k=3, the same config without spec, DRAM contention
python3 tools/nsys_breakdown.py steps profiles/nospec/c8-k2.sqlite [--kind all] [--json out.json]
python3 tools/nsys_tables.py profiles nospec|spec|graph
python3 tools/nsys_breakdown.py gpum profiles/nospec/gpum-c8.sqlite
```

`bench/nsys-chain-s1.sh` holds the exact `NSYS_OPTS` of the graph-level and eager runs. To read a report in the
GUI, copy the `.nsys-rep` from `5090:~/work/edge0-vllm/profiles/`.
