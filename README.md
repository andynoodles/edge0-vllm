# edge0-vllm: Edge0-145B-NVFP4 on vLLM (one RTX 5090)

This repository serves **Edge0-145B-NVFP4** with **vLLM 0.33** (`nvcr.io/nvidia/vllm:26.11-py3`) on one RTX 5090
32 GB (sm_120, PCIe Gen5 x16), a Ryzen 9 9950X (16 cores) and 60 GiB of DDR5. It consists of a vLLM general plugin
(`plugin/edge0_vllm`), a checkpoint converter (`convert/`) and a launch script (`serve.sh`). vLLM itself is not
modified.

**Checkpoint:** [andynoodles/Edge0-145B-NVFP4-vLLM](https://huggingface.co/andynoodles/Edge0-145B-NVFP4-vLLM) on
Hugging Face is the converter's output, ready to serve (a format conversion of
[Edge0/Edge0-145B-NVFP4](https://huggingface.co/Edge0/Edge0-145B-NVFP4); no retraining). Download it and go to
[Quick start](#quick-start); you only need `convert/` to rebuild it from the original release.

Edge0 is a latent MoE built on Qwen3.8-Flash-Next (qwen4exp): 145B parameters, about 10B active per token, 48 layers
with 512 routed experts and top-10 routing per layer, hybrid GDN + QSA attention, and a 51 GB PLE n-gram table. Its
routed experts are 44.3 GiB of NVFP4, more than the GPU holds. Each layer therefore splits its experts between a
**hot pool in VRAM** and a **cold pool in pinned host memory**, which Marlin reads zero-copy over PCIe
([TidalMoE](https://github.com/andynoodles/tidal-moe)'s layout). The work followed TidalMoE's goal, **spend less time loading experts**, and then ported the mechanisms that
make Strata (the engine the model ships with) fast at one request at a time.

**Status (2026-11-20).** Two presets, both from the image `edge0-vllm:dev` (`d0d3bace6b03`), both passing the
correctness gate against Strata (257/261 teacher-forced top-1):

- `./serve.sh` (**`PRESET=tuned`**, the default) for many concurrent requests: 32 sequences, CPU computation of cold
  experts, no speculative decoding. 54.8 / 136.3 / **180.3 tok/s** at c1 / c8 / c32 (1024-token replies); the
  bring-up made 40.1 / 74.3 / 70.1.
- `PRESET=single ./serve.sh` for one request at a time: DFlash2 drafter, a request-reactive expert tier, PLE reads off
  the critical path, small-batch dense kernels, 4 sequences. **At concurrency 1 it beats Strata by 15% on the same
  EOS-stopping load: 140.5 against 122.0 tok/s, TPOT 6.86 against 8.21 ms** (same session, interleaved).

Both listen on port 8125 with the served model name `edge0_145b`.

## Results

### Standard sweep

vllm-bench, ShareGPT (seed 0), chat API, thinking off, temperature 0, 1024 output tokens with `--ignore-eos`,
4 x concurrency prompts, 2 warm-ups. Output tokens/s. Raw JSON stayed on the benchmark box (`~/work/edge0-bench/results/<label>/`, not in this repository; summaries in `results/`); the two
final rows were measured on 2026-11-20 by `bench/final-report.sh` (log `logs/final-report.log`).

| label | configuration | c1 | c2 | c4 | c8 | c16 | c32 |
|---|---|---:|---:|---:|---:|---:|---:|
| `strata-solo` | Strata, one request at a time (MTP; replies end at EOS, 686-820 tokens) | 121.2 | 110.1 | 116.7 | 113.2 | | |
| `strata-batch8` | Strata `--batch 8` (replies end at EOS, 257-735 tokens) | 85.3 | 32.4 | 44.5 | 67.4 | 77.0 | 81.4 |
| `vllm-base` | vLLM bring-up (`PRESET=bringup KV_BYTES=`) | 40.1 | 53.5 | 64.3 | 74.3 | 69.0 | 70.1 |
| `vllm-tuned` | end of the TidalMoE tuning (TUNING-LOG section 5) | 54.9 | 82.0 | 95.8 | 105.3 | 118.4 | 142.0 |
| **`vllm-final-tuned`** | **`PRESET=tuned`** (+ CPU cold experts) | **54.8** | **92.9** | **112.8** | **136.3** | **148.7** | **180.3** |
| **`vllm-final-single`** | **`PRESET=single`** (at most 4 requests run at once) | **140.3** | **188.6** | **212.7** | 214.1 | 212.4 | 214.4 |

Latency of the two final presets (mean TPOT, and median / p99 TTFT, ms):

| conc | `tuned` TPOT | `tuned` TTFT median / p99 | `single` TPOT | `single` TTFT median / p99 | `single` tokens per verify step |
|---:|---:|---:|---:|---:|---:|
| 1 | 18.0 | 185 / 497 | **6.9** | 153 / 391 | 2.70 |
| 2 | 21.0 | 545 / 791 | 10.1 | 476 / 914 | 2.70 |
| 4 | 34.6 | 865 / 1,384 | 17.9 | 531 / 753 | 2.68 |
| 8 | 57.6 | 1,253 / 1,577 | 18.1 | **18,946 / 24,117** | 2.70 |
| 16 | 105.5 | 2,302 / 2,474 | 18.3 | **57,734 / 64,187** | 2.77 |
| 32 | 174.6 | 3,108 / 3,874 | 18.2 | **133,639 / 144,304** | 2.78 |

- **`PRESET=tuned` runs all 32 requests at once** (no preemptions; the GPU peaked at 32,047 of 32,607 MiB). Against
  the end of the TidalMoE tuning it is +26% to +29% from c8 up (c32 142.0 -> 180.3) and +13-18% at c2-c4, with c1
  unchanged. The difference is the CPU cold experts (with round 2's cheaper dispatch kernel) and the PLE table on
  `/data`; a plain c1 decode step has too few cold experts per layer to give the CPU a share.
- **`PRESET=single` makes 2.6x `tuned`'s c1 throughput** and is also faster at c2 and c4. From c8 on it runs only 4
  requests and queues the rest: throughput stays at ~213 tok/s, but requests wait 19-134 s for their first token.
  For 8 or more concurrent users, `tuned` is the preset.
- Against Strata: `PRESET=single` is ahead at every concurrency it serves well (c1-c4), `tuned` is ahead from c8 on.
  Strata's rows are not like for like (shorter, EOS-terminated replies; see "Caveats"); the fair c1 comparison is
  below.

## Single stream against Strata (fair comparison)

Concurrency 1, the same 8 ShareGPT seed-0 prompts, chat API, thinking off, temperature 0, `max_tokens` 1024, and
**replies end at EOS for both engines** (Strata cannot ignore EOS). One session on 2026-11-20 (`bench/final-report.sh`),
Strata, vLLM, then the sweeps and gates, then Strata, vLLM again; each pass on a fresh server after a warm-up pass. Strata: `~/work/edge0-serve.sh`
unchanged, benchmarked on its engine's own HTTP server (`bench/gap-strata.sh ref`). vLLM: `PRESET=single` from the
image (`bench/gap-vllm.sh`). TTFT is to the first real token and "decode TPOT" is the time per token after it, per
request (`tools/c1_compare.py`, `results/final/c1-compare.txt`): Strata sends an empty first chunk at once, so
vllm-bench's own TTFT for Strata is ~1 ms and its TPOT then includes the prefill.

| pass | output tok/s | TPOT (vllm-bench) ms | decode TPOT ms | TTFT median / mean ms | mean reply (tokens) | tokens per step |
|---|---:|---:|---:|---:|---:|---:|
| Strata `final-s1` | 120.9 | 8.26 | 7.61 | 592 / 566 (engine 392) | 869 | |
| vLLM `final-v1-eos` | **140.7** | **6.86** | **6.86** | **204 / 257** | 892 | 2.73 |
| Strata `final-s2` | 123.1 | 8.16 | 7.44 | 532 / 638 (engine 362) | 856 | |
| vLLM `final-v2-eos` | **140.2** | **6.86** | **6.86** | **206 / 260** | 888 | 2.75 |
| **mean: Strata / vLLM** | **122.0 / 140.5 (vLLM +15.1%)** | **8.21 / 6.86 (-16%)** | **7.53 / 6.86 (-9%)** | **~560 / ~205** | 863 / 890 | |
| vLLM, 512 tokens with EOS ignored (`final-v1`, `final-v2`) | 135.3 / 134.2 | 6.92 / 6.99 | | 188 / 190 | 512 | 2.67 / 2.65 |

- **vLLM beats Strata at one request at a time by 15% on the like-for-like load**: 140.5 against 122.0 tok/s, TPOT
  6.86 against 8.21 ms. Per token after the first one vLLM is 9% faster (6.86 vs 7.53 ms); the rest is the prefill
  (TTFT ~205 against ~560 ms, although 28-29% of Strata's prompt tokens came from its conversation cache because the
  warm-up repeats prompts; vLLM's `PRESET=single` has prefix caching off).
- The road there, on this load (TUNING-LOG "Single-stream"): 94.2 tok/s at the start of the single-stream work
  (G1) -> 129.1 (C1: reactive expert tier, dense knobs, 15.7 GiB of experts) -> 131.5 (R1: PLE reads off the critical
  path, 16.2 GiB) -> 134.5 (R2: cheaper CPU-experts dispatch) -> 139.7 (R3: tier decides right after the forward) ->
  140.5 here (the same image as R3, with the PLE table now on `/data`).

## Correctness

Both presets were checked from the image, on the servers that produced the numbers above (`bench/final-report.sh`;
outputs in `results/compare-final-*`, `results/greedy-final-*`).

`correctness/compare_strata.py`: 8 chat prompts against Strata's recorded greedy outputs; teacher-forced on Strata's
text, then free-running. Gate: top-1 = Strata's argmax on >= 250 of 261 positions.

| | prompts pass | top-1 = Strata's argmax | top-1 = Strata's chosen token | mean abs dlogprob | KL (top 5) |
|---|---:|---:|---:|---:|---:|
| `PRESET=tuned` | 8/8 | **257/261** | 250/261 | 0.0336 | 0.0063 |
| `PRESET=single` | 8/8 | **257/261** | 250/261 | 0.0324 | 0.0064 |
| bring-up, for reference | 8/8 | 256/261 | | 0.032 | 0.006 |

(Strata picked a token other than its own argmax 9 times in these texts.)

**Speculative decoding** (`correctness/greedy_same.py`, 8 prompts x 256 greedy tokens, `PRESET=single` with vs
without the drafter, `SPEC_METHOD=none`): **6/8 identical, 2 divergences at near-ties, 0 unexplained.** Prompt 4
splits at token 92 (top-1 minus top-2: 0.0 / 0.125 nat in the two runs) and prompt 5 at token 21 (0.125 / 0.0); both
are the near-ties seen in every earlier gate. Every emitted token is its own pass's argmax. A 23,328-token prompt
was served in 12.4 s (`single`) and 11.3 s (`tuned`), and both servers stayed healthy.

## Quick start

Needs one NVIDIA GPU with 32 GB (tuned on an RTX 5090, sm_120), about 60 GiB of RAM with an unlimited `memlock`
for the pinned cold-expert pool, Docker with the NVIDIA Container Toolkit, and a fast local NVMe SSD: the 51 GB PLE
table `ple-fp8.gguf` is read in place on every decode step.

```bash
# 1. Download the checkpoint (about 112 GB) to a local NVMe SSD, and the DFlash2 drafter (PRESET=single only).
hf download andynoodles/Edge0-145B-NVFP4-vLLM --local-dir /nvme/models/Edge0-145B-NVFP4-vLLM
hf download incoai/Qwen3.8-Flash-Next-DFlash2 --local-dir /nvme/models/Qwen3.8-Flash-Next-DFlash2

# 2. Build the image: NGC vLLM + the plugin + the prebuilt CPU-experts and PLE-reader extensions
#    (again after every plugin change).
git clone https://github.com/andynoodles/edge0-vllm && cd edge0-vllm
docker build -f docker/Dockerfile -t edge0-vllm:dev .

# 3. Serve (tmux session edge0-vllm, port 8125). Ready in 3-4 min.
export MODELS_ROOT=/nvme/models    # or MODEL_DIR=... DFLASH_DIR=... (defaults: ~/models/<name>)
./serve.sh                                   # PRESET=tuned: many concurrent requests
PRESET=single ./serve.sh                     # one request at a time (<= 4 sequences), with the DFlash2 drafter
curl -s localhost:8125/health                # 200 when ready
DRY_RUN=1 PRESET=single ./serve.sh           # print the docker command
docker stop edge0-vllm; tmux kill-session -t edge0-vllm      # stop (killing tmux alone leaves the container)

PRESET=bringup KV_BYTES= ./serve.sh          # the bring-up configuration, for comparisons
EDGE0_DEV=1 ./serve.sh                       # run the plugin source from ./plugin (no rebuild)

# 4. Use it (OpenAI API).
curl -s localhost:8125/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "edge0_145b",
  "messages": [{"role": "user", "content": "Hello"}], "chat_template_kwargs": {"enable_thinking": false}}'
```

`serve.sh` mounts `MODELS_ROOT` at `/models` (a `MODEL_DIR` outside it is mounted on its own), this repository at
`/workspace/edge0` (the placement plans in `plans/` are read from there), and, when `ple-fp8.gguf` is an absolute
symlink to another disk, that disk's directory at the same path. Hardware other than one RTX 5090 + 60 GiB needs
other sizes: `EDGE0_EXPERT_VRAM_GIB`, `KV_BYTES` and the plan in `EDGE0_PLACEMENT` (built by
`tools/plan_placement.py`).

### Rebuilding the checkpoint from the original release

The Hugging Face checkpoint above is exactly this converter's output. To rebuild it, download
[Edge0/Edge0-145B-NVFP4](https://huggingface.co/Edge0/Edge0-145B-NVFP4) to `$MODELS_ROOT/Edge0-145B-NVFP4` and put
the Qwen3.8 HF tokenizer in `$TOKREF` (`convert/run_convert.sh` documents its mounts; `--corpus` and `--strata-ref`
feed the optional checks):

```bash
# Convert the original folder into a vLLM checkpoint (once; CPU only, in the NGC image).
#    522 s on the NVMe plus 53 s of checks; 60.39 GB of tensors, 57 GB on disk (the PLE table is linked, not copied).
cd convert
./run_convert.sh test
CPUS=8 MEM=16g TMUX_SESSION=edge0-convert ./run_convert.sh convert \
  --out /models/Edge0-145B-vllm --jobs 6 --checks full \
  --corpus /bench/datasets/ShareGPT_V3_unfiltered_cleaned_split.json --strata-ref /bench/strata-ref.json
```

Then `MODEL_DIR=$MODELS_ROOT/Edge0-145B-vllm ./serve.sh` (`serve.sh` also falls back to that name by itself).

### Correctness and benchmarks

These need a running Strata server for the reference side (port 8124) and the bench data under `~/work/edge0-bench`.

```bash
# Correctness against Strata (8 prompts, ~15 s; exit 0 = pass), and the speculative greedy check.
cd correctness && python3 -I compare_strata.py
python3 -I greedy_same.py gen --out /tmp/nospec.json     # on PRESET=single SPEC_METHOD=none
python3 -I greedy_same.py gen --out /tmp/spec.json       # on PRESET=single
python3 -I greedy_same.py diff /tmp/nospec.json /tmp/spec.json

# Benchmarks.
cd ..
BASE_URL=http://127.0.0.1:8125 OUT_LEN=1024 EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}' \
  bash bench/bench-openai.sh <label> 1,2,4,8,16,32                 # the standard sweep
python3 bench/summarize.py strata-solo vllm-final-tuned vllm-final-single <label>
DETAILED=1 bash bench/gap-vllm.sh <label> "PRESET=single"          # c1: warm, 512-token and EOS-stopping passes
DETAILED=1 bash bench/gap-strata.sh ref <label>                    # Strata c1, same load (vLLM stopped first)
python3 tools/c1_compare.py <strata label> <label>-eos --metrics results/gap   # TTFT to the first token, decode TPOT
bench/fc-ab.sh <name> "<env A>" "<env B>"                          # interleaved A B B A at c1 (+ c8)
```

Every `serve.sh` switch and plugin knob (`EDGE0_*`) is documented at the top of `serve.sh`.
`bench/final-report.sh` is the job that produced the final numbers in this file.

## The two presets

| setting | `PRESET=tuned` (default) | `PRESET=single` | why |
|---|---|---|---|
| dense weights | Q8_0 served natively (`EDGE0_DENSE=q8`, b12x), as Strata runs them | same, plus for calls of <= 8 tokens: latent D / U only over the routing groups in use (`EDGE0_DU_MASK=1`), fused shared expert (`EDGE0_SHARED_FUSE=1`), LM head re-quantized to Q8_0 (`EDGE0_LMHEAD=q8`) | Q8 is lossless and 4.75 GiB smaller than bf16; the small-batch kernels cut the dense time of a decode step |
| hot experts (VRAM) | 12.8 GiB = 7,101 of 24,576 experts (`plans/plan-s1-12.8.json`) | 16.2 GiB = 8,987 slots: 8,795 hot experts + 4 free tier slots per layer (`plans/plan-s1-16.2.json`) | planned from vLLM's own decode routing on ShareGPT |
| expert cache | `EDGE0_ADAPT=1`: TidalMoE's z-test swaps per epoch of a few seconds | `EDGE0_TIER=1`: Strata's request-reactive policy, decided every step right after the forward, copies while the drafter runs, shadow host copies (`EDGE0_TIER_EARLY=1 EDGE0_TIER_COPY=batch EDGE0_TIER_SPARE=64`) | |
| cold experts | pinned host memory, Marlin zero-copy over PCIe; 55% of each 2-256-token call's cold experts computed by 8 CPU threads (`EDGE0_CPU_EXPERTS=1`) | same | |
| PLE rows | read by the host behind a stream sync before each step | read by a C++ thread while the GPU runs layer 0; layer 1 waits in the graph (`EDGE0_PLE_DEFER=1`) | |
| speculative decoding | off | DFlash2 drafter, 3 draft tokens (`SPEC_METHOD=dflash2 SPEC_K=3`) | the drafter's GDN state costs KV at every batch size (see "Rejected") |
| sequences / KV | 32 / sized by vLLM's profiler (`GPU_UTIL=0.96`, `MAX_BATCHED=4096`) | 4 / 1.75 GiB fixed (`KV_BYTES=1879048192`, `MAX_BATCHED=1024`, `GPU_UTIL=0.94`) | `single` gives the KV memory to hot experts |
| both | bf16 GDN state, prefix caching off, `--max-model-len 32768`, torch.compile + CUDA graphs, `embed_tokens` in pinned host memory | | |

## Memory layout

From the final servers' load logs and `free -m` / `nvidia-smi` with the server up (`logs/final-report.log`, `MEM`
lines). One expert (NVFP4 in Marlin layout, with scales) is 1.85 MiB.

| | `PRESET=tuned` | `PRESET=single` |
|---|---|---|
| **VRAM** (RTX 5090: 32,607 MiB; 31.36 GiB visible to torch) | | |
| hot experts | **12.80 GiB** = 7,101 experts (128-166 per layer) | **16.20 GiB** = 8,987 slots: 8,795 hot experts (147-206 per layer) + 4 free tier slots per layer |
| rest of the target model (Q8_0 dense, attention, router, norms, LM head; by difference) | 10.53 GiB (BF16 LM head, 1.27 GB) | 9.68 GiB (Q8_0 LM head, 0.68 GB) |
| DFlash2 drafter | - | 1.26 GiB (shares the target's embedding and LM head after loading) |
| KV cache (QSA, bf16) + GDN state (bf16) | **4.61 GiB** = 173,812 tokens (5.3 x 32k), 32 sequences | **1.75 GiB** = 46,291 tokens (1.41 x 32k), 4 sequences |
| MoE workspace, CUDA graphs, NCCL communicator | 520 MB (4,096 tokens), 0.37 GiB, 0.49 GiB | 130 MB (1,024 tokens), 0.21 GiB, 0.49 GiB |
| `nvidia-smi` memory.used | 30,961 MiB after start; 32,043-32,045 MiB during the sweep (after a 23k-token prompt) | 31,911 MiB after start, 31,915 MiB peak during the sweep |
| **Host DRAM** (60 GiB DDR5, 61,823 MiB) | | |
| cold experts, pinned (`cudaHostRegister`), read by the GPU over PCIe and by the CPU workers | **31.50 GiB** = 17,475 experts | **34.33 GiB** = 15,781 experts |
| tier spare host slots (shadow copies of swapped-in experts), pinned | - | 3,264 slots = 5.9 GiB |
| `embed_tokens` (pinned), CPU-experts mailbox | 1.18 GiB, 18.4 MB | 1.18 GiB, 18.4 MB |
| used / available with the server up | 40.5 / 21.3 GB | 44.9 / 16.9 GB |
| page cache (caches PLE rows) | 1.4 GB after start | 3.0 GB after start, 12.2 GB after the sweep |
| **NVMe** | | |
| PLE table `ple-fp8.gguf`, 320,001,536 rows x 160 B = 51.2 GB, read in place each step (16 rows per token; 75-90 per c1 verify step) | `/data/edge0` (Crucial P3 Plus, its own NVMe), symlinked from the checkpoint | same |
| checkpoint `~/models/Edge0-145B-vllm` (57 GB, the PLE table linked) | system disk, read at start (~90 s of the 170-210 s start-up) | same |

For comparison, Strata gives its experts 18.67 GiB of VRAM (10,358 slots), keeps all 44.3 GiB of experts in a 45.4
GB pinned arena (so evicting an expert costs nothing), and lends expert slots to prefill buffers during a prompt
(`docs/SINGLE-STREAM-GAP.md` section 5).

## Where the time goes

**Many requests (`PRESET=tuned`).** Nsight Systems profile (`docs/PROFILE.md`) of the tuned preset before CPU
experts were added, steady decode, graph-level step period:

| | c1 | c8 | c32 |
|---|---:|---:|---:|
| decode step | 16.4 ms | 67.0 ms | 165.2 ms |
| **cold-expert Marlin, PCIe-bound** (36 us per cold expert per layer, link at its ~54 GB/s ceiling) | **24%** | **74%** | **86%** |
| dense weight streaming from VRAM (about 10 GB per step; latent D + U alone 18% at c1) | 52% | 14% | 6% |
| GPU idle, almost all of it the host reading PLE rows after a full stream sync | 5.0% | 4.3% | 5.1% |

- **Cold experts are the bottleneck from c8 up.** That is why the work targeted fewer cold bytes per step
  (placement, adaptive cache, more VRAM for hot experts) and then a second reader of the cold pool, the CPU.
- **The CPU path is capped by host DRAM, not by CPU compute.** The GPU's PCIe reads and the CPU's reads share the
  dual-channel DDR5: together about 73 GB/s (GPU alone 50-54 GB/s). The 0.55 split balances the two sides (at c8 the
  CPU takes 812 us and the GPU side 883 us per MoE call), cutting the cold time per call by 20-26%, at that ceiling.
- **A speculative verify step is cold-expert bound**: draft tokens share few cold experts, so a 4-token DFlash2 verify
  costs 1.65x a plain step at c1 and each extra draft position about 3.3 ms.

**One request (`PRESET=single`).** Per output token inside the engine, against Strata (`docs/SINGLE-STREAM-GAP.md`
section 12: node- and graph-level Nsight captures at c1 of round 2's `PRESET=single`, 145 four-token verify steps,
2.65 tokens per step; Strata from its own engine timing):

| ms per output token | Strata | vLLM | |
|---|---:|---:|---|
| cold experts on the critical path | **1.72** | **2.26** | 54 cold expert loads per token against Strata's 40 (2.97 cold experts per layer per step, 42 us each, PCIe + 8 CPU threads) |
| dense path + resident experts + LM head | 5.22 | **3.84** | attention 1.12, latent D/U + shared expert 1.27, hyper-connections + norms 0.54, router + routing 0.45, the rest 0.46 |
| drafter | 0.46 (MTP) | 0.52 (DFlash2) | |
| expert-tier swaps | 0.24 (blocking) | 0 on the GPU | but 41.5 MB H2D + as much D2H per step ran next to the cold reads in layers 1-12 |
| PLE wait in layer 1 + GPU idle | 0.08 | 0.26 | |
| **total** | **7.72** | **6.94** | graph-level period 18.39 ms per step |

- vLLM is faster on everything except cold experts: it gets more tokens per verify step (2.65-2.76 against Strata's
  2.29) and its dense kernels for small batches are as fast or faster.
- **Host DRAM is shared by everything that touches host memory**: an H2D and a D2H copy do not overlap on this box
  (62 MB each way: 1.07 ms one way, 2.0 ms both at once), and the CPU experts and cold Marlin reads use the same DRAM.
  Round 3 therefore moved the tier's copies out of the forward into the drafter window and dropped most write-backs
  (shadow host copies), +2.5% at c1. The final preset was not profiled again after that change.

## What was ported, and what each bought

Gains are interleaved A/B deltas (A B A or A B B A, a fresh server per letter, 512 output tokens with EOS ignored) against
the configuration before each change; they compound. Every kept change passed the correctness gate. Details and
raw numbers: `docs/TUNING-LOG.md`.

**From [TidalMoE](https://github.com/andynoodles/tidal-moe)** (tuning for many requests, `PRESET=tuned`; c1 / c8 / c32):

| TidalMoE mechanism | Edge0 / vLLM port | result |
|---|---|---|
| 1. Per-expert hot/cold placement | Per-layer hot pool in VRAM + cold pool in pinned host memory, cold experts read zero-copy by Marlin; a logical-to-slot `expert_map` instead of permuting the router (routing stays bit-exact); hot and cold Marlin calls on two streams | makes the model run at all on vLLM (stock vLLM cannot run the latent MoE and only offloads whole parameters): bring-up 40.1 / 74.3 / 70.1 tok/s |
| 2. Adaptive expert cache (usage counts + significance test, in-place swaps) | `EDGE0_ADAPT=1`: `[48, 512]` touch counters inside the op, EMA + Poisson z-test per epoch, <= 64 batched swaps, CUDA-graph safe | **kept: +5 / +11 / +2%** |
| 3. Kernel-aware layout, per-layer budgets from a convex cost model | `tools/plan_placement.py` from vLLM's own decode-routing traces (`plans/plan-s1-*.json`) | **kept: +12 / +15 / +14%** (13-24% fewer cold experts per step) |
| 4. Forced MoE kernel tactics | per-tier Marlin M block (`EDGE0_COLD_BSM=64` for calls of > 64 tokens) | **kept: TTFT -23%**, throughput -1 / +6 / -1% |
| 5. VRAM reclaimed for experts | native Q8_0 dense, `embed_tokens` in pinned host memory, torch's nccl2 backend off (2.9 GiB back), KV sized for 32 requests (bf16 GDN state, no prefix caching) | **q8: +19 / +11 / +27%; 32-request KV: -1 / -10 / +32%** (c32 median TTFT 196 s -> 3.5 s) |
| 6. Host-aware persistent MoE kernel | not built: the cold Marlin call already saturates PCIe and the hot GEMMs hide inside it | - |
| 7. Calibrated adaptive verification | vLLM 0.33's own port refuses this model (the GDN backend cannot trim verify lengths per request); the batch-size draft schedule was tested instead | **rejected**: c8 109.6 -> 103.6 tok/s |
| (lever L10) next-layer predictive prefetch | `EDGE0_PREFETCH=8`: pre-gating, VRAM staging slots, copies on a side stream | **rejected**: -13% at c1, +3.5% at c8 |

**From Strata** (each measured on the configuration of its time):

| Strata mechanism | Edge0 / vLLM port | result |
|---|---|---|
| Q8_0 dense weights as shipped | `EDGE0_DENSE=q8` (b12x kernels, no lossy re-quantization) | in both presets (TidalMoE row 5 above) |
| CPU computation of cold experts (pinned busy-polling workers next to the GPU's PCIe reads) | `EDGE0_CPU_EXPERTS=1`: a new AVX-512 kernel that reads vLLM's Marlin layout directly (no second copy of the cold pool; bit-identical to Marlin for single routes), GPU doorbell + host flag inside CUDA graphs (`docs/CPU-EXPERTS.md`); a cheaper dispatch kernel in round 2 | **kept in both presets: +27.7% at c8, +24.5% at c32, c1 neutral; +17.1 / +21.1 / +17.4% with DFlash2 k=3**; dispatch kernel 8.7 -> 1.1 us per call, +3.1% at c1 |
| Speculative decoding (Strata: MTP, `--spec 4` with a confidence cut, 3.1 positions per window on average) | the DFlash2 block drafter (`/models/Qwen3.8-Flash-Next-DFlash2`, fed by layers 5/15/25/35/45), k=3; Edge0's own MTP layer also works | **kept in `single`: +65.3% at c1** (+5.8% at c8, -19.4% at c32); MTP k=3 equal at c1, worse at c8 |
| Request-reactive expert tier (+1 per routed entry, decay, swap when a candidate beats a victim by 1.5) | `EDGE0_TIER=1`: the same policy decided every step from counts a graph-safe kernel collects, copies on the copy engines, maps flipped between steps; later (round 3) decided right after the forward with copies in the drafter window and shadow host copies | **kept in `single`: +7.4% at c1** (c8 +5.0%), **+9.4%** more with the c1 memory split; round 3 +2.5% at c1, +2.8% at c8 |
| VRAM goes to experts, not KV (`--expert-cache auto`: all free VRAM after the model) | `PRESET=single`: KV for 1.4 x 32k tokens, the rest to hot experts (15.2 -> 15.7 -> 16.2 GiB) | part of the +9.4% above; 16.2 GiB: +1.9% at c1, +5.1% at c8 |
| PLE reads off the critical path (Strata reads them with O_DIRECT outside its windows) | `EDGE0_PLE_DEFER=1`: a C++ reader thread reads the step's rows while the GPU runs the launch and layer 0; layer 1 waits in the graph | **kept in `single`: +4.2% at c1, +5.4% at c8** |
| Q8_0 LM head (its GGUF format) | `EDGE0_LMHEAD=q8`, shared with the drafter | with the latent D/U skip and the fused shared expert (`EDGE0_DU_MASK`, `EDGE0_SHARED_FUSE`): **+7.3% at c1 without a drafter, +2.7% with DFlash2, +9.7% at c8**; on `PRESET=single` with the freed VRAM given to experts **+6.4%** |

## Rejected, and why

| idea | why |
|---|---|
| Speculative decoding in `PRESET=tuned` | -19.4% at c32 and 9x the p99 TTFT at c8: vLLM 0.33 keeps one GDN state per draft position per request (~90 MiB each at k=3), so only 6 requests fit next to 12.8 GiB of hot experts. A batch-size draft schedule does not help (the state is sized for the largest k) |
| Draft lengths 4, 5, 7, 15 (DFlash2 and MTP), MTP confidence threshold | every extra position costs 3.2-3.4 ms of verify at c1 (mostly cold experts) while positions 5+ accept <= 0.19: k=7 is 9-23% slower than k=3, k=15 40-52%; k=4 on `PRESET=single` -2.4% |
| Next-layer prefetch (`EDGE0_PREFETCH`) | -13% at c1: ~20 small kernels and a cross-stream join per layer on the critical path |
| Cold-expert block scales in VRAM | -11% bytes per cold touch, but 3.6 GiB of VRAM = 1,940 hot slots, worth more |
| Dense weights re-quantized to NVFP4 | cancelled by the user: dense stays Q8_0, as Strata runs it |
| Cold Marlin CTA cap, hot and cold on one stream | no gain: the cold call is PCIe-bound; two streams are 1-9% faster |
| CPU experts with 6 threads, or a 0.6 / `auto` split; CPU jobs of 2+ cold experts at c1 | 6 threads lose ~8% at c8; the others are within noise of 8 threads / 0.55 |
| PLE reads with more threads (`EDGE0_PLE_IO=threads`), PLE defer without the stream sync | the system disk saturates at ~46k random reads/s; without the sync the host runs ahead and its scheduling lands on the critical path (113 vs 117-119 tok/s) |
| More than 16.2 GiB of experts in `PRESET=single` | 16.7 GiB (KV 1.35 GiB) neutral (+0.2% at c1, -1.0% at c8); 17.0 GiB runs out of memory in the first prefill |
| Fused routing tail (`EDGE0_ROUTE_FUSE=1`), tier swap cap 96, tier decision on the GPU, write-back cost, larger swap cap with a looser rule | +1.4% at c1 / -2.1% at c8 (under the 3% bar); neutral; slower (side stream queues behind the LM head); neutral; -2 to -6% (the copies no longer fit the drafter window) |
| Benchmarking Strata through edge0-infer's front end on :8124 | vllm-bench mis-reads its reused HTTP/1.1 connections (a request counted with 1 token while Strata generated 609); Strata is measured on its engine's own server (`bench/gap-strata.sh`, `DIRECT=1`) |

## Caveats

- **Strata ignores `ignore_eos`.** Its replies always end at EOS (856-869 tokens per reply on average in the c1 comparison, 257-820 in
  the sweep rows), while the
  standard vLLM sweeps generate exactly 1024 tokens per request. Shorter replies keep KV and attention costs lower, so
  the sweep table is not like for like; the c1 comparison above is (EOS-stopping for both, same prompts). Strata's
  server sends an empty first chunk at once, so vllm-bench's TTFT for Strata (1-11 ms) is meaningless; use
  `tools/c1_compare.py`.
- **The Strata sweep rows (`strata-solo`, `strata-batch8`) are from 2026-11-19**, measured by the coordinator through
  edge0-infer's front end on :8124 (bench-openai.sh's default URL). That front end was later found to make vllm-bench
  miscount some streams (TUNING-LOG "Single-stream" C1). `strata-solo`'s c1 (121.2 tok/s) agrees with every later
  direct measurement (120.3-122.2); the short mean replies of `strata-batch8` (257 tokens at c2) may be partly that.
- **The placement plans are ShareGPT-specific.** `plans/plan-s1-*.json` come from vLLM's decode routing on ShareGPT
  prompts (seed 1). For other traffic, re-capture (`bench/capture-routes.sh`) and re-plan (`tools/plan_placement.py
  --gib <budget> --out plans/<name>.json`); the adaptive cache (`tuned`) and the tier (`single`) then follow the
  traffic at run time. A plan must be rebuilt whenever the expert budget changes.
- **`PRESET=single` is for one request at a time.** It runs at most 4 requests and queues the rest (median TTFT
  about 10-12 s at c8 with 512-token replies); use `PRESET=tuned` for 8 or more concurrent requests. It pins about
  5.7 GB more host memory than `tuned` (the tier's spare host slots).
- **c1 varies 2-5% between fresh servers** (the caches warm up, reply lengths and acceptance differ); compare within
  one session with interleaved runs.
- **Greedy text with and without the drafter is not byte-identical**: the verify step runs 4 rows per request, a
  different GEMM shape than a 1-row decode, and bf16 rounding flips near-ties (top-1 minus top-2 <= 0.125-0.25 nat).
  Every emitted token is still the target pass's own argmax. Batched vs one-at-a-time decode without a drafter shows
  the same flips.
- **KV head-room is thin in `PRESET=tuned` at c32** (within ~100-150 MiB of the OOM ceiling with 1024-token outputs;
  vLLM does not profile the GDN/QSA prefill working set). Much longer prompts at c32 may need `MAX_BATCHED=2048`.
- **The PLE table sits on `/data`** (a separate Crucial P3 Plus NVMe; symlinked from the checkpoint dirs, mounted by
  `serve.sh`): about +2.5% at c1 for vLLM, no change for Strata (`bench/plemove.sh`). `bench/plemove.sh switch root`
  moves it back.
- **Licenses.** The code in this directory is part of this repository (Apache-2.0, `../LICENSE`). The CPU expert
  pool follows the design of Strata's CPU pool (MIT, Copyright (c) 2026 Niko1221 and the Strata contributors); no
  Strata source is copied, and the attribution is in `plugin/edge0_vllm/csrc/NOTICE` (the repository-root `NOTICE`
  does not mention Strata yet). The model is not included. Its Edge0 parts are under the **Edge0 Community Model
  License 1.0**, which allows personal, research, benchmarking and internal use but **requires a separate written
  commercial license from Edge0 to operate a commercial model service** (an inference API or hosted endpoint for
  third parties financed by fees, subscriptions or advertising; section 4(b)) or to sell access to the model. The
  Qwen base is under the Qwen Community License 1.0; the DFlash2 drafter and ShareGPT keep their own licenses.

## What remains

1. **Bring the single-stream mechanisms to `PRESET=tuned`.** The tier (`EDGE0_TIER`), the PLE reader
   (`EDGE0_PLE_DEFER`) and the small-batch dense kernels (`EDGE0_DU_MASK`, `EDGE0_SHARED_FUSE`, `EDGE0_LMHEAD=q8`)
   were A/B'd only on `PRESET=single` and at c1 / c8. The Q8 LM head alone frees 0.6 GiB, which gave +9.7% at c8 with
   a drafter (more KV); at c32 a step reads ~512 PLE rows (~10 ms) behind the stream sync. They need A/Bs at c8 / c32
   on `tuned`.
2. **A preset for 4-8 concurrent requests.** `PRESET=single` makes 212.7 tok/s at c4 (`tuned`: 112.8) with a 0.5 s
   median TTFT, but runs only 4 requests at a time. A middle preset (the drafter and the single-stream mechanisms
   with 6-8 sequences and a smaller expert budget) was not measured.
3. **GDN speculative state** keeps speculation out of `tuned`: a GDN variant of vLLM's ReplaySSM (keep the pre-window
   state plus each draft token's small k/v/g/beta inputs) would cut k=3 from ~270 MiB to a few MiB per request, so
   32 requests would fit; the same variable verify lengths would let vLLM's adaptive verification and Strata-style
   confidence-trimmed windows (3.1 positions on average) run.
4. **Single stream, next levers** (TUNING-LOG "Single-stream" R3 item 7; expected 0.05-0.15 ms per token each): flip
   each layer's expert maps in the graph once its own copies have landed (the next forward still waits 0.15-0.4 ms
   per step); fewer write-backs (8-15 per step remain; a full host copy of every expert needs +16.2 GiB pinned); one
   fused routing kernel (top-k, routing tail, CPU hand-off, gather: ~22 us per layer today); a smaller prefill working
   set so that more than 16.2 GiB of experts fits (16.7 GiB fits but was neutral, 17.0 GiB runs out of memory in the
   first prefill).
5. **Long-prompt TTFT**: a 1,024-token prefill chunk spends ~0.35 s reading PLE rows, about 8 s of the 12.4-12.6 s
   TTFT of a 23k-token prompt; the prompt's rows are known on arrival and could be read one chunk ahead.
6. **Memory head-room in `tuned`**: A/B `MAX_BATCHED=2048` (or `GPU_UTIL=0.955`) at c32; the speculative servers'
   CUDA-graph estimate is 0.6 GiB too high (1-2 more requests would fit).
7. A fused host-aware MoE kernel (TidalMoE mechanism 6) remains unbuilt (at most ~1-2% exposed today).
8. `plugin/tests/test_registration_quant.py::test_entry_point_and_register` fails: it expects the plain Qwen4Exp
   config class, but `__init__.py` registers the DFlash wrapper config. The repository-root `NOTICE` does not yet
   mention Strata.

## Files

| Path | What |
|---|---|
| `serve.sh` | starts the server (port 8125, tmux `edge0-vllm`); presets `tuned` (default), `single`, `bringup`; every switch documented at its top |
| `plugin/`, `docs/PLUGIN.md` | the `edge0_vllm` plugin: `Edge0ForCausalLM`, latent MoE block, expert pools, adaptive cache (`adaptive.py`), expert tier (`tier.py`), CPU experts (`cpu_experts.py`, `csrc/`), PLE reader (`ple_defer.py`), Q8 small-batch kernels (`q8gemv.py`, `lmhead_q8.py`), DFlash2 hooks (`draft.py`); tests in `plugin/tests/` |
| `docker/Dockerfile` | `edge0-vllm:dev` = NGC vLLM image + plugin + prebuilt CPU-experts and PLE-reader extensions |
| `convert/`, `docs/CONVERT.md` | converter and validator that produce the checkpoint (published as [andynoodles/Edge0-145B-NVFP4-vLLM](https://huggingface.co/andynoodles/Edge0-145B-NVFP4-vLLM)) |
| `docs/DESIGN.md`, `docs/MODEL-SPEC.md`, `docs/VLLM-BASE.md` | design and contracts; how Strata computes the model; what the vLLM image provides |
| `docs/BRINGUP-LOG.md` | first light: what broke, the fixes, memory, first numbers |
| `docs/TUNING-LOG.md` | every measurement and A/B with numbers (tuning, CPU experts, draft lengths, single-stream rounds, final report) |
| `docs/PROFILE.md`, `docs/CPU-EXPERTS.md`, `docs/SINGLE-STREAM-GAP.md` | Nsight profile of `tuned`; CPU cold experts design; the c1 gap to Strata and its profiles per round |
| `plans/` | hot/cold placement plans built by `tools/plan_placement.py` |
| `correctness/` | `compare_strata.py` (vs Strata), `greedy_same.py` (speculative greedy identity), `long_state_check.py` |
| `bench/` | `bench-openai.sh` (sweep), `gap-vllm.sh` / `gap-strata.sh` (c1 comparison), `ab.sh` / `fc-ab.sh` (A/B), `final-report.sh` (this file's numbers), `summarize.py`, `sweep_table.py`, `specstat.py`, `nsys-capture.sh`, `gap-nsys.sh`, `plemove.sh` |
| `tools/` | expert-IO microbenchmarks, placement planner, tier replay (`sim_tier.py`), Nsight analysis, `c1_compare.py`, CPU-expert and copy benchmarks |
| `results/` | copies of correctness outputs, logs and summaries per stage (`results/final/` for this report) |
