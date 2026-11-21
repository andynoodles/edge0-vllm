# edge0-vllm

A vLLM plugin that serves **Edge0-145B-NVFP4** (145B-parameter latent MoE, ~10B active) on **one RTX 5090 32 GB**.
vLLM itself is not modified.

- Checkpoint: [andynoodles/Edge0-145B-NVFP4-vLLM](https://huggingface.co/andynoodles/Edge0-145B-NVFP4-vLLM)
- The expert-offloading design comes from [TidalMoE](https://github.com/andynoodles/tidal-moe).

## Results

RTX 5090, Ryzen 9 9950X, 60 GiB DDR5. ShareGPT prompts, thinking off, temperature 0. Output tokens/s.

**One request at a time**, replies end at EOS for both engines, same session:

| | tok/s | TPOT |
|---|---:|---:|
| Strata (the engine Edge0 ships with) | 122.0 | 8.21 ms |
| **edge0-vllm `PRESET=single`** | **140.5 (+15%)** | **6.86 ms** |

**Concurrency sweep**, 1024 output tokens per request:

| | c1 | c2 | c4 | c8 | c16 | c32 |
|---|---:|---:|---:|---:|---:|---:|
| Strata, one request at a time\* | 121.2 | 110.1 | 116.7 | 113.2 | | |
| vLLM bring-up (before tuning) | 40.1 | 53.5 | 64.3 | 74.3 | 69.0 | 70.1 |
| `PRESET=tuned` | 54.8 | 92.9 | 112.8 | 136.3 | 148.7 | 180.3 |
| `PRESET=single` | 140.3 | 188.6 | 212.7 | 214.1 | 212.4 | 214.4 |

\* Strata cannot ignore EOS, so its replies are shorter (686-820 tokens); the table above is the fair comparison.

`PRESET=single` runs at most 4 requests and queues the rest, so from c8 on the first token waits 19-134 s (median).
`PRESET=tuned` runs all 32 at once: median time to first token at c32 is 3.1 s.

**Correctness:** both presets match Strata's top-1 token in 257 of 261 teacher-forced positions
(`correctness/compare_strata.py`). With and without the drafter, greedy output is identical for 6 of 8 prompts; the
other 2 split at near-ties (top-1 and top-2 within 0.125 nat).

## Requirements

- NVIDIA GPU with 32 GB (tested: RTX 5090, sm_120), Docker with the NVIDIA Container Toolkit, `tmux`, and the `hf`
  CLI (`pip install -U huggingface_hub`)
- 64 GB of RAM recommended (tested with 60 GiB). 33 GiB (`tuned`) to 41 GiB (`single`) of it is pinned, so `memlock`
  must be unlimited.
- An AVX-512 CPU, for the CPU cold-expert path (on in both presets; turn it off with `EDGE0_CPU_EXPERTS=0`)
- About 112 GB on a local NVMe SSD. The 51 GB PLE table `ple-fp8.gguf` is read from disk on every decode step. To
  keep it on another drive, leave an absolute symlink named `ple-fp8.gguf` in the model folder; `serve.sh` mounts
  the target's directory.

## Quick start

```bash
# Checkpoint, and the DFlash2 drafter (needed by PRESET=single only)
hf download andynoodles/Edge0-145B-NVFP4-vLLM --local-dir /nvme/models/Edge0-145B-NVFP4-vLLM
hf download incoai/Qwen3.8-Flash-Next-DFlash2 --local-dir /nvme/models/Qwen3.8-Flash-Next-DFlash2

# Image: NGC vLLM 0.33 (nvcr.io/nvidia/vllm:26.11-py3) + the plugin
git clone https://github.com/andynoodles/edge0-vllm && cd edge0-vllm
docker build -f docker/Dockerfile -t edge0-vllm:dev .

# Serve on port 8125, in tmux session edge0-vllm (ready in 3-4 min)
export MODELS_ROOT=/nvme/models
./serve.sh                       # PRESET=tuned
PRESET=single ./serve.sh         # or: one request at a time
curl -s localhost:8125/health    # 200 when ready
docker stop edge0-vllm; tmux kill-session -t edge0-vllm

# OpenAI API
curl -s localhost:8125/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "edge0_145b",
  "messages": [{"role": "user", "content": "Hello"}],
  "chat_template_kwargs": {"enable_thinking": false}}'
```

Without `MODELS_ROOT`, `serve.sh` looks in `~/models`; `MODEL_DIR` and `DFLASH_DIR` set the two folders directly.
`DRY_RUN=1 ./serve.sh` prints the docker command. Every switch is documented at the top of `serve.sh`.

## Presets

| | `tuned` (default) | `single` |
|---|---|---|
| for | many concurrent requests: all 32 run at once | 1-4 concurrent requests; more are queued |
| hot experts in VRAM | 12.8 GiB | 16.2 GiB |
| expert swaps | every ~3 s, from recent traffic | every step, from the running requests |
| speculative decoding | off | DFlash2 drafter, up to 3 draft tokens per step |

The hot-expert plans in `plans/` were built for one RTX 5090 and ShareGPT-like traffic. For another GPU or workload,
set `EDGE0_EXPERT_VRAM_GIB` and `KV_BYTES` and build a plan with `bench/capture-routes.sh` and
`tools/plan_placement.py`.

## How it works

The routed experts are 44.3 GiB of NVFP4, more than the GPU holds.

- **Hot/cold experts.** Each layer keeps its most-used experts in VRAM and the rest in pinned host memory, which
  vLLM's Marlin kernel reads directly over PCIe. Placement comes from vLLM's own routing on real traffic and adapts at
  run time.
- **CPU cold experts.** 8 CPU threads compute 55% of each call's cold experts while the GPU reads the rest. The
  AVX-512 kernel reads vLLM's Marlin layout directly, so there is no second copy.
- **Dense weights stay Q8_0**, as Strata runs them, leaving more VRAM for experts. `PRESET=single` adds small-batch
  kernels and a Q8_0 LM head.
- **Speculative decoding** (`single`) with the DFlash2 drafter, fed by five of Edge0's layers.
- **PLE table reads** (`single`) run on a C++ thread while the GPU starts the step.

From c8 up, cold-expert reads over PCIe take most of each step. Measurements, A/B results and rejected ideas:
[`docs/TUNING-LOG.md`](docs/TUNING-LOG.md).

## Rebuilding the checkpoint

The Hugging Face checkpoint is the output of `convert/`. To rebuild it:

1. Download [Edge0/Edge0-145B-NVFP4](https://huggingface.co/Edge0/Edge0-145B-NVFP4) to
   `$MODELS_ROOT/Edge0-145B-NVFP4`.
2. Put the tokenizer files of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
   (`tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`) in a folder and set `TOKREF` to it.
3. Convert. The converter runs in the NGC image without network, CPU only; `--out` is a container path
   (`/models` = `$MODELS_ROOT`).

```bash
export MODELS_ROOT=/nvme/models TOKREF=/path/to/qwen-tokenizer
cd convert && CPUS=8 MEM=16g ./run_convert.sh convert --out /models/Edge0-145B-vllm --jobs 6 --checks full
```

It took about 10 minutes on the test box. `serve.sh` also finds `$MODELS_ROOT/Edge0-145B-vllm`. Details:
[`docs/CONVERT.md`](docs/CONVERT.md).

## Repository

| Path | |
|---|---|
| `plugin/` | the `edge0_vllm` plugin ([`docs/PLUGIN.md`](docs/PLUGIN.md)) |
| `serve.sh`, `docker/`, `convert/`, `plans/` | launcher, image, checkpoint converter, hot-expert plans |
| `correctness/`, `bench/`, `tools/`, `results/` | checks against Strata, benchmarks, analysis tools, recorded outputs |
| `docs/` | design, bring-up and tuning logs, profiles |

## License

The code is MIT ([LICENSE](LICENSE)); third-party notices are in `plugin/edge0_vllm/csrc/NOTICE`. The model is not
included. It is under the Edge0 Community Model License 1.0 and the Qwen Community License 1.0; a commercial model
service needs separate licenses from Edge0 and from Qwen. See the
[model card](https://huggingface.co/andynoodles/Edge0-145B-NVFP4-vLLM).
