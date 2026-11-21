# Edge0-145B on vLLM: expert-load tuning log (TidalMoE-style)

Agent `tune-experts`, 2026-11-19. Goal: cut the time each engine step spends loading experts (fewer cold-expert bytes
per step, a higher VRAM hit rate, overlap of transfers with compute) and maximize vLLM output throughput at
concurrency 1..32 on the RTX 5090 (PCIe Gen5 x16, 32 GB VRAM, 60 GiB RAM).

Protocol (TidalMoE README): vllm-bench, ShareGPT, chat API, thinking off, `--ignore-eos --temperature 0 --seed 0`,
prompts = 4 x concurrency, 2 warm-ups. Baseline and final sweeps use 1024 output tokens. A/B runs use 512 output
tokens at c1 / c8 / c32 (4 / 16 / 64 prompts) and interleave A B A (`bench/ab.sh`; a fresh server per letter): the
two A passes bound the run-to-run noise (it was 0-6%, worst at c8), and each change kept here won by more than that
at some concurrency. A change is kept only if it gains more than ~3% and `correctness/compare_strata.py` still
passes (teacher-forced top-1 >= 250/261, mean |dlogprob| ~0.03).

**Result (section 5):** 40.1 / 53.5 / 64.3 / 74.3 / 69.0 / 70.1 -> **54.9 / 82.0 / 95.8 / 105.3 / 118.4 / 142.0**
output tok/s at c1 / 2 / 4 / 8 / 16 / 32 (+37% to +103%), c32 median TTFT 196 s -> 3.5 s, correctness unchanged.

Tools added for this work (all under `edge0/`):

| Path | What |
|---|---|
| `tools/bench_expert_io.py` | microbenchmarks: copy-engine and SM zero-copy bandwidth, per-touch op cost hot vs cold, Marlin re-reads, CTA cap |
| `tools/plan_placement.py` | placement planner from routing traces (TidalMoE `analyze_routing.py` + `plan_convex.py`, per-expert, PCIe costs) |
| `tools/analyze_locality.py` | c1 request-level locality: LRU staging tier vs the same VRAM as static hot slots |
| `plugin/edge0_vllm/adaptive.py` | adaptive expert cache (`EDGE0_ADAPT=1`), TidalMoE `_AdaptiveCache` port |
| `plugin/edge0_vllm/trace.py` | `EDGE0_TRACE=routes` now records MoE-call boundaries; new `EDGE0_TRACE=predict` (next-layer pre-gating accuracy) |
| `plugin/edge0_vllm/latent_moe.py` | `marlin_tier_direct`: per-tier Marlin M-block / CTA cap (`EDGE0_{HOT,COLD}_BSM`, `EDGE0_COLD_BLOCKS_PER_SM`) |
| `plugin/edge0_vllm/expert_pool.py` | `ExpertPool.swap_many`: batched in-place hot/cold swaps (graph-safe); optional staging slots |
| `plugin/edge0_vllm/prefetch.py` | next-layer prefetch into sticky staging slots (`EDGE0_PREFETCH`, rejected, opt-in) |
| `plans/plan-s1-12.8.json` (`plan-s1-12.1.json`) | the kept placement (and the equal-VRAM one used by the prefetch A/B) |
| `bench/ab.sh`, `bench/summarize.py`, `bench/capture-routes.sh`, `bench/probe-config.sh` | A/B driver, result tables, routing capture, config probe (start + memory lines + correctness + quick c1) |
| `correctness/long_state_check.py` | long-generation drift check for the GDN state dtype |
| `serve.sh` | `PRESET=tuned` (default) / `PRESET=bringup`; new switches `PREFIX_CACHING`, `EDGE0_ADAPT*`, `EDGE0_PREFETCH*`, `EDGE0_*_BSM` |
| `results/tuning/` | copies of the microbenchmark output, correctness runs and A/B summaries |
| `bench/post-baseline.sh`, `post-ab1.sh`, `chain-*.sh`, `final-tuned.sh` | the one-off job chains that ran this log's experiments |

## 1. Baseline (`vllm-base`)

The bring-up configuration exactly (`KV_BYTES= ./serve.sh`: bf16 dense, 9.5 GiB = 5,270 hot experts from
`profile.bin`, `--max-num-seqs 16`, KV sized by the profiler = 3.19 GiB / 80,956 tokens). Full standard sweep,
1024 output tokens, 15:42-16:49. Results: `5090:~/work/edge0-bench/results/vllm-base/`.

| conc | requests | output tok/s | mean TPOT ms | median TTFT ms | P99 TTFT ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 4/4 | 40.1 | 24.5 | 373 | 865 |
| 2 | 8/8 | 53.5 | 36.5 | 996 | 1,245 |
| 4 | 16/16 | 64.3 | 60.9 | 1,374 | 1,990 |
| 8 | 32/32 | 74.3 | 106.2 | 1,558 | 1,781 |
| 16 | 64/64 | 69.0 | 184.3 | 2,712 | 194,086 |
| 32 | 128/128 | 70.1 | 181.7 | 195,563 | 388,845 |

Reading:
- **c16 and c32 do not run 16 or 32 requests.** The engine log shows `Running: 13 reqs, Waiting: 19 reqs, GPU KV
  cache usage: 98.8%`. Every request holds about 7.6% of the 3.19 GiB KV pool (~245 MB): the fp32 GDN state is
  3.2 MB per layer, vLLM sets the attention block to 1600 tokens so attention pages match it, and with prefix
  caching on (mamba_cache_mode `align`) each GDN group keeps 2 state pages per request. So the bring-up config caps
  concurrency at ~13; the rest queues (P99 TTFT 194-389 s). Throughput is flat from c8 on.
- Expert traffic dominates every point (bring-up `EDGE0_TRACE=bw`): at c1 0.52-0.62 GB of cold reads per step
  (~10 of 24.5 ms); at 12-13 running requests 5.5-6.7 GB per step (~105-125 of ~175 ms).

`vllm-stock-offload` (vLLM's own offload as the "stock" reference): not run as a sweep. vLLM's routed-expert path
cannot run Edge0's latent MoE at all (DESIGN.md section 6), and `--cpu-offload-gb` only offloads whole parameters.
A "whole layers hot or cold" layout (what UVA offload does to MoE layers) was not run: it was cheap to emulate as a
plan file but would have cost a GPU hour, so the bring-up config (per-expert `profile.bin` placement) is the
reference throughout (section 4.7).

## 2. Measurements of this box (`tools/bench_expert_io.py`, `logs/expert-io.{txt,json}`)

Run in the `edge0-vllm:dev` image with the GPU free, 16:49. Synthetic pools of 256 hot slots (VRAM) and 256 cold
slots (pinned host, `host_register`), Marlin layout; the op body (`latent_experts_marlin`: gather -> hot Marlin ||
cold Marlin -> combine) captured in a CUDA graph and replayed, as in serving.

**Bandwidth (PCIe Gen5 x16):**

| path | GB/s |
|---|---:|
| copy engine H2D, one 1.85 MB expert record / 32 MiB / 1 GiB | 55.4 / 57.7 / 57.4 |
| copy engine D2H (what a swap writes back), same sizes | 56.2 / 57.4 / 57.4 |
| copy engine, gather of 6 / 10 / 64 records (one memcpy each) | 56.1 / 55.6 / 56.0 |
| SM zero-copy gather kernel (UVA host rows -> VRAM), 6 / 10 / 32 / 128 / 400 records | 53.0 / 53.4 / 53.6 / 53.7 / 53.7 |

Zero-copy reads by SMs reach 53-54 GB/s already at 6 experts (11 MB); the copy engine is ~5% faster on large
transfers. There is no bandwidth headroom left to gain inside the kernels: **the only levers are fewer cold bytes
and overlap**.

**Per-touch cost inside the MoE step** (TidalMoE measured ~45 us per DRAM expert and ~3 us per HBM expert on GB300):
least-squares fit of op time = a + b x cold + c x hot over the touch sweep:

| tokens in the call | a (us) | **b: per cold expert (us)** | **c: per hot expert (us)** |
|---:|---:|---:|---:|
| 1 | 4.1 | **38.0** | 2.7 |
| 8 | 30.4 | 36.3 | 0.5 |
| 16 | 0.1 | 35.9 | 1.2 |
| 32 | 5.7 | 35.9 | 1.2 |
| 64 | 28.0 | 35.8 | 1.2 |

A cold expert costs **~36 us** (= 1,935,360 B at 53.8 GB/s: the cold Marlin call is purely PCIe-bound), a hot one
**~1.2 us**: a 30x ratio (GB300: 15x). Ten experts of one token: 24.7 us all hot, 393.7 us all cold.

**Marlin re-reads cold experts once per M block (new finding).** T tokens all routed to the same 10 cold experts:

| routes per expert | auto `block_size_m` (=8) | M block 16 | M block 64 |
|---:|---:|---:|---:|
| 1 | 380 us | 381 | 388 |
| 8 | 382 | 383 | 389 |
| 16 | 742 | 385 | 390 |
| 32 | 1,468 | 744 | 392 |
| 64 | 3,904 | 1,471 | 399 |
| 128 | 8,304 | 3,858 | 764 |
| 256 | 17,475 | 8,384 | 1,799 |

vLLM's `fused_marlin_moe` picks the M block from the average routes per expert (`R / 512 / bsm < 0.9`), i.e. 8 for
every call below ~2,200 tokens. Each 8-route block of an expert re-reads its 1.85 MB over PCIe (the L2 does not keep
host lines between blocks). Decode rarely gives a cold expert more than 8 routes, but every prefill chunk does: a
300-token prompt sends ~30 routes to a popular cold expert -> 4 reads. Fix (`EDGE0_COLD_BSM=64
EDGE0_BSM_MIN_TOKENS=64`): the cold call of any MoE call above 64 tokens uses one 64-row block per expert (one read
per expert up to 64 routes); decode-sized calls keep vLLM's choice (a 64-row block costs ~2% at 1 route). A/B in
section 4.

**CTA cap and stream split:** limiting the cold call to 1 or 2 CTAs per SM changes nothing (203 / 2,562 / 4,046 /
5,495 us at 1 / 16 / 32 / 64 tokens vs 204 / 2,547 / 4,047 / 5,495 uncapped): the call is bound by PCIe, not SMs.
Running hot and cold on two streams saves 1-9% (the hot part is small).

## 3. Routing on the real workload (`bench/capture-routes.sh`, `EDGE0_TRACE=routes,predict`)

Bring-up config + trace, ShareGPT seed 0, 512 output tokens: 3 prompts at c1, 16 at c4, 64 at c16 (16:50-17:07).
161 shards in `~/.cache/edge0-vllm/edge0-vllm/traces/` (35 MB): 8,643 decode calls, 45,413 decode tokens,
32,423 prefill tokens, with the MoE-call boundaries (`calls`), so per-step distinct experts can be replayed exactly.

**Skew** (TidalMoE's 80/20 test, decode tokens): the top 10 / 20 / 30 / 50% of a layer's experts carry 30 / 48 /
61 / 81% of its selections (layer range at 20%: 39-64%). Flatter than DeepSeek-V4.1 on GB300, so hot slots buy
less per GiB here. **Decode and prefill use different experts:** the per-layer top-20% sets of decode and prefill
tokens overlap with Jaccard 0.33. Strata's `profile.bin` (single-stream ranking) gives layers 55 (L15) to 214 (L0)
hot slots; vLLM's decode traffic wants a much flatter 90-133 per layer.

**Simulator check** (`tools/plan_placement.py`, B decode tokens per step drawn from the trace): the profile plan at
9.5 GiB gives 301 cold + 179 hot distinct experts per step at B=1, the bring-up `EDGE0_TRACE=bw` measured
270-322 + 168-242. At B=13 it predicts ~3,500 cold, measured 2,843-3,466 at 12-13 running requests. Good enough
to rank plans.

**Placement from decode routing vs `profile.bin`** (cold distinct experts per decode step, all 48 layers; GB at
1.935 MB each). In-sample = plan built and scored on the whole trace; held-out = built on the first half of the
decode steps (c1, c4, start of c16), scored on the second half (other prompts):

| VRAM for experts | plan | B=1 | B=8 | B=16 | B=32 |
|---|---|---:|---:|---:|---:|
| 9.5 GiB | profile.bin (bring-up) | 301 (0.58 GB) | 2,214 | 4,027 | 6,694 (12.96 GB) |
| 9.5 GiB | trace plan, in-sample | 230 (-24%) | 1,794 (-19%) | 3,387 (-16%) | 5,938 (-11%) |
| 9.5 GiB | profile.bin, held-out half | 303 | 2,226 | 4,084 | 6,874 |
| 9.5 GiB | trace plan, held-out | 274 (-10%) | 2,003 (-10%) | 3,746 (-8%) | 6,424 (-7%) |
| 12.5 GiB | profile.bin, held-out | 275 | 2,022 | 3,711 | 6,254 |
| 12.5 GiB | trace plan, held-out | 233 (-15%) | 1,699 (-16%) | 3,204 (-14%) | 5,532 (-12%) |
| 14.2 GiB | profile.bin, held-out | 260 | 1,913 | 3,510 | 5,919 |
| 14.2 GiB | trace plan, held-out | 212 (-18%) | 1,543 (-19%) | 2,923 (-17%) | 5,065 (-14%) |

The convex per-layer budget (`plan_convex.py`'s E[max(hot, cold)] objective with the measured 36 / 1.2 us costs)
gives the same allocation as the linear top-H rule here, because a hot touch is 30x cheaper than a cold one
(TidalMoE needed the convex model because its cold experts were only 15x dearer and shared the kernel with hot
tiles). Only about half the in-sample gain survives on unseen prompts, so the A/B plan is built from a separate
capture with different prompts (ShareGPT seed 1) and benchmarked on the seed-0 prompts (section 4).

**Request-level locality at c1** (`tools/analyze_locality.py`, 2,556 c1 decode tokens, static hot set = top 9.5 GiB
of the trace): 29% of a token's cold experts were already used by the previous token, 36% within 2, 43% within 4,
53% within 8, 62% within 16 tokens. An LRU tier of S VRAM slots per layer in front of the cold pool vs the same
VRAM as more static hot slots:

| S slots / layer | VRAM | cold reads per token, LRU tier | cold reads per token, same VRAM as static slots |
|---:|---:|---:|---:|
| 0 | 0 | 167 | 167 |
| 4 | 0.35 GiB | 141 (-16%) | 162 (-3%) |
| 8 | 0.69 GiB | 110 (-34%) | 157 (-6%) |
| 16 | 1.38 GiB | 88 (-48%) | 148 (-11%) |
| 32 | 2.77 GiB | 66 (-61%) | 131 (-22%) |

At low concurrency a recency tier is worth 5x more per GiB than static slots: each request keeps returning to its
own experts.

**Next-layer predictability** (`EDGE0_TRACE=predict`): layer l+1's router applied to layer l's MoE input. Share of
layer l+1's cold routes whose expert is in the predicted top-k' (route recall), and coverage of the step's distinct
cold experts by the union of predicted cold experts, with the over-fetch:

| phase | k'=10 | k'=16 | k'=24 | k'=32 |
|---|---|---|---|---|
| c1 | 65% (1.06x fetched) | 79% (1.8x) | 85% (2.7x) | 89% (3.7x) |
| c16 (12-13 running) | 69-71% (1.02x) | 83% (1.6x) | 89-91% (2.2x) | 93-94% (2.8x) |

Worst layers: L1 and L39 (26-34%), L3 and L23 (~40-49%). So one layer ahead, two thirds of the cold experts can be
fetched before the router asks for them, with almost no wasted bytes at k'=10. At c1 the PCIe link idles about
half of every layer (cold reads are ~10 of 24.5 ms), so this is real overlap (TidalMoE design C, which TidalMoE did
not build). Built as `EDGE0_PREFETCH` (plugin `prefetch.py`), combined with the locality result: the staging slots
are sticky (an LRU-like tier fed by the predictor), see section 4.

## 4. A/B experiments (one change at a time)

All A/B runs: `bench/ab.sh`, ShareGPT seed 0, 512 output tokens, prompts = 4 (c1) / 2 x conc (c8, c32), 2 warm-ups,
a fresh server per letter change. "A" below is the mean of the A passes. Labels `<name>-A1`, `-B2`, ... under
`5090:~/work/edge0-bench/results/`, logs `~/work/edge0-vllm/logs/ab-<name>.log`.

### 4.1 Dense Q8_0 served natively (`EDGE0_DENSE=q8`) and the freed VRAM given to hot experts: KEPT

The checkpoint's dense layers are GGUF Q8_0 blocks; bring-up dequantized them to bf16 at load (15.3 GiB).
`EDGE0_DENSE=q8` (already wired by the model agent, first run here) loads them through vLLM's
`ModelOptBlockQuantLinearMethod("q8_0")` and runs FlashInfer's b12x A16 Q8_0 GEMV/GEMM on sm_120 (276 kernel
signatures warmed at start, +12 s). It is lossless (the same int8 x fp16-scale values).

- Probe at the bring-up budget (9.5 GiB experts): torch allocation after load 24.83 -> **20.08 GiB (-4.75 GiB)**,
  KV pool 3.19 -> 7.96 GiB; c1 TPOT 24.6 -> 21.9 ms (256-token quick bench): the dense part reads half the bytes.
- Correctness: 8/8 pass, teacher-forced top-1 = Strata argmax **257/261** (bring-up 256), mean |dlogprob| 0.033,
  KL(top-5) 0.0065 (`correctness/results/compare-q8-9.5.*`).
- A/B `q8` (ABA): A = bring-up (`KV_BYTES=`), B = `EDGE0_DENSE=q8 EDGE0_EXPERT_VRAM_GIB=14.2` (the freed 4.7 GiB
  as 2,600 more hot slots, 7,869 = 32% of all experts; KV unchanged at ~3.2 GiB).

| conc | A1 | B2 | A3 | B vs A | TPOT A -> B (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 38.2 | **45.7** | 38.4 | **+19%** | 25.3 -> 21.1 |
| 8 | 74.9 | **82.6** | 73.7 | **+11%** | 103 -> 93 |
| 32 | 69.6 | **88.2** | 69.7 | **+27%** | 180 -> 146 |

(c32 still runs only ~13 requests at a time here; section 4.2 fixes that.)

### 4.2 KV sized for 32 concurrent requests (bf16 GDN state, no prefix caching, `--max-num-seqs 32`)

Why ~13 was the ceiling: vLLM makes every attention page as large as one GDN state page. With the fp32 state (3.2 MB
per layer) that is 1,600 tokens per page, and with prefix caching on (`mamba_cache_mode=align`) each GDN group
keeps 2 state pages per request: ~245 MB per request. Changes (all `serve.sh` switches):

- `SSM_DTYPE=bfloat16` (`--mamba-ssm-cache-dtype bfloat16`, which is vLLM's own default for this architecture;
  bring-up had forced fp32 to mirror Strata): pages halve, attention block 800 tokens;
- `PREFIX_CACHING=0`: `mamba_cache_mode=none`, one state page per request (ShareGPT has no reusable prefixes: hit
  rate 0.0% in every run; keep it on for multi-turn chat if its memory is affordable);
- `MAX_SEQS=32`, hot experts 12.8 GiB so that vLLM's profiler leaves a 4.29 GiB KV pool (161,702 tokens).

Probe (`logs/probe-kv32.log`): one short request holds 1.4% of a 7.87 GiB pool = ~113 MB (was ~240 MB). During the
seed-1 capture at c32 the engine ran **32 requests, KV 87-89%** (512-token outputs).

Correctness with bf16 state: teacher-forced top-1 = Strata argmax 257/261, mean |dlogprob| 0.0317, KL 0.0061 (fp32
state: 257/261, 0.033, 0.0065). One free-running prompt (French translation) flips its first token "Il" ->
"Le": Strata's own margin there is 0.13 nat and vLLM scores the two exactly equal (-0.702), so the script's
">= 4 equal tokens before a divergence" rule fails on a near-tie, 7/8 prompts pass. Because 48-token references
cannot show error that builds up in a recurrent state, `correctness/long_state_check.py` generated 6 x 1024
greedy tokens with an fp32-state server and with a bf16-state server and scored both teacher-forced (one prefill
chunk, so the scoring does not depend on the state dtype). Mean log-likelihood per token by 256-token window:

| state | [0,256) | [256,512) | [512,768) | [768,1024) |
|---|---:|---:|---:|---:|
| fp32 (bring-up config) | -0.2652 | -0.2730 | -0.2689 | -0.2659 |
| bf16 (32-sequence config) | -0.2573 | -0.2801 | -0.2802 | -0.2643 |

No drift with length (differences +-0.011 nat, both signs); the texts first diverge after 17-362 tokens at
near-ties (`results/long-state.txt`).

A/B `kv32` (ABA): A = 4.1's winner (q8, 14.2 GiB experts, 16 seqs, fp32 state, prefix caching), B = `SSM_DTYPE=bfloat16
PREFIX_CACHING=0 MAX_SEQS=32 EDGE0_EXPERT_VRAM_GIB=12.8` (q8). **KEPT.**

| conc | A1 | B2 | A3 | B vs A | notes |
|---:|---:|---:|---:|---:|---|
| 1 | 45.7 | 45.4 | 46.3 | -1% | |
| 8 | 84.2 | 78.0 | 89.5 | -10% | A passes alone spread 6%; 1.4 GiB fewer hot experts |
| 32 | 88.1 | **116.7** | 88.1 | **+32%** | median TTFT 83.3 s -> **5.1 s**, P99 167 s -> **5.7 s**; 32 running instead of 13 |

The 32-request config costs ~1,400 hot slots of VRAM (KV pool 4.29 GiB instead of ~3.2 GiB) and pays it back many
times over at c32. Section 4.3 recovers c8 with a better placement.

### 4.3 Per-expert placement from vLLM's own decode routing (`plans/plan-s1-12.8.json`): KEPT

TidalMoE's per-expert hot/cold placement, planned from routing statistics: `tools/plan_placement.py` on the seed-1
capture (`capture-routes.sh routes-s1 1,8,32` on the 4.2 config with **other ShareGPT prompts**, seed 1:
104 shards, decode steps up to B=32), budget 12.8 GiB = 7,101 slots, convex per-layer budgets with the measured
36 / 1.2 us costs. Per layer the hot set is the top-n by decode selection probability; no router change is needed
(logical ids map to slots through the two expert maps, so this is the "expert permutation with router remap" of
TidalMoE without touching the router). Per-layer n_hot 128-166 (profile: 89-240).

Scored offline on the seed-0 traffic (the benchmark prompts, unseen by the planner): cold distinct experts per
decode step 271 -> 206 (-24%) at B=1, 1,993 -> 1,596 (-20%) at B=8, 3,625 -> 2,996 (-17%) at B=16, 6,032 -> 5,229
(-13%) at B=32.

A/B `plan` (ABA): A = 4.2's config (`EDGE0_PLACEMENT=profile`), B = `EDGE0_PLACEMENT=plan:.../plan-s1-12.8.json`.

| conc | A1 | B2 | A3 | B vs A | TPOT A -> B (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 45.1 | **50.4** | 45.1 | **+12%** | 21.4 -> 19.2 |
| 8 | 80.4 | **93.8** | 82.2 | **+15%** | 94 -> 81 |
| 32 | 117.7 | **134.2** | 118.2 | **+14%** | 263 -> 231 |

The measured gains match the simulated cold-traffic cuts almost one to one (cold reads are 70-80% of the step).
Placement does not change numerics (hot and cold run the same Marlin kernel; mixed vs all-hot < 1e-3 in the unit
tests); the final configuration's correctness run is in section 6.

From here on every A/B compares one change against the same baseline **BEST** = 4.1 + 4.2 + 4.3 (q8 dense, 12.8 GiB
plan, 32 sequences, bf16 state, no prefix caching), so the decisions are independent.

### 4.4 Cold-call M block 64 for calls above 64 tokens (`EDGE0_COLD_BSM=64 EDGE0_BSM_MIN_TOKENS=64`): KEPT

The section 2 finding: Marlin reads a cold expert once per 8-route M block, so prefill chunks re-read popular cold
experts over PCIe several times. With the override the cold call of any MoE call above 64 tokens (prefill chunks
and mixed prefill+decode steps) uses one 64-row block per expert; decode-only calls (<= 64 tokens incl. CUDA-graph
padding) keep vLLM's choice. Exact same math (unit test: identical to the stock path within bf16 noise).

A/B `coldbsm` (ABA):

| conc | A1 | B2 | A3 | B vs A | median TTFT A -> B | P99 TTFT A -> B |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 50.5 | 50.1 | 50.5 | -1% | 298 -> **230 ms (-23%)** | 732 -> **487 ms (-33%)** |
| 8 | 94.7 | **101.0** | 95.2 | **+6.4%** | 2,002 -> **1,527 ms (-24%)** | 2,426 -> 2,390 ms |
| 32 | 134.7 | 132.6 | 133.2 | -1% | 4,039 -> 4,039 ms | 5,396 -> 4,987 ms |

Prefill gets faster (TTFT -23% at c1 and c8); c8 throughput +6%; c1 and c32 throughput are within noise (at c32
most cold experts of a mixed step come from the 31 decode tokens, where a 64-row block only adds idle MMA rows).

### 4.5 Adaptive hot/cold swaps (`EDGE0_ADAPT=1`, TidalMoE `_AdaptiveCache` port): KEPT

`plugin/edge0_vllm/adaptive.py`. Inside the op, a graph-safe Triton histogram adds 1 per distinct expert per decode
call (calls up to `max(max_num_seqs, 16)` = 32 tokens; prefill chunks are not counted; padded rows masked) into a
[48, 512] fp32 counter. Every 3 s with >= 30 counted steps (`compute_logits`, engine thread) the counter is read
once, folded into an EMA (half-life 10 epochs), and per layer the hottest cold experts are paired with the
coldest hot ones; a pair swaps when x > 1.25 y and x - y > 3 sqrt(x + y + 1) (TidalMoE policy "z"), at most 64
pairs per epoch, largest gain first. `ExpertPool.swap_many` moves the rows in place with index_select /
index_copy_ on the current stream (cold side = UVA view of the pinned pool; fp8 scales moved as bytes) and
updates the two expert maps: addresses never change, the captured CUDA graphs stay valid. Per-layer budgets are
the plan's. Unit tests: swaps between graph replays keep results equal to the all-hot pool.

Live behaviour (`logs/serve-20261119-202541.log`): the cold share of decode touches at c1/c8 fell from 29% to
16-22% within the first 8 epochs (64 swaps each, 20 ms per epoch incl. the counter read); at c32 epochs stretch
to ~9.5 s (38 counted steps) and the cache keeps making 64 swaps per epoch at ~55% cold share of distinct
touches (most of a c32 step's touches are cold by construction; the swaps are cheap: 64 x 2 x 1.85 MB = 4.5 ms of
PCIe per 9.5 s).

A/B `adapt` (ABA), A = BEST, B = BEST + `EDGE0_ADAPT=1`:

| conc | A1 | B2 | A3 | B vs A | TPOT A -> B (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 50.1 | **52.6** | 50.0 | **+5.1%** | 19.3 -> 18.3 |
| 8 | 94.6 | **105.1** | 94.7 | **+11.0%** | 80.7 -> 72.1 |
| 32 | 134.5 | 136.9 | 133.6 | +2.1% | 231 -> 226 |

The static plan was built from other prompts; the cache tracks the live mix (the benchmark's own prompts), which
is exactly what it is for in production. Saved placements (`EDGE0_ADAPT_SAVE=<path>`, resume with
`EDGE0_PLACEMENT=plan:<path>`) are supported but not used by default, so a fresh start is reproducible.

### 4.6 Next-layer prefetch into sticky VRAM staging slots (`EDGE0_PREFETCH=8`)

`plugin/edge0_vllm/prefetch.py`, built from section 3's two findings (one layer ahead two thirds of the cold
experts are predictable; a recency tier is worth 5x more per GiB than static slots at c1). Each layer's hot tier
gets S = 8 extra VRAM slots (0.69 GiB in all; the B side uses a 12.1 GiB plan so total VRAM equals A). At layer l-1,
layer l's router is applied to layer l-1's MoE input; per expert the largest probability over the step's tokens is
taken, hot experts masked, top-8 with p >= 0.02 kept. Staging is sticky: an expert predicted again keeps its slot
(no copy); a new prediction takes the slot predicted longest ago; residents keep serving hits until evicted. The
copy (w13, w2, s13, s2 rows: 1.85 MB per expert) is an SM gather kernel (32 CTAs) on a low-priority stream that
layer l's MoE op joins; misses are still read zero-copy, so results do not depend on the predictor. In FULL CUDA
graphs the fork/join spans layer l's attention; in PIECEWISE steps (where breakable graphs end a capture segment at
every attention op) nothing is prefetched. Unit tests: eager and graph-captured staging give the all-hot result,
sticky eviction order, rows identical to the cold pool.

Two bugs found on the full model, both fixed: (1) the copy list was a set of main-stream temporaries read by the
prefetch stream after Python had freed them; the caching allocator reused the memory -> illegal address at the
first graph capture (now persistent per-pool buffers); (2) v1 ran the ~20 small bookkeeping kernels per layer on the
main stream.

A/B `prefetch2` (v1, bookkeeping on the main stream), A = BEST + 4.4 + 4.5 (the kept set), B = A with the 12.1 GiB
plan + `EDGE0_PREFETCH=8`: c1 52.3 -> 45.7 (**-13%**, TPOT 18.6 -> 21.4 ms: ~60 us of serial kernels x 48 layers),
c8 105.4 -> 109.1 (+3.5%), c32 138.4 -> 135.6 (-2%). The A3 pass was skipped (c1 is clearly worse).

v2 moves the router GEMV, top-k, bookkeeping, map updates and copies onto the prefetch stream (only a copy of the
layer input into a persistent buffer stays on the main stream) and joins at the start of layer l's op. Probe
(`logs/probe-prefetch3.log`, 256-token quick bench): correctness 8/8, teacher-forced 257/261 (prefetch cannot change
numerics), but **c1 TPOT 22.0 ms** vs 18.6 ms without prefetch: still worse.

**REJECTED for now** (`EDGE0_PREFETCH=0` stays the default; the code stays as an opt-in experiment). The data say the
idea is sound (65-79% of next-layer cold experts predictable, PCIe idle half of each layer at c1), but this
implementation loses more on the critical path than it hides: ~20 tiny kernels per layer plus a cross-stream join
per layer in the c1 graph, and 0.69 GiB taken from static hot slots. Not yet known without an Nsight trace: whether
the low-priority stream is starved so layer l waits at the join, or the graph branches themselves cost the time.
Next step (open issue): one fused Triton kernel for predict + top-k + sticky bookkeeping + copy-list (1 launch per
layer instead of ~20), triggered right after layer l-1's MoE op instead of before its D projection, then Nsight.

### 4.7 Not run as A/Bs (decided by measurement or analysis)

| idea | decision | why |
|---|---|---|
| Cold-call CTA cap (`EDGE0_COLD_BLOCKS_PER_SM=1/2`) | rejected | microbenchmark: identical times; the cold call is PCIe-bound, not SM-bound |
| Hot/cold on one stream (`EDGE0_SPLIT=0`) | rejected (split stays) | microbenchmark: two streams 1-9% faster |
| Cold block scales resident in VRAM (`EDGE0_COLD_SCALES=vram`) | rejected | -11% bytes per cold touch, but 17,475 cold experts x 215 KB = 3.6 GiB of VRAM = 1,940 hot slots; the static slots are worth far more (TidalMoE moved scales the other way) |
| Convex per-layer budgets vs linear top-H | equivalent | a hot touch is 30x cheaper than a cold one, so `plan_convex.py`'s E[max(hot, cold)] objective allocates like the linear rule here; the planner keeps both |
| Dense weights re-quantized to NVFP4 | cancelled by the user | coordinator 15:55: dense stays Q8_0 as Strata runs it (4.1); the code path was removed again |
| `vllm-stock-offload` sweep | not run | stock vLLM cannot run Edge0's latent MoE (DESIGN.md 6) and `--cpu-offload-gb` only offloads whole parameters; the bring-up config with `profile.bin` placement is the closest reference |
| MTP / DFlash2 speculative decoding | out of scope here | next stage (shared task #7); it is the main remaining lever at c1 (Strata's 121 tok/s solo uses MTP) |

## 5. Final configuration (`./serve.sh`, PRESET=tuned, running on :8125)

```
cd ~/work/edge0-vllm && ./serve.sh          # tmux edge0-vllm, container edge0-vllm, image edge0-vllm:dev (rebuilt 21:10)
PRESET=bringup ./serve.sh                   # the bring-up command, for comparisons
```

Differences from the bring-up command (all others unchanged):

| switch | bring-up | tuned | section |
|---|---|---|---|
| `EDGE0_DENSE` | bf16 (dequantized, 15.3 GiB) | **q8** (native Q8_0 via b12x, lossless) | 4.1 |
| `EDGE0_EXPERT_VRAM_GIB` | 9.5 (5,270 hot, 21%) | **12.8 (7,101 hot, 29%)** | 4.1, 4.2 |
| `EDGE0_PLACEMENT` | profile (Strata `profile.bin`) | **plan:/workspace/edge0/plans/plan-s1-12.8.json** | 4.3 |
| `MAX_SEQS` / KV | 16 / 3.19 GiB (~13 requests fit) | **32 / profiler (4.62 GiB, 173,812 tokens; 32 fit)** | 4.2 |
| `SSM_DTYPE` | float32 | **bfloat16** (vLLM's default for the architecture) | 4.2 |
| `PREFIX_CACHING` | 1 | **0** | 4.2 |
| `EDGE0_COLD_BSM`, `EDGE0_BSM_MIN_TOKENS` | (vLLM auto) | **64, 64** | 4.4 |
| `EDGE0_ADAPT` | 0 | **1** | 4.5 |

Memory (21:16 start, `logs/serve-20261119-211606.log`): torch allocation after load 23.32 GiB (12.80 GiB hot experts
+ ~10.5 GiB dense q8 and the rest), KV 4.62 GiB, host: 31.5 GiB of pinned cold experts (was 34.8 GiB, so ~3 GiB more
page cache for the PLE table). Start-up ~170 s.

### Results: full standard sweep, ShareGPT, 1024 output tokens (label `vllm-tuned`, 21:19-21:57)

| conc | bring-up (`vllm-base`) tok/s | **tuned (`vllm-tuned`) tok/s** | gain | TPOT ms base -> tuned | median TTFT base -> tuned | P99 TTFT base -> tuned |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40.1 | **54.9** | **+37%** | 24.5 -> 18.0 | 373 -> 215 ms | 865 -> 471 ms |
| 2 | 53.5 | **82.0** | **+53%** | 36.5 -> 23.8 | 996 -> 622 ms | 1,245 -> 1,031 ms |
| 4 | 64.3 | **95.8** | **+49%** | 60.9 -> 40.7 | 1,374 -> 921 ms | 1,990 -> 1,692 ms |
| 8 | 74.3 | **105.3** | **+42%** | 106.2 -> 74.8 | 1,558 -> 1,299 ms | 1,781 -> 1,454 ms |
| 16 | 69.0 | **118.4** | **+72%** | 184.3 -> 132.9 | 2,712 -> 2,287 ms | 194 s -> **3.0 s** |
| 32 | 70.1 | **142.0** | **+103%** | 181.7 -> 222.3 | 196 s -> **3.5 s** | 389 s -> **4.9 s** |

At c32 the engine now runs 32 requests (KV 82-86% with 1024-token outputs) instead of 13. All 384 requests
succeeded. Strata for reference (coordinator, MTP on, `ignore_eos` ignored so ~700-800-token outputs):
`strata-solo` 121 / 110 / 117 / 113 tok/s at c1 / 2 / 4 / 8 (one request at a time), `strata-batch8` 85 / 32 / 45 /
67 / 77 / 81 at c1..c32. Without speculative decoding, vLLM is now above `strata-batch8` at every concurrency >= 2
and above Strata's best number from c16 on; c1-c8 still trail `strata-solo`, whose MTP is not in this
configuration yet.

Where the gain came from (A/B deltas at 512 tokens, each vs its own baseline; they compound): q8 dense +19 / +11 /
+27% (c1 / c8 / c32), 32-request KV -1 / -10 / +32%, decode-routing plan +12 / +15 / +14%, cold M block -1 / +6 / -1%
(TTFT -23%), adaptive cache +5 / +11 / +2%.

## 6. Correctness of the final configuration

`correctness/compare_strata.py` on the idle tuned server (`results/tuning/compare-tuned.txt`): **8/8 prompts pass**,
teacher-forced top-1 = Strata argmax **257/261** (bring-up 256/261), mean |dlogprob| **0.0328** (bring-up 0.032),
KL(top-5) **0.0065** (bring-up 0.006). Every kept change was also checked on its own: q8 (257/261, 0.033), the
32-request config with bf16 state (257/261, 0.0317; long-generation check without drift), placement / M block /
adaptive swaps do not change numerics (same kernels, unit-tested within bf16 noise).

## 7. Open issues (for the next agents)

1. **Speculative decoding** (task #7, DFlash2 or MTP) is the main lever left at c1-c8; cold experts touched by a
   verify step are shared by its draft tokens, so it also amortizes expert loads.
2. **Prefetch** (4.6) needs a fused single-kernel predictor/bookkeeping and an Nsight trace before it can pay.
3. **Adaptive cache at c32** keeps swapping 64 pairs per ~9.5 s epoch (distinct touches saturate at 32 tokens/step);
   try a larger margin or selection counts for large batches. Each epoch's counter read syncs the GPU once.
4. **Plans are workload-specific.** `plan-s1-12.8.json` comes from ShareGPT (seed 1). For other traffic, re-capture
   (`bench/capture-routes.sh`) and re-plan (`tools/plan_placement.py --gib <budget> --out plans/<name>.json`), or let
   the adaptive cache follow it and save with `EDGE0_ADAPT_SAVE`. A plan must be rebuilt when the expert budget
   changes (n_hot per layer is in the file).
5. **KV head-room at c32 / 1024 tokens is ~15%** (82-86% used). Longer prompts or outputs at 32 concurrent
   requests will queue; trade `EDGE0_EXPERT_VRAM_GIB` (and the plan) against KV if that workload matters.
6. `GPU_UTIL=0.96` leaves ~0.3 GiB unused (~170 hot slots); 0.97 is untested.
7. Prefix caching is off in the tuned preset (0% hit rate on ShareGPT; it doubles GDN state pages). Turn it back on
   (`PREFIX_CACHING=1`) for multi-turn chat if its memory is affordable.

## Coordinator requests (user) — for the speculative-decoding stage

- **DFlash2 drafter instead of MTP** (shared task list #7, details in 5090:~/work/edge0-vllm-COORDINATOR-NOTE.txt): do not load
  Edge0's MTP; serve with the DFlash2 drafter at /models/Qwen3.8-Flash-Next-DFlash2 (vLLM supports `DFlash2DraftModel` natively;
  Edge0ForCausalLM must expose completed outputs of layers 5,15,25,35,45 averaged over the 4 HC streams). Measure acceptance and
  A/B vs no-spec (and vs MTP) at conc 1/8/32; log it in a section "DFlash2 drafter (user request)".
- Dense weights follow Strata (Q8_0, already kept in 4.1); no lossy NVFP4 re-quantization of dense weights.

## 9. Nsight Systems profile of the tuned preset (agent `nsys`, 2026-11-20)

Full write-up: **docs/PROFILE.md**. Tools: `EDGE0_NVTX=1|2` (plugin `nvtx.py`, off by default), `serve.sh NSYS=...`,
`bench/nsys-capture.sh`, `tools/nsys_breakdown.py`, `tools/nsys_tables.py`, `bench/dram-contention.sh`,
`tools/cpu_experts/`. Reports are in `5090:~/work/edge0-vllm/profiles/` and JSON summaries in `results/profile/`.
Nothing in the serving defaults changed, so there is no A/B.

Steady decode, tuned preset. Step period comes from the graph-level trace, the split from the node-level trace:

| | c1 | c8 | c32 |
|---|---:|---:|---:|
| step period | 16.4 ms | 67.0 ms | 165.2 ms |
| cold-expert Marlin (PCIe at its 54 GB/s ceiling, 36 us per cold expert per layer) | 24% | 74% | 86% |
| dense weight streaming (attention proj, router, D, U, shared, HC, lm_head) | 52% (D+U 18%) | 14% | 6% |
| GPU idle: host PLE `preadv` after a full stream sync, graph launch 0.2 ms | 5.0% | 4.3% | 5.1% |
| PCIe RX, mean over the step (GPU metrics) | 13% | 64% | 77% |

Findings for the next agents:
- **In a CUDA-graph replay the cold branch runs on the launch stream and the hot branch on the other stream.** The
  cold Marlin grid (510 CTAs, 3 per SM) holds every SM while it waits on PCIe, so the hot GEMMs run starved but
  hidden inside it: only 0.2-2 ms per step stays exposed. The 4.7 decisions (keep the stream split, no CTA cap)
  still stand; a cold CTA cap could win at most 1.2% at c32.
- **The PLE stall is the cheapest lever.** `stage_checkpoint_rows` syncs the stream and then reads the rows with
  serial `preadv`; most rows miss the page cache, 180-360 us each. That costs 0.6 / 2.5-4.9 / 8 ms per step.
  `EDGE0_PLE_IO=threads` (implemented, never measured) should recover most of it: an A/B is worth doing.
- **DFlash2 k=3, at the 11 GiB / 16-sequence calibration config.** The drafter is 1.7 / 2.1 ms per step (5% / 1.3%
  at c1 / c8). The verify step is cold-expert bound: 4 tokens per request cost 1.65x a plain step at c1 and 2.25x at
  c8, for 2.65 tokens per request. Each extra draft position costs about 3.3-3.9 ms at c1, almost all of it cold
  experts, so k=7 or k=15 are predicted to lose against k=3 on this box (PROFILE.md 4).
- **CPU co-compute of cold experts.** DDR5 is shared: the CPU reads about 62 GB/s alone and the GPU reads about
  50-54 GB/s over PCIe alone, but the two together reach only about 73 GB/s. Under contention Strata's NVFP4
  AVX-512 kernel takes 46.7 us per expert and the GPU's PCIe rate drops to 31 GB/s. So a balanced split can cut cold
  time by at most 28%, an upper bound of +23% at c8 and +30% at c32 and +13% for a k=3 verify at c1, against
  0-2% for plain c1. It is recommended for c >= 8 and for verify steps, after a gating prototype that measures the
  graph-capturable hand-off (target <= 30 us per layer). It needs a CPU kernel for the Marlin record layout, because
  there is no RAM for a second copy of the cold pool (PROFILE.md 5).
- `tests/test_registration_quant.py::test_entry_point_and_register` fails against the current plugin source. It
  expects `MODELS_CONFIG_MAP[ARCH]` to be the Qwen4Exp class, but `__init__.py` now registers the DFlash wrapper
  `Edge0ForCausalLMConfig`. This was not caused by the profiling changes, and the test was not changed here.

## CPU cold experts (user request, agent `cpu-experts`, 2026-11-20)

User: "if needed, also bring Strata's parallel CPU computation of cold experts into vLLM". PROFILE.md 5 recommended it
for c >= 8 and verify steps, behind a gating prototype. Built as `EDGE0_CPU_EXPERTS=1` (default off); design and
mechanism in **docs/CPU-EXPERTS.md**. Code: `plugin/edge0_vllm/cpu_experts.py`, `plugin/edge0_vllm/csrc/`
(`cpu_experts.cpp` AVX-512 kernel + pinned busy-polling pool, `cpu_experts_cuda.cu` dispatch / gather + doorbell /
wait kernels, `mailbox.h`), a third source in `kernels.combine_routes`, a few lines in
`latent_moe.latent_experts_marlin` and `model.py`. Tests: `tests/test_cpu_experts_host.py` (CPU only, numpy port of
vLLM's Marlin repack), `tests/test_cpu_experts.py` (GPU). Bench: `tools/cpu_experts/bench_cpu_split.py` (+
`split_table.py`), `bench/cpu-experts-gate.sh`, `bench/chain-cpuexp.sh`. Results and logs: `results/cpu-experts/`.

How it works: in every decode-sized MoE call (2..256 tokens), a 1-CTA dispatch kernel picks `round(0.55 x C)` of the
call's C cold experts (logical-id order among those with <= 8 routes; deterministic), removes them from the cold
Marlin call's expert map and writes the job into pinned mapped memory; the gather kernel also copies those routes'
activations there, and its last CTA rings a doorbell (sequence number read from VRAM, so CUDA-graph replays ring new
values). 8 pinned CPU workers compute gate/up -> SiLU*up -> down straight from the Marlin bytes of the pinned cold
pool (no second copy) while the GPU reads its share over PCIe; a 1-thread wait kernel spins on the `done` flag (2 s
timeout, never hit), and the combine kernel reads the CPU's rows from mapped memory.

**Numerics.** The kernel reproduces Marlin's epilogue roundings (bf16 at c1, `silu_and_mul`, `bf16(bf16(acc) *
bf16(w g2))`). On real layer-0 experts it is **bit-identical** to the GPU Marlin output for single routes (0 ulp,
4/4 experts) and within 1-2 bf16 ulp on 0.03% of the elements with 5 routes (fp32 summation order). The whole op in a
captured CUDA graph with the CPU split vs GPU only: rel 1e-5..2e-4, 99.8-100% of z elements equal (T = 4/16/32).

**Gate (synthetic calls, `bench_cpu_split.py`: real experts, op body in a CUDA graph, a new random cold set per
replay, 8 threads; us per MoE call, median of 40):**

| config | T=1 C=3 | T=4 C=11 | T=8 C=28 | T=16 C=50 | T=32 C=86 |
|---|---:|---:|---:|---:|---:|
| GPU only | 148 | 431 | 1045 | 1886 | 3241 |
| CPU path, frac 0 (hand-off cost only) | 158 (+7%) | 443 (+3%) | 1057 (+1%) | 1902 (+1%) | 3252 (+0%) |
| frac 0.3 | 144 (-2%) | 394 (-8%) | 925 (-11%) | 1646 (-13%) | 2840 (-12%) |
| frac 0.4 | 145 (-2%) | 378 (-12%) | 878 (-16%) | 1575 (-17%) | 2713 (-16%) |
| frac 0.5 | 120 (-19%) | 337 (-22%) | 825 (-21%) | 1471 (-22%) | 2599 (-20%) |
| frac 0.6 | 119 (-19%) | 323 (-25%) | 789 (-25%) | 1398 (-26%) | 2463 (-24%) |

- Hand-off (dispatch + doorbell + wait, no CPU share): **+11 us per call**, under the 30 us target.
- Per expert under contention: CPU 45-47 us (the same as Strata's own kernel under contention, 46.7 us); GPU 45 us
  at frac 0.3 rising to 68 us at 0.6 (its PCIe rate drops as the CPU streams more). At frac 0.6 the pair moves
  86 x 1.94 MB in 2.46 ms = **68 GB/s** (GPU alone 51 GB/s): -24..-26% per call, the DRAM ceiling the profile
  predicted (-26..-28%). Balance point 0.55-0.6 (at 0.5 the GPU side is longer, at 0.6 the CPU side by 70-130 us).
- 8, 12 and 14 threads give the same times (+-1%), and so does a coarser task size (2 gate/up pairs and 4 w2
  n-tiles per task): at 8 threads the CPU side is DRAM-bound. Default: 8 threads on physical cores 2-9.
- Compute is not the limit: 77-89 us per expert-route on one core (weights in cache), ~10 us per expert on 8 cores
  against ~45 us of DRAM time.

**Correctness in the server** (`bench/probe-config.sh cpuexp-all`, the CPU path forced on every call size incl. T=1
decode and short prefills, 8 threads, frac 0.55): **8/8 prompts pass, teacher-forced top-1 = Strata argmax
256/261** (tuned 257), mean |dlogprob| 0.0324 (tuned 0.0328), KL(top-5) 0.0063; no wait timeouts; the GPU waited
6-9 us per call on average.

**A/B `cpuexp` (ABBA, 512 output tokens, `bench/ab.sh`):** A = tuned preset (`EDGE0_DEV=1`), B = A +
`EDGE0_CPU_EXPERTS=1 EDGE0_CPU_THREADS=8 EDGE0_CPU_FRAC=0.55`.

| conc | A1 | B2 | B3 | A4 | B vs A | TPOT A -> B (ms) | median TTFT A -> B (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 52.8 | 51.3 | 55.0 | 52.3 | +1% (noise) | 18.6 -> 18.4 | 200 -> 172 |
| 8 | 107.1 | **136.1** | **138.7** | 108.1 | **+27.7%** | 71.1 -> 55.4 | 1,776 -> 1,461 |
| 32 | 138.5 | **169.7** | **172.8** | 136.6 | **+24.5%** | 225 -> 180 | 4,113 -> 3,824 |

Live split (B server log): c8 16 CPU experts / 20 routes per call, CPU 812 us vs GPU side 883 us per call; c32 58-65
CPU experts / 72-82 routes per call, CPU 3.0-3.3 ms vs GPU side 3.2-3.5 ms; the GPU waited 7-9 us per call. c1 decode
(T=1) keeps the GPU-only path (`EDGE0_CPU_MIN_TOKENS=2`), so c1 is unchanged by design; its prefills get faster.

**KEPT** (> +3% at c8 and c32, c1 neutral, correctness gate passed). Best setting:
`EDGE0_CPU_EXPERTS=1 EDGE0_CPU_THREADS=8 EDGE0_CPU_FRAC=0.55` (8 and 0.55 are now the knobs' defaults). It is not
in `PRESET=tuned` yet because the spec-len agent's sweep was running on that preset; the coordinator can add
`: "${EDGE0_CPU_EXPERTS:=1}"` to the tuned case of serve.sh (plus `export EDGE0_CPU_EXPERTS`) once nothing else
depends on the preset staying fixed. Speculative decoding with CPU experts and the in-server split tuning: below.

### With DFlash2 k=3 (`bench/chain-cpuexp.sh spec`)

A verify step is a batch of 4 tokens per request, so even at c1 every MoE call has ~11 cold experts and takes the
CPU path (T = 4 >= `EDGE0_CPU_MIN_TOKENS`).

**Speculative gate** (`correctness/greedy_same.py`, 8 strata-ref prompts, 256 greedy tokens; files
`results/greedy-cpuexp-greedy-{1,2}.json`): with CPU experts, DFlash2 on vs off gives **6/8 identical**, the other
2 diverge at exact near-ties (margin 0.0 / 0.125 nat, both passes on their own argmax), 0 unexplained. The same
comparison without CPU experts (the spec-len agent's `greedy-lendf-1` vs `-2`) gives 5/8 identical and 3 near-ties,
so the CPU path does not add divergences (it is bit-identical to Marlin for single routes). Acceptance is unchanged:
2.652 tokens per draft, per position 0.769 / 0.540 / 0.343 (DFlash2 without CPU experts: 2.65, 0.76 / 0.55 / 0.35).

**A/B `cpuexp-df3` (ABBA, 512 output tokens):** A = tuned + `SPEC_METHOD=dflash2 SPEC_K=3`, B = A +
`EDGE0_CPU_EXPERTS=1 EDGE0_CPU_THREADS=8 EDGE0_CPU_FRAC=0.55`.

| conc | A1 | B2 | B3 | A4 | B vs A | TPOT A -> B (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 76.5 | **85.0** | **91.0** | 73.8 | **+17.1%** | 12.9 -> 11.0 |
| 8 | 111.8 | **131.1** | **139.1** | 111.3 | **+21.1%** | 46.5 -> 38.8 |
| 32 | 112.5 | **131.8** | **133.2** | 113.3 | **+17.4%** | 51.0 -> 43.7 |

(c32 with speculation is KV-bound, median TTFT ~96-111 s for both: see the spec-len section.) Live split at c32:
31-35 CPU experts / 55-63 routes per call, CPU 1.6-1.9 ms vs GPU side 1.7-1.9 ms, the GPU waited 9-10 us per call.
The profile's upper bound for a k=3 verify at c1 was +13%; the measured +17% is above it because the bound assumed
a 25 us hand-off per layer, and the hand-off measured 11 us.

Best per concurrency with CPU experts on (512-token A/B passes): c1 **88.0 tok/s with DFlash2 k=3** (tuned no-spec
52.5, Strata solo 121 with MTP), c8 137.4 without speculation / 135.1 with DFlash2 k=3, c32 **171.3 without
speculation**.

### In-server tuning of the split and the threads (`bench/chain-cpuexp.sh tune`, single passes, 512 tokens)

| config (all `EDGE0_CPU_EXPERTS=1`, no speculation) | c1 | c8 | c32 | notes |
|---|---:|---:|---:|---|
| 8 threads, frac 0.55 (the A/B's B passes, for reference) | 51.3 / 55.0 | 136.1 / 138.7 | 169.7 / 172.8 | CPU 0.81 ms vs GPU side 0.88 ms per call at c8 |
| 8 threads, frac 0.6 | - | 130.3 | 169.6 | |
| 8 threads, frac auto | - | 131.3 | 171.9 | settles at frac 0.575 (CPU = GPU-side time); the GPU then waits ~32 us per call |
| 6 threads, frac 0.55 | - | **122.9** | 169.4 | too few: c8 drops ~8% |
| 8 threads, frac 0.55, also single-token decode (`EDGE0_CPU_MIN_TOKENS=1`) | 54.6 | - | - | within the c1 noise (A passes 52.3-52.8, B 51.3-55.0) |

Decisions: 8 threads (6 lose at c8; 12-14 gave nothing in the gate); frac 0.55 fixed (0.6 and auto are within the
pass-to-pass noise of 0.55, and a fixed share keeps runs reproducible; `auto` stays available). The auto controller
equalises the two durations it can measure, which leaves the GPU waiting for the doorbell / done-flag latency (~30 us
per call); a target of CPU time = GPU time - 30 us would be the refinement if auto is ever made the default.
`EDGE0_CPU_MIN_TOKENS` stays 2: plain c1 decode has ~2 cold experts per layer, too few to split, and an ABBA would be
needed to show the +11 us per call hand-off is paid back.

Open items: (1) the spec-len agent's `sched` block re-runs the draft-length question (k=3/5/7) with CPU experts on,
since a cheaper verify can move the optimum; (2) `edge0-vllm:dev` was not rebuilt here because the spec-len sweep
was using it (all runs above used `EDGE0_DEV=1`); the Dockerfile's prebuild step was verified with a test tag (the
image then imports the prebuilt extension in 1.1 s): `cd ~/work/edge0-vllm && nice -n 19 docker build -f
docker/Dockerfile -t edge0-vllm:dev .`; (3) adding it to `PRESET=tuned` (see above).

## Draft length sweep (user request, agent `spec-len`, 2026-11-20)

User: "draft length can be opened to 7 or 15, try it". Scope: fix speculation at concurrency 32 (S1), sweep DFlash2
and MTP draft lengths up to 7 / 15 on the tuned preset (S2), test vLLM's batch-size schedule
`num_speculative_tokens_per_batch_size` (TidalMoE adaptive verification, S3), A/B the best setting against no
speculation (S4), greedy-identity gate (S5). All runs: ShareGPT seed 0, chat API, thinking off, 512 output tokens,
`--ignore-eos`, prompts = 4 (c1) / 16 (c8) / 32 (c32, sweep) or 64 (c32, A/B); `EDGE0_DEV=1` (shared plugin source;
`EDGE0_CPU_EXPERTS` off unless stated).

New tools: `bench/run-configs.sh` now logs `GPUMEM` (peak `nvidia-smi` memory.used per bench window; the caching
allocator keeps its high-water mark, so this is the real step peak) and takes `GREEDY=1` / `STOP_AFTER=1`;
`bench/specstat.py` also counts scheduler preemptions; `bench/sweep_table.py` (markdown table of a sweep log: tok/s,
TPOT, acceptance length, per-position acceptance, step ms = TPOT x acceptance length, KV GiB, peak MiB, max running);
`correctness/greedy_same.py` (the with/without-drafter greedy gate, S5); `bench/chain-speclen.sh` (the job chains).
Logs: `5090:~/work/edge0-vllm/logs/runcfg-{memfix,lendf,lendf2,lenmtp,lenmtp2,lenmtp3,lensch,lenschcpu}.log`,
`ab-speclen.log`; greedy outputs `results/greedy-<label>.json`. Copies in the repo: `results/speclen/` (full sweep
table, A/B summary, compare_strata and greedy-gate outputs).

### S1. The c32 speculation "crash" was a CUDA OOM; fixed in serve.sh

`runcfg-k1s32.log` / `serve-20261119-233402.log` (MTP k=1, 8 GiB hot experts, 32 sequences, GPU_UTIL 0.96): the
engine died at the start of the c32 run with `torch.OutOfMemoryError` (48 MiB requested, 48 MiB free, 31.22 GiB in
use) inside FlashInfer's GDN chunked prefill (`chunk_gated_delta_rule` allocating its output state), in a 4096-token
step that mixed 16 new prompts with spec-decode rows (23 running, 9 waiting). It is unrelated to `EDGE0_SKIP_PAD`
(the "skip-pad" run, `runcfg-knobs.log`, was a no-spec check stopped mid-c32; skip-pad is on by default and only
reroutes CUDA-graph padding rows to hot experts).

Root cause: vLLM's memory profile (V2 runner `profile_run`) runs the dummy forward with `skip_attn=True`, so the GDN
layers' `_forward_core` returns early (stated in `qwen_gdn_linear_attn.py`) and neither the GDN / QSA prefill working
set of a full `max_num_batched_tokens` chunk nor the drafter's attention prefill is counted when the KV cache is
sized. The only slack is (1 - GPU_UTIL) x 31.36 GiB. Measured (`runcfg-memfix.log`, c32 burst, 256-token outputs;
budget = GPU_UTIL x 32,112 MiB, the device size torch sees; the OOM ceiling is ~32,100 MiB of `nvidia-smi` used):

| config (32 sequences) | budget MiB | idle MiB | c32 peak MiB | peak - budget | KV GiB | running at c32 | c32 tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| tuned, no drafter (4096 tokens, 0.96) | 30,828 | 30,955 | 31,997 | +1,169 | 4.61 | 32 | 140.7 |
| MTP k=1, 8 GiB hot, 4096 tokens, 0.92 | 29,543 | 29,879 | 31,389 | **+1,846** | 5.83 | 27 | 117.7 |
| MTP k=1, 8 GiB hot, **2048** tokens, 0.92 | 29,543 | 30,059 | 30,789 | **+1,246** | **6.75** | 31 | 125.4 |
| DFlash2 k=3, 8 GiB hot, 4096 tokens, 0.92 | 29,543 | 29,589 | 30,495 | +952 | 5.78 | 15 | 104.0 |

At GPU_UTIL 0.96 the crashed MTP config needs 30,828 + 1,846 = 32,674 MiB > 32,112: the observed OOM. **Fix
(serve.sh): whenever speculative decoding is on, `MAX_BATCHED` defaults to 2048 (`SPEC_MAX_BATCHED`) and
`GPU_UTIL` to 0.94 (`SPEC_GPU_UTIL`); explicit values still win.** 2048-token chunks halve both the profiled
transient peak (+0.92 GiB of KV at equal GPU_UTIL) and the unprofiled overshoot (1,846 -> 1,246 MiB); 0.94 leaves
~0.8-1.2 GiB of margin. Verified: every speculative server below ran c8 and c32 (MTP k=3 and DFlash2 k=3/5 at c32;
MTP k=1 with 32 sequences ran 31 requests at once), peaks 30,689-31,269 MiB, except MTP k=7 with
`draft_confidence_threshold` (32,037 MiB at c8: that path captures many more CUDA graphs, 1.56 GiB instead of 0.63).
Cost: prompts longer than 2048 tokens prefill in two chunks (more TTFT at bursts); decode is unaffected.

**Open issue for the default preset:** the tuned preset without a drafter also peaks at 31,943-31,997 MiB at c32
(`lendf-1`, `memfix-1`), ~100-150 MiB below the OOM ceiling. Longer prompts at c32 could OOM it the same way;
`MAX_BATCHED=2048` (or GPU_UTIL 0.955) on the no-spec preset needs its own A/B (prefill TTFT).

### S2. Draft length sweep on the tuned preset (12.8 GiB plan, 32 sequences, S1 defaults)

`runcfg-lendf*.log`, `runcfg-lenmtp*.log`; table from `bench/sweep_table.py`. "fits" = the most requests the KV cache
held at once (`Running:` in the server log); "step" = TPOT x acceptance length = one draft + verify step at c1.
Long drafts do not fit the tuned VRAM split with `max_model_len` 32768 (one 32k request needs 1.31 / 1.47 / 1.66 /
2.45 GiB of KV at k=3 / 5 / 7 / 15 and the pool is 2.47 / 1.88 / 1.57 / 1.2 GiB), so k=7 and k=15 run with
`MAX_MODEL_LEN=8192`, k=15 with a 12.1 (DFlash2) / 11.0 GiB (MTP) plan; they ran c1 and c8 only (1-2 requests fit,
so c32 = c8).

| drafter, k | KV GiB | fits | c1 tok/s | c1 acc. len | c1 per-position acceptance | c1 step ms | c8 tok/s | c32 tok/s |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| none (tuned) | 4.61 | 32 | 50.7 | 1 | | 19.3 | 105.9 | **143.9** |
| DFlash2 3 | 2.47 | 6 | 70.3 | 2.69 | .77 .55 .38 | 37.1 | **109.6** | 116.0 |
| DFlash2 5 | 1.88 | 3 | **71.6** | 3.24 | .78 .58 .41 .28 .19 | 43.8 | 79.3 | 81.0 |
| DFlash2 7 | 1.57 | 2 | 63.9 | 3.32 | .77 .53 .38 .26 .18 .12 .07 | 50.3 | 73.6 | - |
| DFlash2 15 (12.1 GiB) | 1.92 | 1 | 42.4 | 3.27 | .76 .52 .33 .24 .16 .11 .07 .04 .03 .01 .01 .00 .00 .00 .00 | 75.6 | 48.0 | - |
| MTP 3 | 2.15 | 5 | 71.1 | 2.64 | .74 .53 .36 | 35.7 | 104.1 | 111.1 |
| MTP 5 | 1.56 | 2 | 63.6 | 3.04 | .72 .52 .36 .26 .19 | 46.4 | 81.3 | 83.5 |
| MTP 7 | 1.37 | 1 | 54.5 | 3.29 | .73 .53 .38 .26 .19 .12 .08 | 58.8 | 60.7 | - |
| MTP 7, confidence 0.5 | 1.22 | 1 | 63.0 | 3.01 | .74 .45 .29 .21 .14 .10 .07 | 46.1 | 64.0 | - |
| MTP 15 (11.0 GiB) | 3.07 | 1 | 34.2 | 3.25 | .72 .50 .34 .24 .16 .11 .06 .04 .03 .02 .01 .01 .00 .00 .00 | 93.0 | 36.4 | - |

(Per-position acceptance is unconditional: the share of drafts whose first i tokens were all accepted. The c8 and
c32 numbers of the k=3/5 rows are KV-limited at 6 / 3 requests at a time; c8 of DFlash2 k=7 also had 10
preemptions. Acceptance at c8 / c32 is within +-0.1 of c1.)

What the sweep shows:

1. **At c1 the optimum is k=4-5 for DFlash2 and k=3 for MTP; 7 loses 9-23% and 15 loses 40-52% against them.** Each
   extra DFlash2 position costs 3.2-3.4 ms of verify (37.1 -> 43.8 -> 50.3 ms; k=15 75.6 ms), almost all of it the
   cold experts the extra token touches; each extra MTP position costs 4.3-6.2 ms (the MTP layer runs once per
   position, plus the full 248k-row lm_head). A position pays only while its acceptance exceeds that cost divided by
   the time per token (3.3 / 13.8 ms = 0.24 for DFlash2, ~0.4 for MTP): DFlash2 a4 = 0.28 pays a little, a5 = 0.19,
   a6 = 0.12, a7 = 0.07 do not. The profile predicted the direction (PROFILE.md 4: ~74 / ~60 tok/s for 7 / 15
   against ~79 for 3); the measured loss is larger.
2. **DFlash2 k=15 is one 16-wide block, not two blocks.** vLLM's dflash2 path has no multi-block chaining (a second
   block would need the target's hidden states for the first block's drafts); `num_speculative_tokens` only sets the
   query width: the drafter is built with `block_size = 1 + num_speculative_tokens` (`qwen3_dflash2.py`, grouped
   conv) and nothing checks it against the trained `dflash_config.block_size` 8. Positions 8-15 are untrained and
   accept 0.04 -> 0.00.
3. **The confidence threshold helps only long MTP chains** (k=7: 54.5 -> 63.0 tok/s by skipping MTP passes), still
   below k=3. It does not shorten the verify (vLLM verifies a uniform length per batch), lowers the acceptance (a2
   0.53 -> 0.45) and multiplies the captured CUDA graphs. MTP k=15 + confidence OOMed in graph capture at 11 GiB /
   0.94 (retried in S3's block).
4. **Draft length costs KV memory at every concurrency.** vLLM 0.33 keeps one GDN recurrent state (36 layers x 48
   heads x 128 x 128 bf16 = 54 MiB, page-padded) per draft position per request so that a rejection can roll back
   (`MambaSpec.num_speculative_blocks = num_speculative_tokens`; ReplaySSM, which avoids this, is Mamba2-only). Per
   request: 118 MiB without a drafter, 387 MiB (DFlash2 k=3), 396 MiB (MTP k=3), 572 MiB (DFlash2 k=5): **+90 MiB
   per draft position**. The CUDA-graph reserve also grows with k (DFlash2 estimate 1.04 / 1.25 / 1.56 GiB at
   k=3/5/7). At the tuned 12.8 GiB of hot experts this leaves room for 6 / 3 / 2 / 1 requests (k=3/5/7/15); 32
   requests at k=3 would need ~12 GiB of KV, i.e. almost no hot experts. So at c32 every speculative config loses
   to no speculation (best 116.0 vs 143.9), and at c8 only DFlash2 k=3 is ahead (+3.5%, 6 running).

### S3. Batch-size schedule, and draft length with CPU cold experts (`runcfg-lensch.log`, `runcfg-lenschcpu.log`)

vLLM's `num_speculative_tokens_per_batch_size` (V2 runner; `SPEC_SCHED`) picks k from the number of requests in the
step; k=0 steps still feed the DFlash2 drafter's context K/V, so it stays in sync. The GDN state is still sized by the
maximum k, so a schedule saves verify work, not memory. CPU cold experts (`EDGE0_CPU_EXPERTS=1`, kept in "CPU cold
experts" above) make each verify cheaper, so they could move the draft-length optimum; tested at c1.

| config (DFlash2, 32 sequences) | hot GiB | fits | c1 tok/s | c1 step ms | c8 tok/s | c32 tok/s | c1 per-position |
|---|---:|---:|---:|---:|---:|---:|---|
| no drafter (S2 reference, `lendf-1`) | 12.8 | 32 | 50.7 | 19.3 | 105.9 | 143.9 | |
| k=3 (S2, `lendf-2`) | 12.8 | 6 | 70.3 | 37.1 | 109.6 | 116.0 | .77 .55 .38 |
| k<=3, schedule `[[1,2,3],[3,32,2]]` | 12.8 | 6 | 70.6 | 36.6 | 103.6 | 104.2 | .76 .55 .36 |
| k<=3, `[[1,2,3],[3,32,2]]` + CPU experts | 12.8 | 6 | 79.5 | 32.3 | 116.5 | 121.5 | .75 .54 .37 |
| k<=3, `[[1,2,3],[3,8,2],[9,32,0]]` + CPU experts | 11.0 | 11 | 75.3 | 34.8 | 132.8 | 119.3 | .77 .56 .39 |
| k=3 + CPU experts (c1 only) | 12.8 | | **81.0** | 31.9 | | | .77 .55 .36 |
| k=5 + CPU experts (c1 only) | 12.8 | | 75.6 | 38.6 | | | .74 .54 .35 .24 .15 |
| k=7 + CPU experts (c1 only, 8192 max len) | 12.8 | | 74.7 | 43.0 | | | .76 .54 .38 .26 .18 .12 .07 |

For reference, the cpu-experts agent's interleaved A/Bs on the same preset (64 prompts at c32): no drafter + CPU
experts 51.3-55.0 / 136.1-138.7 / 169.7-172.8 tok/s at c1 / c8 / c32; DFlash2 k=3 + CPU experts 85.0-91.0 /
131.1-139.1 / 131.8-133.2 (`ab-cpuexp.log`, `ab-cpuexp-df3.log`).

1. **Shorter drafts at mid batch sizes do not pay here.** With k=2 for 3+ requests, c8 drops from 109.6 to 103.6
   tok/s: acceptance 2.65 -> 2.39 (-10%) for a verify only 5% cheaper (129 -> 122 ms). In a batch the draft tokens
   share many cold experts with the other requests' tokens, so one more position costs little. TidalMoE saw the same
   on GB300 (docs/ADAPTIVE-VERIFICATION.md): its static schedule `[[1,4,5],[5,8,3],[9,16,2],[17,32,1]]` lost to
   keeping 5 drafts at every batch size and pruning per request by drafter confidence against a measured cost curve.
   vLLM 0.33 ships that mechanism (`enable_adaptive_verification`, global top-k draft admission from drafter
   confidence against a cost curve profiled at start-up), but it refuses this model: "Adaptive verification trims
   verification requests on device, which the GDNAttentionBackend attention backend does not support"
   (`runcfg-lenav.log`, DFlash2 k=3 and k=5). Per-request verify lengths would need GDN's spec-decode kernels to take
   variable query lengths per request.
2. **k=0 above 8 requests does not rescue c32.** It lets more requests run only if the hot-expert budget gives up
   VRAM for their GDN state (11.0 GiB: 11 fit) and c32 still reaches only 119.3 tok/s, against 143.9 (no drafter)
   and ~170 (no drafter + CPU experts). Under the KV wall of S2.4, the number of requests that fit decides c32, and
   no schedule changes it.
3. **With CPU cold experts the c1 optimum stays at k=3** (81.0 vs 75.6 / 74.7 tok/s for k=5 / 7): CPU experts cut
   the k=3 step from 37.1 to 31.9 ms (+15% c1), but an extra position still costs 2.2-3.4 ms, more than its 0.12-0.24
   acceptance is worth.
4. MTP k=15 + confidence 0.5 could not be started: OOM in CUDA-graph capture at 11 GiB / 0.94 (394 FULL graphs), and
   at 9 GiB / 0.90 vLLM failed in the MTP drafter's piecewise prefill capture with "capturing stream has unjoined
   work" (`target_dependent_ar/speculator.py` `_prefill`; `serve-20261120-054133.log`). Not pursued: MTP k=15 runs at
   34 tok/s without the threshold and k=7 with it at 63, both below k=3.

**Best speculative setting: DFlash2 k=3, no schedule** (`SPEC_METHOD=dflash2 SPEC_K=3`, with S1's
`MAX_BATCHED=2048 GPU_UTIL=0.94`), combined with CPU cold experts. A/B in S4.

### S4. A/B: DFlash2 k=3 on vs off, on top of CPU cold experts (`ab-speclen.log`, ABBA, 06:38-07:11)

Both arms carry the kept CPU cold experts (the best setting without a drafter now), so this isolates the drafter on
the current best stack. A = tuned preset + `EDGE0_CPU_EXPERTS=1 EDGE0_CPU_THREADS=8 EDGE0_CPU_FRAC=0.55` (KV 4.61
GiB, 32 requests fit); B = A + `SPEC_METHOD=dflash2 SPEC_K=3` (so `MAX_BATCHED=2048 GPU_UTIL=0.94`; KV 2.46 GiB,
6 requests fit). 512 output tokens, 4 / 16 / 64 prompts.

| conc | A1 | B2 | B3 | A4 | B vs A | TPOT A -> B (ms) | median TTFT A -> B | P99 TTFT A -> B |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 50.9 | 82.5 | 89.1 | 52.9 | **+65.3%** | 18.9 -> 11.3 | 173 -> 184 ms | 486 -> 422 ms |
| 8 | 131.2 | 132.2 | 144.3 | 130.2 | +5.8% (B passes 9% apart) | 57.9 -> 37.3 | 1.75 -> 1.95 s | **2.1 -> 18-21 s** |
| 32 | 169.9 | 133.1 | 141.5 | 170.9 | **-19.4%** | 180 -> 42 | **4.1 -> 89-96 s** | 5.1 -> 100-106 s |

At c8 two of the eight requests wait for a whole request to finish (6 fit), so B's c8 tail latency is 9x worse even
though its throughput is a little higher; at c32 26 requests queue. Single passes without CPU experts agree (S2:
no drafter 50.7 / 105.9 / 143.9, DFlash2 k=3 70.3 / 109.6 / 116.0 at c1 / c8 / c32).

### S5. Correctness gate

- `compare_strata.py` on DFlash2 k=3 + CPU experts (`lensch-2`; at c1 the schedule always picks k=3): **8/8 pass,
  teacher-forced top-1 = Strata argmax 257/261**, mean |dlogprob| 0.0323, KL(top-5) 0.0064 (tuned: 257/261, 0.0328,
  0.0065). Its 48-token greedy texts are identical to the no-drafter tuned server's on 8/8 prompts. Without CPU
  experts (`lensch-1`) 7/8: the French prompt's first token is an exact tie on the spec server (margin 0.000, "Le"
  vs "Il"), the flip 4.2 documented for the bf16 state.
- `greedy_same.py` (the 8 strata-ref chats, 256 greedy tokens, `ignore_eos`): B vs A of S4 (`greedy-lenschcpu-1` vs
  `greedy-speclen-A`): **5/8 identical**; the 3 others diverge after 20 / 21 / 69 tokens, each at a bf16 near-tie (top-1
  minus top-2 = 0.000-0.125 nat in one of the two runs), and **in every run, every emitted token is that pass's own
  argmax** (0 non-argmax positions in 36,864 generated tokens: the 18 greedy runs that recorded it, 15 of them with a
  drafter). Every other drafter and length in S2/S3 gives the same picture (4-6/8 identical, 0 unexplained
  divergences).
- **Strict identity is not reachable even without a drafter.** Control on A (no drafter): the same 8 prompts sent
  all at once (batched decode) vs one at a time: 6/8 identical, 2 near-tie flips (`greedy-speclen-control.txt`); CPU
  experts on vs off (both no drafter): 4/8. The verify pass runs k+1 rows per request, a different GEMM shape than a
  1-row decode, so bf16 rounding moves the logits by a few 0.125-nat steps, enough to flip these ties. So the gate as
  stated ("identical greedy output") fails for 3/8 prompts for a numerical reason (batch shape), not an acceptance
  error; the lossless property (greedy = the target pass's argmax at every position) holds.

### S6. Decision and open issues

**Draft length 7 or 15: rejected.** On this box a verify position costs 3.2-3.4 ms (DFlash2) or 4.3-6.2 ms (MTP),
mostly cold experts, and positions 5+ accept <= 0.19; k=7 is 9-23% slower than k=3 at c1 and k=15 40-52% slower,
with or without CPU experts. Each position also costs ~90 MiB of GDN state per request, so k=7 / 15 fit only 2 / 1
requests next to the tuned 12.8 GiB of hot experts and cannot start with `max_model_len` 32768.

**Speculation (DFlash2 k=3): not made the default** (c32 -19%: the gate needs > +3% without a regression at the
other concurrencies). It is the best low-concurrency setting: +65% at c1 and +6% at c8 on top of CPU experts
(single-stream 82-89 tok/s, Strata solo 121). For a deployment that runs <= ~6 requests at a time:

    EDGE0_CPU_EXPERTS=1 EDGE0_CPU_THREADS=8 EDGE0_CPU_FRAC=0.55 SPEC_METHOD=dflash2 SPEC_K=3 ./serve.sh
    # serve.sh then sets MAX_BATCHED=2048 GPU_UTIL=0.94 (S1); PRESET=tuned otherwise

Kept: the S1 OOM fix in serve.sh (spec defaults), the measurement tools. Rejected: k=5/7/15 (DFlash2, MTP), MTP
draft-confidence threshold, the batch-size schedules of S3.

Open issues:
1. **GDN spec state is the c32 blocker.** vLLM 0.33 stores one recurrent state per draft position per request
   (`num_speculative_blocks = num_speculative_tokens`), sized by the maximum k even when a schedule picks k=0. A
   GDN variant of ReplaySSM (keep the pre-window state plus the k+1 tokens' small k/v/g/beta inputs and replay the
   accepted ones: 36 layers x 16 KiB per token, ~2.5 MiB for k=3; or a per-step scratch sized by the maximum of B x
   k(B) over batch sizes) would cut k=3 from +270 MiB to a few MiB per request: 32 requests with k=3 would fit
   beside the tuned hot experts. Variable per-request verify lengths in the GDN spec kernels are also what vLLM's
   adaptive verification needs; it refuses GDN today (S3, item 1).
2. The no-drafter tuned preset peaks within ~100-150 MiB of the OOM ceiling at c32 (S1); A/B `MAX_BATCHED=2048`
   (or GPU_UTIL 0.955) there.
3. The CUDA-graph memory estimate is 0.6-0.7 GiB above the real capture at k=3-7 (1.04 vs 0.41 GiB at k=3), so a
   spec server reserves ~0.6 GiB of KV it never uses; an exact estimate would fit one or two more requests.
4. MTP k=15 + `draft_confidence_threshold` cannot start (graph-capture OOM, then a vLLM "capturing stream has unjoined
   work" error in the MTP prefill capture at 9 GiB / 0.90); low value, not pursued.

## Single-stream (goal: match Strata)

User goal: match Strata at concurrency 1 (121 tok/s, TPOT 8.3 ms), then beat it, without hurting c >= 8 badly
(single-stream-only settings go into a separate preset).

### G1. Where Strata wins at c1 (agent `gap`, 2026-11-20): measurement only, no serving default changed

Full write-up: **docs/SINGLE-STREAM-GAP.md**. Tools: `bench/gap-strata.sh` (Strata via `edge0-serve.sh`, or
`serve/server.py` directly with the same engine config and its engine log kept: `STRATA_DECODE_TIMING=1`,
`STRATA_VERIFY_PROFILE=1`, extra engine args), `bench/gap-vllm.sh` (warm + 512-token + EOS-terminated c1 passes with
acceptance and `EDGE0_TRACE=bw` per pass), `bench/gap-nsys.sh` (warm nsys capture), `tools/gap_parse.py`,
`tools/sim_tier.py` (replay of c1 routing through expert-tier policies), `tools/nsys_breakdown.py` now labels the
CPU-experts kernels (`cpu_wait`). Results: `results/gap/`.

Same 8 ShareGPT prompts, c1, thinking off (Strata ends at EOS, ~810 tokens):

| run | tok/s | TPOT ms | tokens / step | cold experts / layer / step | ms / token in the engine |
|---|---:|---:|---:|---:|---:|
| Strata default (`slog`) | 120.8 | 8.3 | 2.29 (window 3.10) | 1.92 (CPU) | 7.72 |
| Strata `--expert-cache 7101` (vLLM's 12.8 GiB) | 103.1 | 9.7 | 2.31 | 3.89 | 9.14 |
| Strata `--adapt-swaps 0` (static tier) | 71.1 | 14.0 | 2.32 | 9.16 | 13.5 |
| vLLM best (DFlash2 k=3 + CPU experts), 512 / EOS | 93.3 / 94.2 | 10.2 / 10.4 | 2.66 / 2.71 | 8.11 / 8.05 | 10.2 |
| vLLM + fast adaptive knobs (`EDGE0_ADAPT_EPOCH_S=0.25 _SWAPS=96 _HALFLIFE=2 _MARGIN=0 _Z=0.5 _MIN_STEPS=4`), 512 / EOS | 93.3 / **99.2** | 10.2 / **9.9** | 2.67 / 2.77 | **5.79 / 5.61** | - |

- The gap (+2.5 ms per token in the engine) is cold experts: vLLM loads 146 per token, Strata 40 (4.5 vs 1.7 ms per
  token). vLLM is ahead on everything else except host idle (PLE reads after a full sync, 0.6 ms per token).
- Strata's lead comes from its request-reactive expert tier (+1 per routed entry, x0.7 per round, <= 96 swaps every
  4 windows; 15.7 swaps per window) and 18.67 GiB = 10,358 expert slots (vLLM 7,101). The replay reproduces Strata's
  measured cold counts within 2% and predicts 9.3 -> 5.25 cold per layer-step for vLLM with that policy at 7,101 slots.
- The fast-knob probe is not a candidate default as is: each 0.25 s epoch costs 28-46 ms of host time (stream sync,
  numpy, 96 swaps on the main stream), ~12% of the wall time, which eats most of the -29% cold experts. It shows the
  lever (EOS pass +5.3% even so); the implementation must be sync-free (GPU-side selection, swaps on a side stream).
- Levers ranked (expected ms per token, from 10.2): reactive tier done on the GPU -1.0..-1.4; c1 preset with more
  expert VRAM (+2.5-3 GiB) -0.6..-0.9; PLE off the critical path -0.4..-0.6; draft length re-tune afterwards
  -0.2..-0.5; CPU-experts dispatch kernel -0.15.

### D1. The dense path at c1 (agent `dense-c1`, 2026-11-20): latent D/U group skipping, fused shared expert, Q8 LM head

Task: shrink the ~9.4 ms of dense weight streaming per step (PROFILE.md: D+U 2.8 ms, shared expert 0.8-0.9, HC 1.4-1.7,
attention projections 2.3, LM head 0.76 + the DFlash2 drafter's use of it 0.75) and the PLE stall. New code (all
opt-in, default off): `plugin/edge0_vllm/q8gemv.py` (Triton Q8_0 GEMV reading b12x's packed weight in place, with
group skipping and split-K epilogues), `plugin/edge0_vllm/lmhead_q8.py`, hooks in `latent_moe.py` (`forward_du`,
`shared_add`; ops `vllm::edge0_latent_moe_du`, `edge0_shared_add`, `edge0_q8_linear`) and `model.py`; tests
`plugin/tests/test_q8gemv.py` (CPU interpreter + GPU); tools `tools/group_usage.py`, `tools/nsys_kernels.py`,
`tools/q8gemv_bench.py`, `tools/q8gemv_sweep.py`, `tools/ple_io_bench.py`, `tools/ple_iops.c`; drivers
`bench/chain-dense.sh`, `bench/chain-dense-run1.sh`; `bench/ab.sh` gained `WARM_PROMPTS=N` (an unrecorded c1 warm
pass after every server start). Logs: `logs/chain-dense.log`, `ab-dense-{spec,nospec2,spec2}.log`, `dense-kbench2.out`,
`dense-sweep.out`; nsys `profiles/denseBw/`, kernel tables `results/nsys-kernels-{denseBw,gapdf3w}.txt`.

**Knobs (serve.sh header):** `EDGE0_DU_MASK=1` (latent D / U over the used routing groups only), `EDGE0_SHARED_FUSE=1`
(shared expert + sigmoid gate + residual add in 4 kernels), `EDGE0_LMHEAD=q8` (LM head re-quantized to Q8_0 at load,
served by b12x, and by the Triton GEMV for calls of <= 16 rows), `EDGE0_Q8_GEMV=1` (every small-M Q8 linear through the
Triton GEMV; neutral, not used). D/U/shared apply to calls of <= 8 tokens (`EDGE0_Q8_GEMV_MAX_M`); prefill and larger
batches keep b12x.

**1. How many latent groups a step uses** (`tools/group_usage.py`, the c1 seed-0 decode routing of section 3, 2,556
tokens): 4.38 of 8 groups per token (55% of the D/U bytes; layers 2.6-6.1), 5.48 per 2 tokens, 6.08 per 3, **6.46 per
4-token verify window (81%)**. D's output rows and U's input columns of unused groups are dead (the gather reads only
xl[t, group(e)] of routed experts; z is exactly 0 there), so skipping them is exact. It pays at plain c1, little at a
k=3 verify.

**2. Kernels** (`tools/q8gemv_bench.py`: CUDA-graph replays of 48 calls, 6 weight copies cycled so the weights are not
L2-resident, as in serving; us per call). The Triton GEMV computes each 32-input block's dot in fp32 with `tl.dot`
(int8 -> bf16 is exact) and scales it by the fp16 base, like b12x's A16 path (bit-equal to b12x on 58% of outputs,
same error against an fp32 reference). The first version was slower than b12x (D 35 vs 29 us); the launch sweep
(`tools/q8gemv_sweep.py`) settled on 4 K-blocks unrolled per iteration, 3 stages, 4 warps, 32 rows per program (D: no
split-K; U: split-K by group, so a skipped group is a skipped split). PDL launches matter only for the small shapes.

| shape (N x K), us per call | M | b12x | Triton | Triton, 4/8 groups used | 6/8 groups |
|---|---:|---:|---:|---:|---:|
| latent D 14336 x 2560 | 1 / 4 | 28.9 / 29.7 | 26.2 / 26.7 | **21.5 / 21.3** | 25.2 / 25.2 |
| latent U 2560 x 14336 | 1 / 4 | 28.0 / 28.1 | 27.1 / 27.9 | **17.6 / 17.9** | 23.4 / 23.5 |
| GDN in_proj_qkvz 16384 x 2560 | 1 / 4 | 30.1 / 30.2 | 29.7 / 29.8 | | |
| shared gate_up 1280 x 2560 ; down 2560 x 640 | 1 / 4 | 3.9 / 4.0 ; 2.1 / 3.9 | 3.1 / 3.2 ; 2.8 / 2.9 | | |
| **shared expert + gate + add**: stock (~10 kernels) vs fused (4) | 1 / 4 | 12.2 / 14.2 | **6.8 / 7.0** | | |
| **LM head 248320 x 2560** (BF16 `F.linear`: 801 / 754 / 757) | 1 / 4 / 8 | Q8 435 / 434 / 437 | **409 / 411 / 414** | | |

**3. PLE (task item 4): the drive is at its random-read ceiling, threads cannot help.** `tools/ple_iops.c` (O_DIRECT
4 KiB random reads over the 51 GB table): 5.6k IOPS with 1 thread (179 us each), 18.7k with 4, 40k with 16, **46-47k
with 32-64** (21 us amortized): the MSI M470 PRO (DRAM-less) saturates at ~46k IOPS over a 51 GB span.
`tools/ple_io_bench.py` (random rows): vLLM's serial reader, which already issues `posix_fadvise(WILLNEED)` for every
row before its `preadv` loop, takes 0.60 ms for 16 rows and 1.70 ms for 64 (26 us per row, close to the drive's
parallel rate); 16 Python threads 0.61 / 1.9-2.2 ms; O_DIRECT + 16 threads 0.65 / 2.2 ms; FADV_RANDOM no better.
`EDGE0_PLE_IO=threads` splits only calls of >= 128 rows, so it is a no-op at c1 (16 rows per token, 64 per k=3 step).
Not A/B'd: parallel reads cannot beat the drive. What would help: overlapping the reads with GPU work (the bonus
token's 16 rows during the drafter's 1.7 ms, or launching the graph before the rows arrive with an in-graph wait before
layer 1, like the CPU-experts doorbell), or moving the table to a faster drive (`/data` is a 3.6 TB Crucial P3 Plus,
not measured).

**4. Correctness** (B = `EDGE0_DU_MASK=1 EDGE0_SHARED_FUSE=1 EDGE0_LMHEAD=q8` on the tuned preset with CPU experts):
`compare_strata.py` (no speculation) **8/8 pass, teacher-forced top-1 = Strata argmax 256/261**, mean |dlogprob| 0.0334
(tuned 0.0328), KL(top-5) 0.0063 (`results/compare-dense-nospec-B.*`). Greedy with vs without the DFlash2 drafter
(`greedy_same.py`, 256 tokens): **5/8 identical, 3 divergences at exact near-ties (margins 0.0/0.25, 0.125/0.0,
0.0/0.125 nat), 0 unexplained** (`results/greedy-dense-{nospec,spec}-B.json`; the same picture as without the knobs).
Re-quantizing the BF16 head (= bf16(q*d) of the GGUF Q8_0 `output.weight`) with ggml's rule plus a search over d's fp16
neighbours recovers 99.8% of the q values exactly on synthetic blocks (`test_lmhead_requant_recovers_q8`).

**5. A/B results** (512 output tokens; c1 = 8 prompts, c8 = 16; A = tuned preset + CPU experts [+ DFlash2 k=3], B = A
+ the three knobs).

| run | protocol | A (tok/s) | B (tok/s) | B vs A | TPOT A -> B (ms) |
|---|---|---|---|---:|---|
| DFlash2 k=3, c1 | ABBA, 4-prompt warm after each start | 90.1 / 88.9 | 91.8 / 97.3 | +5.6% (fresh passes only: +2.6%) | 10.55 -> 10.05 |
| same, **c8** | ABBA | 135.6 / 132.1 | 146.2 / 147.6 | **+9.7%** | 38.4 -> 41.5; median TTFT 2.1 -> 1.2 s |
| DFlash2 k=3, c1 | **ABAB, every pass on a fresh server after an 8-prompt warm pass** | 94.0 / 94.3 | 96.3 / 97.1 | **+2.7%** | 10.1 -> 9.85 |
| no speculation, c1 | ABAB, 8-prompt warm | 58.2 / 57.2 | 61.9 / 61.9 | **+7.3%** | 16.85 -> 15.65 |

(ABBA's B3 runs on a server that has already served the 8 prompts, the other passes on fresh servers; with the
adaptive cache that favours B, so the spec c1 decision uses the ABAB rows, whose pass-to-pass spread is 0.3-0.8%.) At c8
with speculation the verify batch is 32 tokens, so only the head change applies there; most of the +9.7% is the
0.6 GiB the Q8 head frees: the KV pool grew from 61,067 to 76,458 tokens (no speculation: 173,812 -> 196,608), so more
of the 8 requests run at once (median TTFT halves).

**6. Where it lands inside the step** (nsys, warm k=3 c1 capture of B vs the `gap` agent's capture of A, kernel table
`tools/nsys_kernels.py`; per step): target LM head 756 -> 435 us; latent D 28.4 -> 25.9 us per layer and U 28.2 ->
24.6 (6.5 groups used per verify step); shared expert ~19.6 -> ~14.6 us per layer; the drafter only 1.745 -> 1.639 ms:
it shares the target's head but calls it with 16 rows, where b12x takes 638 us (BF16 754) and the Triton GEMV ~410.
Net about -0.8 ms per 27 ms step, which is the measured +2.7%. Without speculation the knobs save 1.2 ms of a 16.9 ms
step (D/U skip ~3.6 of 8 groups, shared, head).

**Decision: KEEP** `EDGE0_DU_MASK=1 EDGE0_SHARED_FUSE=1 EDGE0_LMHEAD=q8` (no speculation +7.3% at c1; DFlash2 k=3 +2.7%
at c1, under the 3% bar on its own, and +9.7% at c8; correctness gates pass). Not put into a preset by this agent
(other agents' A/Bs run on `PRESET=tuned` / `single`); they belong in both: they only touch decode calls of <= 8 tokens
and the head. `EDGE0_Q8_GEMV=1` (generic small-M Q8 linears): REJECTED as a default (qkvz equal to b12x; the gains come
from skipping and fusion, not from the GEMV itself).

### G2. Request-reactive expert tier (`EDGE0_TIER=1`) and the c1 preset (`PRESET=single`) (agent `fast-cache`, 2026-11-20)

(Logged by agent `consolidate` from `logs/fcab-*.log`, `logs/fcprobe-*.log`, `logs/fc-gate.log`: the `fast-cache`
agent measured but was cut off before logging.)

Lever 1 of G1 built as `plugin/edge0_vllm/tier.py` (design in its docstring): Strata's policy (+1 per routed entry,
decay, swap while candidate >= victim + 1.5) decided every step on the host from counts that a graph-safe kernel
(`kernels.tier_count`) accumulates and the host reads one step late from a pinned ring (no stream sync); the swaps
are copy-engine H2D / D2H copies on two side streams into 4 free VRAM slots per layer (`EDGE0_TIER_FREE`, taken from
the plan's hot budget) and spare host slots, and the main stream flips the expert maps after the step's forward with
one `index_copy_`, so captured CUDA graphs stay valid. Knobs `EDGE0_TIER_FREE=4 _SWAPS=48 _DECAY=0.915 _GAIN=1.5
_MIN_USE=2`; supersedes `EDGE0_ADAPT`; works with `EDGE0_CPU_EXPERTS` (evicted experts are computed from their new
host slot). Tests: `plugin/tests/test_tier.py` + the pool / CPU-experts GPU regressions, **29 passed**
(`bench/fc-gate.sh`). Drivers `bench/fc-chain.sh`, `bench/fc-ab.sh` (ABBA, a fresh server per letter and an unscored
8-prompt c1 warm pass, 512 tokens with EOS ignored, one c8 pass per arm), `bench/fc-probe.sh` (one server: warm +
scored c1, then `compare_strata.py`, an optional long prompt, `greedy_same.py gen`).

`PRESET=single` (serve.sh, written by this agent): DFlash2 k=3, CPU experts, `EDGE0_TIER=1`, `MAX_SEQS=4`,
`MAX_BATCHED=1024`, `KV_BYTES=1879048192` (1.75 GiB = 46,291 tokens, 1.41 x 32k requests), and the VRAM this frees
given to hot experts: 15.2 GiB with `plans/plan-s1-15.2.json` (8,432 slots; 8,240 hot + 4 free per layer with the
tier). Calibration probe `fcsinglecal` (`KV_BYTES= GPU_UTIL=0.96` on the 14.5 GiB plan: KV 72,297 tokens, 110.5
tok/s) sized that split.

**What the tier does to the cold count** (tier log lines of the scored c1 passes; per layer and verify step):

| config | hot slots | cold distinct experts | routes from VRAM | swaps / step | host per step (decide + issue) |
|---|---:|---:|---:|---:|---:|
| no tier, 12.8 GiB plan (`fcbase`, `EDGE0_TRACE=bw`) | 7,101 | 8.21 | 70.1% | - | - |
| tier, 12.8 GiB (`fctier`, `fcab2-A`) | 6,909 + 192 free | 5.2-5.9 | 80-84% | 42-48 | 0.5-0.8 + 1.2-1.9 ms |
| tier, 15.2 GiB (`PRESET=single`, `fcsingle`) | 8,240 + 192 free | 4.07-4.26 | 87% | 34 | 0.55 + 1.23 ms |

Reaction within a request (15.2 GiB, cold per layer-step by steps since the prefill): 9-11 (< 4 steps), 7.3-7.9
(< 16), 4.35 (< 64), 3.1 (< 128), 3.6-3.7 (< 256). The replay of G1 predicted 5.25 at 7,101 slots for Strata's policy
with 4-token windows; the served tier gets 5.2-5.9. The host work is issued from `compute_logits` while the GPU runs
the step's forward, so at c1 it is not on the critical path.

**A/B (ABBA, c1 = 8 prompts, c8 = 16 prompts, 512 tokens):**

| run | A | B | c1 A (tok/s) | c1 B (tok/s) | B vs A | TPOT A -> B (ms) | c8 A -> B |
|---|---|---|---|---|---:|---|---|
| `fcab1` | tuned + DFlash2 k=3 + CPU experts | A + `EDGE0_TIER=1` | 95.3 / 92.8 | 101.6 / 100.4 | **+7.4%** | 10.0-10.3 -> 9.3-9.4 | 137.7 -> 144.6 (+5.0%) |
| `fcab2` | A + `EDGE0_TIER=1` (12.8 GiB) | `PRESET=single` (15.2 GiB) | 103.3 / 100.7 | 110.9 / 112.2 | **+9.4%** | 9.1-9.4 -> 8.4-8.5 | 154.5 -> 168.9 (+9.3%) |

Acceptance is unchanged by either (2.62-2.70 tokens per step). `PRESET=single`'s c8 row runs 4 requests at a time
(`MAX_SEQS=4`): its throughput is higher but the median TTFT is 11.5 s against 1.8 s; it is a c1 preset, c >= 8
belongs to `PRESET=tuned`.

**Correctness.** `fctier` (tuned + DFlash2 + CPU experts + tier): `compare_strata.py` 8/8 pass, teacher-forced top-1
= Strata argmax **257/261**, mean |dlogprob| 0.0324, KL(top-5) 0.0063. `fcsingle` (`PRESET=single`): 8/8, **257/261**,
0.0307, 0.0061; a 23,328-token prompt was served (14.0 s) and the server stayed healthy (prefill memory peak with
`MAX_BATCHED=1024` and the small KV). Greedy text with the drafter was recorded (`results/greedy-fctier.json`,
`greedy-fcsingle.json`); the no-drafter counterpart (`fc-chain.sh ab3`) never ran, so the speculative gate of the tier
is the one in C1 below.

**Decision: KEEP** `EDGE0_TIER=1` in `PRESET=single` (+7.4% at c1 on its own, +9.4% more with the c1 memory split;
c8 +5%). Not in `PRESET=tuned`: never measured at c32 there.

### D2. The dense knobs on `PRESET=single`, with the VRAM the Q8 head frees given to experts (`fcab4`, agent `dense-c1`)

(Logged by agent `consolidate`; driver `bench/fc-chain.sh ab4`, logs `logs/fcab-fcab4.log`, `logs/fcprobe-fcsingle2.log`.)
A = `PRESET=single` (15.2 GiB), B = A + `EDGE0_DU_MASK=1 EDGE0_SHARED_FUSE=1 EDGE0_LMHEAD=q8` (D1) +
`EDGE0_EXPERT_VRAM_GIB=15.7 EDGE0_PLACEMENT=plan:/workspace/edge0/plans/plan-s1-15.7.json`: the Q8 head frees 0.6 GiB,
which goes to 278 more expert slots (8,710; 8,518 hot + 4 free per layer) because `PRESET=single` fixes the KV size
(46,291 tokens in both arms; GPU free after load 4.55 -> 4.58 GiB).

| | A1 | B2 | B3 | A4 | B vs A |
|---|---:|---:|---:|---:|---:|
| c1 tok/s (TPOT ms) | 110.3 (8.6) | **118.8 (7.9)** | **117.5 (8.0)** | 111.7 (8.5) | **+6.4%** |
| c8 tok/s (TPOT ms) | 170.1 (21.6) | 175.0 (20.7) | | | +2.9% |

Tier at 15.7 GiB: 3.7-4.4 cold experts per layer-step, 86-89% of routes from VRAM, 33-37 swaps per step (15.2 GiB:
4.1-4.3). A separate probe of B on a fresh server (`fcsingle2`): 118.2 tok/s (TPOT 8.0); `compare_strata.py` 8/8 pass,
top-1 = Strata argmax **257/261**, mean |dlogprob| 0.0321, KL(top-5) 0.0060; the 23,328-token prompt served in 13.8 s;
greedy text with the drafter in `results/greedy-fcsingle2.json`.

`bench/chain-dense-run2.sh` (B+ = B with `EDGE0_QSA_Q8=1` and a 16.0 GiB plan) was written but never ran
(`logs/chain-dense.log` ends with run 1); the Q8 head's Triton GEMV for <= 16-row calls (the drafter's head call,
`EDGE0_LMHEAD_GEMV_ROWS=16`) is already in B. **Decision: KEEP**; made the `PRESET=single` default in C1.

### C1. Parity locked in: gate, `PRESET=single` default, image, Strata vs vLLM in one session (agent `consolidate`, 2026-11-20)

Driver `bench/consolidate.sh build | gate | cmp`, log `logs/consolidate.log`; per-request summary `tools/c1_compare.py`.
Copies of the logs and gate outputs of G2, D2 and C1: `results/single/`; bench JSONs (with per-request data):
`5090:~/work/edge0-bench/results/cmp-*`.

**1. `PRESET=single` default** (serve.sh) is now D2's B: `EDGE0_EXPERT_VRAM_GIB=15.7`, `plans/plan-s1-15.7.json`,
`EDGE0_DU_MASK=1 EDGE0_SHARED_FUSE=1 EDGE0_LMHEAD=q8` (exported to the container like the other preset knobs; an
explicit value still wins, e.g. `EDGE0_DU_MASK=0`). `PRESET=tuned` is unchanged.

**2. Image.** `edge0-vllm:dev` rebuilt from this tree (`6048fe67b3ef`; the previous one is kept as
`edge0-vllm:pre-consolidate`, `0c0843b63bd2`): installed plugin = tree (md5 of all 26 source files), the CPU-experts
extension prebuilt in `/opt/edge0/ext/09b2b7d0edff21f2`. `PRESET=single ./serve.sh` without `EDGE0_DEV` starts in
201 s: 8,518 hot + 192 free expert slots (15.35 GiB), latent D/U skip, Q8 head, tier, 8 CPU-expert workers, KV
46,291 tokens.

**3. Correctness gate of that server** (from the image, no `EDGE0_DEV`): `compare_strata.py` **8/8 pass, teacher-forced
top-1 = Strata argmax 256/261** (= Strata's token 249/261; Strata picked a non-argmax token 9 times), mean |dlogprob|
0.0313, KL(top-5) 0.0060 (`results/compare-cons-single.*`). `greedy_same.py`, `PRESET=single` with vs without the
drafter (`SPEC_METHOD=none`, same memory split): **4/8 identical, 4 divergences at near-ties, 0 unexplained**
(`results/greedy-cons-single-diff.txt`): prompt 7 at token 0 (margins 0.0 / 0.125) and prompt 4 at 92 (0.0 / 0.125)
are the near-ties of every earlier gate (S5, D1); prompts 2 (at 222, 0.75 / 0.125) and 5 (at 21, 0.0 / 0.125) diverge
between end-of-turn special tokens that `ignore_eos` forces past the end of the reply (`<|im_start|>` vs `<|im_end|>`,
`<|endoftext|>` vs `<|im_start|>`). The no-drafter text equals the earlier dense-c1 no-drafter text
(`greedy-dense-nospec-B.json`) on prompts 2 and 5, and the D2 probe's drafter text (`greedy-fcsingle2.json`) on prompt
2: run-to-run order of the tier's swaps and the CPU split moves such near-ties, nothing else. **PASS.**

**4. Measuring Strata: bench its engine server, not edge0-infer's relay.** The first Strata pass of this session
(`cons-s1`, through `~/work/edge0-serve.sh`'s relay on :8124) read 11.2 tok/s with 72-token replies, while Strata's
own log shows every request generated in full (474-1024 tokens at 120-145 tok/s). A 3-request probe with
`--save-detailed` showed why: through the relay (HTTP/1.1, chunked, keep-alive) vllm-bench recorded a request with 1
token after 90 ms while the engine generated 609 for it, and a later request's text started in the middle of a
previous reply; against the engine's own server (`serve/server.py` on the port edge0-infer prints, HTTP/1.0,
`Connection: close`) the same 3 requests were counted exactly (896 / 586 / 453 tokens = Strata's log). So
`bench/gap-strata.sh ref` now benchmarks that port by default (`DIRECT=1`; the server is still started by
`edge0-serve.sh` unchanged). The `gap` agent's `sref` row (113.1 tok/s through the relay) may be affected; its
`slog` row (120.8, `server.py` direct) agrees with the numbers below.

**5. Strata vs vLLM at c1, one session, interleaved S V V S** (each letter a fresh server; same 8 ShareGPT seed-0
prompts, chat API, thinking off, temperature 0, `max_tokens` 1024, replies end at EOS for both engines; Strata:
`~/work/edge0-serve.sh` unchanged, 4-prompt warm pass; vLLM: `PRESET=single` from the image, 8-prompt 512-token warm
pass, then the 512-token EOS-ignored pass, then the EOS pass). TTFT is to the first content token: Strata's server
sends an empty role chunk at once, so vllm-bench's own TTFT for Strata is ~1 ms and its TPOT includes the prefill;
"decode TPOT" = (latency - first content) / (tokens - 1) per request (`tools/c1_compare.py`, `--save-detailed`).

| pass | output tok/s | TPOT (vllm-bench) ms | decode TPOT ms | TTFT median / mean ms | mean reply | tokens / step |
|---|---:|---:|---:|---:|---:|---:|
| Strata `cmp-s1` | 120.4 | 8.31 | 7.67 | 510 / 590 (engine 392) | 883 | - |
| vLLM `cmp-v1-eos` | **129.5** | **7.50** | **7.50** | **203 / 266** | 878 | 2.78 |
| vLLM `cmp-v2-eos` | **128.6** | **7.53** | **7.53** | **206 / 268** | 921 | 2.74 |
| Strata `cmp-s2` | 122.2 | 8.18 | 7.55 | 551 / 574 (engine 390) | 888 | - |
| **Strata mean / vLLM mean** | **121.3 / 129.1 (vLLM +6.4%)** | 8.25 / 7.52 | 7.61 / 7.52 | 530 / 205 (median) | 886 / 900 | |
| vLLM 512 tokens, EOS ignored (`cmp-v1`, `cmp-v2`) | 119.3 / 118.3 | 7.90 / 7.96 | | 200 / 201 | 512 | 2.65 / 2.60 |
| vLLM `PRESET=single` c8 (16 prompts, 512, EOS ignored; `cmp-v2-c8`) | 179.2 | 20.9 | | 11,687 (4 run at a time) | 512 | 2.67 |

- **vLLM now beats Strata at c1 on the like-for-like load: 129.1 vs 121.3 tok/s (+6.4%), TPOT 7.52 vs 8.25 ms.**
  The decode speed proper is at parity (7.52 vs 7.61 ms per token after the first token, -1.2%); the rest is
  prefill: vLLM's TTFT is 205 ms against Strata's ~530 ms (Strata reads these prompts at 70-760 tok/s; 29% of its
  prompt tokens were served from its conversation cache because the warm pass repeats prompts; vLLM's
  `PRESET=single` has prefix caching off).
- The 512-token EOS-ignored number (118.8) is below the EOS one because a 512-token reply amortizes the TTFT less and
  the drafter accepts less early in a reply (2.63 vs 2.76 tokens per step); compare engines on the EOS rows.
- vLLM's tier during the EOS passes: 3.2-3.6 cold distinct experts per layer-step, 89-91% of routes from VRAM, 29-32
  swaps per step: ~3.4 x 48 / 2.76 = **59 cold expert loads per token against Strata's 40** (G1; Strata gives
  experts 10,358 slots, vLLM 8,710). vLLM's decode is level with Strata's despite that because it gets more tokens
  per verify step and its dense path is as fast or faster (G1 section 3).
- c8: `PRESET=single` 179.2 tok/s (D2: 175.0), no regression; it queues 4 of 8 requests (median TTFT 11.7 s), so
  `PRESET=tuned` stays the c >= 8 preset and is untouched.

**6. Next levers (expected ms per output token from 7.52; ranked):**

| # | lever | expected | basis |
|---|---|---:|---|
| 1 | PLE rows off the critical path: read the bonus token's rows while the drafter runs, or launch the graph before the rows arrive with an in-graph wait before layer 1 (like the CPU-experts doorbell); the drive is at its IOPS ceiling, so overlap, not parallel reads | -0.4 to -0.6 | nsys (G1): 1.64 ms of GPU idle per step in `ple.read` / `ple.stage`; D1 item 3 |
| 2 | More expert VRAM in `PRESET=single`: KV for one 32k request (1.75 -> ~1.25 GiB), `GPU_UTIL` 0.94 -> 0.96, `EDGE0_QSA_Q8=1` (never measured), `MAX_BATCHED` 1024 -> 512; ~+1.5 GiB = ~+800 slots (new `plan-s1-*.json`) | -0.2 to -0.4 | replay (G1 section 8): 3.85 -> 2.61 cold per layer-step from 8,579 to 10,337 slots, x 48 x ~31 us per step / 2.76 |
| 3 | Draft length k=4 or confidence-trimmed windows (Strata: 3.1 positions), now that a verify position costs less | -0.1 to -0.3 | per-position acceptance 0.79 / 0.57 / 0.41 at k=3; S2 measured k=5 +2% at c1 on the old stack |
| 4 | Tier parameters: the 48-swap cap is hit on 21-38% of steps; `EDGE0_TIER_SWAPS=96`, `EDGE0_TIER_FREE=8`, decay | -0.1 to -0.2 | tier log: "cap reached" 103-185 per ~490 steps |
| 5 | CPU-experts dispatch kernel (1 CTA, 0.52 ms per step) | -0.15 | nsys (G1) |
| 6 | Prefill / TTFT is already 2.6x faster than Strata's; not a lever for tok/s at these prompt lengths | 0 | section 5 |

### R1. Beyond parity, round 1 (agent `exceed-1`, 2026-11-20): PLE rows off the critical path, 16.2 GiB of experts

Drivers: `bench/gap-nsys.sh` (profile), `bench/exceed1-gnsys.sh` (graph-level captures), `bench/fc-probe.sh` /
`bench/fc-ab.sh` (probes, ABBA), `bench/exceed1.sh build | gate | cmp`. Logs: `logs/fcab-{r1ple,r1vram}.log`,
`logs/fcprobe-{pledefer1,pledefer2,r1vram162,r1swap96}.log`, `logs/exceed1.log`, `logs/nsys-{r1single,r1defer,g1base,
g1defer}*.log`; nsys `profiles/{r1single,r1singlew,r1defer,r1deferw,g1basew,g1deferw}/`, step tables
`results/nsys-*.json`, kernel table `results/nsys-kernels-r1singlew.txt`.

**1. Profile of C1's `PRESET=single`** (docs/SINGLE-STREAM-GAP.md section 10): graph-level step 19.52 ms per 4-token
verify step (7.23 ms per token at 2.70 tokens per step), GPU idle 1.03 ms per step (PLE `ple.read` 0.53 + `ple.stage`
0.18 + launch 0.32). Per token against Strata: dense path + resident experts + head 3.77 vs 5.22 ms (vLLM ahead), cold
experts 2.45 vs 1.72 (behind, 55-76 cold loads per token vs 40), host idle 0.38 vs 0.08 (behind), drafter 0.52 vs 0.46.

**2. `EDGE0_PLE_DEFER=1`: PLE rows off the critical path** (`plugin/edge0_vllm/ple_defer.py`, `csrc/ple_reader.cpp`,
`csrc/ple_wait.cu`; test `plugin/tests/test_ple_defer.py`). Edge0's only PLE layer is decoder layer 1
(`ple_layer_ids` [2]), so a step can run layer 0 before it needs its ~64 rows. `stage_checkpoint_rows` is replaced:
the n-gram ids are computed on the GPU as before, a one-CTA kernel (`post_ids`) writes them into pinned, mapped host
memory and rings a doorbell; a C++ reader thread (no GIL, no CUDA call: it spins on the doorbell) reads the rows
exactly as vLLM's `_FilePlane.read` does (fadvise WILLNEED for all, then pread each) into mapped memory and publishes
the step's sequence number; the main thread launches the step's graph without waiting for the read; inside the graph
the PLE embedding's forward runs `wait_copy` (spin until the flag reaches the step's number, written to device memory by
a fill kernel before the launch, then copy the rows into vLLM's staging buffer). Same ids, same bytes, same staging
buffer: outputs unchanged. Every captured graph contains the wait (the reader is created when the table is bound,
before the profile run); a wait over 2 s gives up and is counted (0 timeouts in every run).

- **First version without the stream sync was slower** (`pledefer1`: 113.2 tok/s scored c1 against 117-119 for the
  C1 config). Graph-level nsys (`g1basew` vs `g1deferw`) showed why: with no sync in `prepare_inputs` the host runs
  ahead, then blocks before its output processing and scheduling, which then sit on the critical path: GPU idle 1.03
  -> 2.10 ms per step, attributed to `runner.execute_model` 1.13 + `runner.prepare_inputs` 0.40 + unlabeled 0.48.
  The reader (event wake-up then) also spun in `cudaEventSynchronize`.
- **Kept version** (`EDGE0_PLE_DEFER_SYNC=1`, default): keep vLLM's stream sync at the same place, so only the launch
  follows it, and wake the reader with the GPU doorbell (`EDGE0_PLE_DEFER_WAKE=doorbell`, default; `event` kept as an
  option). Reader per step in the scored passes: read 0.69-1.17 ms for ~64 rows; the GPU waited in layer 1 in 39-70% of
  steps, 0.41-0.72 ms per step on average (before: PLE 0.71 + launch 0.32 ms of idle per step). Probe `pledefer2`:
  123.1 tok/s.
- **Correctness** (`pledefer1`, the same data path): `compare_strata.py` 8/8 pass, teacher-forced top-1 = Strata argmax
  257/261, mean |dlogprob| 0.0314, KL(top-5) 0.0061. Final gate of the kept configuration: item 5.

| ABBA `r1ple` (512 tokens, EOS ignored; fresh server per letter + 8-prompt warm pass) | A1 | B2 | B3 | A4 | B vs A |
|---|---:|---:|---:|---:|---:|
| A = C1's `PRESET=single`, B = A + `EDGE0_PLE_DEFER=1`: c1 tok/s (TPOT ms) | 117.1 (8.1) | **123.8 (7.6)** | **121.2 (7.8)** | 118.1 (8.0) | **+4.2%** |
| c8 tok/s (TPOT ms) | 172.8 (21.0) | 182.2 (20.0) | | | +5.4% |

Acceptance unchanged (2.57-2.67 tokens per step), tier cold counts unchanged (2.7-4.4 per layer-step). **KEEP** in
`PRESET=single`. Not in `PRESET=tuned` (not measured at c32, where a 32-token step reads ~512 rows = ~10 ms).

**3. Tier swap cap 96** (`EDGE0_TIER_SWAPS=96`, probe `r1swap96`, on top of 2): 123.6 tok/s (B of `r1ple`: 121.2-123.8),
swaps 33-44 per step instead of 25-38, cold per layer-step unchanged (3.2-4.0). REJECTED (neutral; more PCIe copy
traffic for nothing).

**4. 16.2 GiB of expert VRAM** (`plans/plan-s1-16.2.json`, made with `tools/plan_placement.py` and the cost settings of
the 15.7 GiB plan; 8,987 slots = 8,795 hot + 192 free, +277 slots; KV unchanged at 46,291 tokens; GPU free after load
4.57 -> 4.11 GiB). Probe `r1vram162`: 126.0 tok/s.

| ABBA `r1vram` | A1 | B2 | B3 | A4 | B vs A |
|---|---:|---:|---:|---:|---:|
| A = C1's `PRESET=single` + `EDGE0_PLE_DEFER=1` (15.7 GiB), B = A with 16.2 GiB: c1 tok/s (TPOT ms) | 122.6 (7.7) | **124.4 (7.6)** | **125.2 (7.5)** | 122.4 (7.7) | **+1.9%** |
| c8 tok/s (TPOT ms) | 181.9 (20.1) | 191.2 (18.9) | | | **+5.1%** |

Both B passes above both A passes at c1, +5.1% at c8 (the 4-request batches touch more experts); cold per layer-step
3.1-4.0 vs 3.2-4.5. **KEEP** in `PRESET=single` (c1 small but consistent, c8 better), provided the 23k-token prompt still
fits (item 5).

**5. New `PRESET=single` default, image, gate** (`bench/exceed1.sh build | gate`). serve.sh `PRESET=single` now adds
`EDGE0_PLE_DEFER=1` and uses `EDGE0_EXPERT_VRAM_GIB=16.2` with `plans/plan-s1-16.2.json` (explicit values still win);
`PRESET=tuned` unchanged. Image `edge0-vllm:dev` rebuilt from this tree (installed plugin = tree, 29 files; both
extensions prebuilt: cpu-experts `09b2b7d0edff21f2`, ple-defer `e2a23920da7f1aad`; the previous image is kept as
`edge0-vllm:pre-exceed1` = C1's `6048fe67b3ef`; after the gate and item 6 it was rebuilt once more for a docstring-only
change of `ple_defer.py`: `9878d5d64464`, same extension keys). Gate from the image, no `EDGE0_DEV`:
- `compare_strata.py` (`results/compare-r1-single.*`): **teacher-forced top-1 = Strata argmax 257/261 (= Strata's token
  250/261)**, mean |dlogprob| 0.0311, KL(top-5) 0.0059 (C1: 256/261, 0.0313, 0.0060). The free-running rule passes 7/8:
  prompt 7 diverges at token 0 on an exact tie in vLLM (`'Le'` -0.702 vs `'Il'` -0.702; Strata's margin 0.128), the
  near-tie that greedy_same has flagged in every gate (S5, D1, C1); the 16.2 GiB hot set moves which experts are
  computed on the CPU / over PCIe / in VRAM and with it the last bf16 bit there. Passes the workflow's criterion
  (teacher-forced >= 250/261).
- `greedy_same.py`, with vs without the DFlash2 drafter (`results/greedy-r1-single-diff.txt`): **6/8 identical, 2
  divergences at near-ties (prompt 1 at token 58, margins 0.125 / 0.0; prompt 4 at 92, 0.125 / 0.125, as in every
  gate), 0 unexplained** (C1: 4/8 identical). PASS.
- A 23,328-token prompt (`LONG` 6,000 words, as G2 / D2) served in 12.6 s (D2: 13.8 s), server healthy afterwards: the
  0.5 GiB less headroom still holds the prefill peak (`MAX_BATCHED` 1024). Its PLE rows are 16,384 per 1,024-token chunk
  and take 0.35 s per chunk to read (~8 s of the 12.6 s): a TTFT lever for long prompts (see next levers).

**6. Strata vs vLLM at c1 with the round-1 `PRESET=single`** (`bench/exceed1.sh cmp`, from the image; same 8 ShareGPT
seed-0 prompts, chat API, thinking off, `max_tokens` 1024, replies end at EOS for both engines; vLLM twice, each on a
fresh server after the 512-token warm pass; Strata's passes are C1's `cmp-s1` / `cmp-s2` of the same day and box,
reused, as the box did not change; `tools/c1_compare.py`).

| pass | output tok/s | TPOT (vllm-bench) ms | decode TPOT ms | TTFT median / mean ms | mean reply | tokens / step |
|---|---:|---:|---:|---:|---:|---:|
| Strata `cmp-s1` / `cmp-s2` (C1) | 120.4 / 122.2 | 8.31 / 8.18 | 7.67 / 7.55 | 510 / 590, 551 / 574 | 883 / 888 | - |
| vLLM `r1-v1-eos` | **132.0** | **7.38** | **7.38** | **204 / 259** | 888 | 2.74 |
| vLLM `r1-v2-eos` | **130.9** | **7.36** | **7.36** | **193 / 259** | 892 | 2.73 |
| **Strata mean / vLLM mean** | **121.3 / 131.5 (vLLM +8.4%)** | 8.25 / 7.37 | 7.61 / 7.37 (-3.2%) | ~530 / ~199 (median) | 886 / 890 | |
| vLLM C1 config (C1, for reference) | 129.1 | 7.52 | 7.52 | 205 | 900 | 2.76 |
| vLLM 512 tokens, EOS ignored (`r1-v1`, `r1-v2`) | 126.5 / 125.0 (C1: 119.3 / 118.3) | 7.45 / 7.55 | | 183 / 182 | 512 | 2.69 / 2.64 |
| vLLM `PRESET=single` c8 (16 prompts, 512, EOS ignored; `r1-v2-c8`) | 187.2 (C1: 179.2) | 19.7 | | 10,914 (4 run at a time) | 512 | 2.67 |

- **vLLM now leads Strata at c1 on decode speed proper as well as end to end: 131.5 vs 121.3 tok/s (+8.4%), decode TPOT
  7.37 vs 7.61 ms (-3.2%)**; C1 had decode at parity (7.52). The EOS load gained less than the 512-token A/B passes
  (+1.8% vs +5.8% over C1): the EOS replies run to ~890 tokens, past what the warm pass put in the page cache, so
  their PLE rows are read from NVMe and more of the read outlasts layer 0.
- c8 +4.5% over C1 (PLE overlap and the larger hot set); `PRESET=tuned` untouched.

**7. Next levers (expected ms per output token from 7.37; ranked):**

| # | lever | expected | basis |
|---|---|---:|---|
| 1 | Cold experts are the one remaining gap to Strata (2.45 vs 1.72 ms per token in section 10 of the gap doc): free more VRAM for experts in `PRESET=single` (KV for one 32k request: 1.75 -> ~1.35 GiB; `EDGE0_QSA_Q8=1`, never measured) and/or make the CPU share faster (dispatch kernel 1 CTA, 0.56 ms per step = 8% of the cold time; caching `frac` in VRAM and one system fence instead of 512) | -0.1 to -0.3 | profile R1; replay G1 section 8 |
| 2 | PLE: read the bonus token's 16 rows while the drafter runs (it is known after rejection sampling, ~1.4 ms before the drafts), so only the 48 draft rows remain after the sync; or put the PLE table on the faster `/data` NVMe (Crucial P3 Plus; not measured: needs a 51 GB copy onto a root-owned volume, a user decision) | -0.1 to -0.2 | reader: the GPU still waits 0.4-0.9 ms per step in layer 1 (more on new text) |
| 3 | Long-prompt TTFT: prompt tokens are known when the request arrives, so the PLE rows of chunk n+1 can be read while chunk n computes (now 0.35 s per 1,024-token chunk, ~8 s of a 23k-token prompt's 12.6 s) | TTFT only | item 5 |
| 4 | `EDGE0_PLE_DEFER=1` in `PRESET=tuned`: measure at c8 / c32 (a 32-token step reads ~512 rows, ~10 ms, behind the sync today) | c >= 8 | not measured |
| 5 | Draft length / confidence-trimmed windows: this vLLM's confidence stop (`confidence_stop.py`) serves autoregressive drafters only; DFlash2 would need its own per-position cut and variable-length verify graphs | -0.1 to -0.3 | per-position acceptance 0.75-0.78 / 0.52-0.57 / 0.34-0.40 |

### R2. Beyond parity, round 2 (agent `exceed-2`, 2026-11-20): a cheaper CPU-experts dispatch kernel; route fusion, draft length 4 and more expert VRAM measured

Driver: `bench/exceed2.sh prof | probe | vram | build | gate | cmp` (log `logs/exceed2.log`), ABBAs with
`bench/fc-ab.sh` (`logs/fcab-r2*.log`), probes `logs/fcab-r2p-*.log`. Tools: `tools/dispatch_bench.py` (the dispatch /
gather kernels in a graph of 48 calls). Tests: `plugin/tests/test_cpu_experts.py` (new: calls with and without a CPU
share alternating in one captured graph, fixed and auto split; 8 passed), `plugin/tests/test_route_post.py` (33
passed). Results and logs copied to `results/r2/`.

**1. Profile of R1's `PRESET=single`** (docs/SINGLE-STREAM-GAP.md section 11; node-level `profiles/r2singlew/`,
graph-level `profiles/g2singlew/`): graph-level step 19.02 ms (median 18.38) per 4-token verify step, GPU idle 0.41 ms
(R1 before the PLE defer: 1.03). Per token (2.65 tokens per step) against Strata: dense path + resident experts + head
3.86 vs 5.22 ms (vLLM ahead), cold experts 2.72 vs 1.72 (behind: 3.5 cold per layer-step = 63 cold loads per token vs
40), drafter 0.52 vs 0.46, PLE wait in layer 1 + idle 0.29 vs 0.08. Inside the cold branch, the CPU-experts dispatch
kernel costs 11.6 us per layer on every call (0.56 ms per step), although only ~20% of the c1 calls give the CPU a
share.

**2. Probes** (one warm + one scored c1 pass each, 512 tokens EOS ignored; one server per line; `bench/exceed2.sh
probe`; ms per step = TPOT x tokens per step):

| probe | tok/s | TPOT ms | tokens / step | ms / step | cold / layer-step | notes |
|---|---:|---:|---:|---:|---:|---|
| `r2p-base`: R1 `PRESET=single` (image) | 121.3 | 7.8 | 2.63 | 20.5 | 3.1-3.9 | |
| `r2p-dv2`: + new dispatch kernel (item 3; `EDGE0_DEV=1`) | 129.5 | 7.2 | 2.70 | 19.4 | 3.1-4.0 | |
| `r2p-dv2mc2`: + `EDGE0_CPU_MIN_COLD=2` | 127.0 | 7.4 | 2.61 | 19.3 | 2.8-3.9 | CPU jobs 15.5k -> 27.5k per pass (3.1 experts each), neutral per step |
| `r2p-dv2mc2f45`: + `EDGE0_CPU_FRAC=0.45` | 125.8 | 7.5 | 2.59 | 19.4 | | neutral |
| `r2p-dv2k4`: dispatch + `SPEC_K=4` | 126.4 | 7.4 | 2.93 | 21.7 | 3.5-5.0 | +0.25 tokens per step (4th position 0.25) but +2.3 ms per step: more cold experts (4.5-5.0 per layer-step), 92 PLE rows; **REJECTED** |

Splitting smaller calls with the CPU (`EDGE0_CPU_MIN_COLD` 2, frac 0.45) moves work without shortening the step: a
1-3-expert cold Marlin call costs about the same as the CPU's share plus its hand-off. Not pursued (defaults kept).

**3. CPU-experts dispatch kernel, round 2** (`csrc/cpu_experts_cuda.cu`, `cpu_experts.py`; docs/CPU-EXPERTS.md 3.2).
The old kernel (1 CTA x 512 threads) cost 8.7 us per call in a graph even with no CPU share: all 512 threads ended with
`__threadfence_system` and every warp read `frac` from mapped memory over PCIe. Now: a fixed split passes `frac` as a
kernel argument (`EDGE0_CPU_FRAC=auto` passes -1 and reads the mailbox only when the call has enough cold experts),
the cold / eligible counts share one block scan, a call without a CPU share returns right after the choice (no mapped
writes; `n_cpu` = 0 for gather / wait), only threads that wrote mapped memory fence, and the gather skips its fences
and CTA counter without a share. Same choice of CPU experts (deterministic, logical-id order), same job layout.
`tools/dispatch_bench.py` (graph of 48 calls, T=4, 4 cold experts per call):

| us per call | old | new |
|---|---:|---:|
| dispatch, no CPU share | 8.73 | **1.11** |
| dispatch, CPU share | 8.80 | 2.05 |
| dispatch + gather, no CPU share | 10.84 | 2.35 |
| dispatch + gather, CPU share | 11.83 | 5.09 |

| ABBA `r2disp` (512 tokens, EOS ignored; fresh server per letter + 8-prompt warm pass) | A1 | B2 | B3 | A4 | B vs A |
|---|---:|---:|---:|---:|---:|
| A = R1 `PRESET=single` (image), B = A + new dispatch kernel (`EDGE0_DEV=1`): c1 tok/s (TPOT ms) | 123.1 (7.6) | **126.2 (7.5)** | **127.4 (7.4)** | 122.8 (7.7) | **+3.1%** |
| tokens per step / ms per step | 2.61 / 19.9 | 2.65 / 19.9 | 2.64 / 19.6 | 2.67 / 20.6 | -2.5% per step |
| c8 tok/s (TPOT ms) | 201.2 (18.1) | 189.6 (19.1) | | | -5.8% (see below) |

Both B passes above both A passes at c1. The c8 pair moved the other way, with more cold experts in B's pass (the CPU
took 10.6 vs 9.7 experts per job at frac 0.55, i.e. ~9% more cold experts per call) and a longer PLE wait (1.6 vs 1.3
ms per step): the kernel does strictly less work per call, and R1's c8 passes of the same preset were 191.2 / 187.2, so
this reads as the pass-to-pass spread of c8 (only 16 prompts, 4 at a time). A second c8 sample with the new kernel
(`r2v162-A1-c8`, item 4) made **191.7** tok/s, R1's level. **KEEP** (c1 +3.1%), correctness in item 6.

**4. More expert VRAM (17.0 GiB, `plans/plan-s1-17.0.json`): blocked by load-time transients, not served.** Replay
(`tools/sim_tier.py`, the c1 seed-0 routing, the tier as served): 3.12 cold per layer-window at 16.2 GiB, 2.93 at 16.7,
2.82 at 17.0 (-10%, ~0.6 ms per step). Plans for 16.7 / 17.0 made like the 16.2 one (`tools/plan_placement.py`
regimes 1:1,8:1,32:1, the same costs; the 16.2 plan regenerates bit-identically; a c1-only regime 4:1 gives almost the
same per-layer budgets, so the plan's regime mix is not a lever).
- `r2v170` (17.0 GiB): OOM while loading the target: the Marlin scale factor of a layer (`marlin_nvfp4.scale_factor`,
  vLLM's `_nvfp4_compute_scale_factor` on the whole stacked block-scale tensor) needs a 1.64 GiB fp32 + mask transient.
  **Fixed**: the factor is now computed chunk by chunk from the largest positive scale with the same float32 tensor ops
  (bit-identical factor; `plugin/tests/test_scale_factor.py` against vLLM's function, 5 passed; `test_gpu_experts.py`
  18 passed on real layer-0 experts). Load-time only, nothing changes at run time.
- `r2v170b` (17.0 GiB with that fix): the target loads (GPU free after load 3.29 GiB), then the DFlash2 drafter's
  constructor allocates its own BF16 `embed_tokens` and `lm_head` (2 x 1.19 GiB) before vLLM replaces both with the
  target's (`_maybe_share_lm_head`; the drafter checkpoint has neither): OOM on the `lm_head`.
- `r2v162` (16.2 GiB, new dispatch kernel; GPU memory sampled every 0.2 s): c1 127.3 tok/s (7.4 ms), c8 191.7; the
  23,328-token prompt in 12.5 s; **peak `memory.used` 31,897 MiB of 32,607** (the CUDA caching allocator keeps what the
  prefill and the drafter's constructor took). So 17.0 GiB would need both transients gone (the drafter's embedding
  and head built on the meta device when they are to be shared) and leaves ~0.7 GiB for the prefill working set
  instead of ~1.5: not attempted in this round; a next lever (below).

**5. `EDGE0_ROUTE_FUSE=1`: the routing tail in one kernel** (`kernels.route_post`, hooked in
`Edge0LatentMoeBlock.route` for calls of <= 1,024 routes when no trace / adapt / predict hook is on; `tier.count_row`).
skip_pad's two `torch.where` + `zeros_like` and the tier's `full_like` + `where` + `tier_count` become one Triton
kernel with the same results bit for bit (`plugin/tests/test_route_post.py`, 33 passed): 5.13 -> 1.15 us per call in a
graph of 48, i.e. ~0.19 ms per step.

| ABBA `r2fuse` | A1 | B2 | B3 | A4 | B vs A |
|---|---:|---:|---:|---:|---:|
| A = R1 `PRESET=single` + new dispatch (`EDGE0_DEV=1`), B = A + `EDGE0_ROUTE_FUSE=1`: c1 tok/s (TPOT ms) | 125.9 (7.5) | 127.8 (7.4) | 127.4 (7.4) | 125.7 (7.5) | +1.4% |
| tokens per step | 2.61 | 2.66 | 2.62 | 2.66 | |
| c8 tok/s (TPOT ms) | 196.4 (18.6) | 192.2 (19.2) | | | -2.1% |

Both B passes above both A passes, as the microbenchmark predicts (~1%), but under the ~3% bar and not better at c8:
**not in the preset** (rule); the knob stays (default off). Together with the rest of the MoE plumbing (vLLM's
`topk_softmax` 3.9 us, the gather, two `moe_align_block_size` + `count_and_sort` pairs, `wait`, combine: ~22 us per
layer, ~1 ms per step) it is the start of a fused routing kernel (next levers). A1-c8 is a third c8 sample of the new
dispatch kernel (196.4; with item 3's 189.6 and item 4's 191.7).

**6. Image and gate** (`bench/exceed2.sh build | gate`). `PRESET=single` itself is unchanged (the dispatch kernel is
code, always on; `EDGE0_ROUTE_FUSE` stays off). Image `edge0-vllm:dev` rebuilt from this tree = `d0aa833bb901`
(installed plugin = tree, 29 files; extensions prebuilt: cpu-experts `9f510b166444f412` (new sources), ple-defer
`e2a23920da7f1aad`); the previous image is kept as `edge0-vllm:pre-exceed2` (R1's `9878d5d64464`). Gate from the image,
no `EDGE0_DEV`:
- `compare_strata.py` (`results/compare-r2-single.*`): **teacher-forced top-1 = Strata argmax 257/261 (= Strata's token
  250/261)**, mean |dlogprob| 0.0328, KL(top-5) 0.0063 (R1: 257/261, 0.0311, 0.0059). Free-running rule 7/8: prompt 7
  diverges at token 0 on the exact tie seen in every gate (`'Le'` vs `'Il'`, vLLM margin 0.0, Strata's 0.128). Passes
  the workflow criterion (>= 250/261).
- `greedy_same.py`, with vs without the DFlash2 drafter (`results/greedy-r2-single-diff.txt`): **5/8 identical, 3
  divergences at near-ties** (prompt 4 at token 92, margins 0.125 / 0.0, as in every gate; prompt 5 at 21, 0.125 /
  0.125; prompt 6 at 24, 0.25 / 0.0), **0 unexplained**. PASS.
- The 23,328-token prompt: 12.7 s, server healthy afterwards.

**7. Strata vs vLLM at c1 with the round-2 `PRESET=single`** (`bench/exceed2.sh cmp`, from the image; same 8 ShareGPT
seed-0 prompts, chat API, thinking off, `max_tokens` 1024, replies end at EOS for both engines; vLLM twice, each on a
fresh server after the 512-token warm pass; Strata's passes are C1's `cmp-s1` / `cmp-s2` of the same day and box, reused
because the box did not change; `tools/c1_compare.py`).

| pass | output tok/s | TPOT (vllm-bench) ms | decode TPOT ms | TTFT median / mean ms | mean reply | tokens / step |
|---|---:|---:|---:|---:|---:|---:|
| Strata `cmp-s1` / `cmp-s2` (C1) | 120.4 / 122.2 | 8.31 / 8.18 | 7.67 / 7.55 | 510 / 590, 551 / 574 | 883 / 888 | - |
| vLLM `r2-v1-eos` | **134.6** | **7.18** | **7.18** | **185 / 256** | 924 | 2.73 |
| vLLM `r2-v2-eos` | **134.4** | **7.20** | **7.20** | **198 / 255** | 898 | 2.72 |
| **Strata mean / vLLM mean** | **121.3 / 134.5 (vLLM +10.9%)** | 8.25 / 7.19 | 7.61 / 7.19 (-5.5%) | ~530 / ~192 (median) | 886 / 911 | |
| vLLM R1 `PRESET=single` (R1 item 6, for reference) | 131.5 | 7.37 | 7.37 | 199 | 890 | 2.74 |
| vLLM 512 tokens, EOS ignored (`r2-v1`, `r2-v2`) | 126.2 / 128.2 (R1: 126.5 / 125.0) | 7.45 / 7.33 | | 183 / 189 | 512 | 2.69 / 2.63 |
| vLLM `PRESET=single` c8 (16 prompts, 512, EOS ignored; `r2-v2-c8`) | 190.8 (R1: 187.2) | 19.0 | | 10,037 (4 run at a time) | 512 | 2.74 |

- **vLLM now leads Strata at c1 by 10.9% on the EOS load (134.5 vs 121.3 tok/s), with decode TPOT 7.19 vs 7.61 ms
  (-5.5%)** and TTFT ~192 vs ~530 ms. Round 2 adds +2.3% over R1 on the EOS load (131.5 -> 134.5) and +1.2% on the
  512-token passes (the ABBA's +3.1% came from fresh servers with the same warm pass; the two cmp servers are within
  the R1 spread on the 512-token load). c8 +1.9% over R1. `PRESET=tuned` untouched.

**8. Next levers (expected ms per output token from 7.19; ranked):**

| # | lever | expected | basis |
|---|---|---:|---|
| 1 | **A tier that makes a swap resident one step earlier**: mapped per-layer counters read by a host thread while the step runs, decisions and H2D copies per layer as the counts arrive, the flip enqueued before the next launch (the copies must overlap the dense parts of the step, not the cold Marlin calls; the swaps now issued at step start land on layers 1-12, whose cold calls are the longest) | -0.25 (today's rule) to -1.0 (LRU-like rule, same PCIe bytes) | replay (gap doc section 11): 3.12 -> 2.85 cold per layer-window; 1.83 with every touched cold expert swapped in; cold experts are 2.72 ms per token |
| 2 | **More expert VRAM (17.0 GiB)**: build the DFlash2 drafter's BF16 `embed_tokens` / `lm_head` on the meta device when vLLM will share the target's (2 x 1.19 GiB load transient), then check the prefill working set (`torch.cuda.max_memory_allocated` around the 23k-token prompt; the caching allocator already holds 31.9 of 32.6 GB at 16.2 GiB) | -0.2 | replay: 3.12 -> 2.82 cold per layer-window (-10%) |
| 3 | **Fused MoE routing kernel**: `topk_softmax` + the routing tail (`EDGE0_ROUTE_FUSE`, measured +1.4%) + the CPU-experts dispatch + gather in one or two kernels; the rest of the plumbing (two `moe_align_block_size` + `count_and_sort`, `wait`, combine) is ~22 us per layer | -0.1 to -0.15 | kernel table `results/nsys-kernels-r2singlew.txt` |
| 4 | PLE: the GPU still waits 0.33-0.6 ms per step in layer 1 for the step's ~78 rows (more on new text); read the bonus token's 16 rows while the drafter runs | -0.05 to -0.1 | `edge0 ple-defer` lines of every pass |
| - | not levers at c1 (measured): `EDGE0_CPU_MIN_COLD` 2 / frac 0.45 (neutral), `SPEC_K=4` (-2.4%), tier swap cap / gain / decay / free slots (replay within a few %), a c1-only placement regime (same budgets) | | items 2, 4; gap doc section 11 |

### R3. Beyond parity, round 3 (agent `exceed-3`, 2026-11-20): the tier decides right after the forward and copies in the drafter window; shadow host copies; more expert VRAM measured

Box state: between R2 and R3 the coordinator A/B'd the PLE table on `/data` (`logs/plemove.log`: vLLM `PRESET=single`
EOS load 134.1 on the system disk vs 137.7 on `/data`; Strata 121.5 on `/data`, its system-disk run failed) and left
it on the system disk with the page cache dropped. Driver: `bench/exceed3.sh test | prof | probe | vram | build | gate |
cmp` (log `logs/exceed3.log`), ABBAs with `bench/fc-ab.sh` (`logs/fcab-r3early.log`, `logs/fcab-r3vram.log`), probes
`logs/fcab-r3p-*.log`, `logs/fcab-r3v*.log`. New tool `tools/tier_copy_bench.py`. Tests: `plugin/tests/test_tier.py`
(16 passed; new: early mode with both copy paths and with / without the CPU path, shadow eviction without write-back
with / without the CPU path, shadow stealing, GPU decision = numpy decision, batch copies move the same bytes). Logs and
results copied to `results/r3/`.

**1. Profile of R2's `PRESET=single`** (docs/SINGLE-STREAM-GAP.md section 12; `profiles/r3singlew/`, `profiles/g3singlew/`):
graph-level period 18.39 ms (median 17.88) per 4-token verify step, GPU idle 0.44. Per token (2.65 per step) against
Strata: dense path + resident experts + head 3.84 vs 5.22 ms, cold experts 2.26 vs 1.72 (2.97 cold per layer-step in
the capture = 54 cold loads per token vs 40), drafter 0.52 vs 0.46, PLE wait + idle 0.26 vs 0.08; total 6.94 vs 7.72.
The tier's copies (41.5 MB H2D + as much D2H per step) run during layers 1-12 of the forward.

**2. The copies themselves** (`tools/tier_copy_bench.py`, `logs/exceed3-copybench.txt`): 32 experts (61.9 MB) H2D
take 1.18 ms with `cudaMemcpyBatchAsync` (1.30 with one `cudaMemcpyAsync` per array, 1.07 as one contiguous copy);
H2D and D2H at the same time take 2.10 ms, i.e. **the two directions do not overlap on this box** (host DRAM is the
shared limit, ~62 GB/s of copies in total). A victim's write-back costs as much as the swap. Issue cost on the host:
0.05 ms per batch vs 0.20 ms per-array.

**3. `EDGE0_TIER_EARLY=1`** (`tier.py`; the stage hook wraps `model_state.stage_checkpoint_rows`, i.e. EDGE0_PLE_DEFER's
staging, before and after its stream sync): `on_step` only snapshots the counts; at the next step's staging the host
waits for the forward's snapshot event (not for the drafter), folds it in, decides with the served rule and issues
the copies; the flip (scales, maps) runs on the copy stream behind both copy streams, and the next forward waits for
it (`post_sync`). Counts of step N place experts from step N+1 (served tier: N+2), and the copies run during the LM
head, sampler and drafter (~2 ms without cold reads) instead of during the forward. `EDGE0_TIER_COPY=batch`: one
`cudaMemcpyBatchAsync` per direction. **`EDGE0_TIER_SHADOW`** (default on with EARLY): a swapped-in expert keeps its
host slot (nothing writes it while the expert is hot), so evicting it again needs no D2H; a victim that never had a
host copy is written back into a spare host slot or the shadow of the most used hot expert (`EDGE0_TIER_SPARE` extra
spare host slots per layer). The flip wait (GPU time the next forward waits for the flip) and the write-backs per
step are logged (`edge0 tier: EARLY:` lines).

Probes (one warm + one scored c1 pass each, 512 tokens EOS ignored; `bench/exceed3.sh probe`):

| probe | tok/s | TPOT ms | tokens / step | flip wait us / step | write-backs / step | notes |
|---|---:|---:|---:|---:|---:|---|
| `r3p-base`: R2 `PRESET=single` (image) | 124.7 | 7.6 | 2.62 | - | (= swaps, during the forward) | page cache still cold from the drop |
| `r3p-early`: + EARLY, batch copies | 126.7 | 7.4 | 2.67 | 720-1,050 | = swaps (~30) | copies run at half speed: H2D + D2H together |
| `r3p-early64`: + cap 64, gain 0.5, min usage 1 (replay: -10% cold) | 122.0 | 7.7 | 2.67 | 1,960-2,650 | ~56 | **rejected**: the window cannot hold 2 x 100 MB |
| `r3p-early2`: + shadows, no extra spares | 130.5 | 7.2 | 2.70 | 600-1,025 | 27-35 (98% of swaps) | shadows capped at the 4 spares per layer: victims are plan-hot experts |
| `r3p-early2x`: shadows + cap 64 rule | 121.2 | 7.8 | 2.67 | 2,380-2,680 | 58-61 | rejected |
| `r3p-early3`: + 64 extra spare host slots per layer (+5.7 GB pinned) | 129.0 | 7.3 | 2.60 | 240-595 | 9-15 | |
| `r3p-early4`: + H2D batch issued before the victims' bookkeeping, vectorized steal | **134.1** | 7.0 | 2.66 | 35-418 | 8-15 | host decide + issue 0.29 + 0.44-0.61 ms |
| `r3p-early4w`: + `EDGE0_TIER_WB_COST=0.5` (victims without a shadow count 0.5 more) | 130.8 | 7.2 | 2.60 | 68-161 | 0.7-1.3 | more churn (34-36 swaps per step); knob kept, off |
| `r3p-early5`: + `EDGE0_TIER_GPU_DECIDE=1` (rule on a side stream after the forward) | 129.9 | 7.2 | - | 230-548 | 9-16 | the host waits 0.57 ms for the side stream (queued behind the LM head): **slower**, knob kept, off |

| ABBA `r3early` (512 tokens, EOS ignored; fresh server per letter + 8-prompt warm pass) | A1 | B2 | B3 | A4 | B vs A |
|---|---:|---:|---:|---:|---:|
| A = R2 `PRESET=single` (image), B = A + `EDGE0_TIER_EARLY=1 EDGE0_TIER_COPY=batch EDGE0_TIER_SPARE=64` (`EDGE0_DEV=1`): c1 tok/s (TPOT ms) | 125.5 (7.5) | **129.7 (7.2)** | **131.5 (7.1)** | 129.4 (7.3) | **+2.5%** |
| tokens per step | 2.61 | 2.63 | 2.64 | 2.65 | |
| PLE: GPU waited (share of steps, us per wait) | 51%, 552 | 29%, 224 | 30%, 221 | 39%, 440 | |
| c8 tok/s (TPOT ms) | 189.8 (18.9) | **195.1 (19.0)** | | | **+2.8%** |

Both B passes above both A passes; the A passes drift up by 3% (the page cache refills after the coordinator's drop),
which the ABBA order cancels. Cold experts per layer-step are within the window-to-window spread (A 2.7-4.0, B 2.9-3.8:
the replay's -9% is not resolvable here); what B gains is the forward without copies next to its cold reads, and a
shorter PLE wait (the rows are read while no swap traffic competes for host DRAM: 0.53-0.56 vs 0.70-0.87 ms per read).
It still waits 0.15-0.4 ms per step for the flip (copies that outlast the ~2 ms window), more at c8 (0.6-0.8 ms).
**KEEP** (+2.5% at c1, under the ~3% bar but with every B pass ahead, and +2.8% at c8): `PRESET=single` now sets
`EDGE0_TIER_EARLY=1 EDGE0_TIER_COPY=batch EDGE0_TIER_SPARE=64`. Correctness in item 5.

**4. More expert VRAM.** `EDGE0_DFLASH_SHARE_VOCAB=1` (`draft.install_dflash`): the DFlash2 drafter is built with
parameter-less placeholders for its `embed_tokens` / `lm_head` (vLLM's `load_dflash_model` replaces both with the
target's right after loading; the checkpoint ships neither): no 2.36 GiB load transient. Memory probes
(`bench/exceed3.sh vram`, early tier on):
- `r3v170` (17.0 GiB, KV 1.75 GiB): loads (GPU free after the target 3.29 GiB, the drafter fits), then the b12x
  kernel warm-up runs out of memory (244 MiB; 1.67 GiB left after weights and KV).
- `r3v170k` (17.0 GiB, KV 1.35 GiB = 35,368 tokens, 1.08 x 32k): starts (graphs 0.21 GiB), then the first prefill runs
  out of memory in the QSA indexer (128 MiB, 70 MiB free).
- `r3v167k` (16.7 GiB, `plans/plan-s1-16.7.json`, KV 1.35 GiB): c1 133.8 tok/s, c8 192.7, the 23,328-token prompt in
  12.5 s, peak `memory.used` 31,997 of 32,607 MiB.

| ABBA `r3vram` | A1 | B2 | B3 | A4 | B vs A |
|---|---:|---:|---:|---:|---:|
| A = R2 `PRESET=single` + early tier (16.2 GiB), B = A + `EDGE0_DFLASH_SHARE_VOCAB=1`, 16.7 GiB, KV 1.35 GiB: c1 tok/s | 133.4 | 129.7 | 134.4 | 130.1 | +0.2% |
| c8 tok/s | 195.5 | 193.5 | | | -1.0% |

Neutral: **REJECTED** (the replay's -6% cold experts at 16.7 GiB does not show at this resolution, and the smaller KV
costs c8 TTFT: 11.7 vs 10.3 s). `EDGE0_DFLASH_SHARE_VOCAB` stays as a knob (off): it is what makes > 16.2 GiB load at all.

**5. Image and gate** (`bench/exceed3.sh build | gate`). `PRESET=single` = R2's plus `EDGE0_TIER_EARLY=1
EDGE0_TIER_COPY=batch EDGE0_TIER_SPARE=64` (shadows on with EARLY; +5.7 GB of pinned host memory, the box still had
~19 GB available with the server up). Image `edge0-vllm:dev` rebuilt from this tree = `d0d3bace6b03` (installed plugin =
tree, 29 files; extensions unchanged: cpu-experts `9f510b166444f412`, ple-defer `e2a23920da7f1aad`); the previous image
is kept as `edge0-vllm:pre-exceed3` (R2's `d0aa833bb901`). Gate from the image, no `EDGE0_DEV`:
- `compare_strata.py` (`results/compare-r3-single.*`): **8/8 prompts pass; teacher-forced top-1 = Strata argmax 257/261
  (= Strata's token 250/261)**, mean |dlogprob| 0.0331, KL(top-5) 0.0066 (R2: 257/261, 0.0328, 0.0063). Passes the
  workflow criterion (>= 250/261).
- `greedy_same.py`, with vs without the DFlash2 drafter (`results/greedy-r3-single-diff.txt`): 3/8 identical, 4
  divergences at near-ties (prompt 1 at token 58, margins 0.0 / 0.125; prompt 4 at 117, 0.0 / 0.125; prompt 5 at 21 and
  prompt 6 at 50, 0.125 / 0.125), and prompt 2 at token 222, which the script counts as unexplained (margins 0.375 /
  0.75) but which lies after the reply ended: `<|im_end|>` at token 213 and EOS at 215, then `ignore_eos` forces
  special tokens, and the two runs split between them (the same case C1 documented for prompt 2). **0 unexplained
  divergences inside the replies.** PASS.
- The 23,328-token prompt: 12.6 s, server healthy afterwards.

**6. Strata vs vLLM at c1 with the round-3 `PRESET=single`** (`bench/exceed3.sh cmp`, from the image; same 8 ShareGPT
seed-0 prompts, chat API, thinking off, `max_tokens` 1024, replies end at EOS for both engines; vLLM twice, each on a
fresh server after the 512-token warm pass; `tools/c1_compare.py`, `results/r3-c1-compare.txt`). The box changed
since C1 (PLE table moved to `/data` and back, page cache dropped; the coordinator's Strata run on the system disk
failed), so Strata was re-run once on the system disk, after the vLLM passes (`r3-s1`, `bench/gap-strata.sh ref`,
engine server direct).

| pass | output tok/s | TPOT (vllm-bench) ms | decode TPOT ms | TTFT median / mean ms | mean reply | tokens / step |
|---|---:|---:|---:|---:|---:|---:|
| Strata `r3-s1` (this round, system disk) | **120.3** | 8.36 | 7.63 | 643 / 671 (engine 389) | 876 | - |
| Strata earlier today: `cmp-s1` / `cmp-s2` (C1) / `pm-s-data` (PLE on `/data`) | 120.4 / 122.2 / 121.5 | 8.31 / 8.18 / 8.23 | 7.67 / 7.55 / 7.74 | 510 / 551 / 403 | 883 / 888 / 893 | - |
| vLLM `r3-v1-eos` | **136.3** | **7.11** | **7.11** | **199 / 259** | 904 | 2.70 |
| vLLM `r3-v2-eos` | **143.1** | **6.75** | **6.75** | **193 / 254** | 846 | 2.76 |
| **Strata / vLLM mean** | **120.3 / 139.7 (vLLM +16.1%)** | 8.36 / 6.93 | 7.63 / 6.93 (-9.2%) | 643 / 196 (median) | 876 / 875 | |
| vLLM R2 `PRESET=single` (R2 item 7; coordinator's `pm-v-root2-eos` on the same disk: 134.1) | 134.5 | 7.19 | 7.19 | 192 | 911 | 2.73 |
| vLLM 512 tokens, EOS ignored (`r3-v1`, `r3-v2`) | 131.4 / 132.1 (R2: 126.2 / 128.2) | 7.15 / 7.10 | | 185 / 194 | 512 | 2.67 / 2.68 |
| vLLM `PRESET=single` c8 (16 prompts, 512, EOS ignored; `r3-v2-c8`) | 196.8 (R2: 190.8) | 18.3 | | 10,230 (4 run at a time) | 512 | 2.70 |

- **vLLM leads Strata at c1 by 16.1% on the EOS load (139.7 vs 120.3 tok/s; against the mean of all four Strata
  passes today, 121.1: +15.4%), decode TPOT 6.93 vs 7.63 ms (-9.2%), TTFT ~196 vs ~640 ms.** Round 3 adds +3.9% over
  R2 on the EOS load (134.5 -> 139.7; the two passes differ by 5%: `r3-v2-eos`'s replies are shorter and accept more,
  2.76 vs 2.70 tokens per step) and +3.6% on the 512-token passes (127.2 -> 131.8). c8 +3.1% over R2. `PRESET=tuned`
  untouched.

**7. Next levers (expected ms per output token from 6.93; ranked):**

| # | lever | expected | basis |
|---|---|---:|---|
| 1 | **The flip wait**: the next forward still waits 0.15-0.4 ms per step at c1 (0.6-1.0 at c8) for copies that outlast the ~2 ms window. Make the flip per layer (layer l's maps flip in-graph once its copies landed, as the PLE wait does; the rest of the copies overlap the next forward's first layers, which are dense-only for the experts already resident), or issue the copies in layer order with a per-layer flag | -0.05 to -0.15 | `edge0 tier: EARLY` lines (flip wait, write-backs) |
| 2 | **Fewer write-backs**: 8-15 per step remain (the churn set exceeds 4 + 64 spare host slots per layer). A full host copy of every expert (+16.2 GiB pinned; ~19 GB were available) would make them 0; or `EDGE0_TIER_WB_COST` tuned (0.5 cut them to ~1 per step but added churn: neutral in one probe) | -0.05 to -0.1 | probes `r3p-early3/4/4w` |
| 3 | **Fused MoE routing kernel** (unchanged from R2): `topk_softmax` + routing tail + dispatch + gather; the routing plumbing is ~22 us per layer | -0.1 to -0.15 | `results/nsys-kernels-r3singlew.txt` |
| 4 | **More expert VRAM needs a smaller prefill working set**: 16.7 GiB fits (KV 1.35 GiB) but was neutral; 17.0 GiB runs out of memory in the first prefill (QSA indexer). `MAX_BATCHED=512` or a smaller indexer workspace would make room | -0.1 (replay -6 to -10% cold) | item 4 |
| 5 | Copy-rate ceiling: H2D and D2H share ~62 GB/s of host DRAM, which the CPU experts and the cold Marlin reads also use; any write-back or D2H in the forward costs cold-read bandwidth | | item 2 |
| - | not levers at c1 (measured this round): swap cap 64 / gain 0.5 / min usage 1 (-2 to -6%: the window cannot hold the copies), the decision on the GPU (slower: the side stream queues behind the LM head), 16.7 GiB + KV 1.35 GiB (neutral) | | items 3, 4 |

### F. Final report (agent `report`, 2026-11-20): standard sweeps of both presets, gates, Strata vs vLLM at c1

Driver `bench/final-report.sh all` (log `logs/final-report.log`; copies of it, the gap logs and the c1 table in
`results/final/`). Image `edge0-vllm:dev` `d0d3bace6b03` (= R3, plugin unchanged since), PLE table on `/data` (the
coordinator switched it after R3), page cache not dropped. Order: Strata `final-s1`, `PRESET=single` (c1 passes
`final-v1`, gate, 23k-token prompt, sweep), `PRESET=single SPEC_METHOD=none` (greedy text), `PRESET=tuned` (gate,
23k-token prompt, sweep), Strata `final-s2`, `PRESET=single` (c1 passes `final-v2`, left running on :8125).

**1. Standard sweeps** (vllm-bench, ShareGPT seed 0, chat API, thinking off, 1024 tokens, `--ignore-eos`, 4 x c
prompts, 2 warm-ups; one vllm-bench call per point):

| conc | `vllm-final-tuned` tok/s | TPOT ms | TTFT median / p99 ms | `vllm-final-single` tok/s | TPOT ms | TTFT median / p99 ms | single: tokens per step |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 54.8 | 18.0 | 185 / 497 | **140.3** | 6.9 | 153 / 391 | 2.70 |
| 2 | 92.9 | 21.0 | 545 / 791 | **188.6** | 10.1 | 476 / 914 | 2.70 |
| 4 | 112.8 | 34.6 | 865 / 1,384 | **212.7** | 17.9 | 531 / 753 | 2.68 |
| 8 | 136.3 | 57.6 | 1,253 / 1,577 | 214.1 | 18.1 | 18,946 / 24,117 | 2.70 |
| 16 | 148.7 | 105.5 | 2,302 / 2,474 | 212.4 | 18.3 | 57,734 / 64,187 | 2.77 |
| 32 | **180.3** | 174.6 | 3,108 / 3,874 | 214.4 | 18.2 | 133,639 / 144,304 | 2.78 |

- `tuned` runs 32 requests at once, 0 preemptions, GPU peak 32,047 MiB. Against `vllm-tuned` (section 5, 2026-11-19):
  c1 54.9 -> 54.8, c2 +13%, c4 +18%, c8 +29%, c16 +26%, c32 +27% (142.0 -> 180.3); the differences are CPU cold
  experts (with R2's dispatch kernel) and the PLE table on `/data`.
- `single` runs at most 4 requests (`MAX_SEQS=4`), 0 preemptions, GPU peak 31,915 MiB, per-position acceptance
  0.75-0.76 / 0.54-0.58 / 0.38-0.44. It beats `tuned`'s throughput at every concurrency, but from c8 on the requests
  queue (median TTFT 19-134 s), so `tuned` stays the preset for c >= 8. At c4 (212.7 tok/s, TTFT 0.53 s) it is a
  far better c4 server than `tuned` (112.8): a middle preset (drafter + single-stream mechanisms, 6-8 sequences) was
  not measured.

**2. Gates** (both from the image, no `EDGE0_DEV`):
- `compare_strata.py`: `tuned` 8/8 pass, teacher-forced top-1 = Strata argmax **257/261** (Strata's token 250/261),
  mean |dlogprob| 0.0336, KL(top-5) 0.0063; `single` 8/8, **257/261** (250/261), 0.0324, 0.0064
  (`results/compare-final-{tuned,single}.*`).
- `greedy_same.py`, `PRESET=single` with vs without the drafter: **6/8 identical, 2 near-ties (prompt 4 at token 92,
  margins 0.0 / 0.125; prompt 5 at 21, 0.125 / 0.0), 0 unexplained**, 0 non-argmax tokens
  (`results/greedy-final-single-diff.txt`). PASS.
- 23,328-token prompt: `single` 12.4 s, `tuned` 11.3 s, both healthy afterwards; 0 tracebacks in both serve logs.

**3. Strata vs vLLM at c1, EOS-stopping, same 8 prompts** (`tools/c1_compare.py`, `results/final/c1-compare.txt`;
Strata on its engine server, `DIRECT=1`):

| pass | tok/s | TPOT (vllm-bench) ms | decode TPOT ms | TTFT median / mean ms | mean reply | tokens / step |
|---|---:|---:|---:|---:|---:|---:|
| Strata `final-s1` | 120.9 | 8.26 | 7.61 | 592 / 566 (engine 392, 28% of prompt tokens reused) | 869 | - |
| vLLM `final-v1-eos` | 140.7 | 6.86 | 6.86 | 204 / 257 | 892 | 2.73 |
| Strata `final-s2` | 123.1 | 8.16 | 7.44 | 532 / 638 (engine 362, 29% reused) | 856 | - |
| vLLM `final-v2-eos` | 140.2 | 6.86 | 6.86 | 206 / 260 | 888 | 2.75 |
| **mean Strata / vLLM** | **122.0 / 140.5 (+15.1%)** | **8.21 / 6.86** | **7.53 / 6.86 (-8.8%)** | ~560 / ~205 | 863 / 890 | |
| vLLM 512 tokens, EOS ignored (`final-v1`, `final-v2`) | 135.3 / 134.2 | 6.92 / 6.99 | | 188 / 190 | 512 | 2.67 / 2.65 |

R3 measured 139.7 (system disk); the coordinator's `/data` A/B found +2.5% for vLLM and none for Strata, which fits
140.5 here and Strata's 122.0 (earlier today 120.3-122.2).

**4. Memory with the server up** (`MEM` lines): `tuned` 30,961 MiB after start (torch after load 23.33 GiB: 12.80
hot experts; KV 4.61 GiB = 173,812 tokens), host 40.5 GB used / 21.3 available (31.50 GiB cold experts pinned);
`single` 31,911 MiB (torch after load 25.88 GiB: 16.20 hot + free slots; drafter 1.26 GiB; KV 1.75 GiB = 46,291
tokens), host 44.9 GB used / 16.9 available (34.33 GiB cold experts + 3,264 spare host slots pinned).

README.md rewritten from these numbers. Left running: `PRESET=single` on :8125 (tmux `edge0-vllm`), GPU lock released.

