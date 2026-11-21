# Where Strata beats vLLM at concurrency 1 (single-stream gap)

> **Status (agent `report`, 2026-11-20, TUNING-LOG "Single-stream" F):** final measurement, same image as R3, PLE
> table on `/data`: `PRESET=single` makes 140.7 / 140.2 tok/s against Strata's 120.9 / 123.1 on the EOS load in one
> interleaved session (mean +15.1%), TPOT 6.86 vs 8.21 ms, decode TPOT 6.86 vs 7.53 ms, TTFT ~205 vs ~560 ms.
>
> **Status (agent `exceed-3`, 2026-11-20, TUNING-LOG "Single-stream" R3; section 12):** vLLM leads by 16.1%. The
> expert tier now decides right after the forward and copies during the drafter, with shadow host copies so evictions
> need no write-back (`EDGE0_TIER_EARLY=1`, ABBA +2.5% at c1, +2.8% at c8): `PRESET=single` makes 139.7 tok/s against
> Strata's 120.3 (re-measured this round) on the EOS load, decode TPOT 6.93 vs 7.63 ms, TTFT ~196 vs ~643 ms. New
> finding: H2D and D2H copies share host DRAM on this box (they do not overlap). 16.7 GiB of experts was neutral.
>
> **Status (agent `exceed-2`, 2026-11-20, TUNING-LOG "Single-stream" R2; section 11):** vLLM leads by 10.9%. The
> CPU-experts dispatch kernel no longer costs 11.6 us per layer (now ~1-2 us; ABBA +3.1%): `PRESET=single` makes
> 134.5 tok/s against Strata's 121.3 on the EOS load, decode TPOT 7.19 vs 7.61 ms, TTFT ~192 vs ~530 ms. Cold
> experts remain the one per-token gap (2.72 vs 1.72 ms); the replay points at a tier that makes swaps resident one
> step earlier.
>
> **Status (agent `exceed-1`, 2026-11-20, TUNING-LOG "Single-stream" R1; section 10):** vLLM leads. `PRESET=single`
> now also reads the PLE rows off the critical path (`EDGE0_PLE_DEFER=1`) and has 16.2 GiB of expert VRAM: 131.5 tok/s
> against Strata's 121.3 on the EOS load (+8.4%), decode TPOT 7.37 vs 7.61 ms, TTFT ~199 vs ~530 ms. The remaining
> gap to Strata inside the step is cold experts (2.45 vs 1.72 ms per token).
>
> **Status (agent `consolidate`, 2026-11-20, TUNING-LOG "Single-stream" C1):** the gap is closed. `PRESET=single`
> (levers 1 and 2 below: the request-reactive tier `EDGE0_TIER`, 15.7 GiB of expert VRAM, plus D1's dense knobs) makes
> 129.1 tok/s against Strata's 121.3 on this load in one interleaved session (TPOT 7.52 vs 8.25 ms; decode per token
> at parity, TTFT 205 vs ~530 ms). Strata's `sref` row below went through edge0-infer's relay, which vllm-bench
> mis-reads (C1 item 4); `bench/gap-strata.sh ref` now benchmarks Strata's engine server directly.

Agent `gap`, 2026-11-20. Both engines measured on the 5090 box under the same load: vllm-bench, ShareGPT seed 0, chat
API, thinking off, temperature 0, concurrency 1, 8 prompts per pass after a warm-up pass. Strata ignores `ignore_eos`,
so its replies end at EOS (about 800 tokens with `max_tokens` 1024); vLLM ran both the A/B protocol (512 tokens, EOS
ignored) and the same EOS-terminated load as Strata (`max_tokens` 1024, mean reply 881 tokens). vLLM config: the best
c1 setting of TUNING-LOG (tuned preset + `SPEC_METHOD=dflash2 SPEC_K=3` + `EDGE0_CPU_EXPERTS=1 EDGE0_CPU_THREADS=8
EDGE0_CPU_FRAC=0.55`). Strata config: `~/work/edge0-serve.sh` unchanged (MTP `--spec 4 --spec-min-p 0.5`,
`--expert-cache auto`, `--pool-workers 24 --pcie-mode dma --pcie-frac 0.1 --vram-reserve-mib 700 --kv-resident 32768`).

## 1. Answer in short

| c1, same 8 prompts | Strata | vLLM best | |
|---|---:|---:|---|
| output tok/s (bench) | **120.8** (ref path 113.1-124.2) | **93.3** (512, EOS ignored) / **94.2** (EOS) | Strata +28% |
| TPOT ms (bench) | **8.3** | **10.2** / **10.4** | +1.9..2.1 ms/token |
| decode ms per output token inside the engine | **7.72** | **10.2** | +2.5 ms/token |
| tokens per verify step (positions verified) | 2.29 (3.10) | 2.66 (4) | vLLM drafts better |
| **cold (non-VRAM) expert loads per output token** | **40** | **146** | **3.6x** |
| VRAM for experts | **18.67 GiB = 10,358 slots (42%)** | 12.80 GiB = 7,101 slots (29%) | +46% slots |
| expert-tier swaps per verify step | **15.7** (blocking, 96 per round every 4 windows) | 0.6 (64 per 3 s epoch) | 27x |

1. **The whole gap is cold experts.** Per verify step both engines stream the same Q8 dense weights (Strata 9.6 ms
   per window without its U stage, vLLM 9.4 ms per step including U), and per token vLLM's GPU path is even 1.1 ms
   cheaper because its kernels are as fast or faster and it gets 2.66 tokens per step instead of 2.29. But vLLM pays **4.5 ms per token for cold experts against Strata's 1.7**:
   it loads 146 cold experts per token, Strata 40. The price of one cold expert is not the problem (vLLM ~31 us per
   expert-layer with PCIe + 8 CPU threads, Strata 41 us with 24 CPU threads); the count is.
2. **Why Strata has 3.6x fewer cold experts: a request-reactive expert tier, plus 46% more VRAM for it.** Strata's
   adaptive tier adds 1 per routed entry, decays the counts by 0.7 every round and swaps up to 96 experts every 4
   verify windows (`--adapt-min-gain 1.5`), i.e. it follows the current request within tens of tokens. Measured on
   Strata itself: **turning it off (`--adapt-swaps 0`) drops Strata to 71.1 tok/s** (CPU experts per layer-window
   1.92 -> 9.16, 17.7 -> 31.3 ms per window), below vLLM. **With vLLM's VRAM budget (`--expert-cache 7101`) Strata
   still makes 103.1 tok/s** (3.89 cold per layer-window). So at Strata's VRAM the reactive tier is worth 1.70x
   (71.1 -> 120.8), and with the tier on the extra 5.9 GiB of expert VRAM is worth 1.17x (103.1 -> 120.8).
3. **vLLM's adaptive cache is effectively static within a request** (64 swaps per 3 s epoch with an EMA half-life of
   10 epochs = 30 s). A live probe with the existing knobs (`EDGE0_ADAPT_EPOCH_S=0.25 EDGE0_ADAPT_SWAPS=96
   EDGE0_ADAPT_HALFLIFE=2 EDGE0_ADAPT_MARGIN=0 EDGE0_ADAPT_Z=0.5`) cut cold experts per layer-step 8.1 -> 5.8 (-29%)
   and the EOS pass went 94.2 -> 99.2 tok/s, although each 0.25 s epoch costs 28-46 ms of host time (counter read
   with a stream sync + numpy + 96 swaps on the main stream), i.e. ~12% of the wall time. A GPU-side, sync-free
   version of Strata's policy is the first lever (section 7).
4. **Second-order: host idle and memory layout.** vLLM idles 1.6 ms per step waiting for PLE `preadv` after a full
   stream sync (0.6 ms per token); Strata's window shows no PLE time at all (O_DIRECT reads off the critical path).
   vLLM's non-expert VRAM is 18.6 GiB against Strata's 12.2 GiB: KV sized for 6 x 32k requests (2.43 GiB), a 1.42
   GiB activation reserve (Strata lends 3,350 expert slots to prefill instead), ~1.9 GiB left by GPU_UTIL 0.94, a
   1.25 GiB drafter (Strata's MTP 0.83 GiB).
5. **Strata's "expert cache 100.0% hit" is not a hit rate.** It prints 100.0% even with the adaptive tier off at
   71 tok/s and 9.16 CPU experts per layer-window: on this decode path the DONE counters see only VRAM entries
   (`serve/server.py` hit_rate = hits / lookups; the CPU pool's entries are counted in `multi_misses` /
   `multi_entries`, which only `STRATA_DECODE_TIMING=1` prints).

## 2. What was run

| label | engine, config | what it gives |
|---|---|---|
| `sref` | `~/work/edge0-serve.sh` (edge0-infer serve -> server.py -> strata), warm 4 + 8 prompts | the reference path; per-request history (`GET /metrics`) |
| `slog` | the same engine config (copied from edge0-infer's temp config), `serve/server.py` run directly with a `log` file, `STRATA_DECODE_TIMING=1` | Strata's own per-request decode timing line (window time split, CPU experts / VRAM hits / PCIe per layer-window, swaps per window) |
| `sprof` | `slog` + `STRATA_VERIFY_PROFILE=1` | Strata's GPU stage table per window (slows decode ~5%) |
| `snoad` | `slog` + `--adapt-swaps 0` | Strata with its adaptive tier off (static profile.bin tier) |
| `s7101` | `slog` + `--expert-cache 7101` | Strata with vLLM's 12.8 GiB expert budget |
| `vdf3cpu` | vLLM best c1 config + `EDGE0_TRACE=bw EDGE0_CPU_LOG_S=10`, warm 8 + 8 (512, EOS ignored) + 8 (EOS, 1024) | tok/s, acceptance (`/metrics`), cold / hot distinct experts per step, CPU split |
| `gapdf3w` | vLLM best c1 config under nsys (node-level graph trace, `EDGE0_NVTX=1`), captured after an 8-prompt warm pass | per-step GPU breakdown (150 steps) |
| `vfast` | `vdf3cpu` + fast adaptive knobs (above) | live probe of a request-reactive tier in vLLM |

All result files: `results/gap/` (bench summaries `gap-*.log`, Strata `GET /metrics` snapshots, parsed summaries
`strata-<label>.summary.json` / `vllm-<label>.summary.json`, Strata engine lines, the engine config, the nsys step
table `nsys-gapdf3w.json`). Bench JSONs: `5090:~/work/edge0-bench/results/{sref,slog,sprof,snoad,s7101,vdf3cpu,
vdf3cpu-eos,vfast,vfast-eos}/`.

| pass | tok/s | TPOT ms | mean reply | tokens per step | cold experts per layer-step |
|---|---:|---:|---:|---:|---:|
| Strata `sref` (ref path; warm pass) | 113.1 (124.2) | 8.2 (8.1) | 796 | - | - |
| Strata `slog` | **120.8** | **8.3** | 807 | 2.29 (window 3.10) | **1.92** (CPU) + 0.02 (PCIe) |
| Strata `sprof` (GPU profile on) | 116.5 | 8.6 | 822 | 2.32 | 2.05 |
| Strata `s7101` (vLLM's expert VRAM) | 103.1 | 9.7 | 868 | 2.31 | 3.89 + 0.10 |
| Strata `snoad` (adaptive tier off) | 71.1 | 14.0 | 842 | 2.32 | 9.16 + 0.52 |
| vLLM `vdf3cpu` (512, EOS ignored) | **93.3** | **10.2** | 512 | 2.66 (4) | **8.11** |
| vLLM `vdf3cpu-eos` (EOS, max 1024) | 94.2 | 10.4 | 881 | 2.71 | 8.05 |
| vLLM `vfast` (512) / `vfast-eos` | 93.3 / **99.2** | 10.2 / 9.9 | 512 / 932 | 2.67 / 2.77 | **5.79 / 5.61** |

(Strata's engine decode rate over the same 10 requests is 129.1 tok/s = 7.75 ms per token; the remaining 0.55 ms
per token of its TPOT is its Python server and the edge0-infer proxy. vLLM's engine-side generation rate over the
pass is 10.2-10.4 ms per token, the same as its TPOT. vLLM's cold counts come from `EDGE0_TRACE=bw` and include the
8 prefill steps of the pass, about +0.4 per layer-step.)

## 3. Per output token (ms): the gap table

Strata: its decode timing line (`slog`, 3,650 windows, 2.29 tokens per window), divided by tokens per window. vLLM:
the nsys step composition (`gapdf3w`, steady decode, 4-token verify steps) with the cold-expert time calibrated to the
bench average (27.1 ms per step = TPOT 10.2 x 2.66): the nsys window (25.0 ms per step, 2.68 tokens per step) was a
cheap stretch of the two warm-up prompts the adaptive cache had seen repeatedly (cold time 8.6 ms instead of 11.9),
and node-level tracing inflates `cudaGraphLaunch` (2.1 ms of idle per step; graph-level about 0.4 ms, PROFILE.md 3.3).

| per output token, ms | Strata | vLLM | vLLM - Strata | evidence |
|---|---:|---:|---:|---|
| GPU dense path + VRAM-resident experts + U / combine + LM head | 5.22 | 4.15 | **-1.07** | Strata: GPU-reach wait 11.26 + window tail 0.69 per window; vLLM: dense 9.38 + MoE bookkeeping 1.40 + other 0.25 per step |
| ...of which dense weight streaming per verify step (not per token) | 9.6 ms / window + U | 9.4 ms / step incl. U | vLLM faster | both stream the same Q8_0 weights (section 4) |
| **cold experts on the critical path** | **1.72** (CPU pool) | **4.47** (PCIe Marlin + CPU split) | **+2.75** | Strata: 1.92 per layer-window x 41 us; vLLM: 8.1 per layer-step x ~31 us |
| drafter (+ rejection sampling) | 0.46 (MTP) | 0.67 (DFlash2) | +0.21 | Strata draft 1.06 ms / window; vLLM 1.74 + 0.03 ms / step |
| expert-tier swaps | 0.24 (blocking, 15.7 swaps / window) | 0.10 (3 s epochs, ~30 ms each) | -0.14 | Strata adapt wait 0.54 ms / window; vLLM adapt log |
| host gaps: PLE reads, launch, stage, commit | 0.08 | 0.80 | **+0.72** | vLLM idle: `ple.read` 1.29 + `ple.stage` 0.35 + launch ~0.4 + rest 0.1 ms / step |
| **total, inside the engine** | **7.72** | **10.2** | **+2.5** | |
| bench TPOT | 8.3 | 10.2 / 10.4 (EOS) | +1.9 / +2.1 | Strata's server adds 0.55 ms / token |

Reading:
- **Cold experts explain more than the whole gap (+2.75 of +2.5 ms per token).** vLLM is better at everything that
  does not depend on the cold count: more tokens per step (2.66 vs 2.29) amortise the dense path, its per-step dense
  path is as fast as Strata's (D in b12x: 1.41 ms per step against ~2.5 ms of Strata's `iq_mmvq` D), and a cold expert
  costs it ~31 us (PCIe zero-copy and 8 CPU threads in parallel) against 41 us for Strata's 24 CPU threads alone.
- Cold expert loads per token: vLLM 8.11 x 48 / 2.66 = **146**, Strata 1.92 x 48 / 2.29 = **40**. Strata without its
  adaptive tier: 9.16 x 48 / 2.32 = 190; at vLLM's VRAM: 3.89 x 48 / 2.31 = 81.
- Strata's window time against its cold count is a straight line over the three Strata runs (window minus swap
  wait: 17.15 / 20.26 / 31.31 ms at 1.92 / 3.89 / 9.16 cold per layer-window): **41 us per cold expert per layer,
  13.0 ms per window with no cold expert**. vLLM's cold time with the CPU split is ~31 us per expert-layer
  (nsys: cold Marlin 8.43 + CPU wait 0.20 + dispatch 0.52 ms per step).

## 4. Inside one verify step: GPU work, side by side

Strata `sprof` (GPU stamps per window, 3.1 tokens, profile on), vLLM `gapdf3w` (exclusive GPU time per 4-token step):

| work per step | Strata ms / window | vLLM ms / step |
|---|---:|---:|
| attention: projections + GDN / QSA kernels | 2.93 | 2.36 + 0.60 = 2.96 |
| hyper-connections + norms | 1.75 + HC up (in the next row) | 1.45 |
| latent D (Strata: HC up + D + D2H of the latent rows + doorbell) | 3.10 | 1.41 |
| router (Strata: + plan + doorbell) | 0.55 | 0.22 |
| shared expert | 0.80 (runs while the CPU computes) | 0.94 |
| VRAM-resident experts | 1.43 + waits 0.59 | 0.15 exposed (6.1 hidden under the cold branch) |
| U + combine (+ Strata: the wait for the CPU rows) | 5.33 (U and combine of all rows, and the wait for the CPU rows; not split by the profile) | 1.39 + 0.12 |
| cold experts | on the CPU (above) | **8.43** cold Marlin + **0.20** CPU wait |
| MoE route / gather / dispatch | in "router" | 0.99 (dispatch kernel 0.52) + 0.14 |
| LM head + sampler | 0.44 | 0.76 |
| drafter | 1.06 (host-timed) | 1.74 |
| GPU idle | ~0 (window timing: 0.04 ms other) | 3.82 (PLE 1.64, launch 2.08 node-traced, adapt 0.05) |

Strata's per-layer critical path: GPU work up to the doorbell (HC, attention, HC, D, router: ~174 us per layer from the
profile) -> the CPU pool computes the missed experts (~82 us per layer at 1.92 per layer; the GPU meanwhile runs the
shared expert and the VRAM hits) -> the GPU applies U and combines (~61 us per layer: 11.26 ms / 48 - 174 us) -> next
layer. Its decode timing line shows it: the host waits 11.26 ms per window for the GPU and works 3.94 ms per window
on CPU experts; the CPU part grows 1:1 with the cold count (18.25 ms at 9.16 cold per layer-window with the tier off)
while the GPU part stays at 10.6-11.3 ms. Per verify step, vLLM's GPU work is smaller (dense 9.4 ms incl. U against
Strata's 9.6 ms + U); its cold branch (8.4 ms) is what Strata does not have.

## 5. Memory layout and hit rates

| | Strata | vLLM best c1 |
|---|---|---|
| VRAM for experts | **18.67 GiB = 10,358 slots** (42% of 24,576), pre-filled from `profile.bin` in 0.6 s | **12.80 GiB = 7,101 slots** (29%), `plan-s1-12.8.json` |
| how the size is chosen | `--expert-cache auto`: all free VRAM after the model (19.88 GiB) minus a 700 MiB reserve, the draft head and 256 MiB | `EDGE0_EXPERT_VRAM_GIB=12.8`, KV from what GPU_UTIL 0.94 leaves |
| dense + embedding + runtime | ~12.2 GiB in all (dense Q8 skeleton + D/U Q8, embedding 644 MiB in VRAM, MTP 835 MiB, graphs), 448 MiB free | dense + drafter 11.56 GiB (drafter 1.25), non-torch 1.26 GiB, activation peak 1.42 GiB, CUDA graphs 0.42 GiB, ~1.9 GiB unused (GPU_UTIL 0.94) |
| KV | int8; 32,768 of 131,072 cells per QSA layer in VRAM, the rest streamed from 1.55 GiB of pinned RAM (99.83% of block reads hit VRAM here) | bf16 paged, 2.43 GiB = 61,067 tokens (1.86 x 32k requests at k=3) |
| prompt buffers | borrows 3,350 expert slots (6.04 GiB) during a prompt, refills them after | a permanent 1.42 GiB activation peak (`MAX_BATCHED` 2048) |
| host RAM | 45.4 GB pinned arena with **all** experts (an eviction costs nothing), KV 1.55 GiB | 31.5 GiB pinned cold pool (cold experts only; a swap writes the evicted expert back) |
| expert tier policy | +1 per routed entry, x0.7 per round, every 4 windows <= 96 swaps where cand >= victim + 1.5; blocking (0.54 ms per window), async tier available but not engaged | 3 s epochs, EMA half-life 10 epochs, x > 1.25 y and z-test, <= 64 swaps per epoch |
| swaps per output token | 6.9 | 0.2 |
| cold experts per layer per verify step | **1.92** (2.33 routed entries) + 0.02 over PCIe | **8.11** (+ 19.2 hot distinct) |
| routed entries served from VRAM | **92.5%** (2.33 of 31.0 entries per layer-window on the CPU) | ~71% (8.1 cold x 1.44 routes per cold expert of 40) |
| distinct experts served from VRAM | - | 70.3% (`EDGE0_TRACE=bw`), adaptive log: cold share of touches 30% |

## 6. Strata's mechanisms, measured

- **Adaptive tier, A/B on Strata itself** (same prompts, same binary): on 120.8 tok/s / 17.69 ms per window / 1.92
  CPU experts per layer-window; off (`--adapt-swaps 0`, static profile tier) 71.1 tok/s / 31.31 ms / 9.16. At vLLM's
  VRAM (`--expert-cache 7101`, adaptive on) 103.1 tok/s / 21.12 ms / 3.89, 22.9 swaps and 0.86 ms of blocking swap
  wait per window.
- **How fast it reacts within one request** (Strata keeps no per-window log, so this is the replay of section 8, which
  reproduces Strata's measured cold counts within 2%): cold experts per layer-window by position in a 512-token
  request, 7,101 slots, 4-token windows: static plan 8.6 / 7.9 / 7.4 / 9.8 / 9.1 / 11.2 for tokens <16 / <32 / <64 /
  <128 / <256 / <512; Strata's policy 8.4 / 6.7 / 6.2 / 5.9 / 4.5 / 5.0. It is ~15% below static from token 16 on,
  ~40% from token 64 and 50-55% from ~128 tokens on. vLLM's served policy stays within 5-10% of static.
- **CPU pool**: 24 workers on 15 physical cores (Strata itself warns: 16 cores, "Recommended: --pool-workers 15") +
  the host thread; 41 us per cold expert per layer on the critical path (DRAM-bound); `--pcie-frac 0.1` moves only
  0.02 experts per layer-window to PCIe at this miss rate (0.52 with the tier off).
- **Verify windows**: MTP `--spec 4` with `--spec-min-p 0.5` (and prompt lookup, `--suffix-draft 3`): windows of 1-6
  tokens (graphs captured for 1/2/3/4/6), 3.10 positions on average, 2.29 tokens per window, 61% of drafts accepted.
  vLLM DFlash2 k=3: always 4 positions, 2.66 tokens per step, per position 0.77 / 0.54 / 0.36.
- **Drafter**: MTP 1.06 ms per window (835 MiB); DFlash2 1.74 ms per step (1.25 GiB).
- **PLE**: `--ple-io direct` (O_DIRECT, probe 32,751 rows/s). No PLE term in the window: the timing line accounts for
  99.9% of the window without one. vLLM: 1.64 ms of GPU idle per step in `ple.read` / `ple.stage`.
- **Dense path**: equal per verify step (9.6 vs 9.4 ms). Strata's latent D is slower (HC up + D 3.10 ms per window
  vs vLLM's D 1.41 ms per step), its attention projections are as fast, its router and shared expert hide under the
  CPU pool's time.

## 7. Levers, ranked by expected ms saved per output token (vLLM at c1, from 10.2 ms per token)

| # | lever | expected | evidence |
|---|---|---:|---|
| 1 | **Request-reactive expert tier** (Strata's policy: per-entry counts, decay 0.7 per round, <= ~96 swaps every ~4 steps, gain >= 1.5) done without a host sync: counts and pair selection on the GPU (or read one step late from mapped memory), swaps as copies on a side stream between steps, evicted experts written back on the D2H direction (or kept: a full pinned copy is 44.3 GiB) | **-1.0 to -1.4** | Strata A/B: tier off 31.3 -> on 17.7 ms per window (-44%); replay: vLLM 9.3 -> 5.25 cold per layer-step at 7,101 slots; live probe with the existing knobs: 8.1 -> 5.8 (-29%) and EOS 94.2 -> 99.2 tok/s despite ~12% host epoch overhead. 2.3-2.9 fewer cold per layer-step x 48 x ~31 us = 3.4-4.3 ms per step, minus ~0.3-0.9 ms per step of swap traffic (24 x 1.94 MB) if not overlapped |
| 2 | **More VRAM for experts in a c1-only preset** (KV for 1-2 requests, `MAX_BATCHED` 1024 or lend expert slots to prefill as Strata does, GPU_UTIL ~0.97, MTP instead of DFlash2 saves 0.4 GiB): +2.5-3 GiB = +1,400-1,650 slots | **-0.6 to -0.9** | Strata 7,101 -> 10,358 slots: 21.1 -> 17.7 ms per window (-1.5 ms per token); replay with lever 1: 5.25 -> 3.85 cold per layer-step (+1,478 slots), without it 9.26 -> 7.54 |
| 3 | **PLE rows off the critical path** (`EDGE0_PLE_IO=threads`, never measured; or read the drafts' rows during `propose`; no full stream sync) | **-0.4 to -0.6** | nsys: 1.64 ms idle per step (`ple.read` 1.29, `ple.stage` 0.35); Strata has none |
| 4 | **Re-tune the draft length once cold experts are cheaper** (k=4-5, or confidence-trimmed windows like Strata's 3.1 positions) | -0.2 to -0.5 | each extra position costs ~3.3 ms per step today, mostly cold experts (TUNING-LOG S2: k=5 already +2% at c1); levers 1-2 roughly halve that |
| 5 | **CPU-experts dispatch kernel** (1 CTA, 11 us per call x 48): more CTAs or fold into the gather | -0.15 | nsys: `dispatch_kernel` 0.52 ms per step |
| 6 | drafter cost (DFlash2 1.74 ms per step vs MTP 1.06 per window) | -0.1 | DFlash2 keeps the better acceptance (2.66 vs 2.29); not worth trading |
| - | not levers: dense path (equal per step, more tokens per step in vLLM), cost per cold expert (vLLM 31 us < Strata 41 us), a Strata-style CPU-only split (Strata moves 0.02 experts per layer-window to PCIe), cold-call CTA caps | 0 | sections 3-4 |

Levers 1 + 2 + 3 together: about 10.2 -> 7.5-8.2 ms per token inside the engine, i.e. Strata's 7.7 (bench 8.3).
Levers 1 and 2 also help c8 (fewer cold experts per step); lever 2 must stay a separate preset because it gives up
KV capacity.

## 8. The replay that predicts the cold counts (`tools/sim_tier.py`)

Replays the c1 decode routing of the seed-0 benchmark prompts (`EDGE0_TRACE=routes`, 2,556 tokens, TUNING-LOG 3)
through expert-tier policies, from the plan's hot set and per-layer budgets: verify windows cover the next k+1 tokens'
routes and advance by an accepted length drawn from the per-position acceptance. Policies: `static`, `vllm` (the served
EDGE0_ADAPT rule), `strata` (Strata's `adapt` lambda in `generate.cpp`: usage += 1 per routed entry, every 4 windows
per layer the non-resident experts with usage >= 2 paired against the least-used residents, swap while cand >= vict +
1.5, <= 96 per round, then usage x 0.7), `lru` (copy on miss over the whole budget).

**Validation against Strata's own measurements** (Strata-like 3-token windows): 7,101 slots: replay 3.94 cold per
layer-window and 23.8 swaps per window, measured 3.89 and 22.9; 10,337 slots: replay 1.91 and 18.8, measured 1.92
and 15.7.

vLLM's 4-token verify steps (k=3, 2.70 tokens per step):

| slots | static | vllm (served) | strata | lru (bound) |
|---:|---:|---:|---:|---:|
| 7,101 (now) | 10.08 | 9.26 | **5.25** (23.9 swaps / step) | 2.95 |
| 8,579 (+1,478) | 8.28 | 7.54 | **3.85** | 2.10 |
| 9,077 (+1,976) | 7.70 | 6.99 | 3.46 | 1.87 |
| 10,337 (Strata's) | 6.45 | 5.79 | 2.61 | 1.40 |
| 11,075 | 5.83 | 5.16 | 2.22 | 1.18 |

Single-token steps (no speculation): 3.66 / 3.22 / **1.42** / 1.09 at 7,101 slots. (The live vLLM count, 8.1, is
below the replay's 9.3 because the A/B prompts repeat every pass and the served cache has learned them; the live
`vfast` probe, 5.8 with 9-step epochs, sits next to the replay's 5.25 with 4-step rounds.)

## 9. Reproduce

```
# on the 5090 host, holding the GPU lock; vLLM stopped for the Strata runs and the other way round
cd ~/work/edge0-vllm
bash bench/gap-strata.sh ref sref                                   # edge0-serve.sh as is; saves the engine config
bash bench/gap-strata.sh log slog STRATA_DECODE_TIMING=1            # server.py direct, engine log kept
bash bench/gap-strata.sh log sprof "STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1"
bash bench/gap-strata.sh log snoad STRATA_DECODE_TIMING=1 --adapt-swaps 0
bash bench/gap-strata.sh log s7101 STRATA_DECODE_TIMING=1 --expert-cache 7101
python3 tools/gap_parse.py strata logs/gap-strata-slog.engine.log --last 10 --metrics results/gap/slog.metrics.json
bash bench/gap-vllm.sh vdf3cpu "EDGE0_DEV=1 SPEC_METHOD=dflash2 SPEC_K=3 EDGE0_CPU_EXPERTS=1 EDGE0_CPU_THREADS=8 EDGE0_CPU_FRAC=0.55 EDGE0_TRACE=bw EDGE0_CPU_LOG_S=10"
python3 tools/gap_parse.py vllm <serve log> <l0> <l1>               # the SLOG line of each pass
bash bench/gap-nsys.sh gapdf3 "EDGE0_DEV=1 EDGE0_NVTX=1 SPEC_METHOD=dflash2 SPEC_K=3 EDGE0_CPU_EXPERTS=1 EDGE0_CPU_THREADS=8 EDGE0_CPU_FRAC=0.55"
python3 tools/nsys_breakdown.py steps profiles/gapdf3w/c1-k1.sqlite  # now knows the CPU-experts kernels (cpu_wait)
python3 tools/sim_tier.py --trace '<traces>/routes-*.npz' --from-ts 1795078406 --to-ts 1795078500 \
    --plan plans/plan-s1-12.8.json --extra-slots 0,1500,3257 --k 3 --curve     # --k 2 --acc .66,.42,.25: Strata-like windows
```

`bench/gap-strata.sh floor` (Strata's `--gpu-only-full`) was written but not run: the GPU stage profile answered the
dense-path question.

## 10. Round 1 beyond parity (agent `exceed-1`, 2026-11-20): where each output token goes now

`PRESET=single` as C1 left it (DFlash2 k=3, CPU experts, reactive tier, 15.7 GiB of experts, latent D/U group skip,
fused shared expert, Q8 head), profiled at c1 after an 8-prompt warm pass (`bench/gap-nsys.sh`; node-level capture
`profiles/r1singlew/` for the kernel split, graph-level capture `profiles/g1basew/` for the true step period and the
idle time; `results/nsys-{r1singlew,g1basew}.json`, kernel table `results/nsys-kernels-r1singlew.txt`). Steps are
4-token verify steps; per token = per step / 2.70 (the measured acceptance of the 512-token passes, 2.60-2.78).

| per output token, ms | Strata (G1) | vLLM C1 config | vLLM - Strata | per verify step (vLLM) |
|---|---:|---:|---:|---|
| cold experts on the critical path | 1.72 | **2.45** | **+0.73** | cold Marlin 5.94 + CPU-experts wait 0.11 + dispatch kernel 0.56 = 6.61 ms (3.1-4.3 cold per layer-step, ~38 us each) |
| attention (projections, GDN, QSA) | | 1.10 | | 2.36 + 0.42 + 0.18 |
| latent D + U + shared expert | | 1.23 | | D 1.26 (Triton GEMV, group skip), U + shared 2.05 |
| hyper-connections + norms | | 0.53 | | 1.44 |
| router + top-k + routing / tier bookkeeping kernels | | 0.45 | | 0.22 + 0.99 (topk, tier counts, small elementwise) |
| hot experts (exposed) + gather + combine | | 0.21 | | 0.31 (4.4 hidden under the cold branch) + 0.16 + 0.11 |
| LM head + sampling + rejection | | 0.17 | | 0.42 + 0.03 |
| other GPU (memsets, small copies) | | 0.08 | | 0.21 |
| **subtotal: dense path + resident experts + head** | **5.22** | **3.77** | **-1.45** | 10.2 ms |
| drafter | 0.46 (MTP) | 0.52 (DFlash2) | +0.06 | 1.39 |
| expert-tier swaps | 0.24 (blocking) | 0 (copy engines, overlapped) | -0.24 | 25-38 swaps per step |
| GPU idle: PLE rows + launch | 0.08 | **0.38** | **+0.30** | graph-level: `ple.read` 0.53 + `ple.stage` 0.18 + launch 0.32 = 1.03 ms |
| **total, inside the engine** | **7.72** | **7.23** (period 19.52 ms / step) | **-0.49** | |
| bench decode TPOT (C1, EOS load) | 7.61 | 7.52 | | |

- vLLM is now ahead on the dense path by 1.45 ms per token (more tokens per verify step, faster D/U and head) and
  behind on exactly two things: **cold experts (+0.73 ms per token: 3.1-4.3 cold per layer-step x 48 / 2.70 = 55-76
  cold expert loads per token against Strata's 40)** and **host idle (+0.30: the PLE row reads behind a stream sync, and the graph launch)**.
- The PLE idle here (0.70 ms per step) is the warm-bench value: the scored pass repeats the warm pass's prompts, so
  many of its rows are in the page cache. Rows of new text cost more: the reader measured 0.9-1.5 ms per step for
  ~64 rows (the drive's ~46k IOPS), and a 1,024-token prefill chunk needs 16,384 rows = 0.35 s.
- Inside the cold branch, the CPU-experts dispatch kernel (1 CTA, 11.6 us x 48) is 8% of the cold time.

**Round-1 levers taken** (TUNING-LOG "Single-stream" R1): (1) PLE rows off the critical path (`EDGE0_PLE_DEFER=1`): the
step's graph is launched right after the stream sync and the rows are read by a C++ thread while the GPU runs layer 0;
the PLE layer (decoder layer 1) waits for them inside the graph. (2) More expert VRAM (16.2 GiB). Results in R1:

| | C1 config | + `EDGE0_PLE_DEFER=1` | + 16.2 GiB (= round-1 `PRESET=single`) |
|---|---:|---:|---:|
| c1, 512 tokens EOS ignored, ABBA means (tok/s) | 117.6 | 122.5 (+4.2%, `r1ple`) | 124.8 (+1.9%, `r1vram`) |
| c8 (tok/s) | 172.8 | 182.2 (+5.4%) | 191.2 (+5.1%) |
| c1 EOS load vs Strata (tok/s; decode TPOT ms) | 129.1; 7.52 (C1) | | **131.5; 7.37** (Strata 121.3; 7.61) |

The first PLE version without the stream sync was slower (113 tok/s): the host ran ahead, then blocked before its
output processing and scheduling, which landed on the critical path (graph-level nsys `g1deferw`: GPU idle 1.03 ->
2.10 ms per step). Keeping vLLM's sync at the same place and only moving the read behind the launch is what gains. The
GPU still waits 0.4-0.9 ms per step in layer 1 for rows of new text (the drive's ~46k IOPS): the bonus token's rows
during the drafter, or a faster drive, are the next PLE steps. A tier swap cap of 96 (`EDGE0_TIER_SWAPS=96`) was
neutral (rejected).

## 11. Round 2 (agent `exceed-2`, 2026-11-20): where each output token goes now

`PRESET=single` as round 1 left it (DFlash2 k=3, CPU experts, reactive tier, 16.2 GiB of experts, D/U group skip,
fused shared expert, Q8 head, PLE rows deferred), profiled at c1 after an 8-prompt warm pass (`bench/exceed2.sh prof`):
node-level capture `profiles/r2singlew/` (`EDGE0_NVTX=1 EDGE0_TRACE=bw`; kernel split, `results/nsys-r2singlew.json`,
kernel table `results/nsys-kernels-r2singlew.txt`) and graph-level capture `profiles/g2singlew/` (`EDGE0_NVTX=1`; true
step period and idle, `results/nsys-g2singlew.json`). 145 four-token verify steps each. During the node-level capture
the tier logged 3.48 cold experts per layer-step (the scored passes of this round: 3.0-3.9). Per token = per step /
2.65 (tokens per step of this round's 512-token passes, 2.59-2.70).

| per output token, ms | Strata (G1) | vLLM R1 `PRESET=single` | vLLM - Strata | per verify step (vLLM) |
|---|---:|---:|---:|---|
| cold experts on the critical path | 1.72 | **2.72** | **+1.00** | cold Marlin 6.52 + CPU-experts wait 0.13 + dispatch kernel 0.56 = 7.21 ms (3.5 cold per layer-step = 168 cold expert-layers per step = 63 per token, Strata 40; 43 us each) |
| attention (projections, GDN, QSA) | | 1.12 | | 2.36 + 0.42 + 0.18 |
| latent D + U + shared expert | | 1.27 | | D 1.25 (Triton GEMV, group skip) + U and shared 2.11 |
| hyper-connections + norms | | 0.54 | | 1.44 |
| router + top-k + routing / tier bookkeeping | | 0.46 | | 0.22 + 1.00 (of which ~0.2 are `EDGE0_TRACE=bw`'s own counters, not in production) |
| hot experts (exposed) + gather + combine | | 0.22 | | 0.28 (4.97 hidden under the cold branch) + 0.17 + 0.12 |
| LM head + sampling + rejection | | 0.17 | | 0.41 + 0.03 |
| other GPU (memsets, small copies) | | 0.08 | | 0.22 |
| **subtotal: dense path + resident experts + head** | **5.22** | **3.86** | **-1.36** | 10.22 ms |
| drafter | 0.46 (MTP) | 0.52 (DFlash2) | +0.06 | 1.38 (its Q8 LM head alone 0.41) |
| expert-tier swaps | 0.24 (blocking) | 0 on the GPU | -0.24 | copy engines: 25-35 swaps = 50-70 MB H2D per step, sharing the PCIe link with the cold Marlin reads |
| PLE rows: in-graph wait in layer 1 | | 0.14 | | 0.33-0.39 ms per step (`edge0 ple-defer` of the scored passes: GPU waited in 36-40% of steps; 0.5-0.6 ms on new text) |
| GPU idle: launch + PLE staging | 0.08 | 0.15 | +0.07 | graph-level: 0.41 ms (`graph.full` 0.35 + `ple.stage` 0.05; R1 before the PLE defer: 1.03) |
| **total, inside the engine** | **7.72** | **7.39** | **-0.33** | 19.59 ms (graph-level period 19.02 mean, 18.38 median) |
| bench decode TPOT (EOS load) | 7.61 | 7.37 (R1) | | |

- **Cold experts are the whole remaining gap, and more than all of it**: +1.00 ms per token, against a dense path that
  is 1.36 ms per token faster than Strata's. 63 cold expert loads per token against Strata's 40.
- **The CPU-experts dispatch kernel is a fixed 11.6 us per layer** (0.56 ms per step, 0.21 ms per token), paid on every
  call, although at c1 only ~20% of the MoE calls give the CPU a share (`edge0 cpu-experts`: ~15.5k jobs per ~1,900
  steps x 48 layers; `EDGE0_CPU_MIN_COLD=4` with 3-4 cold experts per layer-step). In isolation
  (`tools/dispatch_bench.py`, a graph of 48 calls) it costs 8.7 us without a CPU share: all 512 threads run
  `__threadfence_system` and read `frac` from mapped memory. Round-2 lever 1.
- The routing tail of each MoE call (skip_pad's two `where` + `zeros_like`, the tier's `full_like` + `where` +
  `tier_count`) is six one-CTA kernels: 5.1 us per call in a graph, 1.15 us fused (`EDGE0_ROUTE_FUSE=1`,
  `kernels.route_post`, bit-identical): 0.19 ms per step.
- The swaps' 50-70 MB per step share the PCIe link with the cold Marlin's ~330 MB per step of zero-copy reads, and
  they land on the coldest layers: the tier issues them at the start of the step, so they run during layers 1-12,
  while the per-layer cold Marlin time is 200-234 us in layers 0-2 against 45-140 us in most later layers (node-level
  capture; the hot set covers early layers worst: 9-14 cold experts per step in L0-L4, ~3.5 on average).

**What the replay says about the tier** (`tools/sim_tier.py`, the c1 seed-0 routing, 16.2 GiB plan, k=3; cold experts
per layer-window, the served policy = 3.12):

| variant | cold / layer-window | swaps / window |
|---|---:|---:|
| as served (`EDGE0_TIER`: decided from step N-1's counts, copied during step N, resident from N+1) | 3.12 | 30 |
| swap cap 16 / 24 / 32 per step | 3.85 / 3.53 / 3.31 | 16 / 22 / 26 |
| gain 3 (half the swaps) | 3.32 | 18 |
| free slots per layer 8, cap 96 | 3.02 | 33 |
| decay 0.85 / 0.95 | 3.15 / 3.09 | 27 / 33 |
| **resident one window earlier** (counts of step N acted on during step N, resident from N+1; Strata's blocking tier does this) | **2.85** | 30 |
| ... and swap every touched cold expert (min usage 1, gain 0.5, up to 12 per layer) | **1.83** | 83 |
| LRU over the whole budget (copy on miss) | 1.91 | 92 |
| more VRAM: 16.7 / 17.0 GiB (as served) | 2.93 / 2.82 | 29 / 28 |

The knobs of the current policy move the count by a few per cent; what matters is **when** a swapped expert becomes
resident. A tier that reads each layer's counts while the step is still running (mapped counters, a host thread
deciding per layer as they arrive), copies during the same step and flips before the next launch would cut cold
experts by 9% with today's rule and by ~40% with an LRU-like rule, at the same total PCIe bytes (each miss read once by
Marlin and once by the copy), provided the copies overlap the dense parts of the step rather than the cold Marlin calls.

**Round-2 results** (TUNING-LOG "Single-stream" R2):

| change | c1 ABBA (512 tokens) | c8 | decision |
|---|---:|---:|---|
| CPU-experts dispatch kernel, round 2 (8.7 -> 1.1 us per call without a CPU share) | **+3.1%** (`r2disp`) | 189.6 / 191.7 / 196.4 (R1: 187-191) | **kept** (code, always on) |
| `EDGE0_ROUTE_FUSE=1` (routing tail 5.1 -> 1.15 us per call) | +1.4% (`r2fuse`) | -2.1% | knob, off (under the 3% bar) |
| `EDGE0_CPU_MIN_COLD=2`, frac 0.45 | neutral per step (probes) | | defaults kept |
| `SPEC_K=4` | -2.4% (probe: +9% tokens per step, +12% step time) | | rejected |
| 17.0 GiB of experts | not served: the drafter's constructor allocates a BF16 embedding and LM head (2 x 1.19 GiB) before vLLM shares the target's; a 1.64 GiB load transient of the Marlin scale factor was fixed on the way | | next lever |

Fair c1 comparison of the round-2 `PRESET=single` (EOS-stopping, same 8 prompts; Strata C1's passes reused): **vLLM 134.6 /
134.4 tok/s, decode TPOT 7.18 / 7.20 ms, against Strata's 120.4 / 122.2 tok/s, 7.67 / 7.55 ms: +10.9%, -5.5% per
token** (R1: 131.5 tok/s, 7.37 ms). Next levers, ranked: TUNING-LOG R2 item 8.

## 12. Round 3 (agent `exceed-3`, 2026-11-20): where each output token goes now

`PRESET=single` as round 2 left it (image `d0aa833bb901`; PLE table on the system disk, page cache dropped by the
coordinator's PLE-move A/B just before), profiled at c1 after an 8-prompt warm pass (`bench/exceed3.sh prof`):
node-level capture `profiles/r3singlew/` (`EDGE0_NVTX=1 EDGE0_TRACE=bw`; `results/nsys-r3singlew.{json,txt}`, kernel
table `results/nsys-kernels-r3singlew.txt`) and graph-level capture `profiles/g3singlew/` (`results/nsys-g3singlew.*`),
145 four-token verify steps each. The tier logged 2.97 cold experts per layer-step during the node-level capture.
Per token = per step / 2.65.

| per output token, ms | Strata (G1) | vLLM R2 `PRESET=single` | vLLM - Strata | per verify step (vLLM) |
|---|---:|---:|---:|---|
| cold experts on the critical path | 1.72 | **2.26** | **+0.54** | cold Marlin 5.79 + CPU-experts wait 0.11 + dispatch kernel 0.10 = 6.00 ms (2.97 cold per layer-step = 143 cold expert-layers per step = 54 per token, Strata 40; 42 us each) |
| attention (projections, GDN, QSA) | | 1.12 | | 2.36 + 0.42 + 0.18 |
| latent D + U + shared expert | | 1.27 | | D 1.26 + U 0.03 + shared 2.09 |
| hyper-connections + norms | | 0.54 | | 1.44 |
| router + top-k + routing / tier bookkeeping | | 0.45 | | 0.22 + 0.98 (incl. ~0.2 of `EDGE0_TRACE=bw` counters) |
| hot experts (exposed) + gather + combine | | 0.20 | | 0.31 (4.47 hidden under the cold branch) + 0.11 + 0.11 |
| LM head + sampling + rejection | | 0.17 | | 0.42 + 0.04 |
| other GPU (memsets, small copies) | | 0.08 | | 0.22 |
| **subtotal: dense path + resident experts + head** | **5.22** | **3.84** | **-1.38** | 10.17 ms |
| drafter | 0.46 (MTP) | 0.52 (DFlash2) | +0.06 | 1.39 |
| expert-tier swaps | 0.24 (blocking) | 0 on the GPU, but see below | -0.24 | copy engines during layers 1-12: 41.5 MB H2D + as much D2H per step (memcpy kernels 1.95 + 1.93 ms) |
| PLE rows: in-graph wait in layer 1 + GPU idle (launch, staging) | 0.08 | 0.26 | +0.18 | PLE wait ~0.24 ms per step in the scored passes (47% of steps, 0.51 ms each; page cache cold after the drop) + graph-level idle 0.44 |
| **total, inside the engine** | **7.72** | **6.94** (graph-level period 18.39 ms, median 17.88) | **-0.78** | |

**New finding: an H2D and a D2H copy do not overlap on this box** (`tools/tier_copy_bench.py`,
`logs/exceed3-copybench.txt`; host_register'd pinned memory, the tier's four arrays per expert):

| 32 experts (61.9 MB) per direction | cudart, one call per array | `cudaMemcpyBatchAsync` | one contiguous copy |
|---|---:|---:|---:|
| H2D only: GPU ms (GB/s) | 1.30 (47.5) | 1.18 (52.4) | 1.07 (57.7) |
| H2D and D2H on two streams at once: GPU ms | 2.10 | 2.10 | 2.01 |
| host time to issue the H2D (ms) | 0.20 | 0.05 | 0.01 |

The two directions share one limit (host DRAM: dual-channel DDR5 serves ~62 GB/s of copies in total), so the served
tier's victim write-backs (D2H, one per swap) cost as much as its swaps, and both ran during the forward's first
layers, next to the cold Marlin reads that need the same DRAM. A concurrent GPU compute load does not slow the copies.

**Round-3 lever: the tier decides right after the forward and copies in the drafter window** (`EDGE0_TIER_EARLY=1`,
`EDGE0_TIER_COPY=batch`, shadow host copies with `EDGE0_TIER_SPARE=64`; TUNING-LOG "Single-stream" R3 item 3). The
decision waits for the forward's counts (not for the drafter), the copies run during the LM head, sampler and drafter
(~2 ms with no cold reads), and the flip is ordered before the next launch, so a step's counts place experts from the
next step on (the replay's "resident one window earlier"). Because the two copy directions share host DRAM, the
victims' write-backs had to go: a swapped-in expert keeps its host slot as a shadow and is evicted back to it with no
D2H. What it took to fit the window (probes, c1 tok/s): served tier 124.7 -> early 126.7 (flip wait 0.7-1.0 ms per
step) -> + shadows 130.5 -> + 64 spare host slots per layer 129.0 (write-backs 98% -> ~45% of swaps) -> + the H2D batch
issued before the victims' bookkeeping **134.1** (flip wait 0.04-0.4 ms). A larger swap cap with a looser rule (replay:
-10% cold) made it worse (122.0: 2 x 100 MB do not fit the window).

| change | c1 ABBA (512 tokens) | c8 | decision |
|---|---:|---:|---|
| early tier + batch copies + shadows (64 spare host slots per layer) | **+2.5%** (`r3early`: 125.5 / 129.4 -> 129.7 / 131.5, both B above both A) | +2.8% (189.8 -> 195.1) | **kept** (`PRESET=single`) |
| 16.7 GiB of experts + KV 1.35 GiB + drafter vocab placeholders (`EDGE0_DFLASH_SHARE_VOCAB=1`) | +0.2% (`r3vram`) | -1.0% | rejected (knob kept: it is what lets > 16.2 GiB load; 17.0 GiB runs out of memory in the first prefill) |
| decision on the GPU (`EDGE0_TIER_GPU_DECIDE`), write-back cost 0.5 (`EDGE0_TIER_WB_COST`) | probes 129.9 / 130.8 vs 134.1 | | knobs, off |

Cold experts per layer-step do not drop measurably at this resolution (A 2.7-4.0, B 2.9-3.8 per 10 s window); the
gain is the forward without swap traffic next to its cold reads (copies moved to the drafter window) and a shorter PLE
wait (rows read while no swap competes for host DRAM: GPU waited in 29-30% of steps, 0.22 ms, vs 39-51%, 0.44-0.55
ms). Gate (image `d0d3bace6b03`): teacher-forced top-1 = Strata argmax 257/261 (Strata's token 250/261), 8/8 prompts;
greedy with vs without drafter 3/8 identical, 4 near-ties, 1 split after EOS; 23k-token prompt 12.6 s.

**Fair c1 comparison, round 3** (EOS-stopping, same 8 prompts; Strata re-run once on the system disk because the box
changed): **vLLM 136.3 / 143.1 tok/s (mean 139.7), decode TPOT 7.11 / 6.75 ms, against Strata's 120.3 tok/s, 7.63 ms:
+16.1%, -9.2% per token**, TTFT ~196 vs ~643 ms (R2: 134.5 tok/s, 7.19 ms). Next levers: TUNING-LOG R3 item 7 (the
remaining flip wait, write-backs, a fused routing kernel, a smaller prefill working set for more expert VRAM).

