# CPU computation of cold experts inside vLLM (`EDGE0_CPU_EXPERTS=1`)

Agent `cpu-experts`, 2026-11-20. User request: bring Strata's parallel CPU computation of cold experts into the
vLLM port. This file is the design, written before the code; measurements and decisions are in TUNING-LOG.md
section "CPU cold experts (user request)".

## 1. Why, and the ceiling

PROFILE.md: cold experts read zero-copy over PCIe take 24 / 74 / 86% of a decode step at c1 / c8 / c32 and 55% of a
DFlash2 k=3 verify step at c1. The link is at its ~54 GB/s ceiling during those phases. The CPU can compute an
expert from the same pinned host bytes: DDR5 gives the CPU ~62 GB/s alone, and the CPU and GPU together reach
~73 GB/s (GPU 31 + CPU 42 GB/s). So a split of each MoE call's cold experts between PCIe and the CPU can cut cold
time by at most ~26-28%; the ceiling is DRAM, not CPU FLOPs. Upper bounds: +23% at c8, +30% at c32, +13% for a
k=3 verify at c1, ~0 for plain c1 (2 cold experts per layer cannot be split usefully).

## 2. What Strata does (src/kernels/cpu, src/core, src/kernels/cuda; MIT)

- `ExpertPool` (`pool.cpp`): N pinned workers (one per physical core, the first core left to the host thread),
  busy-polling an epoch cache line with `_mm_pause`, then parking after `STRATA_POOL_SPIN_US`. Tasks are rows of
  experts, grabbed in order.
- GPU -> CPU: a 1-thread "doorbell" kernel increments a sequence number in mapped pinned memory after a
  `__threadfence_system()` (the increment is read from memory, never a captured literal, so CUDA-graph replays
  ring new values). CPU -> GPU: the pool raises a flag in mapped memory; a 1-thread spin kernel (or the first
  thread of the merge kernel) spins until `flag == seq`, then `__threadfence_system()`, then reads the CPU rows
  from mapped memory (`add_hits_mapped_kernel`). Timeouts release the GPU spinners on errors (#267).
- `--pcie-frac` / auto split: the share of VRAM misses read over PCIe instead of computed on the CPU, refined from
  the pool's measured time per expert, `L / (L + 1.85 R)`.
- Kernels (`nvfp4_avx512.cpp`): Strata's own NVFP4 row layout, Q8_0 activations, `vpdpbusd`.

## 3. Design for the vLLM plugin

### 3.1 Data: the cold pool as it is (no second copy)

The cold tier is 31.5 GiB of pinned, mapped host memory in **Marlin** layout (expert_pool.py; 60 GiB RAM leave no
room for a Strata-layout copy). The CPU kernel reads Marlin tiles directly:

- Weights `int32 [K/16, 2N or K*2]`: tile (k-tile `kt`, n-tile `nt`) of 16 x 64 values = 128 words at
  `(kt * n_tiles + nt) * 128`. Word `w = th*4 + warp` holds column `n0 = warp*16 + th/4` (+8) and rows
  `(th%4)*2 + {0,1,8,9}`; nibble `i` holds `vals[{0,2,4,6,1,3,5,7}[i]]` (vLLM `gptq_marlin_repack.cu`). So 16
  consecutive words (one zmm) share `th/4`: lane `l` covers columns `(l%4)*16 + v (+8)` and rows
  `(l/4)*2 + {0,1,8,9}`. Per zmm and nibble, `vpermps` on the low 4 bits decodes E2M1 (signed table), and FMAs
  against 4 permuted copies of the 16 activations of the k-tile accumulate 4 k-partials per column.
- Block scales `fp8 [K/16, N]`, one per (k-tile, column), permuted within 64-column chunks
  (`marlin_permute_scales` then `[0,2,1,3]` interleave) and stored as the high byte of `fp16(s * factor * 2^7) << 1`
  ("S0E5M3"). Decode: `float bits = (b << 20) + (112 << 23)`, 0 for b = 0. For vector `v` the 16 scales are the
  8 bytes `8v .. 8v+7` of the chunk (byte `8v + 4(c>>1) + (c&1) + 2h` for column group c = l%4, half h).
- Global scales (`g13`, `g2`, fp32 per slot, Marlin-processed) live in VRAM; the dispatch kernel copies the
  selected slots' values into the job, so the CPU never needs a synchronised host copy (adaptive swaps move them).

Numerics follow the Marlin epilogue exactly where it rounds (marlin_moe_wna16/marlin_template.h):
GEMM1 `c1 = bf16(acc * g13)`, `silu_and_mul`: `h = bf16(float(bf16(silu(gate))) * up)`, GEMM2
`y = bf16(bf16(acc) * bf16(w_route * g2))`, with `acc` the fp32 sum of `(e2m1 * scale) * x` (bf16 activations,
exact products; Marlin's internal power-of-two `2^-126` is folded into the global scale). Only the fp32
summation order and `expf` differ, so CPU and GPU outputs should agree to ~1 bf16 ulp. No Q8_0 activations.

### 3.2 One MoE call (inside the FULL CUDA graph)

```
main:  topk -> [dispatch] -> [gather + mapped copy + doorbell] -> hot Marlin --------------> join -> [wait] -> combine(+cpu rows)
side:                                                        \-> cold Marlin (cold_map_eff) -/
CPU pool:                                    doorbell seen -> G1 tasks -> G2 tasks -> done flag
```

1. `dispatch` (1 CTA, 512 threads, one per logical expert): histogram of the call's routes; cold experts =
   `cold_map[e] >= 0` and touched; eligible = cold and `routes <= EDGE0_CPU_MAX_ROUTES`. If the call has at least
   `EDGE0_CPU_MIN_COLD` cold experts, the CPU takes `round(frac * C)` of them: the eligible experts in logical-id
   order (deterministic, timing-independent). `EDGE0_CPU_MAX_ROUTES` defaults to 8: above that the CPU's
   per-route compute approaches its DRAM time, while Marlin reads an expert once per 8 routes anyway. Writes `cold_map_eff` (cold map minus the CPU experts), `cpu_sel[e]`,
   the job (slot, route list, weights, `g13`, `g2` per expert) into mapped memory, and `n_cpu` into VRAM. A fixed
   split passes `frac` as a kernel argument; `EDGE0_CPU_FRAC=auto` passes -1 and the kernel reads `frac` from mapped
   memory (only when the call has enough cold experts), so the auto split can change it without recapturing graphs.
   Round 2 (agent `exceed-2`, TUNING-LOG "Single-stream" R2): a call without a CPU share returns right after the
   choice (no mapped writes, `n_cpu` = 0), the cold / eligible counts share one block scan, and only threads that
   wrote mapped memory fence. The old kernel cost 8.7 us per call even without a CPU share (512
   `__threadfence_system` + a mapped read of `frac`), now 1.1 us (2.1 us with a share; `tools/dispatch_bench.py`).
2. `gather` (one CTA per route) replaces the Triton gather: writes the Marlin input row in VRAM and, for CPU
   routes, the same bf16 row into the mapped job (zero-copy writes, the otherwise idle D2H direction). With a CPU
   share, CTAs of CPU routes fence (`__threadfence_system`) and every CTA counts in; the last one increments the
   sequence number in VRAM and rings the doorbell (mapped). Without a share (`n_cpu` = 0) it only writes the Marlin
   rows.
3. Hot Marlin (main) and cold Marlin with `cold_map_eff` (side stream) as before.
4. `wait` (1 thread): returns at once if `n_cpu == 0`, else spins on the mapped `done == seq` with `__nanosleep`,
   with a timeout (default 2 s) that records an error instead of hanging the graph.
5. `combine` (Triton) gets a third source: route rows of CPU experts are read from the mapped `y` with `.cv`
   (uncached) loads.

The decision to use the CPU path is static per captured graph: calls with `EDGE0_CPU_MIN_TOKENS <= T <=
EDGE0_CPU_MAX_TOKENS` (defaults 2 and 256) go through it; prefill chunks and single-token calls keep the original
path (their kernel launches cost more than the CPU can save; prefill experts have many routes each).

### 3.3 CPU pool (C++ extension `edge0_vllm._cpu_experts`)

- `EDGE0_CPU_THREADS` workers (default 8, the measured saturation point) pinned to `EDGE0_CPU_CORES` (default
  physical cores 2..9, so cores 0-1, 10-15 and all SMT siblings stay free for the engine thread, the API server, CUDA's threads and PLE reads; the profile
  measured a collapse with 32 threads plus the GPU feeder).
- Workers spin on the mapped sequence number with `_mm_pause`; after `EDGE0_CPU_SPIN_MS` (default 200 ms) without a
  job they sleep in 50 us steps (an idle server burns no cores; the first layer after idle pays <= 50 us).
- A job = the CPU experts of one MoE call. Task list: G1 tasks (expert, gate n-tile j and up n-tile j: 64
  intermediate features, `silu * up` fused) for all experts, then G2 tasks (expert, 2 output n-tiles of w2). A G2
  task waits (spin) until its expert's 10 G1 tasks are done; G1 tasks never wait, and every G1 task is grabbed
  before any G2 task, so there is no deadlock. Tasks are grabbed with a CAS on a 64-bit (seq, index) ticket that a
  per-job leader publishes after resetting the per-job counters, so a straggler of the previous job can never take
  a task of the next one. The worker that completes the last task stores `done = seq` (release).
- Per task, routes are processed one at a time (weights of the task, 23-126 KB, stay in L2 after the first route);
  software prefetch a few k-tiles ahead, because each k-tile row of a column is a 512 B block 10-14 KB from the next.

### 3.4 Split control

`EDGE0_CPU_FRAC` (default 0.55, measured) fixes the share; it is the deterministic mode used for the correctness gate.
`EDGE0_CPU_FRAC=auto`: the wait kernel writes the GPU-side duration (doorbell -> wait start) and its own spin time
per call into mapped stats; the pool measures its job time; a host thread moves `frac` towards the balance point
`f* = (f/Tc) / (f/Tc + (1-f)/Tg)` with damping (Strata's auto PCIe share, per window). Experts are whole units;
the floor `EDGE0_CPU_MIN_COLD` (default 4) keeps tiny calls on the GPU.

### 3.5 Safety

- Mailbox, `seq`, `cpu_sel`, `cold_map_eff` are process-wide (layers run one after the other on one stream; the
  side stream joins before the next layer's dispatch).
- Not combined with `EDGE0_PREFETCH` (staging maps) or `EDGE0_MOE_IMPL=torch` (refused at start).
- If the extension cannot be built or the CPU lacks AVX-512 (F/BW/VL/DQ/VBMI), `EDGE0_CPU_EXPERTS=1` fails at
  start, loudly; it is never silently ignored.
- Timeouts of the wait kernel are counted in mapped memory and logged by the engine thread every few seconds.

## 4. Gate before integration

1. Unit tests: CPU kernel vs Marlin GPU output and vs the fp32 ModelOpt reference on random experts (identical up
   to ~1 bf16 ulp; same error to the fp32 reference).
2. Microbenchmark (`tools/cpu_experts/bench_cpu_split.py`): synthetic cold pool, the op body captured in a CUDA
   graph, C cold experts per call, GPU only vs split at several fractions and thread counts. Gate: hand-off
   (dispatch + doorbell + wait with an empty CPU share) <= 30 us per call, and the split must beat GPU-only by
   > 15% at C >= 28 (c8-sized calls).
3. Only then the server: correctness gate (`compare_strata.py` >= 250/261), A/B vs the tuned preset at c1/8/32.

## 5. Outcome (details and tables: TUNING-LOG.md "CPU cold experts (user request)")

- Kernel: bit-identical to Marlin on real experts for single routes, <= 2 bf16 ulp on 0.03% of elements otherwise.
- Gate passed: hand-off +11 us per call; per call -24..-26% at 11-86 cold experts with frac 0.6 (CPU 45-47 us per
  expert, GPU 45-68 us per expert, 68 GB/s together against 51 GB/s for the GPU alone). 8 threads saturate.
- Server: correctness 8/8, teacher-forced 256/261; ABBA vs the tuned preset **+27.7% at c8, +24.5% at c32**, c1
  neutral (single-token decode keeps the GPU path).
- With DFlash2 k=3 (verify steps take the CPU path even at c1): ABBA **+17.1% / +21.1% / +17.4% at c1 / c8 / c32**
  (c1 75.2 -> 88.0 tok/s); greedy outputs with and without the drafter: 6/8 identical, 2 near-ties (5/8 and 3
  near-ties without CPU experts).
- Tuning in the server: 8 threads (6 lose 8% at c8), frac 0.55 (0.6 and `auto`, which settles at 0.575, are within
  noise).
- Run it: `EDGE0_CPU_EXPERTS=1 ./serve.sh` (defaults: 8 threads on cores 2-9, frac 0.55, calls of 2..256 tokens).

### How to change it

- Another expert geometry: `mailbox.h` constants (LAT, FF) and the tile loops; Marlin's 16 x 64 tile and the scale
  permutation are fixed by vLLM's repack, and `tests/test_cpu_experts_host.py` ports that repack to numpy.
- A different vLLM / Marlin version: rerun `tests/test_cpu_experts.py` on the GPU; it compares against the live
  Marlin kernel, so a layout or epilogue change shows up as a non-zero ulp count at once.
- More cores or another CPU: rerun `tools/cpu_experts/bench_cpu_split.py --threads ... --fracs ...` (needs the GPU
  lock and no server) and set `EDGE0_CPU_THREADS` / `EDGE0_CPU_FRAC` (or `EDGE0_CPU_FRAC=auto`).
