"""Expert-load microbenchmarks on the RTX 5090 (TidalMoE's "per-touch cost" measurement, adapted to PCIe Gen5).

Runs in the edge0-vllm image with the GPU free of the server:

  docker run --rm --gpus all --ipc=host --ulimit memlock=-1 -v /home/andychen/work/edge0-vllm:/workspace/edge0 \
    -v /home/andychen/.cache/edge0-vllm:/root/.cache -e PYTHONPATH=/workspace/edge0/plugin --entrypoint python \
    edge0-vllm:dev /workspace/edge0/tools/bench_expert_io.py [--quick] [--json out.json]

Parts:
  bw      copy-engine H2D / D2H from pinned host memory (1 record = 1,935,360 B, 32 MiB, 1 GiB), per-record
          cudaMemcpyAsync gathers, and an SM zero-copy gather kernel (host -> VRAM rows) by CTA count.
  touch   the real op body (``latent_experts_marlin``: gather -> hot Marlin || cold Marlin -> combine) on synthetic
          Marlin pools (256 hot slots in VRAM, 256 cold slots in pinned host memory, zero weights), inside a CUDA
          graph like serving: time vs distinct hot / cold experts at T = 1 .. 64 tokens -> marginal us per touch.
  reread  T tokens all routed to the same 10 cold experts (routes per expert = T): does Marlin re-read an expert
          once per M-block (auto block_size_m) and does a larger M block (64) avoid it?
  ctas    cold-call CTA cap (blocks_per_sm) and EDGE0_SPLIT on a mixed decode-like step.
Uses <= 2.5 GiB of pinned host memory and <= 3 GiB of VRAM.
"""
from __future__ import annotations

import argparse
import json
import math
import types

import torch

REC = 1_935_360
GIB = 1 << 30
RES: dict = {}


def say(k, v):
    print(f"{k:64s} {v}", flush=True)


def gtime(fn, reps=20, warm=3) -> float:
    """seconds per call of fn (eager), events on the current stream."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps / 1e3


def graph_time(fn, reps=50) -> float:
    """seconds per replay of fn captured in a CUDA graph (how decode runs in the server)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return gtime(g.replay, reps=reps, warm=3)


# ---------------------------------------------------------------------------------------------- bandwidth
def part_bw(host):
    from edge0_vllm.kernels import _HAS_TRITON
    dev = torch.device("cuda")
    vram = torch.empty(GIB, dtype=torch.uint8, device=dev)
    say("== bw: copy engine (cudaMemcpyAsync) from pinned host", "")
    for n, lab in ((REC, "1 record (1.85 MB)"), (32 << 20, "32 MiB"), (GIB, "1 GiB")):
        t = gtime(lambda: vram[:n].copy_(host.cpu[:n], non_blocking=True), reps=10 if n >= GIB else 50)
        say(f"H2D {lab}", f"{n / t / 1e9:6.1f} GB/s  ({t * 1e6:8.1f} us)")
        RES[f"h2d_{n}"] = n / t / 1e9
        t = gtime(lambda: host.cpu[:n].copy_(vram[:n], non_blocking=True), reps=10 if n >= GIB else 50)
        say(f"D2H {lab}", f"{n / t / 1e9:6.1f} GB/s  ({t * 1e6:8.1f} us)")
        RES[f"d2h_{n}"] = n / t / 1e9
    for k in (6, 10, 64):
        idx = torch.randperm(host.nbytes // REC)[:k].tolist()

        def gather():
            for i, j in enumerate(idx):
                vram[i * REC:(i + 1) * REC].copy_(host.cpu[j * REC:(j + 1) * REC], non_blocking=True)
        t = gtime(gather, reps=20)
        say(f"copy-engine gather of {k} records (one memcpy each)", f"{k * REC / t / 1e9:6.1f} GB/s ({t * 1e6:8.1f} us)")
        RES[f"ce_gather_{k}"] = k * REC / t / 1e9
    if not _HAS_TRITON:
        return
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _sm_gather(src, dst, rows, REC_I32: tl.constexpr, BLOCK: tl.constexpr):
        # one program per (record, chunk); 16 B vector loads from the UVA host view
        pid = tl.program_id(0)
        nchunk = tl.cdiv(REC_I32, BLOCK)
        r = pid // nchunk
        c = pid % nchunk
        row = tl.load(rows + r).to(tl.int64)
        offs = c * BLOCK + tl.arange(0, BLOCK)
        m = offs < REC_I32
        v = tl.load(src + row * REC_I32 + offs, mask=m)
        tl.store(dst + r.to(tl.int64) * REC_I32 + offs, v, mask=m)

    src = host.dev.view(torch.int32)
    dst = vram.view(torch.int32)
    rec_i32 = REC // 4
    say("== bw: SM zero-copy gather kernel (host rows -> VRAM), one record = 1.85 MB", "")
    nrows = host.nbytes // REC
    for k in (6, 10, 32, 128, 400):
        rows = torch.randperm(nrows, device=dev)[:k].to(torch.int32)
        for block in (1024, 4096, 16384):
            grid = (k * triton.cdiv(rec_i32, block),)
            t = gtime(lambda: _sm_gather[grid](src, dst, rows, REC_I32=rec_i32, BLOCK=block, num_warps=8), reps=20)
            say(f"SM gather k={k:3d} block={block:5d} ({grid[0]} programs)",
                f"{k * REC / t / 1e9:6.1f} GB/s ({t * 1e6:8.1f} us)")
            RES[f"sm_gather_{k}_{block}"] = k * REC / t / 1e9
    del vram


# ---------------------------------------------------------------------------------------------- Marlin op
def make_pool(n_hot=256, n_cold=256):
    from edge0_vllm.expert_pool import Tier
    dev = torch.device("cuda")
    hot = Tier.in_vram(n_hot, dev)
    cold = Tier.in_host(n_cold, dev, "host_register")
    for t in (hot, cold):
        t.g13.fill_(1.0)
        t.g2.fill_(1.0)
    hot_map = torch.full((512,), -1, dtype=torch.int32, device=dev)
    cold_map = torch.full((512,), -1, dtype=torch.int32, device=dev)
    hot_map[:n_hot] = torch.arange(n_hot, dtype=torch.int32, device=dev)
    cold_map[n_hot:n_hot + n_cold] = torch.arange(n_cold, dtype=torch.int32, device=dev)
    return types.SimpleNamespace(n_hot=n_hot, n_cold=n_cold, hot=hot, cold=cold, hot_map=hot_map, cold_map=cold_map)


def make_ids(t, dh, dc, n_hot=256, seed=0):
    """[t, 10] int32 ids with exactly min(dh, ..) distinct hot and dc distinct cold experts (cycled), no duplicate
    expert within a token. Hot ids 0..n_hot-1, cold ids n_hot..511."""
    g = torch.Generator().manual_seed(seed)
    hot = torch.randperm(n_hot, generator=g)[:dh].tolist() if dh else []
    cold = (torch.randperm(512 - n_hot, generator=g)[:dc] + n_hot).tolist() if dc else []
    slots = t * 10
    # share of routes going to cold: proportional to distinct counts, but at least dc routes so all are touched
    n_cold_routes = min(slots, max(dc, round(slots * dc / max(dh + dc, 1)))) if dc else 0
    if dh == 0:
        n_cold_routes = slots
    assert n_cold_routes == 0 or len(cold) >= 1
    ids = []
    ci = hi = 0
    for tok in range(t):
        row = []
        for j in range(10):
            r = tok * 10 + j
            pick_cold = r < n_cold_routes
            src, idx = (cold, ci) if pick_cold else (hot, hi)
            # avoid duplicates within a token
            for _ in range(len(src)):
                e = src[idx % len(src)]
                idx += 1
                if e not in row:
                    break
            row.append(e)
            if pick_cold:
                ci = idx
            else:
                hi = idx
        ids.append(row)
    out = torch.tensor(ids, dtype=torch.int32, device="cuda")
    return out


def op_runner(pool, t, ids, split=True):
    from edge0_vllm.latent_moe import _MoeWorkspace, latent_experts_marlin
    dev = torch.device("cuda")
    ws = _MoeWorkspace(max(t, 64) * 10, dev, torch.bfloat16)
    grp = (torch.arange(512, device=dev) % 8).to(torch.int32)
    xl = torch.randn(t, 8 * 1792, device=dev).to(torch.bfloat16)
    w = torch.full((t, 10), 0.1, dtype=torch.float32, device=dev)
    z = torch.empty_like(xl)

    def fn():
        latent_experts_marlin(pool, grp, xl, ids, w, ws, z, split=split)
    return fn


def distinct(ids, n_hot=256):
    u = torch.unique(ids[ids >= 0])
    return int((u < n_hot).sum()), int((u >= n_hot).sum())


def part_touch(pool, quick):
    say("== touch: op time (CUDA graph replay) vs distinct hot / cold experts", "")
    rows = []
    # T = 1: 10 routes, h hot + (10-h) cold
    for h in range(0, 11, 1 if not quick else 2):
        ids = make_ids(1, h, 10 - h)
        dh, dc = distinct(ids)
        t = graph_time(op_runner(pool, 1, ids))
        rows.append((1, dh, dc, t))
        say(f"T=1  hot {dh:3d} cold {dc:3d}", f"{t * 1e6:8.1f} us")
    for T in ((8, 32) if quick else (8, 16, 32, 64)):
        for dc in (0, 8, 16, 32, 64, 128, 256):
            if dc > T * 10:
                continue
            dh = min(64, T * 10 - dc) if dc < T * 10 else 0
            ids = make_ids(T, dh, dc)
            dh, dc = distinct(ids)
            t = graph_time(op_runner(pool, T, ids))
            rows.append((T, dh, dc, t))
            say(f"T={T:<3d} hot {dh:3d} cold {dc:3d}", f"{t * 1e6:8.1f} us")
        for dh in (10, 64, 128, 256):
            if dh > T * 10:
                continue
            ids = make_ids(T, dh, 0)
            dh_, dc_ = distinct(ids)
            t = graph_time(op_runner(pool, T, ids))
            rows.append((T, dh_, dc_, t))
            say(f"T={T:<3d} hot {dh_:3d} cold {dc_:3d} (hot only)", f"{t * 1e6:8.1f} us")
    RES["touch"] = rows
    # least squares per T: t = a + b*cold + c*hot
    import numpy as np
    for T in sorted({r[0] for r in rows}):
        sub = [r for r in rows if r[0] == T]
        A = np.array([[1.0, r[2], r[1]] for r in sub])
        y = np.array([r[3] * 1e6 for r in sub])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        say(f"fit T={T}: us = a + b*cold + c*hot", f"a={coef[0]:7.1f}  b(cold)={coef[1]:6.2f}  c(hot)={coef[2]:6.2f}")
        RES[f"fit_T{T}"] = [float(x) for x in coef]


def part_reread(pool, quick):
    from edge0_vllm import latent_moe as lm
    say("== reread: T tokens all on the same 10 cold experts (routes per expert = T)", "")
    out = []
    for T in ((1, 16, 64, 256) if quick else (1, 8, 16, 32, 64, 128, 256)):
        ids = torch.stack([torch.arange(256, 266, dtype=torch.int32, device="cuda")] * T)
        res = {}
        for lab, bsm in (("auto", 0), ("bsm16", 16), ("bsm64", 64)):
            if bsm:
                lm.set_tier_tuning("cold", bsm)
            else:
                lm.set_tier_tuning("cold", 0)
            t = graph_time(op_runner(pool, T, ids))
            res[lab] = t
        lm.set_tier_tuning("cold", 0)
        say(f"T={T:<4d} auto bsm={lm._auto_bsm(T * 10)}", "  ".join(f"{k} {v * 1e6:8.1f} us" for k, v in res.items()))
        out.append((T, {k: v * 1e6 for k, v in res.items()}))
    RES["reread"] = out


def part_ctas(pool, quick):
    from edge0_vllm import latent_moe as lm
    say("== ctas: decode-like mixed steps; split on/off; cold blocks_per_sm cap", "")
    out = []
    for T, dh, dc in ((1, 5, 5), (16, 40, 70), (32, 60, 110), (64, 80, 150)):
        ids = make_ids(T, dh, dc)
        dh_, dc_ = distinct(ids)
        res = {"split": graph_time(op_runner(pool, T, ids, split=True)),
               "seq": graph_time(op_runner(pool, T, ids, split=False))}
        for bps in (1, 2):
            lm.set_tier_tuning("cold", 0, bps)
            res[f"cold_bps{bps}"] = graph_time(op_runner(pool, T, ids, split=True))
        lm.set_tier_tuning("cold", 0)
        say(f"T={T:<3d} hot {dh_:3d} cold {dc_:3d}", "  ".join(f"{k} {v * 1e6:7.1f}" for k, v in res.items()))
        out.append((T, dh_, dc_, {k: v * 1e6 for k, v in res.items()}))
    RES["ctas"] = out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--parts", default="bw,touch,reread,ctas")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    torch.manual_seed(0)
    p = torch.cuda.get_device_properties(0)
    say("device", f"{p.name} sm_{p.major}{p.minor} {p.multi_processor_count} SMs, {p.total_memory / GIB:.1f} GiB")
    parts = a.parts.split(",")
    if "bw" in parts:
        from edge0_vllm import hostmem
        host = hostmem.alloc(2 * GIB, "host_register", torch.device("cuda"))
        host.cpu.fill_(1)
        part_bw(host)
        del host
    pool = make_pool() if any(x in parts for x in ("touch", "reread", "ctas")) else None
    if "touch" in parts:
        part_touch(pool, a.quick)
    if "reread" in parts:
        part_reread(pool, a.quick)
    if "ctas" in parts:
        part_ctas(pool, a.quick)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(RES, f, indent=1)
        say("written", a.json)


if __name__ == "__main__":
    main()
