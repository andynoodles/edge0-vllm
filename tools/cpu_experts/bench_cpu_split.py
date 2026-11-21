"""Gate for EDGE0_CPU_EXPERTS (docs/CPU-EXPERTS.md section 4): the MoE op body (gather -> hot || cold Marlin ->
combine) captured in a CUDA graph, GPU only vs with a CPU share of the cold experts.

  python bench_cpu_split.py [--threads 8,12,14] [--fracs 0.3,0.4,0.5,0.6] [--reps 40] [--out results.json]

Real layer-0 experts of $EDGE0_CKPT in a pool with 128 hot (VRAM) and 384 cold (pinned host) slots. Each case is
(T tokens, C distinct cold experts per call, ~1.5 routes per cold expert; the other routes go to hot experts); every
replay draws a new random cold set, so neither the GPU's L2 nor the CPU's L3 keep experts between replays. Times
are CUDA-event medians of graph replays. Run in the edge0-vllm image under the GPU lock, no server running:

  docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --network host -v /home/andychen/models:/models \
    -v ~/work/edge0-vllm:/workspace/edge0 -v ~/.cache/edge0-vllm:/root/.cache -e PYTHONPATH=/workspace/edge0/plugin \
    -e EDGE0_CKPT=/models/Edge0-145B-vllm --entrypoint python edge0-vllm:dev \
    /workspace/edge0/tools/cpu_experts/bench_cpu_split.py
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "plugin", "tests"))

CASES = [(1, 3), (4, 11), (8, 28), (16, 50), (32, 86)]   # (tokens, cold experts): c1, c1 k=3 verify, c8, c16, c32


def routing(t: int, c: int, cold_ids: list[int], hot_ids: list[int], g: torch.Generator, rpe: float = 1.5):
    perm = torch.randperm(len(cold_ids), generator=g)[:c].tolist()
    cold = [cold_ids[i] for i in perm]
    rc = min(10 * t, c * t, max(c, int(round(c * rpe))))
    reps = [1] * c
    for i in range(rc - c):
        reps[i % c] += 1
    seq = [e for e, k in zip(cold, reps) for _ in range(k)]
    slots: list[list[int]] = [[] for _ in range(t)]
    for i, e in enumerate(seq):
        tt = i % t
        while e in slots[tt] or len(slots[tt]) >= 10:
            tt = (tt + 1) % t
        slots[tt].append(e)
    hp = torch.randperm(len(hot_ids), generator=g).tolist()
    for tt in range(t):
        k = 0
        while len(slots[tt]) < 10:
            e = hot_ids[hp[(tt * 10 + k) % len(hp)]]
            k += 1
            if e not in slots[tt]:
                slots[tt].append(e)
    ids = torch.tensor(slots, dtype=torch.int32)
    w = torch.rand(t, 10, generator=g) + 0.05
    w = (w / w.sum(1, keepdim=True)).float()
    return ids.cuda(), w.cuda()


def settings(threads: int, frac: float, min_cold: int = 1, tiles=(1, 2), cores: str = ""):
    return types.SimpleNamespace(cpu_min_tokens=1, cpu_max_tokens=64, cpu_min_cold=min_cold, cpu_max_routes=64,
                                 cpu_max_experts=512, cpu_timeout_ms=2000.0, cpu_cores=cores, cpu_threads=threads,
                                 cpu_frac=str(frac), cpu_frac_start=frac, cpu_spin_ms=500.0,
                                 cpu_g1_pairs=tiles[0], cpu_g2_tiles=tiles[1])


class Log:
    def info(self, *a):
        print(a[0] % a[1:] if len(a) > 1 else a[0], flush=True)

    warning = info


def run_case(pool, grp, ws, t, c, cx, reps, seed, cold_ids, hot_ids):
    from edge0_vllm.latent_moe import latent_experts_marlin
    g = torch.Generator().manual_seed(seed)
    gx = torch.Generator(device="cuda").manual_seed(seed)
    inputs = [routing(t, c, cold_ids, hot_ids, g) for _ in range(reps + 3)]
    xl = (torch.randn(t, 8 * 1792, generator=gx, device="cuda") * 0.5).to(torch.bfloat16)
    si, sw = inputs[0][0].clone(), inputs[0][1].clone()
    z = torch.empty_like(xl)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        latent_experts_marlin(pool, grp, xl, si, sw, ws, z, split=True, cpu=cx)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        latent_experts_marlin(pool, grp, xl, si, sw, ws, z, split=True, cpu=cx)
    times = []
    st0 = cx.stats() if cx is not None else None
    for i, (ids, w) in enumerate(inputs):
        si.copy_(ids)
        sw.copy_(w)
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        if i >= 3:
            times.append(a.elapsed_time(b) * 1000)
    res = {"T": t, "C": c, "us_med": statistics.median(times), "us_mean": statistics.mean(times)}
    if cx is not None:
        st = cx.stats()
        d = {k: st[k] - st0[k] for k in st}
        n = max(1, d["jobs"])
        res.update(jobs=d["jobs"], cpu_experts=d["experts"] / n, cpu_us=1e6 * d["cpu_s"] / n,
                   gpu_side_us=1e6 * d["gpu_s"] / max(1, d["gpu_calls"]),
                   wait_us=1e6 * d["wait_s"] / max(1, d["gpu_calls"]), timeouts=st["timeouts"])
    del graph
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="8,12,14")
    ap.add_argument("--fracs", default="0,0.3,0.4,0.5,0.6")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--cases", default="")
    ap.add_argument("--tiles", default="1:2", help="G1 pairs : G2 tiles per CPU task, comma-separated")
    ap.add_argument("--cores", default="", help="EDGE0_CPU_CORES")
    ap.add_argument("--no-gpu-only", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import test_gpu_experts as gx
    from edge0_vllm.cpu_experts import CpuExperts, load
    ext = load(verbose=True)
    cases = CASES if not a.cases else [tuple(int(v) for v in x.split(":")) for x in a.cases.split(",")]
    t0 = time.time()
    pool = gx._pool(128)
    print(f"pool loaded in {time.time() - t0:.0f} s: {pool.n_hot} hot, {pool.n_cold} cold", flush=True)
    order = pool.order
    hot_ids, cold_ids = order[:pool.n_hot], order[pool.n_hot:]
    ws = gx._ws(64)
    grp = gx._layer_tensors()["expert_group"].to("cuda")
    rows = []

    def show(r):
        extra = ""
        if "cpu_us" in r:
            extra = (f" cpu {r['cpu_experts']:.1f} exp {r['cpu_us']:.0f} us | gpu side {r['gpu_side_us']:.0f} us, "
                     f"waited {r['wait_us']:.0f} us")
        print(f"{r['label']:>24} T={r['T']:>2} C={r['C']:>3}: {r['us_med']:8.1f} us{extra}", flush=True)

    for t, c in ([] if a.no_gpu_only else cases):
        r = run_case(pool, grp, ws, t, c, None, a.reps, 1000 + t, cold_ids, hot_ids)
        r["label"] = "gpu-only"
        rows.append(r)
        show(r)
    tiles = [tuple(int(v) for v in x.split(":")) for x in a.tiles.split(",")]
    for th in [int(x) for x in a.threads.split(",")]:
        for tl in tiles:
            for fr in [float(x) for x in a.fracs.split(",")]:
                cx = CpuExperts(settings(th, fr, tiles=tl, cores=a.cores), torch.device("cuda"), Log(), ext=ext)
                cx.register(0, pool)
                try:
                    for t, c in cases:
                        r = run_case(pool, grp, ws, t, c, cx, a.reps, 1000 + t, cold_ids, hot_ids)
                        r["label"] = f"th={th} t={tl[0]}:{tl[1]} f={fr}"
                        rows.append(r)
                        show(r)
                finally:
                    cx.pool.stop()
                time.sleep(0.2)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
