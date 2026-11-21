#!/usr/bin/env python3
"""Microbenchmark of the CPU-experts dispatch (+ gather) kernels (agent exceed-2, TUNING-LOG "Single-stream" R2).

Runs inside the edge0-vllm container on the GPU (no server). One extension per process:
  python3 tools/dispatch_bench.py [--so /opt/edge0/ext/<key>/edge0_cpu_experts.so] [--build-dir d]
Default: build / import the extension from the plugin tree on PYTHONPATH (cpu_experts.load()).

A CUDA graph of 48 dispatch (and dispatch + gather) calls with c1-verify-like inputs (T=4 tokens x top-10 routes,
~4 cold experts per call), replayed 200 times; prints us per call for: no CPU share (min_cold above the cold count),
and a CPU share (min_cold 1, frac 0.55; nothing consumes the job, the wait kernel is not run).
"""
import argparse
import importlib.machinery
import importlib.util

import torch


def load_ext(so, build_dir):
    if so:
        loader = importlib.machinery.ExtensionFileLoader("edge0_cpu_experts", so)
        spec = importlib.util.spec_from_file_location("edge0_cpu_experts", so, loader=loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        return mod
    from edge0_vllm import cpu_experts
    return cpu_experts.load(build_dir=build_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", default="")
    ap.add_argument("--build-dir", default=None)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--cold", type=int, default=4, help="cold experts per call")
    ap.add_argument("--frac", type=float, default=0.55, help="new kernel: fixed frac argument (-1 = mailbox)")
    a = ap.parse_args()
    from edge0_vllm import hostmem
    ext = load_ext(a.so, a.build_dir)
    dev = torch.device("cuda")
    T, K, E, L = 4, 10, 512, 48
    max_routes = 256 * K
    lay = dict(zip(("tasks", "route_ids", "route_w", "x", "y", "total", "header"), ext.layout(max_routes)))
    buf = hostmem.alloc(lay["total"], "host_register", dev)
    buf.cpu.zero_()
    buf.cpu[192:196].view(torch.float32).fill_(0.55)          # MboxHeader.frac (line 3)
    mbox_dev = int(buf.dev.data_ptr())
    g = torch.Generator().manual_seed(0)
    ids_l, cold_l = [], []
    for _ in range(L):
        ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)]).to(torch.int32)
        flat = torch.unique(ids.flatten())
        cold_e = flat[torch.randperm(flat.numel(), generator=g)[:a.cold]]
        cm = torch.full((E,), -1, dtype=torch.int32)
        cm[cold_e.long()] = torch.arange(a.cold, dtype=torch.int32)
        ids_l.append(ids.to(dev))
        cold_l.append(cm.to(dev))
    w = torch.rand(T, K, generator=g).to(dev)
    g13 = torch.rand(E, generator=g).to(dev)
    g2 = torch.rand(E, generator=g).to(dev)
    group = torch.randint(0, 8, (E,), generator=g, dtype=torch.int32).to(dev)
    xl = torch.randn(T, 8 * 1792, generator=g).to(torch.bfloat16).to(dev)
    out = torch.empty(max_routes, 1792, dtype=torch.bfloat16, device=dev)
    state = torch.zeros(16, dtype=torch.int64, device=dev)
    sel = torch.zeros(E, dtype=torch.int32, device=dev)
    eff = torch.full((E,), -1, dtype=torch.int32, device=dev)
    doc = (ext.dispatch.__doc__ or "").split("\n")[0]
    extra = ()
    if "arg14" in doc:           # new signature: (..., state, frac): fixed frac literal (>= 0) or -1 = read the mailbox
        extra = (float(a.frac),)
    print("dispatch args:", 15 + len(extra), "frac arg:", extra)

    def body(min_cold, with_gather):
        for i in range(L):
            ext.dispatch(ids_l[i], w, cold_l[i], g13, g2, i, min_cold, 8, 512, mbox_dev, max_routes, eff, sel, state,
                         *extra)
            if with_gather:
                ext.gather(xl, ids_l[i], group, out, sel, mbox_dev, max_routes, state)

    for name, mc, wg in (("dispatch, no CPU share", 99, False), ("dispatch, CPU share", 1, False),
                         ("dispatch+gather, no CPU share", 99, True), ("dispatch+gather, CPU share", 1, True)):
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            body(mc, wg)
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=s):
                body(mc, wg)
        torch.cuda.synchronize()
        for _ in range(10):
            gr.replay()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(a.reps):
            gr.replay()
        e1.record()
        torch.cuda.synchronize()
        us = e0.elapsed_time(e1) * 1e3 / (a.reps * L)
        print(f"{name:34s} {us:7.2f} us per call (graph of {L}, {a.reps} replays); n_cpu of the last call {int(state[1]) & 0xffffffff}")


if __name__ == "__main__":
    main()
