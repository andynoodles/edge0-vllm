#!/usr/bin/env python3
"""How fast are the expert tier's swap copies (EDGE0_TIER_EARLY, TUNING-LOG "Single-stream" R3)?

A round of n swaps = n experts H2D (host_register'd pinned slots -> VRAM slots) + n experts D2H (VRAM -> pinned
spare slots), each expert = the four Marlin arrays (w13 1,146,880 B, w2 573,440, s13 143,360, s2 71,680), on two
side streams, as tier.py issues them. Prints host issue time and GPU time (events) per mode:

  cudart   one cudaMemcpyAsync per array (late tier's path)
  batch    one cudaMemcpyBatchAsync per direction (EDGE0_TIER_COPY=batch)
  contig   one contiguous copy of the same bytes per direction (the bandwidth ceiling)
  kernel   an SM copy kernel (UVA reads / writes) per direction, for comparison

with and without a concurrent compute load on the main stream (a stand-in for the drafter).
Run inside the image with the GPU free:  python3 tools/tier_copy_bench.py [--n 16,32,48]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plugin"))
from edge0_vllm import hostmem  # noqa: E402
from edge0_vllm.config import S2_BYTES, S13_BYTES, W2_BYTES, W13_BYTES  # noqa: E402

ARR = (W13_BYTES, W2_BYTES, S13_BYTES, S2_BYTES)
SLOTS = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="16,32,48,64")
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    from cuda.bindings import runtime as rt
    dev = torch.device("cuda")
    host = [hostmem.alloc(SLOTS * nb, "host_register", dev) for nb in ARR]
    vram = [torch.empty(SLOTS * nb, dtype=torch.uint8, device=dev) for nb in ARR]
    for h in host:
        h.cpu.fill_(7)
    hp = [h.cpu.data_ptr() for h in host]
    vp = [v.data_ptr() for v in vram]
    h2d, d2h = torch.cuda.Stream(), torch.cuda.Stream()
    attr = rt.cudaMemcpyAttributes()
    attr.srcAccessOrder = rt.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderStream
    kind = rt.cudaMemcpyKind.cudaMemcpyDefault
    big_h = hostmem.alloc(64 * sum(ARR), "host_register", dev)
    big_v = torch.empty(64 * sum(ARR), dtype=torch.uint8, device=dev)
    load_a = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)

    def lists(n, direction):
        src_slots = list(range(0, 2 * n, 2))
        dst_slots = list(range(1, 2 * n, 2))
        d, s, z = [], [], []
        for i in range(n):
            for k, nb in enumerate(ARR):
                if direction == "h2d":
                    d.append(vp[k] + dst_slots[i] * nb)
                    s.append(hp[k] + src_slots[i] * nb)
                else:
                    d.append(hp[k] + dst_slots[i] * nb)
                    s.append(vp[k] + src_slots[i] * nb)
                z.append(nb)
        return d, s, z

    def issue(mode, n, direction, strm):
        if mode == "contig":
            nb = n * sum(ARR)
            with torch.cuda.stream(strm):
                if direction == "h2d":
                    big_v[:nb].copy_(big_h.cpu[:nb], non_blocking=True)
                else:
                    big_h.cpu[:nb].copy_(big_v[:nb], non_blocking=True)
            return
        if mode == "kernel":
            nb = n * sum(ARR)
            with torch.cuda.stream(strm):
                if direction == "h2d":
                    big_v[:nb].copy_(big_h.dev[:nb])           # device view of host memory: an SM copy kernel
                else:
                    big_h.dev[:nb].copy_(big_v[:nb])
            return
        d, s, z = lists(n, direction)
        if mode == "batch":
            err = rt.cudaMemcpyBatchAsync(d, s, z, len(d), [attr], [0], 1, strm.cuda_stream)
            assert int(err[0]) == 0, err
        else:
            for x, y, k in zip(d, s, z):
                err = rt.cudaMemcpyAsync(x, y, k, kind, strm.cuda_stream)
                assert int(err[0]) == 0, err

    def run(mode, n, load: bool, both: bool):
        main = torch.cuda.current_stream()
        torch.cuda.synchronize()
        t_host, t_gpu = [], []
        for _ in range(a.reps):
            ev0 = torch.cuda.Event(enable_timing=True)
            ev0.record(main)
            for s in (h2d, d2h):
                s.wait_event(ev0)
            if load:
                for _ in range(12):                    # ~2 ms of matmuls on the main stream, concurrent with the copies
                    load_a @ load_a
            t0 = time.perf_counter()
            issue(mode, n, "h2d", h2d)
            if both:
                issue(mode, n, "d2h", d2h)
            t1 = time.perf_counter()
            e1, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e1.record(h2d)
            e2.record(d2h)
            torch.cuda.synchronize()
            t_host.append(t1 - t0)
            t_gpu.append(max(ev0.elapsed_time(e1), ev0.elapsed_time(e2) if both else 0.0))
        t_host.sort()
        t_gpu.sort()
        mid = len(t_gpu) // 2
        return t_host[mid] * 1e3, t_gpu[mid]

    print(f"{'mode':>7s} {'n':>3s} {'MB/dir':>7s} {'load':>5s} {'both':>5s} {'host ms':>8s} {'GPU ms':>7s} {'GB/s/dir':>9s}")
    for n in (int(x) for x in a.n.split(",")):
        mb = n * sum(ARR) / 1e6
        for load in (False, True):
            for both in (False, True):
                for mode in ("cudart", "batch", "contig", "kernel"):
                    h, g = run(mode, n, load, both)
                    gbs = mb / 1e3 / (g / 1e3) if not load else float("nan")
                    print(f"{mode:>7s} {n:>3d} {mb:>7.1f} {int(load):>5d} {int(both):>5d} {h:>8.3f} {g:>7.3f} {gbs:>9.1f}",
                          flush=True)


if __name__ == "__main__":
    main()
