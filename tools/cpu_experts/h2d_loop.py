"""GPU reads of pinned host memory in a loop, GB/s every 0.5 s (docs/PROFILE.md section 5: DRAM contention between
the GPU's PCIe reads of cold experts and CPU expert compute).

  python h2d_loop.py <mode> <GiB> <seconds>
  mode ce: copy engine (pinned host -> VRAM, 64 MiB copies)
  mode sm: SM zero-copy reads through a UVA view of host-registered memory (what the cold Marlin call does),
           a sum over 64 MiB chunks

Run in the edge0-vllm image with the GPU free (PYTHONPATH=/workspace/edge0/plugin for the sm mode).
"""
import sys
import time

import torch


def main():
    mode, gib, seconds = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    n = int(gib * 2**30) // (64 << 20) * (64 << 20)
    chunk = 64 << 20
    dev = torch.device("cuda")
    if mode == "ce":
        host = torch.empty(n, dtype=torch.uint8, pin_memory=True)
        host.fill_(1)
        dst = torch.empty(chunk, dtype=torch.uint8, device=dev)

        def one(o):
            dst.copy_(host[o:o + chunk], non_blocking=True)
    elif mode == "sm":
        from edge0_vllm import hostmem
        buf = hostmem.alloc(n, "host_register", dev)
        buf.cpu.fill_(1)
        uva = buf.dev.view(torch.int32)
        acc = torch.zeros((), dtype=torch.int64, device=dev)
        per = chunk // 4

        def one(o):
            acc.add_(uva[o // 4:o // 4 + per].sum(dtype=torch.int64))
    else:
        raise SystemExit("mode: ce | sm")
    torch.cuda.synchronize()
    t_end = time.time() + seconds
    t_last = time.time()
    moved = 0
    total, t0 = 0, time.time()
    o = 0
    while time.time() < t_end:
        for _ in range(8):
            one(o)
            o = (o + chunk) % n
            moved += chunk
        torch.cuda.synchronize()
        t = time.time()
        if t - t_last >= 0.5:
            print(f"{t:.3f} gpu-{mode} {moved / (t - t_last) / 1e9:.1f} GB/s", flush=True)
            total += moved
            moved, t_last = 0, t
    total += moved
    print(f"mean gpu-{mode} {total / (time.time() - t0) / 1e9:.1f} GB/s over {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
