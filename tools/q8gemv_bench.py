#!/usr/bin/env python3
"""b12x Q8_0 dense GEMM vs edge0_vllm.q8gemv (Triton GEMV over the same packed weight) at decode sizes.

Run in the edge0-vllm image with the GPU (holding the GPU lock):
  docker run --rm --gpus all -v ~/work/edge0-vllm:/workspace/edge0 -e PYTHONPATH=/workspace/edge0/plugin \
      --entrypoint python3 edge0-vllm:dev /workspace/edge0/tools/q8gemv_bench.py [--m 1,4] [--shapes D,U,...]

Per Edge0 Q8 shape: random Q8_0 blocks (fp16 bases like the checkpoint's), ``COPIES`` weight copies cycled so the
weights never sit in the 96 MB L2 (as in serving, where each weight is read once per step), timed as CUDA-graph
replays of 48 calls. Reports us per call and achieved GB/s (weight bytes incl. bases), and the max relative error
against an fp32 reference of the dequantized weight. ``--mask`` adds the latent D/U group-skip variants with route
ids drawn so that ``--groups`` of the 8 groups are used.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, "/workspace/edge0/plugin")

SHAPES = {   # name: (N, K)
    "D": (14336, 2560), "U": (2560, 14336), "qkvz": (16384, 2560), "gdn_out": (2560, 6144),
    "sh_gu": (1280, 2560), "sh_down": (2560, 640), "lm_head": (248320, 2560), "qsa_qkv": (13312, 2560),
}


def rand_q8(n, k, dev):
    kb = k // 32
    blocks = torch.empty((n, kb, 34), dtype=torch.uint8, device=dev)
    d = (torch.rand((n, kb), device=dev) * 2e-3 + 1e-4).to(torch.float16)
    blocks[..., :2] = d.view(torch.uint8).view(n, kb, 2)
    blocks[..., 2:] = torch.randint(-127, 128, (n, kb, 32), device=dev, dtype=torch.int8).view(torch.uint8)
    return blocks


def ref_weight(blocks):
    n, kb, _ = blocks.shape
    d = blocks[..., :2].contiguous().view(torch.float16).float()
    q = blocks[..., 2:].contiguous().view(torch.int8).float()
    return (q * d).reshape(n, kb * 32)


def timed(fn, reps=48, iters=20):
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3):
            fn(i)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        for i in range(reps):
            fn(i)
    g.replay()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        st.record()
        g.replay()
        en.record()
        en.synchronize()
        ts.append(st.elapsed_time(en) * 1e3 / reps)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="1,2,4,8")
    ap.add_argument("--shapes", default="D,U,qkvz,gdn_out,sh_gu,sh_down")
    ap.add_argument("--copies", type=int, default=6)
    ap.add_argument("--mask", action="store_true")
    ap.add_argument("--groups", default="4,6,8")
    ap.add_argument("--fma", action="store_true", help="also time the CUDA-core (non-dot) variant")
    ap.add_argument("--cfg", default="", help="BLOCK_N:SPLIT overrides to try, comma-separated")
    ap.add_argument("--shared", action="store_true", help="also the fused shared expert + add")
    ap.add_argument("--nopdl", action="store_true", help="launch the Triton kernels without PDL")
    a = ap.parse_args()
    from vllm.utils.b12x import get_b12x_blockscaled
    from edge0_vllm import q8gemv
    api = get_b12x_blockscaled()
    if a.nopdl:
        q8gemv._PDL = False
    dev = torch.device("cuda")
    torch.manual_seed(0)
    ms = [int(x) for x in a.m.split(",")]
    for name in a.shapes.split(","):
        n, k = SHAPES[name]
        packs, refs = [], []
        copies = 2 if name == "lm_head" else a.copies
        for c in range(copies):
            b = rand_q8(n, k, dev)
            if c == 0:
                refs.append(ref_weight(b) if name != "lm_head" else ref_weight(b[:4096]))
            packs.append(api.pack_weight(b, recipe="q8_0"))
            del b
        wbytes = n * k + n * (k // 32) * 2
        query = api.BlockscaledQuery(recipe="q8_0", num_tokens=2048, in_features=k, padded_in_features=k,
                                     out_features=n, activation_mode="a16")
        plan = api.plan_regimes(query, exact_m=tuple(ms))
        for m in ms:
            xs = [torch.randn((m, k), device=dev, dtype=torch.bfloat16) for _ in range(copies)]
            nr = refs[0].shape[0]
            y_ref = xs[0].float() @ refs[0].t()
            y_b = api.mm(xs[0], packs[0], plan=plan)[:, :nr]
            y_t = q8gemv.q8_gemv(xs[0], packs[0].values, packs[0].metadata, n, k)[:, :nr]
            eb = ((y_b.float() - y_ref).abs().max() / y_ref.abs().max()).item()
            et = ((y_t.float() - y_ref).abs().max() / y_ref.abs().max()).item()
            same = (y_b == y_t).float().mean().item()
            tb = timed(lambda i: api.mm(xs[i % copies], packs[i % copies], plan=plan))
            tt = timed(lambda i: q8gemv.q8_gemv(xs[i % copies], packs[i % copies].values,
                                                packs[i % copies].metadata, n, k))
            line = (f"{name:8s} N={n:5d} K={k:5d} M={m}: b12x {tb:6.1f} us ({wbytes / tb / 1e3:5.0f} GB/s) | "
                    f"triton {tt:6.1f} us ({wbytes / tt / 1e3:5.0f} GB/s)  relerr b12x {eb:.1e} triton {et:.1e} "
                    f"bitsame {same:.3f}")
            if a.fma:
                tf = timed(lambda i: q8gemv.q8_gemv(xs[i % a.copies], packs[i % a.copies].values,
                                                    packs[i % a.copies].metadata, n, k, use_dot=False))
                line += f" | fma {tf:6.1f} us"
            if name == "lm_head":
                wbf = torch.randn((n, k), device=dev, dtype=torch.bfloat16)
                tl_ = timed(lambda i: torch.nn.functional.linear(xs[0], wbf), reps=8)
                line += f" | bf16 F.linear {tl_:6.1f} us"
                del wbf
            print(line, flush=True)
            for cfg in [c for c in a.cfg.split(",") if c]:
                bn, sp = (int(v) for v in cfg.split(":"))
                try:
                    tc = timed(lambda i: q8gemv.q8_gemv(xs[i % a.copies], packs[i % a.copies].values,
                                                        packs[i % a.copies].metadata, n, k, block_n=bn, split=sp))
                    print(f"    cfg BLOCK_N={bn} SPLIT={sp}: {tc:6.1f} us ({wbytes / tc / 1e3:5.0f} GB/s)")
                except Exception as e:  # noqa: BLE001
                    print(f"    cfg {cfg}: {type(e).__name__} {str(e)[:100]}")
            if a.mask and name in ("D", "U"):
                eg = torch.arange(512, device=dev, dtype=torch.int32) % 8
                for ng in [int(v) for v in a.groups.split(",")]:
                    # route ids that use exactly groups 0..ng-1
                    ids = torch.randint(0, 64, (m, 10), device=dev, dtype=torch.int32) * 8 + \
                        torch.randint(0, ng, (m, 10), device=dev, dtype=torch.int32)
                    ids[0, :ng] = torch.arange(ng, device=dev, dtype=torch.int32)
                    mode = 1 if name == "D" else 2
                    xm = xs[0].clone()
                    if mode == 2:   # z is zero on unused groups
                        xm.view(m, 8, -1)[:, ng:] = 0
                    y_m = q8gemv.q8_gemv(xm, packs[0].values, packs[0].metadata, n, k, ids=ids, expert_group=eg,
                                         mask_mode=mode)
                    y_full = q8gemv.q8_gemv(xm, packs[0].values, packs[0].metadata, n, k)
                    if mode == 1:
                        ok = torch.equal(y_m[:, :ng * 1792], y_full[:, :ng * 1792])
                    else:
                        ok = ((y_m.float() - y_full.float()).abs().max() / y_full.float().abs().max()).item() < 1e-2
                    xms = [x.clone() for x in xs]
                    tm = timed(lambda i: q8gemv.q8_gemv(xms[i % a.copies], packs[i % a.copies].values,
                                                        packs[i % a.copies].metadata, n, k, ids=ids,
                                                        expert_group=eg, mask_mode=mode))
                    print(f"    masked {ng}/8 groups: {tm:6.1f} us (x{tm / tb:.2f} of b12x)  match {ok}")
        del packs, refs
        torch.cuda.empty_cache()
    if a.shared:
        bench_shared(api, q8gemv, ms, a.copies, dev)


def bench_shared(api, q8gemv, ms, copies, dev):
    """Shared expert + residual add: stock sequence (b12x gate_up, silu_and_mul, b12x down, bf16 gate GEMV,
    sigmoid, mul, add) vs q8gemv.shared_expert_add (4 kernels)."""
    import torch.nn.functional as F
    gus, dns = [], []
    for _ in range(copies):
        gus.append(api.pack_weight(rand_q8(1280, 2560, dev), recipe="q8_0"))
        dns.append(api.pack_weight(rand_q8(2560, 640, dev), recipe="q8_0"))
    wg = (torch.randn(2560, device=dev) * 0.02).to(torch.bfloat16)
    for m in ms:
        qa = api.BlockscaledQuery(recipe="q8_0", num_tokens=2048, in_features=2560, padded_in_features=2560,
                                  out_features=1280, activation_mode="a16")
        qb = api.BlockscaledQuery(recipe="q8_0", num_tokens=2048, in_features=640, padded_in_features=640,
                                  out_features=2560, activation_mode="a16")
        pa = api.plan_regimes(qa, exact_m=(m,))
        pb = api.plan_regimes(qb, exact_m=(m,))
        xs = [torch.randn((m, 2560), device=dev, dtype=torch.bfloat16) for _ in range(copies)]
        rs = [torch.randn((m, 2560), device=dev, dtype=torch.bfloat16) for _ in range(copies)]

        def stock(i):
            x, r = xs[i % copies], rs[i % copies]
            gu = api.mm(x, gus[i % copies], plan=pa)
            h = torch.empty((m, 640), device=dev, dtype=torch.bfloat16)
            torch.ops._C.silu_and_mul(h, gu)
            d = api.mm(h, dns[i % copies], plan=pb)
            g = F.linear(x, wg.view(1, -1))
            return r + torch.sigmoid(g) * d

        def fused(i):
            return q8gemv.shared_expert_add(xs[i % copies], rs[i % copies], gus[i % copies], dns[i % copies], wg)
        y0, y1 = stock(0), fused(0)
        err = ((y0.float() - y1.float()).abs().max() / y0.float().abs().max()).item()
        ts, tf = timed(stock), timed(fused)
        print(f"shared expert + add M={m}: stock {ts:6.1f} us | fused {tf:6.1f} us  relerr {err:.1e}", flush=True)


if __name__ == "__main__":
    main()
