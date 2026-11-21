#!/usr/bin/env python3
"""Launch-parameter sweep of edge0_vllm.q8gemv on the latent D / U shapes vs b12x (same harness as q8gemv_bench).

  python3 q8gemv_sweep.py [--m 1,4] [--shapes D,U]
Prints us per call for BLOCK_N x SPLIT x UNROLL x warps x stages; U splits are multiples of the 8 groups (mask-safe).
"""
import argparse
import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q8gemv_bench import SHAPES, rand_q8, timed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="1,4")
    ap.add_argument("--shapes", default="D,U")
    ap.add_argument("--copies", type=int, default=6)
    a = ap.parse_args()
    from vllm.utils.b12x import get_b12x_blockscaled
    from edge0_vllm import q8gemv
    api = get_b12x_blockscaled()
    dev = torch.device("cuda")
    grid = {
        "D": dict(bn=(32, 64, 128), sp=(1, 2, 4), un=(1, 2, 4), w=(2, 4, 8), st=(3, 4, 6)),
        "U": dict(bn=(32, 64), sp=(8, 16, 32), un=(1, 2, 4), w=(2, 4, 8), st=(3, 4, 6)),
        "qkvz": dict(bn=(32, 64), sp=(1, 2), un=(1, 2, 4), w=(4, 8), st=(4, 6)),
    }
    for name in a.shapes.split(","):
        n, k = SHAPES[name]
        packs = [api.pack_weight(rand_q8(n, k, dev), recipe="q8_0") for _ in range(a.copies)]
        query = api.BlockscaledQuery(recipe="q8_0", num_tokens=2048, in_features=k, padded_in_features=k,
                                     out_features=n, activation_mode="a16")
        ms = [int(v) for v in a.m.split(",")]
        plan = api.plan_regimes(query, exact_m=tuple(ms))
        for m in ms:
            xs = [torch.randn((m, k), device=dev, dtype=torch.bfloat16) for _ in range(a.copies)]
            tb = timed(lambda i: api.mm(xs[i % a.copies], packs[i % a.copies], plan=plan))
            res = []
            g = grid[name]
            for bn, sp, un, w, st in itertools.product(g["bn"], g["sp"], g["un"], g["w"], g["st"]):
                kb = k // 32
                if kb % sp or (kb // sp) % un or n % bn:
                    continue
                try:
                    t = timed(lambda i: q8gemv.q8_gemv(xs[i % a.copies], packs[i % a.copies].values,
                                                       packs[i % a.copies].metadata, n, k, block_n=bn, split=sp,
                                                       unroll=un, warps=w, stages=st), iters=10)
                except Exception as e:  # noqa: BLE001
                    print(f"  {name} bn{bn} sp{sp} un{un} w{w} st{st}: {type(e).__name__} {str(e)[:80]}")
                    continue
                res.append((t, bn, sp, un, w, st))
            res.sort()
            print(f"{name} M={m}: b12x {tb:.1f} us; best triton:", flush=True)
            for t, bn, sp, un, w, st in res[:6]:
                print(f"    {t:6.1f} us  BLOCK_N={bn} SPLIT={sp} UNROLL={un} warps={w} stages={st}", flush=True)
        del packs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
