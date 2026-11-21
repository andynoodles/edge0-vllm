#!/usr/bin/env python3
"""Request-level expert locality at concurrency 1 (TidalMoE EXPERT-CACHE-ANALYSIS "what this analysis can't answer
yet": does a fast LRU staging tier pay?). On this PCIe box a copy-on-miss into VRAM costs the same PCIe bytes as the
zero-copy read it replaces, so a dynamic tier pays whenever its hit rate beats what the same VRAM buys as static hot
slots.

  python3 analyze_locality.py --trace '<dir>/routes-*.npz' --from-ts <t0> --to-ts <t1> --gib 9.5 \
      [--plan plan.json | --profile profile.bin] [--p-trace '<glob of batched shards>']

Replays the decode tokens of single-request phases (calls with exactly 1 real token, consecutive = same request,
request boundaries ignored) per layer, with the static hot set of the plan, and compares per layer:
  * reuse: share of cold touches at token t already touched in the previous W tokens;
  * dynamic: an LRU of S VRAM slots per layer in front of the cold pool (copy on miss);
  * static: the same 48 x S slots added to the static hot set (next experts by decode probability).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import OrderedDict

import numpy as np

L, E = 48, 512
SLOT = 1_935_360


def load_c1(pattern, t0, t1):
    toks = [[] for _ in range(L)]
    for f in sorted(glob.glob(pattern)):
        m = os.path.getmtime(f)
        if not (t0 <= m <= t1):
            continue
        d = np.load(f)
        calls = d["calls"]
        lay = [d[f"layer{l}"] for l in range(L)]
        off = 0
        for rows, real in calls.tolist():
            if real == 1:
                for l in range(L):
                    r = lay[l][off:off + rows]
                    r = r[r[:, 0] >= 0]
                    if r.shape[0]:
                        toks[l].append(r[0])
            off += rows
    return [np.stack(t) if t else np.zeros((0, 10), np.int16) for t in toks]


def decode_prob(pattern):
    cnt = np.zeros((L, E))
    n = 0
    for f in sorted(glob.glob(pattern)):
        d = np.load(f)
        calls = d["calls"]
        lay = [d[f"layer{l}"] for l in range(L)]
        off = 0
        for rows, real in calls.tolist():
            if real <= 64:
                for l in range(L):
                    r = lay[l][off:off + rows]
                    r = r[r[:, 0] >= 0].astype(np.int64)
                    cnt[l] += np.bincount(r.ravel(), minlength=E)[:E]
                n += real
            off += rows
    return cnt / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--from-ts", type=float, default=0)
    ap.add_argument("--to-ts", type=float, default=1e12)
    ap.add_argument("--gib", type=float, default=9.5)
    ap.add_argument("--plan", default="")
    ap.add_argument("--p-trace", default="", help="shards for the decode probability ranking (default: --trace)")
    a = ap.parse_args()
    toks = load_c1(a.trace, a.from_ts, a.to_ts)
    T = toks[0].shape[0]
    print(f"c1 decode tokens: {T}")
    if T < 50:
        return
    p = decode_prob(a.p_trace or a.trace)
    H = int(a.gib * (1 << 30) // SLOT)
    if a.plan:
        d = json.load(open(a.plan))
        hot = np.zeros((L, E), bool)
        for l in range(L):
            ent = d["layers"][str(l)]
            hot[l, ent["order"][:ent["n_hot"]]] = True
        rank = [np.array([e for e in d["layers"][str(l)]["order"]]) for l in range(L)]
    else:
        flat = np.argsort(-p.ravel(), kind="stable")[:H]
        hot = np.zeros((L, E), bool)
        hot.ravel()[flat] = True
        rank = [np.argsort(-p[l], kind="stable") for l in range(L)]
    cold_touch = sum(int((~hot[l, toks[l].astype(np.int64)]).sum()) for l in range(L))
    print(f"static hot set: {hot.sum()} slots; cold touches per token {cold_touch / T:.1f} (all layers)")
    # reuse distance
    for W in (1, 2, 4, 8, 16, 32):
        reuse = 0
        for l in range(L):
            tl = toks[l].astype(np.int64)
            for t in range(1, T):
                c = [e for e in tl[t] if not hot[l, e]]
                if not c:
                    continue
                prev = set(tl[max(0, t - W):t].ravel().tolist())
                reuse += sum(1 for e in c if e in prev)
        print(f"  reuse within previous {W:2d} tokens: {reuse / max(cold_touch, 1) * 100:5.1f}% of cold touches")
    print(f"{'S/layer':>8s} {'VRAM GiB':>9s} {'LRU cold/token':>15s} {'static+S cold/token':>20s}")
    for S in (4, 8, 16, 32, 64):
        miss = 0
        for l in range(L):
            lru = OrderedDict()
            for row in toks[l].astype(np.int64):
                for e in row:
                    if hot[l, e]:
                        continue
                    if e in lru:
                        lru.move_to_end(e)
                    else:
                        miss += 1
                        lru[e] = 1
                        if len(lru) > S:
                            lru.popitem(last=False)
        # same VRAM as static slots: per layer the next S experts by rank that are not hot
        st = 0
        for l in range(L):
            extra = [e for e in rank[l] if not hot[l, e]][:S]
            h2 = hot[l].copy()
            h2[extra] = True
            st += int((~h2[toks[l].astype(np.int64)]).sum())
        print(f"{S:8d} {48 * S * SLOT / 2**30:9.2f} {miss / T:15.1f} {st / T:20.1f}")


if __name__ == "__main__":
    main()
