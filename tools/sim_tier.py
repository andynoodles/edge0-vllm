#!/usr/bin/env python3
"""Replay c1 decode routing through expert-tier policies: cold (non-VRAM) distinct experts per layer per verify window
(docs/SINGLE-STREAM-GAP.md, lever "Strata's adaptive tier").

  python3 sim_tier.py --trace '<dir>/routes-*.npz' --from-ts T0 --to-ts T1 --plan plans/plan-s1-12.8.json \
      [--extra-slots 0,2000,4000] [--acc .77,.55,.38] [--k 3]

Tokens: the single-request decode calls of the window (consecutive = same request; boundaries ignored). A verify
window covers the routes of the next k+1 real tokens (the drafts' own routes are not in the trace; the real
continuation stands in for them) and advances by an accepted length drawn from the per-position acceptance (seeded).
Policies, all starting from the plan's hot set and per-layer budgets (+ extra slots spread like the plan's):
  static     the plan, no swaps
  vllm       EDGE0_ADAPT as served: every EPOCH_TOK tokens (3 s at the measured rate), EMA half-life 10 epochs of
             distinct touches per call, per-layer pairs hottest cold vs coldest hot, swap if x > 1.25 y and
             x - y > 3 sqrt(x + y + 1), at most 64 pairs per epoch (largest gain first)
  strata     Strata's serve tier (generate.cpp adapt): usage += 1 per routed entry; every 4 windows per layer the
             candidates (non-resident, usage >= 2) sorted down are paired with the residents sorted up, swap while
             cand >= vict + 1.5, at most 96 per round (largest gain first) for all layers, then usage *= 0.7; a swapped
             expert is resident from the next window on
  lru        per-layer LRU over the whole budget (copy on miss), the bound of a pure recency tier
  fast       the plugin's EDGE0_TIER (tier.py): Strata's counts and gain rule, decided by the host one window late
             at the next step's boundary, copied while that step runs, resident from the step after it; every
             ``every`` windows, <= ``cap`` pairs per round and <= ``layer_cap`` per layer (= free VRAM slots),
             decay per window (``--fast k=v,...``)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import OrderedDict

import numpy as np

L, E = 48, 512
CURVE_BINS = (16, 32, 64, 128, 256, 512)     # --curve: token position inside a 512-token request (upper bounds)


PREFILL = {}      # decode-token index -> per-entry routing counts [L, E] of the prefill chunks just before it


def load_c1(pattern, t0, t1):
    toks = [[] for _ in range(L)]
    pend = None
    for f in sorted(glob.glob(pattern)):
        if not (t0 <= os.path.getmtime(f) <= t1):
            continue
        d = np.load(f)
        lay = [d[f"layer{l}"] for l in range(L)]
        off = 0
        for rows, real in d["calls"].tolist():
            if real == 1:
                if pend is not None:
                    PREFILL[len(toks[0])] = pend
                    pend = None
                for l in range(L):
                    r = lay[l][off:off + rows]
                    r = r[r[:, 0] >= 0]
                    if r.shape[0]:
                        toks[l].append(r[0].astype(np.int64))
            elif real > 1:
                pend = np.zeros((L, E)) if pend is None else pend
                for l in range(L):
                    r = lay[l][off:off + rows]
                    r = r[r[:, 0] >= 0].ravel()
                    np.add.at(pend[l], r[(r >= 0) & (r < E)].astype(np.int64), 1.0)
            off += rows
    T = min(len(t) for t in toks)
    return np.stack([np.stack(t[:T]) for t in toks])          # [L, T, 10]


def windows(T, k, acc, seed=0):
    rng = np.random.default_rng(seed)
    cond = [acc[0]] + [acc[i] / acc[i - 1] for i in range(1, len(acc))]   # P(pos i accepted | pos i-1 accepted)
    out, p = [], 0
    while p < T:
        a = 0
        while a < k and rng.random() < cond[a]:
            a += 1
        w = min(k + 1, T - p)
        out.append((p, w, min(a + 1, w)))
        p += min(a + 1, w)
    return out


def plan_hot(plan, extra):
    d = json.load(open(plan))
    n = np.array([d["layers"][str(l)]["n_hot"] for l in range(L)], float)
    add = np.floor(extra * n / n.sum()).astype(int)
    hot = np.zeros((L, E), bool)
    for l in range(L):
        o = d["layers"][str(l)]["order"]
        hot[l, o[:int(n[l]) + add[l]]] = True
    return hot


FAST = dict(every=1, cap=24, layer_cap=2, lag=0, delay=1, decay=0.7 ** 0.25, gain=1.5, min_use=2.0, pf=0.0)


def run_fast(toks, hot0, wins, curve=None, **kw):
    """A round after window wi sees the counts of windows <= wi - lag; its experts are resident from window
    wi + 1 + delay. Strata (blocking swaps): lag 0, delay 0. EDGE0_TIER: the host decides at the next step's
    boundary from the counts of the finished step (lag 0), the copies run during that step and the maps flip after
    it (delay 1)."""
    a = dict(FAST)
    a.update(kw)
    hot = hot0.copy()
    usage = np.zeros((L, E))
    hist = []                       # per-window count arrays not yet visible to the host
    pending = []                    # (first window resident, layer, in, out)
    cold_dist = swaps = 0
    pend_in = np.zeros((L, E), bool)
    def do_round(act):
        nonlocal swaps
        busy_out = np.zeros((L, E), bool)
        for _act, l, ce, ve in pending:
            busy_out[l, ve] = True
        res = hot | pend_in
        cand_all = []
        for l in range(L):
            c = np.flatnonzero(~res[l] & (usage[l] >= a["min_use"]))
            if not c.size:
                continue
            v = np.flatnonzero(hot[l] & ~busy_out[l])
            c = c[np.argsort(-usage[l, c], kind="stable")][:a["layer_cap"]]
            v = v[np.argsort(usage[l, v], kind="stable")][:c.size]
            for ce, ve in zip(c, v):
                if usage[l, ce] < usage[l, ve] + a["gain"]:
                    break
                cand_all.append((usage[l, ce] - usage[l, ve], l, ce, ve))
        cand_all.sort(key=lambda x: -x[0])
        for _, l, ce, ve in cand_all[:a["cap"]]:
            pending.append((act, l, ce, ve))
            pend_in[l, ce] = True
            swaps += 1

    pf_starts = {wi: PREFILL[p] for wi, (p, _w, _a) in enumerate(wins) if p in PREFILL}
    for wi, (p, w, _a) in enumerate(wins):
        if a["pf"] > 0 and wi in pf_starts:          # the prefill step's counts: one round before this window
            usage += a["pf"] * pf_starts[wi]
            do_round(wi + a["delay"])
        keep = []
        for act, l, ce, ve in pending:
            if act <= wi:
                hot[l, ce], hot[l, ve] = True, False
                pend_in[l, ce] = False
            else:
                keep.append((act, l, ce, ve))
        pending = keep
        c0 = cold_dist
        cnt = np.zeros((L, E))
        for l in range(L):
            ids = toks[l, p:p + w].ravel()
            dist = np.unique(ids)
            cold_dist += int((~hot[l, dist]).sum())
            np.add.at(cnt[l], ids, 1.0)
        if curve is not None:
            b = next(x for x in CURVE_BINS if p % 512 < x)
            curve.setdefault(b, [0, 0])
            curve[b][0] += cold_dist - c0
            curve[b][1] += 1
        hist.append(cnt)
        while len(hist) > a["lag"]:
            usage = usage * a["decay"] + hist.pop(0)
        if (wi + 1) % a["every"]:
            continue
        do_round(wi + 1 + a["delay"])
    return cold_dist / len(wins) / L, swaps / len(wins)


def run(policy, toks, hot0, wins, epoch_tok, curve=None, fast=None):
    if policy == "fast":
        return run_fast(toks, hot0, wins, curve, **(fast or {}))
    """curve: dict bin -> [cold, windows], filled with the window's position inside its 512-token request block."""
    hot = hot0.copy()
    cold_dist, swaps = 0, 0
    if policy == "lru":
        lrus = []
        for l in range(L):
            od = OrderedDict((int(e), None) for e in np.flatnonzero(hot[l]))
            lrus.append(od)
        for (p, w, a) in wins:
            for l in range(L):
                od, cap = lrus[l], int(hot0[l].sum())
                for e in set(toks[l, p:p + w].ravel().tolist()):
                    if e in od:
                        od.move_to_end(e)
                    else:
                        cold_dist += 1
                        swaps += 1
                        od[e] = None
                        if len(od) > cap:
                            od.popitem(last=False)
        return cold_dist / len(wins) / L, swaps / len(wins)
    usage = np.zeros((L, E))
    ema = np.zeros((L, E))
    epoch_cnt = np.zeros((L, E))
    tok_since = 0
    for wi, (p, w, a) in enumerate(wins):
        c0 = cold_dist
        for l in range(L):
            ids = toks[l, p:p + w].ravel()
            dist = np.unique(ids)
            cold_dist += int((~hot[l, dist]).sum())
            np.add.at(usage[l], ids, 1.0)
            epoch_cnt[l, dist] += 1.0
        if curve is not None:
            b = next(x for x in CURVE_BINS if p % 512 < x)
            curve.setdefault(b, [0, 0])
            curve[b][0] += cold_dist - c0
            curve[b][1] += 1
        tok_since += a
        if policy == "strata" and (wi + 1) % 4 == 0:
            cand_all = []
            for l in range(L):
                c = np.flatnonzero(~hot[l] & (usage[l] >= 2.0))
                if not c.size:
                    continue
                v = np.flatnonzero(hot[l])
                c = c[np.argsort(-usage[l, c], kind="stable")]
                v = v[np.argsort(usage[l, v], kind="stable")][:c.size]
                for ce, ve in zip(c, v):
                    if usage[l, ce] < usage[l, ve] + 1.5:
                        break
                    cand_all.append((usage[l, ce] - usage[l, ve], l, ce, ve))
            cand_all.sort(key=lambda x: -x[0])
            for _, l, ce, ve in cand_all[:96]:
                hot[l, ce], hot[l, ve] = True, False
                swaps += 1
            usage *= 0.7
        if policy == "vllm" and tok_since >= epoch_tok:
            tok_since = 0
            ema = ema * 0.5 ** (1 / 10) + epoch_cnt
            epoch_cnt[:] = 0
            pairs = []
            for l in range(L):
                c = np.flatnonzero(~hot[l])
                v = np.flatnonzero(hot[l])
                c = c[np.argsort(-ema[l, c], kind="stable")][:64]
                v = v[np.argsort(ema[l, v], kind="stable")][:64]
                for ce, ve in zip(c, v):
                    x, y = ema[l, ce], ema[l, ve]
                    if not (x > 1.25 * y and x - y > 3 * math.sqrt(x + y + 1)):
                        break
                    pairs.append((x - y, l, ce, ve))
            pairs.sort(key=lambda t: -t[0])
            for _, l, ce, ve in pairs[:64]:
                hot[l, ce], hot[l, ve] = True, False
                swaps += 1
    return cold_dist / len(wins) / L, swaps / len(wins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--from-ts", type=float, default=0)
    ap.add_argument("--to-ts", type=float, default=1e13)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--extra-slots", default="0")
    ap.add_argument("--acc", default=".77,.55,.38")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--tok-s", type=float, default=85.0, help="decode rate for the vllm policy's 3 s epochs")
    ap.add_argument("--policies", default="static,vllm,strata,lru")
    ap.add_argument("--curve", action="store_true", help="also print the reaction curve inside each request")
    ap.add_argument("--fast", action="append", default=[],
                    help="fast-policy variant as k=v,k=v (every, cap, layer_cap, lag, delay, decay, gain, min_use); "
                         "repeatable, each adds a policy 'fast<i>'")
    a = ap.parse_args()
    toks = load_c1(a.trace, a.from_ts, a.to_ts)
    T = toks.shape[1]
    acc = [float(x) for x in a.acc.split(",")][:a.k]
    wins = windows(T, a.k, acc) if a.k > 0 else [(p, 1, 1) for p in range(T)]
    tpw = T / len(wins)
    print(f"c1 decode tokens {T}, windows {len(wins)} (k={a.k}, {tpw:.2f} tokens/window, width {np.mean([w for _, w, _ in wins]):.2f})")
    print(f"{'extra slots':>11s} {'slots':>6s} {'policy':>7s} {'cold/layer/window':>18s} {'cold/token (all L)':>19s} {'swaps/window':>13s}")
    curves = {}
    for extra in (int(x) for x in a.extra_slots.split(",")):
        hot0 = plan_hot(a.plan, extra)
        variants = [(pol, None) for pol in a.policies.split(",") if pol]
        for i, spec in enumerate(a.fast):
            kw = {k: float(v) for k, v in (x.split("=") for x in spec.split(",") if x)}
            kw = {k: (int(v) if k in ("every", "cap", "layer_cap", "lag", "delay") else v) for k, v in kw.items()}
            print(f"  fast{i}: {dict(FAST, **kw)}")
            variants.append((f"fast{i}", kw))
        for pol, kw in variants:
            cv = {} if a.curve else None
            c, s = run("fast" if kw is not None else pol, toks, hot0, wins, a.tok_s * 3.0, cv, kw)
            print(f"{extra:>11d} {int(hot0.sum()):>6d} {pol:>7s} {c:>18.2f} {c * L / tpw:>19.1f} {s:>13.1f}")
            if cv is not None:
                curves[(extra, pol)] = cv
    if curves:   # the trace's c1 phase is 512-token requests back to back (2 warm-ups + 3 prompts at 512 tokens)
        print("cold per layer-window by token position inside the request:")
        print(f"{'slots':>6s} {'policy':>7s} " + " ".join(f"{'<' + str(b):>7s}" for b in CURVE_BINS))
        for (extra, pol), cv in curves.items():
            print(f"{int(plan_hot(a.plan, extra).sum()):>6d} {pol:>7s} " +
                  " ".join(f"{cv[b][0] / max(cv[b][1], 1) / L:>7.2f}" if b in cv else f"{'-':>7s}" for b in CURVE_BINS))


if __name__ == "__main__":
    main()
