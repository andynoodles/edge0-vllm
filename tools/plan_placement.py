#!/usr/bin/env python3
"""Hot/cold placement planner for Edge0 on vLLM, from routing traces of the real workload (TidalMoE
``tools/analyze_routing.py`` + ``tools/plan_convex.py``, adapted to per-expert slots and the RTX 5090's PCIe costs).

Input: ``EDGE0_TRACE=routes`` shards (``routes-<pid>-<t>-NNNNN.npz``: ``layer<l>`` [rows, 10] int16 logical ids, -1
= padding; ``calls`` [n, 2] = rows and real tokens of every MoE call of layer 0, in order).

  python3 plan_placement.py --trace '<dir>/routes-*.npz' --gib 9.5 --out plan.json [--profile profile.bin]
        [--decode-max 64] [--regimes 1:1,8:1,32:1] [--cost-cold-us 36 --cost-hot-us 1.0 --fixed-hot-us 40 ...]

What it does:
1. Splits the trace into MoE calls; calls with <= ``--decode-max`` real tokens are decode steps, the rest prefill.
2. p[l, e] = per-token selection probability in decode tokens (what the hot set should serve; TidalMoE found
   prefill-weighted scores pull the hot set toward prompt experts that prefill touches anyway).
3. Per layer, experts are ordered by p (the touch probability ``1 - (1 - p)^B`` is monotone in p, so this order is
   optimal for every batch size B); only the per-layer budgets n_hot[l] depend on the batch-size mix.
4. Per-layer budgets: greedy marginal allocation of the H hot slots (H = GiB x 2^30 / 1,935,360) under the layer cost
   ``E[max(hot part, cold part)]`` (hot and cold Marlin calls overlap on two streams), summed over the regimes
   (decode batch sizes B with weights, each normalized by its step time), using the normal approximation of
   ``plan_convex.py``. ``--linear`` instead takes the global top-H (layer, expert) pairs by touch probability.
5. Reports simulated distinct cold experts and cold GB per decode step at B = 1, 8, 16, 32 for: the profile.bin plan
   (bring-up), the linear plan and the convex plan; with ``--holdout`` the plan is built on the first half of the
   trace and evaluated on the second.
Writes ``edge0-placement-v1`` JSON (``EDGE0_PLACEMENT=plan:<path>``).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

L, E, K = 48, 512, 10
SLOT_BYTES = 1_935_360


def load_trace(pattern: str, decode_max: int):
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"no trace files match {pattern}")
    dec = [[] for _ in range(L)]
    pre = [[] for _ in range(L)]
    steps = []            # per decode call: list over layers of id arrays [n, 10]
    n_calls = n_dec_calls = 0
    for f in files:
        d = np.load(f)
        lay = [d[f"layer{l}"] for l in range(L)]
        calls = d["calls"] if "calls" in d.files else None
        if int(d["lost"][0]) if "lost" in d.files else 0:
            print(f"warning: {f}: {int(d['lost'][0])} rows lost to ring overflow; skipped", file=sys.stderr)
            continue
        if calls is None or calls.shape[0] == 0 or int(calls[:, 0].sum()) != lay[0].shape[0] or \
                any(a.shape[0] != lay[0].shape[0] for a in lay):
            print(f"warning: {f}: call ring does not match rows ({None if calls is None else calls[:, 0].sum()} vs "
                  f"{lay[0].shape[0]}); skipped", file=sys.stderr)
            continue
        off = 0
        for rows, real in calls.tolist():
            n_calls += 1
            sl = [a[off:off + rows] for a in lay]
            off += rows
            sl = [a[a[:, 0] >= 0] for a in sl]                    # drop padding rows
            if sl[0].shape[0] == 0:
                continue
            if real <= decode_max:
                n_dec_calls += 1
                for l in range(L):
                    dec[l].append(sl[l])
                steps.append(sl)
            else:
                for l in range(L):
                    pre[l].append(sl[l])
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros((0, K), np.int16)
    dec = [cat(x) for x in dec]
    pre = [cat(x) for x in pre]
    print(f"trace: {len(files)} shards, {n_calls} calls, {n_dec_calls} decode calls, "
          f"{dec[0].shape[0]} decode tokens, {pre[0].shape[0]} prefill tokens")
    return dec, pre, steps


def sel_prob(rows_per_layer) -> np.ndarray:
    p = np.zeros((L, E))
    for l, a in enumerate(rows_per_layer):
        if a.shape[0]:
            p[l] = np.bincount(a.astype(np.int64).ravel(), minlength=E)[:E] / a.shape[0]
    return p


def read_profile_order(path: str, H: int):
    blob = Path(path).read_bytes()
    assert blob[:4] == b"STRP"
    ver, nl, ne, slots, n = struct.unpack_from("<5I", blob, 4)
    ranked = np.frombuffer(blob, "<u2", count=2 * n, offset=24).reshape(n, 2).astype(np.int64)
    hot = np.zeros((L, E), bool)
    taken = 0
    for layer, expert in ranked.tolist():
        if taken >= H:
            break
        if layer < L and expert < E and not hot[layer, expert]:
            hot[layer, expert] = True
            taken += 1
    return hot


# ------------------------------------------------------------------------------------------- cost model
def Phi(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def phi(x):
    return np.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)


def layer_cost_table(p_l: np.ndarray, B: float, a: dict) -> np.ndarray:
    """cost[n] (us) for n = 0..E hot experts (the n most probable), decode step of B tokens:
    E[max(fixed_hot + hot_us * H_touch, fixed_cold + cold_us * C_touch)] with H/C sums of Bernoulli touches
    (normal approximation, as plan_convex.py). With no cold experts the cold call does not run, and vice versa."""
    order = np.argsort(-p_l, kind="stable")
    q = 1.0 - (1.0 - np.clip(p_l[order], 0, 1)) ** B
    v = q * (1 - q)
    cq = np.concatenate([[0.0], np.cumsum(q)])
    cv = np.concatenate([[0.0], np.cumsum(v)])
    n = np.arange(E + 1)
    mh = a["fixed_hot"] * (n > 0) + a["hot_us"] * cq
    vh = a["hot_us"] ** 2 * cv
    mc = a["fixed_cold"] * (n < E) + a["cold_us"] * (cq[-1] - cq)
    vc = a["cold_us"] ** 2 * (cv[-1] - cv)
    if a.get("mode", "max") == "sum":
        return mh + mc
    mu = mc - mh
    s = np.sqrt(np.maximum(vh + vc, 1e-9))
    return mh + mu * Phi(mu / s) + s * phi(mu / s)


def plan_budgets(p: np.ndarray, H: int, regimes, a: dict, linear: bool = False, floor: np.ndarray | None = None):
    """n_hot[l] for the H slots. regimes: [(B, weight)]."""
    if linear:
        w = sum(wt * (1.0 - (1.0 - np.clip(p, 0, 1)) ** B) for B, wt in regimes)
        flat = np.argsort(-w.ravel(), kind="stable")[:H]
        n = np.bincount(flat // E, minlength=L)
        return n
    tabs = []
    for B, wt in regimes:
        t = np.stack([layer_cost_table(p[l], B, a) for l in range(L)])           # [L, E+1]
        step = t[:, 0].sum()                                                      # all-cold step (normalizer)
        tabs.append(wt * t / step)
    tab = sum(tabs)
    n = np.zeros(L, int) if floor is None else floor.copy()
    gain = tab[np.arange(L), n] - tab[np.arange(L), np.minimum(n + 1, E)]
    left = H - n.sum()
    import heapq
    heap = [(-gain[l], l) for l in range(L) if n[l] < E]
    heapq.heapify(heap)
    while left > 0 and heap:
        g, l = heapq.heappop(heap)
        n[l] += 1
        left -= 1
        if n[l] < E:
            heapq.heappush(heap, (-(tab[l, n[l]] - tab[l, n[l] + 1]), l))
    # local search: move single slots between layers while the objective drops
    obj = lambda nn: tab[np.arange(L), nn].sum()
    cur = obj(n)
    for _ in range(4 * L):
        best = None
        dec_gain = tab[np.arange(L), np.maximum(n - 1, 0)] - tab[np.arange(L), n]       # cost increase of -1
        inc_gain = tab[np.arange(L), n] - tab[np.arange(L), np.minimum(n + 1, E)]       # cost decrease of +1
        dec_gain[n == 0] = np.inf
        inc_gain[n == E] = -np.inf
        a_l, b_l = int(np.argmin(dec_gain)), int(np.argmax(inc_gain))
        if a_l != b_l and inc_gain[b_l] - dec_gain[a_l] > 1e-12:
            n[a_l] -= 1
            n[b_l] += 1
            cur = obj(n)
        else:
            break
    return n


def hot_mask_from_budgets(p: np.ndarray, n: np.ndarray) -> np.ndarray:
    hot = np.zeros((L, E), bool)
    for l in range(L):
        hot[l, np.argsort(-p[l], kind="stable")[:n[l]]] = True
    return hot


def simulate(hot: np.ndarray, steps, B_list=(1, 8, 16, 32), p=None, seed=0):
    """Distinct cold experts per decode step of B tokens, by bootstrap: B tokens drawn from different recorded
    decode steps (independent requests), per layer. Returns {B: (cold/step, hot/step)}."""
    rng = np.random.default_rng(seed)
    tok = [np.concatenate([s[l] for s in steps]) for l in range(L)] if steps else None
    out = {}
    for B in B_list:
        reps = 400 if B <= 8 else 200
        cold = hotc = 0.0
        n = tok[0].shape[0]
        for _ in range(reps):
            idx = rng.choice(n, size=B, replace=False)
            for l in range(L):
                u = np.unique(tok[l][idx].astype(np.int64))
                h = hot[l, u].sum()
                hotc += h
                cold += u.size - h
        out[B] = (cold / reps, hotc / reps)
    return out


def selftest(d: str) -> str:
    """Write a synthetic Zipf-routed trace (c1 shard + batched shards with graph padding) into ``d``."""
    rng = np.random.default_rng(0)
    pops = []
    for _ in range(L):
        w = 1.0 / np.arange(1, E + 1) ** 0.9
        rng.shuffle(w)
        pops.append(w / w.sum())
    Path(d).mkdir(parents=True, exist_ok=True)
    for shard in range(3):
        calls, rows = [], [[] for _ in range(L)]
        for _ in range(120):
            b = 1 if shard == 0 else int(rng.integers(4, 17))
            pad = 16 - b if b not in (1, 2, 4, 8, 16) else 0
            for _ in range(b):
                for l in range(L):
                    rows[l].append(rng.choice(E, K, replace=False, p=pops[l]))
            for _ in range(pad):
                for l in range(L):
                    rows[l].append(-np.ones(K, np.int64))
            calls.append((b + pad, b))
        np.savez(f"{d}/routes-1-1-{shard:05d}.npz", calls=np.array(calls), lost=np.array([0]),
                 **{f"layer{l}": np.array(rows[l], np.int16) for l in range(L)})
    return f"{d}/routes-*.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="", help="glob of routes-*.npz shards")
    ap.add_argument("--selftest", default="", help="write a synthetic trace into this dir and plan on it")
    ap.add_argument("--gib", type=float, help="VRAM for hot experts (GiB)")
    ap.add_argument("--slots", type=int, help="hot slots (instead of --gib)")
    ap.add_argument("--out", default="")
    ap.add_argument("--profile", default="/models/Edge0-145B-vllm/profile.bin")
    ap.add_argument("--decode-max", type=int, default=64)
    ap.add_argument("--regimes", default="1:1,8:1,32:1", help="B:weight,...")
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--eval-trace", default="", help="score the plans on this capture instead of --trace")
    ap.add_argument("--cold-us", type=float, default=36.0)
    ap.add_argument("--hot-us", type=float, default=1.0)
    ap.add_argument("--fixed-hot-us", type=float, default=40.0)
    ap.add_argument("--fixed-cold-us", type=float, default=40.0)
    ap.add_argument("--mode", default="max", choices=("max", "sum"))
    ap.add_argument("--min-hot", type=int, default=0, help="floor of n_hot per layer")
    a = ap.parse_args()
    if a.selftest:
        a.trace = selftest(a.selftest)
    H = a.slots if a.slots else int(a.gib * (1 << 30) // SLOT_BYTES)
    regimes = [(float(b), float(w)) for b, w in (x.split(":") for x in a.regimes.split(","))]
    cost = {"cold_us": a.cold_us, "hot_us": a.hot_us, "fixed_hot": a.fixed_hot_us, "fixed_cold": a.fixed_cold_us,
            "mode": a.mode}
    dec, pre, steps = load_trace(a.trace, a.decode_max)
    if a.eval_trace:                         # build on --trace, score on another capture (e.g. other prompts)
        _d, _p, eval_steps = load_trace(a.eval_trace, a.decode_max)
        build_steps = steps
        p = sel_prob(dec)
    elif a.holdout:
        half = len(steps) // 2
        build_steps, eval_steps = steps[:half], steps[half:]
        p = sel_prob([np.concatenate([s[l] for s in build_steps]) for l in range(L)])
    else:
        build_steps = eval_steps = steps
        p = sel_prob(dec)
    p_pre = sel_prob(pre)
    print(f"budget H = {H} hot slots ({H * SLOT_BYTES / 2**30:.2f} GiB, {H / (L * E) * 100:.1f}% of experts); "
          f"regimes {regimes}; cost {cost}")
    # skew diagnostics (TidalMoE 80/20 test)
    s = -np.sort(-p, axis=1)
    for frac in (0.1, 0.2, 0.3, 0.5):
        k = int(frac * E)
        sh = s[:, :k].sum(1) / np.maximum(s.sum(1), 1e-12)
        print(f"  top {int(frac * 100):2d}% of experts carry {sh.mean() * 100:5.1f}% of decode selections "
              f"(layer min {sh.min() * 100:4.1f}%, max {sh.max() * 100:4.1f}%)")
    if p_pre.sum():
        k20 = int(0.2 * E)
        jac = np.mean([len(set(np.argsort(-p[l])[:k20]) & set(np.argsort(-p_pre[l])[:k20])) /
                       len(set(np.argsort(-p[l])[:k20]) | set(np.argsort(-p_pre[l])[:k20])) for l in range(L)])
        print(f"  decode vs prefill top-20% sets: Jaccard {jac:.2f}")
    plans = {}
    if a.profile and Path(a.profile).exists():
        plans["profile"] = read_profile_order(a.profile, H)
    n_lin = plan_budgets(p, H, regimes, cost, linear=True)
    plans["linear"] = hot_mask_from_budgets(p, n_lin)
    floor = np.full(L, a.min_hot) if a.min_hot else None
    n_cvx = plan_budgets(p, H, regimes, cost, floor=floor)
    plans["convex"] = hot_mask_from_budgets(p, n_cvx)
    print(f"{'plan':10s} " + " ".join(f"{'B=' + str(B):>22s}" for B in (1, 8, 16, 32)) +
          "   (simulated per decode step, all 48 layers: cold distinct [hot distinct], cold GB)")
    sims = {}
    for name, hot in plans.items():
        sim = simulate(hot, eval_steps)
        sims[name] = sim
        print(f"{name:10s} " + " ".join(f"{c:7.0f} [{h:5.0f}] {c * SLOT_BYTES / 1e9:5.2f}GB" for B, (c, h)
                                         in sim.items()))
        sel_cold = float((p * ~hot).sum() / p.sum())
        print(f"{'':10s} cold share of decode selections {sel_cold * 100:.1f}%; n_hot per layer min/max "
              f"{hot.sum(1).min()}/{hot.sum(1).max()}")
    print("per-layer n_hot convex :", " ".join(str(int(x)) for x in n_cvx))
    if "profile" in plans:
        print("per-layer n_hot profile:", " ".join(str(int(x)) for x in plans["profile"].sum(1)))
    if a.out:
        hot = plans["linear" if a.linear else "convex"]
        layers = {}
        for l in range(L):
            o = np.argsort(-p[l], kind="stable")
            h = [int(e) for e in o if hot[l, e]]
            c = [int(e) for e in o if not hot[l, e]]
            layers[str(l)] = {"order": h + c, "n_hot": len(h)}
        d = {"format": "edge0-placement-v1", "num_layers": L, "num_experts": E, "slot_bytes": SLOT_BYTES,
             "source": f"plan_placement.py {'linear' if a.linear else 'convex'} H={H} trace={a.trace} "
                       f"regimes={a.regimes} cost={cost}",
             "simulated": {k: {str(B): [round(c, 1), round(h, 1)] for B, (c, h) in v.items()} for k, v in sims.items()},
             "layers": layers}
        Path(a.out).write_text(json.dumps(d))
        print("written", a.out)


if __name__ == "__main__":
    main()
