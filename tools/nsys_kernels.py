#!/usr/bin/env python3
"""Kernel-level table of steady decode steps (single-stream dense-path work, TUNING-LOG "Single-stream").

  nsys_kernels.py <db.sqlite> [--steps N] [--layer-seq]

Per category (nsys_breakdown.structural_labels / moe_tiers) and kernel (short name + grid): kernels per step, mean
duration (us), total per step (ms). ``--layer-seq`` also prints the main-stream kernel sequence of one GDN layer
and one QSA layer (name, grid, us), so fusion candidates are visible.
"""
import argparse
import collections
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nsys_breakdown as nb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--layer-seq", action="store_true")
    a = ap.parse_args()
    tr = nb.Trace(a.db)
    steps = nb.find_steps(tr)
    work = nb.assign_work(tr, steps)
    idx = nb.nvtx_index(tr)
    agg = collections.defaultdict(list)       # (cat, name, grid) -> durations
    nsteps = 0
    ctx_cache = {}
    seq_done = False
    for i, s in enumerate(steps):
        if s["ctx"] != 0 or i not in work:
            continue
        ks = sorted((k for k in work[i] if k["kind"] == "k"), key=lambda k: k["start"])
        spec = set()
        for k in ks:
            ap_ = k["api"]
            if ap_ is None:
                continue
            if ap_["corr"] not in ctx_cache:
                st = nb.nvtx_stack_at(idx, ap_["tid"], ap_["start"])
                ctx_cache[ap_["corr"]] = "spec.propose" in st or "spec.reject" in st
            if ctx_cache[ap_["corr"]]:
                spec.add(id(k))
        fwd = [k for k in ks if k["graph"] is not None and id(k) not in spec]
        if len(fwd) < 1000:
            continue
        tiers = nb.moe_tiers(fwd)
        lab = nb.structural_labels(fwd)
        nsteps += 1
        for k in ks:
            c = ("drafter" if id(k) in spec else tiers.get(id(k)) or lab.get(id(k)) or
                 nb.report_category(k, set(), None, None))
            agg[(c, k["name"], k["grid"])].append((k["end"] - k["start"]) / 1e3)
        if a.layer_seq and not seq_done and nsteps == 3:
            seq_done = True
            main_s = collections.Counter(k["stream"] for k in fwd).most_common(1)[0][0]
            mk = [k for k in fwd if k["stream"] == main_s]
            hc = [j for j, k in enumerate(mk) if k["name"] == "_hc_combine_norm_kernel"]
            # layers 0..3: print from the 3rd HC block (layer 1 attention side) for ~4 HC blocks (2 layers)
            for title, b in (("layer 2 (GDN?)", 4), ("layer 3 (QSA?)", 6)):
                if len(hc) > b + 2:
                    print(f"--- {title}: main-stream kernels from HC block {b}")
                    for k in mk[hc[b]:hc[b + 2]]:
                        print(f"  {lab.get(id(k), '?'):15s} {(k['end'] - k['start']) / 1e3:7.1f} us  "
                              f"{str(k['grid']):16s} {k['name'][:90]}")
        if nsteps >= a.steps:
            break
    print(f"{nsteps} decode steps")
    bycat = collections.defaultdict(float)
    rows = []
    for (c, n, g), ds in agg.items():
        per = len(ds) / nsteps
        tot = sum(ds) / nsteps / 1e3
        bycat[c] += tot
        rows.append((c, tot, per, statistics.mean(ds), n, g))
    for c in sorted(bycat, key=lambda c: -bycat[c]):
        print(f"== {c}: {bycat[c]:.3f} ms/step")
        for r in sorted([r for r in rows if r[0] == c], key=lambda r: -r[1]):
            if r[1] < 0.005:
                continue
            print(f"   {r[1]:7.3f} ms  {r[2]:6.1f}/step  {r[3]:7.2f} us  {str(r[5]):16s} {r[4][:100]}")


if __name__ == "__main__":
    main()
