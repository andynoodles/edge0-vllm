#!/usr/bin/env python3
"""Aggregate the c1 measurements of bench/gap-strata.sh and bench/gap-vllm.sh (docs/SINGLE-STREAM-GAP.md).

  python3 gap_parse.py strata <engine.log> [--metrics <pass>.metrics.json --last N]
      Strata's engine stderr (STRATA_DECODE_TIMING=1 / STRATA_VERIFY_PROFILE=1): per-request decode timing lines and
      GPU stage lines, weighted by verify windows over the last N requests (default: all); decode expert-cache hit
      rate lines; adaptive tier totals. With --metrics: the server's per-request history (GET /metrics) of the last N.
  python3 gap_parse.py vllm <serve.log> <line0> <line1>
      the vLLM serve log lines of one bench pass (gap-vllm.sh prints "SLOG <label> <file> <l0> <l1>"):
      EDGE0_TRACE=bw per-step hot/cold distinct experts, cpu-experts split, adaptive epochs.
Standard library only.
"""
import json
import re
import sys

NUM = r"([-0-9.]+)"


def wavg(rows, key, wkey="windows"):
    w = sum(r[wkey] for r in rows if key in r)
    return sum(r[key] * r[wkey] for r in rows if key in r) / w if w else float("nan")


TIMING = re.compile(
    r"strata decode timing: (\d+) windows, avg T " + NUM + r", " + NUM + r" tokens/window, " + NUM + r" ms/window = "
    r"adapt wait " + NUM + r" \+ verify " + NUM + r" \(GPU-reach wait " + NUM + r" \+ per-layer host " + NUM +
    r" \[plan " + NUM + r" actq " + NUM + r" jobs " + NUM + r" CPU " + NUM + r"\] \+ stage " + NUM +
    r"\) \+ commit/emit " + NUM + r" \+ draft " + NUM + r" \+ adapt join " + NUM + r" \+ other " + NUM +
    r"; per layer-window: CPU experts " + NUM + r" \(" + NUM + r" entries\), VRAM hits " + NUM + r", PCIe " + NUM +
    r"; " + NUM + r" swaps/window")
TKEYS = ["windows", "T", "tok_per_window", "ms_window", "adapt_wait", "verify", "gpu_reach_wait", "per_layer_host",
         "plan", "actq", "jobs", "cpu", "stage", "commit_emit", "draft", "adapt_join", "other", "cpu_experts_lw",
         "cpu_entries_lw", "vram_hits_lw", "pcie_lw", "swaps_window"]
HIT = re.compile(r"decode expert cache hit rate: " + NUM + r"% \((\d+) hits / (\d+) lookups\)(?:; (\d+) more read by the "
                 r"GPU over PCIe or from another GPU \(" + NUM + r"% of all (\d+) routed\))?")


def strata(args):
    path = args[0]
    last = int(args[args.index("--last") + 1]) if "--last" in args else 0
    lines = open(path, errors="replace").read().splitlines()
    timing, stages, hits = [], [], []
    for ln in lines:
        m = TIMING.search(ln)
        if m:
            r = dict(zip(TKEYS, (float(x) for x in m.groups())))
            timing.append(r)
            continue
        if "strata decode GPU stages (ms/window):" in ln:
            body = ln.split("(ms/window):", 1)[1]
            tot = re.search(r"total " + NUM + r" ms/window over (\d+) windows", body)
            r = {"windows": float(tot.group(2)) if tot else 0.0, "total": float(tot.group(1)) if tot else 0.0}
            for part, tag in ((body.split("|")[0], "gdn"), (body.split("|")[1] if "|" in body else "", "qsa")):
                part = part.replace("GDN layers:", "").replace("QSA layers:", "")
                for name, v in re.findall(r"\s*(.+?)\s+([0-9]+\.[0-9]+)(?=\s|$)", part):
                    r[f"{tag}:{name.strip()}"] = float(v)
            stages.append(r)
            continue
        m = HIT.search(ln)
        if m:
            hits.append([float(x) if x is not None else 0.0 for x in m.groups()])
    if last:
        timing, stages, hits = timing[-last:], stages[-last:], hits[-last:]
    out = {}
    if timing:
        out["timing"] = {k: round(wavg(timing, k), 3) for k in TKEYS if k != "windows"}
        out["timing"]["windows"] = sum(r["windows"] for r in timing)
        out["timing"]["requests"] = len(timing)
    if stages:
        keys = sorted({k for r in stages for k in r if k != "windows"})
        out["gpu_stages_ms_window"] = {k: round(wavg(stages, k), 3) for k in keys}
        out["gpu_stages_ms_window"]["windows"] = sum(r["windows"] for r in stages)
    if hits:
        h = sum(x[1] for x in hits)
        lk = sum(x[2] for x in hits)
        off = sum(x[3] for x in hits)
        out["hit_rate"] = {"hits": h, "lookups": lk, "pcie": off, "vram_share_of_lookups": round(h / lk, 4) if lk else None,
                           "pcie_share_of_routed": round(off / (lk + off), 4) if lk + off else None, "requests": len(hits)}
    for ln in reversed(lines):
        if "asynchronous adaptive tier:" in ln:
            out["adapt_async"] = ln.split("strata serve: ", 1)[-1]
            break
    if "--metrics" in args:
        m = json.load(open(args[args.index("--metrics") + 1]))
        hist = m.get("requests") or m.get("history") or []
        if isinstance(hist, dict):
            hist = list(hist.values())
        hist = hist[-last:] if last else hist
        keys = ("output_tokens", "decode_ms", "decode_tok_s", "hit_rate", "pcie_share", "drafts_offered",
                "drafts_accepted", "prompt_tokens", "prompt_ms", "ram_blobs", "file_blobs", "file_mb")
        out["requests"] = [{k: r.get(k) for k in keys} for r in hist]
        tok = sum(r.get("output_tokens") or 0 for r in hist)
        dms = sum(r.get("decode_ms") or 0 for r in hist)
        out["requests_total"] = {"n": len(hist), "output_tokens": tok, "decode_ms": round(dms, 1),
                                 "decode_tok_s": round(1000 * tok / dms, 1) if dms else None,
                                 "drafts_offered": sum(r.get("drafts_offered") or 0 for r in hist),
                                 "drafts_accepted": sum(r.get("drafts_accepted") or 0 for r in hist)}
        out["engine_info"] = m.get("engine") or m.get("info")
    print(json.dumps(out, indent=1, ensure_ascii=False))


BW = re.compile(r"edge0 trace bw: " + NUM + r"s (\d+) steps (\d+) tok \| per step: hot " + NUM + r" cold " + NUM +
                r" distinct experts, cold " + NUM + r" MB")
CPU = re.compile(r"edge0 cpu-experts: (\d+) jobs, " + NUM + r" experts / " + NUM + r" routes per job, CPU " + NUM +
                 r" us per job, GPU side " + NUM + r" us per call, GPU waited " + NUM + r" us per call")
ADAPT = re.compile(r"edge0 adapt: epoch (\d+) \(" + NUM + r"s, (\d+) counted steps\): cold share of decode touches " +
                   NUM + r"%, (\d+) candidates, (\d+) swaps")


def vllm(args):
    path, l0, l1 = args[0], int(args[1]), int(args[2])
    lines = open(path, "rb").read().decode(errors="replace").split("\n")[l0:l1]   # wc -l numbering (tqdm writes \r)
    bw, cpu, ad = [], [], []
    for ln in lines:
        m = BW.search(ln)
        if m:
            dt, steps, tok, hot, cold, mb = (float(x) for x in m.groups())
            bw.append({"dt": dt, "steps": steps, "tok": tok, "hot": hot, "cold": cold})
        m = CPU.search(ln)
        if m:
            j, e, r, c, g, w = (float(x) for x in m.groups())
            cpu.append({"jobs": j, "experts": e, "routes": r, "cpu_us": c, "gpu_us": g, "wait_us": w})
        m = ADAPT.search(ln)
        if m:
            ep, dt, st, cs, cand, sw = (float(x) for x in m.groups())
            ad.append({"epoch": ep, "steps": st, "cold_share": cs, "swaps": sw})
    out = {}
    if bw:
        st = sum(r["steps"] for r in bw)
        out["trace_bw"] = {"steps": st, "tok_per_step": round(sum(r["tok"] for r in bw) / st, 3),
                           "hot_distinct_per_step": round(sum(r["hot"] * r["steps"] for r in bw) / st, 1),
                           "cold_distinct_per_step": round(sum(r["cold"] * r["steps"] for r in bw) / st, 1),
                           "cold_per_layer_step": round(sum(r["cold"] * r["steps"] for r in bw) / st / 48, 2),
                           "hot_per_layer_step": round(sum(r["hot"] * r["steps"] for r in bw) / st / 48, 2),
                           "flushes": len(bw)}
        h, c = out["trace_bw"]["hot_distinct_per_step"], out["trace_bw"]["cold_distinct_per_step"]
        out["trace_bw"]["vram_share_of_distinct"] = round(h / (h + c), 4) if h + c else None
    if cpu:
        j = sum(r["jobs"] for r in cpu)
        out["cpu_experts"] = {k: round(sum(r[k] * r["jobs"] for r in cpu) / j, 2)
                              for k in ("experts", "routes", "cpu_us", "gpu_us", "wait_us")}
        out["cpu_experts"]["jobs"] = j
    if ad:
        s = sum(r["steps"] for r in ad)
        out["adapt"] = {"epochs": len(ad), "swaps": sum(r["swaps"] for r in ad),
                        "cold_share_of_touches_pct": round(sum(r["cold_share"] * r["steps"] for r in ad) / s, 2) if s else None}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"strata": strata, "vllm": vllm}[sys.argv[1]](sys.argv[2:])
