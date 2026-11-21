#!/usr/bin/env python3
"""Per-engine-step time breakdown from Nsight Systems captures of the vLLM engine (docs/PROFILE.md).

Input: sqlite exports (``nsys export --type sqlite -o x.sqlite x.nsys-rep``) of captures made by
``bench/nsys-capture.sh`` (vLLM ``--profiler-config {"profiler":"cuda"}``: every ``execute_model`` is an NVTX range
``execute_context_N(T)_generation_M(T)``; ``EDGE0_NVTX=1`` adds host ranges for the runner phases, PLE staging,
scheduler and speculative decoding; ``--cuda-graph-trace=node`` gives every kernel of a graph replay).

  nsys_breakdown.py names <db>                      kernel names by total time (to extend the classifier)
  nsys_breakdown.py steps <db> [--calib <eager db>] [--json out.json] [--kind decode|all]

A step = the GPU work launched from one ``execute_model`` start to the next (forward graph + sampling + input
prep). Its period is measured on the GPU (first activity of step k to first activity of step k+1); busy time is
attributed exclusively per category with priority (cold Marlin > hot Marlin > rest; the cold side stream overlaps
the hot main-stream Marlin call inside each MoE op), idle = period - busy, and idle segments are attributed to the
engine thread's innermost host NVTX range while the GPU waited.

Kernel categories come from the kernel name + stream (``classify``); kernels whose name alone is ambiguous (the Q8
GEMMs of D, U, the shared expert and the attention projections) are labelled by an eager calibration capture
(``--calib``, ``EDGE0_NVTX=2`` + ``EAGER=1``), whose module / edge0 NVTX ranges give each kernel's owner: the k-th
kernel of a graph-replayed decode step is matched to the eager step with the same token count by sequence
alignment of kernel names.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import difflib
import json
import re
import sqlite3
import statistics
import sys

# ------------------------------------------------------------------------------------------------- loading


class Trace:
    def __init__(self, path: str):
        self.con = con = sqlite3.connect(path)
        con.row_factory = sqlite3.Row
        self.tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
        self.s = dict(con.execute("select id, value from StringIds"))
        self.kernels = []
        if "CUPTI_ACTIVITY_KIND_KERNEL" in self.tables:
            cols = self._cols("CUPTI_ACTIVITY_KIND_KERNEL")
            gid = "graphNodeId" if "graphNodeId" in cols else "NULL"
            for r in con.execute(f"select start, end, streamId, correlationId, shortName, demangledName, gridX, "
                                 f"gridY, gridZ, blockX, blockY, blockZ, {gid} as gid from CUPTI_ACTIVITY_KIND_KERNEL "
                                 f"order by start"):
                self.kernels.append(dict(start=r["start"], end=r["end"], stream=r["streamId"],
                                         corr=r["correlationId"], name=self.s.get(r["shortName"], "?"),
                                         full=self.s.get(r["demangledName"], "?"),
                                         grid=(r["gridX"], r["gridY"], r["gridZ"]),
                                         block=(r["blockX"], r["blockY"], r["blockZ"]), graph=r["gid"], kind="k"))
        self.copies = []
        for tab, kind in (("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
            if tab not in self.tables:
                continue
            cols = self._cols(tab)
            ck = "copyKind" if "copyKind" in cols else "NULL"
            for r in con.execute(f"select start, end, streamId, correlationId, bytes, {ck} as ck from {tab}"):
                self.copies.append(dict(start=r["start"], end=r["end"], stream=r["streamId"], corr=r["correlationId"],
                                        bytes=r["bytes"], copykind=r["ck"], kind=kind,
                                        name=f"{kind}:{_COPYKIND.get(r['ck'], r['ck'])}"))
        # --cuda-graph-trace=graph: each graph replay is one activity (no per-kernel records inside)
        if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in self.tables:
            for r in con.execute("select start, end, streamId, correlationId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
                self.copies.append(dict(start=r["start"], end=r["end"], stream=r["streamId"], corr=r["correlationId"],
                                        bytes=0, copykind=None, kind="graph", name="graph:forward"))
        self.api = {}
        self.api_list = []
        for tab in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
            if tab not in self.tables:
                continue
            for r in con.execute(f"select start, end, nameId, correlationId, globalTid from {tab}"):
                a = dict(start=r["start"], end=r["end"], name=self.s.get(r["nameId"], "?"), corr=r["correlationId"],
                         tid=r["globalTid"])
                self.api_list.append(a)
                if r["correlationId"]:
                    self.api.setdefault(r["correlationId"], a)
        self.api_list.sort(key=lambda a: a["start"])
        self.nvtx = []
        if "NVTX_EVENTS" in self.tables:
            for r in con.execute("select start, end, text, textId, globalTid, eventType from NVTX_EVENTS "
                                 "where end is not null and eventType in (59, 60, 70, 71)"):
                text = r["text"] if r["text"] is not None else self.s.get(r["textId"], "")
                self.nvtx.append(dict(start=r["start"], end=r["end"], text=text or "", tid=r["globalTid"]))
        self.nvtx.sort(key=lambda n: (n["start"], -n["end"]))
        self.osrt = []
        if "OSRT_API" in self.tables:
            for r in con.execute("select start, end, nameId, globalTid from OSRT_API"):
                self.osrt.append(dict(start=r["start"], end=r["end"], name=self.s.get(r["nameId"], "?"),
                                      tid=r["globalTid"]))
        self.threads = {}
        if "ThreadNames" in self.tables:
            for r in con.execute("select nameId, globalTid from ThreadNames"):
                self.threads[r["globalTid"]] = self.s.get(r["nameId"], "?")
        for k in self.kernels + self.copies:
            a = self.api.get(k["corr"])
            k["api"] = a
            k["tid"] = a["tid"] if a else None

    def _cols(self, tab):
        return {r[1] for r in self.con.execute(f"pragma table_info({tab})")}


_COPYKIND = {1: "HtoD", 2: "DtoH", 8: "DtoD", 10: "PtoP", 0: "?", None: "?"}

# ------------------------------------------------------------------------------------------------- classification
# Name rules (first match wins). The Q8 dense GEMMs and bf16 GEMMs are "dense" here and are split into latent D/U,
# shared expert, attention projections etc. by the eager calibration labels when --calib is given.
RULES = [
    ("cold_marlin", None),            # decided by stream (side stream of the MoE op), see classify()
    ("marlin", r"Marlin|marlin"),
    ("moe_align", r"moe_align|count_and_sort|sort_expert|align_block"),
    ("moe_route", r"topk|topK|top_k|_route_topk|gating"),
    ("moe_gather", r"gather_routes|^gather_kernel"),   # gather_kernel: EDGE0_CPU_EXPERTS=1 (csrc/cpu_experts_cuda.cu)
    ("cpu_wait", r"^wait_kernel"),                    # EDGE0_CPU_EXPERTS=1: the GPU spins on the CPU's done flag
    ("moe_route", r"^dispatch_kernel"),               # EDGE0_CPU_EXPERTS=1: the CPU/GPU split of the cold experts
    ("moe_combine", r"combine_routes"),
    ("moe_touch", r"touch_count|_touch"),
    ("act", r"silu_and_mul|act_and_mul|act_mul"),
    ("gdn", r"gated_delta|delta_rule|chunk_|recurrent|causal_conv|conv1d|fused_gdn|gdn|l2norm|fused_sigmoid_gating"),
    ("qsa", r"qsa|sparse|indexer|flash|fmha|attn|attention|paged|mla|topk_select|_select"),
    ("dense", r"b12x|q8|Q8|gemv|gemm|Gemm|GEMM|cutlass|sm120|nvjet|cublas|splitK|matmul|xmma|ampere_|sm90_|sm100_"),
    ("ple", r"ngram|_ple_|engram|lookup_nvfp4_ple"),
    ("hc", r"hyper|_hc_|hc_|sinkhorn"),
    ("norm", r"norm|rms"),
    ("sample", r"sampl|argmax|softmax|logit|penalt|multinomial|gumbel|rejection|cumsum"),
]
_RULES_RE = [(c, re.compile(p)) for c, p in RULES if p]


def classify_name(name: str) -> str:
    for cat, rx in _RULES_RE:
        if rx.search(name):
            return cat
    return "other"


# Calibration label (innermost meaningful NVTX range of an eager kernel) -> category
def label_category(stack: list[str]) -> str | None:
    """stack: NVTX range texts enclosing the launch, outermost first."""
    e0 = [s for s in stack if s.startswith("edge0.")]
    if e0:
        inner = e0[-1]
        return {"edge0.cold": "cold_marlin", "edge0.hot": "hot_marlin", "edge0.gather": "moe_gather",
                "edge0.combine": "moe_combine", "edge0.route": "moe_route", "edge0.D": "latent_D",
                "edge0.U": "latent_U", "edge0.gate": "router_gate", "edge0.shared": "shared_expert",
                "edge0.moe": "moe_other"}.get(inner)
    mods = [s[4:] for s in stack if s.startswith("mod:")]
    if not mods:
        return None
    m = mods[-1]
    if re.search(r"\.(linear_attn|self_attn|attn)\.(in_proj|qkv|q_proj|kv_proj|k_proj|v_proj|o_proj|out_proj|"
                 r"wq|wkv|wo|q_a|q_b|kv_a|kv_b|fused_qkv)", m):
        return "attn_proj"
    if re.search(r"\.linear_attn", m):
        return "gdn"
    if re.search(r"\.(self_attn|attn)", m):
        return "qsa"
    if re.search(r"ngram|ple|engram", m):
        return "ple"
    if re.search(r"hyper_connection|\.hc|_hc", m):
        return "hc"
    if re.search(r"norm", m):
        return "norm"
    if re.search(r"lm_head|logits", m):
        return "lm_head"
    if re.search(r"embed", m):
        return "embed"
    if re.search(r"\.mlp\.gate$", m):
        return "router_gate"
    if re.search(r"\.mlp\.(shared_expert|shared_expert_gate)", m):
        return "shared_expert"
    if re.search(r"\.mlp\.latent_down", m):
        return "latent_D"
    if re.search(r"\.mlp\.latent_up", m):
        return "latent_U"
    if re.search(r"mtp|draft|dflash", m):
        return "drafter"
    return "model_other"


# Report categories, in table order. "cold_marlin" = all side-stream work of the MoE op (align + 2 Marlin GEMMs +
# SiLU on the cold tier); "hot_marlin" = the same on the main stream for the hot tier.
CATS = ["attn_gdn", "attn_qsa", "attn_proj", "dense_other", "router_gate", "latent_D", "latent_U", "shared_expert",
        "moe_route", "moe_gather", "hot_marlin", "cold_marlin", "cpu_wait", "moe_combine", "ple_gpu", "hc_norm", "lm_head_sample",
        "drafter", "spec_reject", "other_gpu", "memcpy_h2d", "memcpy_d2h", "memset", "forward_graph"]
PRIO = {c: i for i, c in enumerate(["cold_marlin", "hot_marlin"])}   # lower = higher priority; others = 10


def report_category(k, side_streams, calib_cat=None, tier=None) -> str:
    if k["kind"] == "memcpy":
        return {"HtoD": "memcpy_h2d", "DtoH": "memcpy_d2h"}.get(k["name"].split(":")[1], "memcpy_h2d" if
                                                                k["name"].endswith("PtoP") else "other_gpu")
    if k["kind"] == "memset":
        return "memset"
    if k["kind"] == "graph":
        return "forward_graph"
    if tier is not None:
        return tier
    if k["stream"] in side_streams:
        return "cold_marlin"
    if calib_cat:
        c = calib_cat
        return {"gdn": "attn_gdn", "qsa": "attn_qsa", "ple": "ple_gpu", "hc": "hc_norm", "norm": "hc_norm",
                "lm_head": "lm_head_sample", "embed": "other_gpu", "moe_other": "moe_route",
                "model_other": "other_gpu"}.get(c, c)
    c = classify_name(k["name"])
    if k.get("graph") is None and k["kind"] == "k" and c in ("dense", "sample"):
        return "lm_head_sample"      # outside the forward graph: compute_logits (lm_head GEMM) + sampler
    return {"marlin": "hot_marlin", "moe_align": "hot_marlin", "act": "hot_marlin", "moe_touch": "moe_route",
            "gdn": "attn_gdn", "qsa": "attn_qsa", "dense": "dense_other", "ple": "ple_gpu", "hc": "hc_norm",
            "norm": "hc_norm", "sample": "lm_head_sample", "other": "other_gpu"}.get(c, c)


# ------------------------------------------------------------------------------------------------- steps

_STEP_RE = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")


def find_steps(tr: Trace):
    steps = [n for n in tr.nvtx if _STEP_RE.search(n["text"])]
    steps.sort(key=lambda n: n["start"])
    out = []
    for n in steps:
        m = _STEP_RE.search(n["text"])
        out.append(dict(cpu_start=n["start"], cpu_end=n["end"], tid=n["tid"], ctx=int(m[1]), ctx_tok=int(m[2]),
                        gen=int(m[3]), gen_tok=int(m[4]), text=n["text"]))
    return out


_GATHER = {"_gather_routes_kernel", "gather_kernel"}   # Triton gather, or the EDGE0_CPU_EXPERTS=1 CUDA gather
_BRANCH = {"Marlin", "act_and_mul_kernel", "moe_align_block_size_kernel", "count_and_sort_expert_tokens_kernel"}


def moe_tiers(ks):
    """Hot / cold label of the expert-branch kernels of every MoE op in one step (kernels sorted by start).

    The op forks after the route gather: cold tier on the side stream, hot tier on the main stream, then joins at
    the combine. In a CUDA graph replay the driver chooses which branch runs on the launch stream (here: the cold
    one), so the stream id does not tell the tier. The capture order does: the cold branch is captured first, so its
    graph nodes have the lower ids. Eager runs (no node ids): the non-launch (side) stream is cold."""
    out = {}
    launch = collections.Counter(k["stream"] for k in ks).most_common(1)[0][0] if ks else None
    sec = None
    for k in ks:
        n = k["name"]
        if n in _GATHER:
            sec = []
        elif n == "_combine_routes_kernel" and sec is not None:
            by = collections.defaultdict(list)
            for x in sec:
                by[x["stream"]].append(x)
            if len(by) == 2:
                (sa, xa), (sb, xb) = by.items()
                na = min((x["graph"] for x in xa if x["graph"] is not None), default=None)
                nb = min((x["graph"] for x in xb if x["graph"] is not None), default=None)
                if na is not None and nb is not None:
                    cold = sa if na < nb else sb
                else:
                    cold = sb if sa == launch else sa
                for x in sec:
                    out[id(x)] = "cold_marlin" if x["stream"] == cold else "hot_marlin"
            elif len(by) == 1:      # one tier only (all hot or all cold) or split off: leave to the name rules
                pass
            sec = None
        elif sec is not None and n in _BRANCH:
            sec.append(k)
    return out


_GEMM = re.compile(r"b12x_libdense|DenseGemm|nvjet|CuteSkinnyGemm|skinny_gemm|splitKreduce|gemv|Gemv|cutlass.*[Gg]emm")


def structural_labels(ks):
    """Category of every non-branch kernel of one decode step from the Qwen4Exp layer structure (main stream order;
    learned from the EDGE0_NVTX=2 eager capture, where every kernel carries its module range):

      [embed / PLE] then per layer:
        HC:   _hc_combine_norm, hc_down_silu GEMM, input_mix_weight_up GEMM, _hc_gate_mix
        attention block: GDN (in_proj_qkvz b12x, in_proj_ba skinny, conv1d, gated delta rule, norm, out_proj b12x)
                         or QSA (qkv_proj nvjet, indexer, sparse attention, merge, o_proj b12x)
        HC (mlp side), then the MoE block: router gate GEMM (+split-K reduce), D (last b12x before topkGating),
        route kernels, gather, [hot || cold branches], combine, U (first GEMM after combine), shared expert
      [final HC / norm / lm_head / sampling]
    Graph replays use different GEMM variants than eager runs (other grids, 2-7x faster b12x configs), so the
    labels are positional, not per kernel signature."""
    out = {}
    main = [k for k in ks if k["name"] not in _BRANCH]
    # segments between HC groups
    i, n = 0, len(main)
    seg_start = 0
    segments = []           # (lo, hi) index ranges of non-HC kernels
    while i < n:
        if main[i]["name"] == "_hc_combine_norm_kernel":
            if i > seg_start:
                segments.append((seg_start, i))
            j = i
            while j < n:
                out[id(main[j])] = "hc_norm"
                if main[j]["name"] == "_hc_gate_mix_kernel":
                    break
                j += 1
            i = seg_start = j + 1
            continue
        i += 1
    if seg_start < n:
        segments.append((seg_start, n))
    moe_segs = [s for s in segments if any(main[x]["name"] in _GATHER for x in range(*s))]
    last_moe = moe_segs[-1][1] if moe_segs else -1
    first_layer = segments[1][0] if len(segments) > 1 else 0
    for lo, hi in segments:
        names = [main[x]["name"] for x in range(lo, hi)]
        gname = next((x for x in names if x in _GATHER), None)
        if gname is not None:
            t = names.index("topkGating") if "topkGating" in names else names.index(gname)
            g = names.index(gname)
            c = names.index("_combine_routes_kernel") if "_combine_routes_kernel" in names else hi - lo
            # router gate = bf16 nvjet split-K GEMM + its splitKreduce; D = the b12x Q8 GEMM (+ its split-K reduce)
            # right before topkGating; U = the first b12x GEMM after the combine (+ its reduce)
            u = next((x for x in range(c + 1, hi - lo) if "b12x_libdense" in names[x]), None)
            u_end = u
            while u_end is not None and u_end + 1 < hi - lo and "SplitKReduce" in names[u_end + 1]:
                u_end += 1
            d0 = next((x for x in range(t) if "b12x_libdense" in names[x]), t)
            for x in range(hi - lo):
                k = main[lo + x]
                if x < t:
                    cat = "router_gate" if x < d0 else "latent_D"
                elif x < g:
                    cat = "moe_route"
                elif x == g:
                    cat = "moe_gather"
                elif x < c:
                    cat = "cpu_wait" if names[x] == "wait_kernel" else "moe_route"
                elif x == c:
                    cat = "moe_combine"
                elif u is not None and u <= x <= u_end:
                    cat = "latent_U"
                else:
                    cat = "shared_expert"
                out[id(k)] = cat
        elif lo >= last_moe and last_moe >= 0:
            for x in range(lo, hi):
                nm = main[x]["name"]
                out[id(main[x])] = "ple_gpu" if classify_name(nm) == "ple" else "lm_head_sample"
        elif lo < first_layer:
            for x in range(lo, hi):
                nm = main[x]["name"]
                out[id(main[x])] = "ple_gpu" if classify_name(nm) == "ple" else "other_gpu"
        else:
            kind = "attn_gdn" if any("gated_delta" in nm or "conv1d" in nm for nm in names) else "attn_qsa"
            for x in range(lo, hi):
                nm = main[x]["name"]
                out[id(main[x])] = "attn_proj" if _GEMM.search(nm) else kind
    return out


def side_streams_of(tr: Trace) -> set:
    """Streams that run Marlin kernels but are not the stream with the most kernels (= the MoE side stream)."""
    per = collections.Counter(k["stream"] for k in tr.kernels)
    if not per:
        return set()
    main = per.most_common(1)[0][0]
    marlin = collections.Counter(k["stream"] for k in tr.kernels if re.search(r"[Mm]arlin", k["name"]))
    return {s for s in marlin if s != main and marlin[s] > 0.2 * max(marlin.values())}


def nvtx_stack_at(index, tid, t):
    """NVTX ranges on thread tid containing time t, outermost first (index: tid -> list sorted by start)."""
    rs = index.get(tid, [])
    starts = [r["start"] for r in rs]
    i = bisect.bisect_right(starts, t)
    st = [r for r in rs[max(0, i - 400):i] if r["end"] >= t]
    st.sort(key=lambda r: (r["start"], -r["end"]))
    return [r["text"] for r in st]


def nvtx_index(tr: Trace):
    idx = collections.defaultdict(list)
    for n in tr.nvtx:
        idx[n["tid"]].append(n)
    for v in idx.values():
        v.sort(key=lambda r: r["start"])
    return idx


def assign_work(tr: Trace, steps):
    """GPU activities -> step index by the CPU time of their launching API call (window = execute start k ..
    execute start k+1, any thread of the engine process)."""
    starts = [s["cpu_start"] for s in steps]
    work = collections.defaultdict(list)
    for k in tr.kernels + tr.copies:
        a = k["api"]
        if a is None:
            continue
        i = bisect.bisect_right(starts, a["start"]) - 1
        if 0 <= i < len(steps):
            work[i].append(k)
    return work


def sweep(intervals):
    """intervals: (start, end, cat) -> ({cat: exclusive ns}, busy ns, [(gap_start, gap_end)]) with PRIO."""
    ev = []
    for s, e, c in intervals:
        if e > s:
            ev.append((s, 1, c))
            ev.append((e, -1, c))
    ev.sort(key=lambda x: (x[0], x[1]))
    active = collections.Counter()
    out = collections.Counter()
    gaps = []
    last_t = None
    busy = 0
    for t, d, c in ev:
        if last_t is not None and t > last_t:
            live = [x for x, n in active.items() if n > 0]
            if live:
                best = min(live, key=lambda x: (PRIO.get(x, 10), CATS.index(x) if x in CATS else 99))
                out[best] += t - last_t
                busy += t - last_t
            else:
                gaps.append((last_t, t))
        active[c] += d
        last_t = t
    return out, busy, gaps


# ------------------------------------------------------------------------------------------------- calibration

def calibrate(cal: Trace):
    """Eager capture -> per decode step (gen tokens T): list of (kernel name, category) in main-stream order."""
    idx = nvtx_index(cal)
    steps = find_steps(cal)
    work = assign_work(cal, steps)
    side = side_streams_of(cal)
    seqs = {}
    for i, s in enumerate(steps):
        if s["ctx"] != 0:
            continue
        ks = sorted((k for k in work.get(i, []) if k["kind"] == "k" and k["name"] not in _BRANCH),
                    key=lambda k: k["start"])
        seq = []
        for k in ks:
            st = nvtx_stack_at(idx, k["tid"], k["api"]["start"]) if k["api"] else []
            cat = label_category(st)
            seq.append((k["name"], k["grid"], cat))
        if seq and s["gen_tok"] not in seqs:
            seqs[s["gen_tok"]] = seq
    return seqs


def calib_labels(ks, side, seqs, tokens):
    """Label the kernels of one graph-mode step from the eager sequence with the same token count."""
    ref = seqs.get(tokens)
    lab = [None] * len(ks)
    if not ref:
        return lab
    pos = [i for i, k in enumerate(ks) if k["name"] not in _BRANCH]
    a = [(ks[i]["name"], ks[i]["grid"]) for i in pos]
    b = [(n, g) for n, g, _ in ref]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for blk in sm.get_matching_blocks():
        for j in range(blk.size):
            lab[pos[blk.a + j]] = ref[blk.b + j][2]
    return lab


# ------------------------------------------------------------------------------------------------- main

HOST_RANGES = ["ple.read", "ple.stage", "model_state.prepare_inputs", "runner.prepare_inputs", "graph.full",
               "graph.pw", "runner.sample", "runner.postprocess", "runner.sample_tokens", "runner.execute_model",
               "spec.propose", "spec.reject", "sched.schedule", "sched.update_from_output", "edge0.adapt.epoch"]


def analyze(tr: Trace, seqs=None, kind="decode"):
    steps = find_steps(tr)
    work = assign_work(tr, steps)
    side = side_streams_of(tr)
    idx = nvtx_index(tr)
    eng_tid = collections.Counter(s["tid"] for s in steps).most_common(1)[0][0] if steps else None
    rows = []
    first_gpu = {}
    for i in range(len(steps)):
        ws = work.get(i, [])
        if ws:
            first_gpu[i] = min(k["start"] for k in ws)
    # host ranges per step (all engine-process threads)
    host = collections.defaultdict(lambda: collections.Counter())
    starts = [s["cpu_start"] for s in steps]
    for n in tr.nvtx:
        base = n["text"]
        if base in HOST_RANGES:
            i = bisect.bisect_right(starts, n["start"]) - 1
            if 0 <= i < len(steps):
                host[i][base] += n["end"] - n["start"]
    sync_api = collections.defaultdict(lambda: collections.Counter())
    for a in tr.api_list:
        if "Synchronize" in a["name"] or a["name"].startswith("cudaGraphLaunch") or "MemcpyAsync" in a["name"]:
            i = bisect.bisect_right(starts, a["start"]) - 1
            if 0 <= i < len(steps):
                key = "sync" if "Synchronize" in a["name"] else ("graph_launch" if "GraphLaunch" in a["name"]
                                                                  else "memcpy_api")
                sync_api[i][key] += a["end"] - a["start"]
    ctx_cache = {}
    for i, s in enumerate(steps):
        if i + 1 >= len(steps) or i not in first_gpu or (i + 1) not in first_gpu:
            continue
        if kind == "decode" and s["ctx"] != 0:
            continue
        t0, t1 = first_gpu[i], first_gpu[i + 1]
        ws = work[i]
        ks = sorted((k for k in ws if k["kind"] == "k"), key=lambda k: k["start"])
        labels = calib_labels(ks, side, seqs, s["gen_tok"]) if seqs else [None] * len(ks)
        lab = {id(k): l for k, l in zip(ks, labels)}
        # host context of each launch (cached per correlation id: all nodes of a graph replay share one)
        spec = {}
        for k in ws:
            a = k["api"]
            if a is None:
                continue
            if a["corr"] not in ctx_cache:
                st = nvtx_stack_at(idx, a["tid"], a["start"])
                ctx_cache[a["corr"]] = ("drafter" if "spec.propose" in st else
                                        "spec_reject" if "spec.reject" in st else None)
            if ctx_cache[a["corr"]]:
                spec[id(k)] = ctx_cache[a["corr"]]
        tiers = moe_tiers([k for k in ks if id(k) not in spec])
        if not seqs:
            fwd = [k for k in ks if k["graph"] is not None and id(k) not in spec] or ks   # target forward graph
            struct = structural_labels(fwd)
            lab = {id(k): struct.get(id(k)) for k in ks}
        iv = []
        sums = collections.Counter()
        names = collections.Counter()
        for k in ws:
            c = spec.get(id(k)) or report_category(k, side, lab.get(id(k)), tiers.get(id(k)))
            ks_, ke = max(k["start"], t0), min(k["end"], t1)
            iv.append((ks_, ke, c))
            sums[c] += k["end"] - k["start"]
            names[(c, k["name"])] += k["end"] - k["start"]
        # also count work of the NEXT step that started before t1? (none: t1 = its first activity)
        excl, busy, gaps = sweep(iv)
        period = t1 - t0
        idle = period - busy
        # idle attribution: innermost host range on the engine thread at each gap's midpoint
        gap_attr = collections.Counter()
        for g0, g1 in gaps:
            st = nvtx_stack_at(idx, eng_tid, (g0 + g1) // 2)
            inner = next((x for x in reversed(st) if x in HOST_RANGES or x.startswith("execute_")), "none")
            if inner.startswith("execute_"):
                inner = "execute_model(other)"
            gap_attr[inner] += g1 - g0
        h2d = sum(k["bytes"] for k in ws if k["kind"] == "memcpy" and k["name"].endswith("HtoD"))
        rows.append(dict(step=i, text=s["text"], ctx=s["ctx"], gen=s["gen"], gen_tok=s["gen_tok"],
                         ctx_tok=s["ctx_tok"], period=period, busy=busy, idle=idle, excl=dict(excl),
                         sums=dict(sums), gap_attr=dict(gap_attr), host=dict(host[i]), api=dict(sync_api[i]),
                         n_kernels=len(ks), h2d_bytes=h2d, cpu_exec=s["cpu_end"] - s["cpu_start"],
                         names={f"{c}|{n}": v for (c, n), v in names.items()},
                         unlabelled=sum(1 for k, l in zip(ks, labels) if l is None and k["name"] not in _BRANCH)
                if seqs else None))
    return rows, side


def summarize(rows):
    if not rows:
        return {}
    n = len(rows)
    agg = lambda key: {c: sum(r[key].get(c, 0) for r in rows) / n / 1e6 for c in
                       sorted({c for r in rows for c in r[key]})}
    names = collections.Counter()
    for r in rows:
        for k, v in r["names"].items():
            names[k] += v / n / 1e6
    return dict(steps=n, period_ms=statistics.mean(r["period"] for r in rows) / 1e6,
                period_med_ms=statistics.median(r["period"] for r in rows) / 1e6,
                busy_ms=statistics.mean(r["busy"] for r in rows) / 1e6,
                idle_ms=statistics.mean(r["idle"] for r in rows) / 1e6,
                excl_ms=agg("excl"), sums_ms=agg("sums"), gap_ms=agg("gap_attr"), host_ms=agg("host"),
                api_ms=agg("api"), h2d_MB=statistics.mean(r["h2d_bytes"] for r in rows) / 1e6,
                cpu_exec_ms=statistics.mean(r["cpu_exec"] for r in rows) / 1e6,
                kernels=statistics.mean(r["n_kernels"] for r in rows),
                gen_tok=collections.Counter(r["gen_tok"] for r in rows).most_common(3),
                unlabelled=(statistics.mean(r["unlabelled"] for r in rows) if rows[0]["unlabelled"] is not None
                            else None),
                top_names=names.most_common(40))


def print_summary(sm, title=""):
    print(f"== {title}: {sm['steps']} steps, gen tokens {sm['gen_tok']}, kernels/step {sm['kernels']:.0f}, "
          f"unlabelled {sm['unlabelled']}")
    p = sm["period_ms"]
    print(f"period {p:.2f} ms (median {sm['period_med_ms']:.2f}), GPU busy {sm['busy_ms']:.2f}, idle "
          f"{sm['idle_ms']:.2f} ({100 * sm['idle_ms'] / p:.1f}%), H2D {sm['h2d_MB']:.2f} MB/step, "
          f"execute_model CPU {sm['cpu_exec_ms']:.2f} ms")
    print("exclusive GPU time per category (ms, % of period) | summed kernel time (ms):")
    for c in CATS:
        e = sm["excl_ms"].get(c, 0)
        t = sm["sums_ms"].get(c, 0)
        if e or t:
            print(f"  {c:16s} {e:8.3f} {100 * e / p:5.1f}%   sum {t:8.3f}")
    print(f"  {'GPU idle':16s} {sm['idle_ms']:8.3f} {100 * sm['idle_ms'] / p:5.1f}%")
    print("GPU idle attributed to the engine thread's host range (ms):")
    for c, v in sorted(sm["gap_ms"].items(), key=lambda x: -x[1]):
        print(f"  {c:28s} {v:8.3f}")
    print("host ranges per step (ms, all engine threads):")
    for c, v in sorted(sm["host_ms"].items(), key=lambda x: -x[1]):
        print(f"  {c:28s} {v:8.3f}")
    print("CUDA API per step (ms):", {k: round(v, 3) for k, v in sm["api_ms"].items()})
    print("top kernels (category | name: ms per step):")
    for k, v in sm["top_names"][:30]:
        print(f"  {v:8.3f}  {k[:150]}")


def cmd_names(db):
    tr = Trace(db)
    side = side_streams_of(tr)
    tot = collections.Counter()
    cnt = collections.Counter()
    grids = collections.defaultdict(collections.Counter)
    for k in tr.kernels:
        key = (k["name"], k["stream"] in side)
        tot[key] += k["end"] - k["start"]
        cnt[key] += 1
        grids[key][(k["grid"], k["block"])] += 1
    print(f"kernels {len(tr.kernels)}, copies {len(tr.copies)}, nvtx {len(tr.nvtx)}, side streams {side}, "
          f"streams {collections.Counter(k['stream'] for k in tr.kernels).most_common(6)}")
    for (n, sd), v in tot.most_common(80):
        print(f"{v / 1e6:9.2f} ms {cnt[(n, sd)]:7d}x {'SIDE ' if sd else '     '}{classify_name(n):11s} {n[:110]} "
              f"{grids[(n, sd)].most_common(2)}")
    texts = collections.Counter(n["text"][:60] for n in tr.nvtx)
    print("nvtx:", texts.most_common(40))


def cmd_gpum(db):
    """Mean of every GPU metric of a host-side --gpu-metrics-devices capture (PCIe RX/TX, SM active, DRAM ...)."""
    con = sqlite3.connect(db)
    tabs = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    names = {}
    if "TARGET_INFO_GPU_METRICS" in tabs:
        names = {r[0]: r[1] for r in con.execute("select metricId, metricName from TARGET_INFO_GPU_METRICS")}
    rows = con.execute("select metricId, avg(value), max(value), count(*), min(timestamp), max(timestamp) "
                       "from GPU_METRICS group by metricId").fetchall()
    for mid, avg, mx, n, t0, t1 in rows:
        print(f"{names.get(mid, mid)[:60]:60s} mean {avg:12.2f} max {mx:12.2f} n {n} ({(t1 - t0) / 1e9:.1f} s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["names", "steps", "gpum"])
    ap.add_argument("db")
    ap.add_argument("--calib")
    ap.add_argument("--json")
    ap.add_argument("--kind", default="decode", choices=["decode", "all"])
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    if a.cmd == "names":
        return cmd_names(a.db)
    if a.cmd == "gpum":
        return cmd_gpum(a.db)
    seqs = calibrate(Trace(a.calib)) if a.calib else None
    rows, side = analyze(Trace(a.db), seqs, a.kind)
    sm = summarize(rows)
    if not sm:
        print("no steps found", file=sys.stderr)
        return 1
    print_summary(sm, a.title or a.db)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(summary=sm, rows=[{k: v for k, v in r.items() if k != "names"} for r in rows]), f,
                      indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
