#!/usr/bin/env python3
"""Speculative-decoding gate: identical greedy text with and without the drafter on the strata-ref prompts.

  python3 -I greedy_same.py gen --out base.json [--max-tokens 256]      # on the server without a drafter
  python3 -I greedy_same.py gen --out spec.json                          # on the server with the drafter
  python3 -I greedy_same.py diff base.json spec.json                     # exit 0 when every prompt is identical
  python3 -I greedy_same.py gen --conc 8 --out base-c8.json              # control: batched decode, no drafter

``gen`` sends the 8 chat requests of strata-ref.json one at a time (temperature 0, thinking off, ``ignore_eos``,
``max_tokens``), with ``return_token_ids`` and top-2 logprobs, so ``diff`` can show the margin at a first divergence:
greedy decoding of the verify batch (k+1 rows) and of a 1-row decode step run different GEMM shapes, so an exact
near-tie (margin ~0) may flip in bf16 without any bug; a real acceptance bug shows up as a divergence with a clear
margin. ``gen`` also records per position the gap between the pass's top-1 logprob and the emitted token's (0 = the
emitted token is that pass's argmax); ``diff`` reports both margins and gaps at a divergence and calls it a near-tie
when either margin is <= --tie (0.25 nat) and both passes emitted their own argmax there. Exit 0 only when every
prompt is identical. Standard library only.
"""
import argparse
import json
import sys
import urllib.request


def post(url, body, timeout=1800):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def gen(a):
    ref = json.load(open(a.ref))
    if a.conc > 1:      # all prompts in flight at once: batched decode steps (a control for batch-shape numerics)
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(max_workers=a.conc) as ex:
            out = list(ex.map(lambda ir: gen_one(a, *ir), enumerate(ref["records"])))
    else:
        out = [gen_one(a, i, rec) for i, rec in enumerate(ref["records"])]
    json.dump(out, open(a.out, "w"), ensure_ascii=False)


def gen_one(a, i, rec):
    r = post(a.url + "/v1/chat/completions", {
        "model": a.model, "messages": rec["messages"], "max_tokens": a.max_tokens, "temperature": 0,
        "ignore_eos": True, "return_token_ids": True, "logprobs": True, "top_logprobs": 2,
        "chat_template_kwargs": {"enable_thinking": False}})
    ch = r["choices"][0]
    lp = ch["logprobs"]["content"]
    margins = [round(e["top_logprobs"][0]["logprob"] - e["top_logprobs"][1]["logprob"], 4)
               if len(e.get("top_logprobs") or []) > 1 else None for e in lp]
    # gap = top-1 logprob - logprob of the emitted token: 0 when the emitted token is this pass's argmax (a tie
    # counts as argmax); > 0 would mean greedy emitted a non-argmax token (an acceptance bug)
    gaps = [round(max(0.0, e["top_logprobs"][0]["logprob"] - e["logprob"]), 4) if e.get("top_logprobs") else None
            for e in lp]
    print(f"[{i}] {len(ch.get('token_ids') or [])} tokens: {ch['message']['content'][:70]!r}", flush=True)
    return {"i": i, "token_ids": ch.get("token_ids"), "text": ch["message"]["content"], "margins": margins,
            "gaps": gaps}


def diff(a):
    x, y = (json.load(open(f)) for f in a.files[:2])
    same = 0
    rows = []
    for p, q in zip(x, y):
        s, t = p["token_ids"], q["token_ids"]
        n = 0
        while n < min(len(s), len(t)) and s[n] == t[n]:
            n += 1
        ident = s == t
        same += ident
        row = {"i": p["i"], "identical": ident, "common_prefix": n, "len": [len(s), len(t)]}
        if not ident and n < min(len(s), len(t)):
            row["margin_a"] = p["margins"][n]
            row["margin_b"] = q["margins"][n]
            ga = (p.get("gaps") or [None] * (n + 1))[n]
            gb = (q.get("gaps") or [None] * (n + 1))[n]
            row["gap_a"], row["gap_b"] = ga, gb
            # explained: a bf16 near-tie (top-1 - top-2 <= --tie nat in either pass) and neither pass emitted a
            # non-argmax token there
            ms = [m for m in (row["margin_a"], row["margin_b"]) if m is not None]
            row["near_tie"] = bool(ms) and min(ms) <= a.tie and not (ga or 0) > 0 and not (gb or 0) > 0
        for nm, rec in (("nonargmax_a", p), ("nonargmax_b", q)):
            if rec.get("gaps") is not None:
                row[nm] = sum(1 for g in rec["gaps"][:n + 1] if g and g > 0)
        rows.append(row)
        print(json.dumps(row))
    res = {"prompts": len(rows), "identical": same, "tokens_compared": sum(min(r["len"]) for r in rows),
           "tokens_common": sum(r["common_prefix"] for r in rows),
           "diverged_at_near_tie": sum(1 for r in rows if r.get("near_tie")),
           "diverged_unexplained": sum(1 for r in rows if not r["identical"] and not r.get("near_tie"))}
    print("SUMMARY " + json.dumps(res))
    sys.exit(0 if same == len(rows) else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("gen", "diff"))
    ap.add_argument("files", nargs="*")
    ap.add_argument("--ref", default="/home/andychen/work/edge0-bench/strata-ref.json")
    ap.add_argument("--url", default="http://127.0.0.1:8125")
    ap.add_argument("--model", default="edge0_145b")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--conc", type=int, default=1, help="gen: requests in flight (default 1 = one at a time)")
    ap.add_argument("--tie", type=float, default=0.25, help="near-tie threshold, nat (2 bf16 ulps of a ~20 logit)")
    a = ap.parse_args()
    gen(a) if a.mode == "gen" else diff(a)


if __name__ == "__main__":
    main()
