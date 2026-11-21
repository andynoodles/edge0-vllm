#!/usr/bin/env python3
"""Is speculative decoding lossless on this server? (TidalMoE ADAPTIVE-VERIFICATION.md "argmax re-score".)

Greedy speculative decoding must emit, at every position, the target model's own argmax. This check generates greedy
outputs (one at a time, or many concurrently so the verify batches are large and drafts get rejected mid-batch) and
then re-scores every output teacher-forced (``/v1/completions`` with ``prompt_logprobs``: one prefill pass, no
drafter involved). A position "is not argmax" when the generated token is not the re-score's top-1; the gap is
``logprob(top-1) - logprob(generated)``. Decode and prefill run different kernels and batch shapes, so a few
near-ties flip even without speculation: run the same check on the non-speculative server and compare the rates
and the gaps. A real bug (wrong acceptance, stale state after a rejection) shows up as many large gaps.

  python3 -I spec_check.py gen --out spec-c1.json [--n 12 --max-tokens 512 --conc 1]
  python3 -I spec_check.py gen --out spec-c16.json --conc 16 --repeat 2       # 24 concurrent-ish requests
  python3 -I spec_check.py rescore spec-c1.json [--ref base-c1.json]          # on any server (prefill only)
Standard library only.
"""
import argparse
import concurrent.futures as cf
import json
import math
import sys
import urllib.request

PROMPTS = [
    "Write a detailed, step-by-step guide to setting up a home vegetable garden, covering soil, planning, planting, "
    "watering, pests and harvest.",
    "Explain how a CPU executes instructions, from fetch and decode to pipelining, caches and branch prediction. Be "
    "thorough.",
    "請詳細說明台灣的全民健康保險制度，包括歷史、財務、給付範圍、優缺點與未來挑戰。",
    "Write a long short story about a lighthouse keeper who finds a mysterious message in a bottle.",
    "Compare Python, Rust and Go for building web backends in depth: performance, safety, ecosystem, tooling, hiring.",
    "Explique en détail l'histoire de la Révolution française, ses causes, ses étapes principales et ses conséquences.",
    "Write a Python module that implements an LRU cache with TTL expiry, with type hints, docstrings and unit tests.",
    "Solve step by step: a train leaves at 9:40 at 72 km/h, a second train leaves the same station at 10:15 at "
    "96 km/h on the same track. When and where does the second catch up? Then generalize the method.",
    "List 40 interesting facts about the ocean, numbered, one sentence each.",
    "Translate into English and explain the grammar of each sentence: 「春眠不覺曉，處處聞啼鳥。夜來風雨聲，花落知多少。」",
    "Draft a professional email to a landlord about a broken heater, then a firmer follow-up a week later.",
    "Explain the transformer architecture to a new engineer: attention, positional encodings, layer norm, training.",
]


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def gen_one(a, i, p):
    r = post(a.url + "/v1/chat/completions", {
        "model": a.model, "messages": [{"role": "user", "content": p}], "max_tokens": a.max_tokens,
        "temperature": 0, "ignore_eos": True, "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False}})
    ch = r["choices"][0]
    ids = ch.get("token_ids")
    if ids is None:
        sys.exit("server did not return token_ids (needs return_token_ids)")
    return {"i": i, "prompt": p, "token_ids": ids, "prompt_token_ids": r.get("prompt_token_ids"),
            "text": ch["message"]["content"]}


def gen(a):
    jobs = [(k * len(PROMPTS[:a.n]) + i, p) for k in range(a.repeat) for i, p in enumerate(PROMPTS[:a.n])]
    with cf.ThreadPoolExecutor(max_workers=a.conc) as ex:
        out = list(ex.map(lambda j: gen_one(a, *j), jobs))
    for rec in out:
        print(f"{rec['i']:3d} {len(rec['token_ids']):5d} tokens: {rec['text'][:70]!r}", flush=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False)


def chat_prompt_ids(a, prompt):
    r = post(a.url + "/tokenize", {"model": a.model, "messages": [{"role": "user", "content": prompt}],
                                   "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": False}})
    return r["tokens"]


def rescore_one(a, rec):
    pids = rec.get("prompt_token_ids") or chat_prompt_ids(a, rec["prompt"])
    ids = rec["token_ids"]
    r = post(a.url + "/v1/completions", {"model": a.model, "prompt": pids + ids, "max_tokens": 1,
                                        "prompt_logprobs": 1, "temperature": 0})
    plp = r["choices"][0]["prompt_logprobs"][len(pids):]
    gaps = []
    for tok, d in zip(ids, plp):
        if not d:
            continue
        chosen = d.get(str(tok))
        best = max(d.values(), key=lambda v: v["logprob"] if isinstance(v, dict) else v)
        best_lp = best["logprob"] if isinstance(best, dict) else best
        ch_lp = chosen["logprob"] if isinstance(chosen, dict) else chosen
        if ch_lp is None or best_lp is None:
            continue
        gaps.append(max(0.0, best_lp - ch_lp))
    return gaps


def rescore(a):
    run = json.load(open(a.files[0]))
    ref = json.load(open(a.ref)) if a.ref else None
    all_gaps, prefixes = [], []
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        per = list(ex.map(lambda rec: rescore_one(a, rec), run))
    for rec, gaps in zip(run, per):
        all_gaps += gaps
        if ref is not None:
            b = next((r["token_ids"] for r in ref if r["prompt"] == rec["prompt"]), None)
            if b is not None:
                n = 0
                while n < min(len(b), len(rec["token_ids"])) and b[n] == rec["token_ids"][n]:
                    n += 1
                prefixes.append(n)
    n = len(all_gaps)
    bad = [g for g in all_gaps if g > 1e-6]
    res = {"file": a.files[0], "positions": n, "not_argmax": len(bad), "not_argmax_pct": round(100 * len(bad) / max(n, 1), 2),
           "max_gap": round(max(bad), 3) if bad else 0.0, "mean_gap_when_not_argmax": round(sum(bad) / len(bad), 3) if bad else 0.0,
           "gaps_over_1": sum(g > 1.0 for g in bad), "gaps_over_2": sum(g > 2.0 for g in bad)}
    if prefixes:
        res["common_prefix_with_ref"] = prefixes
    print(json.dumps(res))
    if a.out:
        json.dump(res, open(a.out, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("gen", "rescore"))
    ap.add_argument("files", nargs="*")
    ap.add_argument("--url", default="http://127.0.0.1:8125")
    ap.add_argument("--model", default="edge0_145b")
    ap.add_argument("--out", default="")
    ap.add_argument("--ref", default="")
    ap.add_argument("--n", type=int, default=len(PROMPTS))
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args()
    gen(a) if a.mode == "gen" else rescore(a)


if __name__ == "__main__":
    main()
