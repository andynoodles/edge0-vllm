#!/usr/bin/env python3
"""Does a lower-precision GDN recurrent state (``--mamba-ssm-cache-dtype bfloat16``) degrade long generations?

The 8 Strata reference prompts are 48 tokens long, too short to show error that builds up in the recurrent state over
many decode steps. This check generates long greedy outputs on two servers (e.g. fp32 state, then bf16 state) and
scores every output by teacher forcing on one server (``/v1/completions`` with ``prompt_logprobs``). A prompt under
4,096 tokens is prefilled in one chunk, where the GDN kernel keeps the state in fp32, so the scoring does not depend
on the state dtype. If the bf16-state generations drift, their per-token log-likelihood falls behind the fp32-state
generations in the later windows.

  python3 -I long_state_check.py gen   --out fp32.json [--n 6 --max-tokens 1024]     # on the fp32-state server
  python3 -I long_state_check.py gen   --out bf16.json                                # on the bf16-state server
  python3 -I long_state_check.py score fp32.json bf16.json [--window 256]             # on either server
Standard library only.
"""
import argparse
import json
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
]


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def gen(a):
    out = []
    for p in PROMPTS[:a.n]:
        r = post(a.url + "/v1/chat/completions", {
            "model": a.model, "messages": [{"role": "user", "content": p}], "max_tokens": a.max_tokens,
            "temperature": 0, "ignore_eos": True, "logprobs": True, "top_logprobs": 1, "return_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False}})
        ch = r["choices"][0]
        ids = ch.get("token_ids")
        if ids is None:
            sys.exit("server did not return token_ids (needs return_token_ids)")
        out.append({"prompt": p, "token_ids": ids, "prompt_token_ids": r.get("prompt_token_ids"),
                    "text": ch["message"]["content"]})
        print(f"{len(ids):5d} tokens: {ch['message']['content'][:80]!r}", flush=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False)


def chat_prompt_ids(a, prompt):
    r = post(a.url + "/tokenize", {"model": a.model, "messages": [{"role": "user", "content": prompt}],
                                   "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": False}})
    return r["tokens"]


def score_one(a, rec):
    pids = rec.get("prompt_token_ids") or chat_prompt_ids(a, rec["prompt"])
    ids = rec["token_ids"]
    r = post(a.url + "/v1/completions", {"model": a.model, "prompt": pids + ids, "max_tokens": 1,
                                        "prompt_logprobs": 0, "temperature": 0})
    plp = r["choices"][0]["prompt_logprobs"][len(pids):]
    lps = []
    for tok, d in zip(ids, plp):
        v = d.get(str(tok)) if d else None
        lps.append(v["logprob"] if isinstance(v, dict) else (v if v is not None else float("nan")))
    return lps


def score(a):
    runs = [json.load(open(f)) for f in a.files]
    w = a.window
    print(f"mean log-likelihood per token of each run's greedy output, scored teacher-forced on {a.url}")
    print(f"{'run':24s} " + " ".join(f"[{i * w}-{(i + 1) * w})".rjust(12) for i in range(a.max_tokens // w)) +
          "   agree-with-first-run prefix")
    base_ids = None
    for f, run in zip(a.files, runs):
        cols = [[] for _ in range(a.max_tokens // w)]
        prefixes = []
        for i, rec in enumerate(run):
            lps = score_one(a, rec)
            for j, v in enumerate(lps):
                if j // w < len(cols) and v == v:
                    cols[j // w].append(v)
            if base_ids is not None:
                b = base_ids[i]
                n = 0
                while n < min(len(b), len(rec["token_ids"])) and b[n] == rec["token_ids"][n]:
                    n += 1
                prefixes.append(n)
        if base_ids is None:
            base_ids = [r["token_ids"] for r in run]
        print(f"{f[-24:]:24s} " + " ".join(f"{(sum(c) / len(c) if c else float('nan')):12.4f}" for c in cols) +
              (f"   {prefixes}" if prefixes else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("gen", "score"))
    ap.add_argument("files", nargs="*")
    ap.add_argument("--url", default="http://127.0.0.1:8125")
    ap.add_argument("--model", default="edge0_145b")
    ap.add_argument("--out", default="")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--window", type=int, default=256)
    a = ap.parse_args()
    gen(a) if a.mode == "gen" else score(a)


if __name__ == "__main__":
    main()
