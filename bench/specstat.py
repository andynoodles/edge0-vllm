#!/usr/bin/env python3
"""Speculative-decoding counters from vLLM's /metrics (Prometheus), for one bench window.

  python3 specstat.py snap [--url http://127.0.0.1:8125] > before.json
  python3 specstat.py diff before.json after.json     -> one line: drafts, draft tokens, accepted, acceptance length,
                                                         per-position acceptance rates (unconditional, TidalMoE a_i),
                                                         scheduler preemptions in the window (KV cache ran out)
Acceptance length = 1 + accepted / drafts (tokens emitted per verify step and request, the bonus token included).
Standard library only.
"""
import json
import re
import sys
import urllib.request

KEYS = ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total")
PREEMPT = "vllm:num_preemptions_total"
POS = re.compile(r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{.*position="(\d+)".*\}\s+([0-9.eE+-]+)$')


def snap(url):
    try:
        text = urllib.request.urlopen(url + "/metrics", timeout=10).read().decode()
    except OSError:
        return {}
    out = {"pos": {}}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = POS.match(line)
        if m:
            out["pos"][m.group(1)] = float(m.group(2))
            continue
        for k in KEYS + (PREEMPT,):
            if line.startswith(k + "{"):
                out[k] = out.get(k, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def diff(a, b):
    pre = int(b.get(PREEMPT, 0) - a.get(PREEMPT, 0)) if b else 0
    if not b or KEYS[0] not in b:
        return {"spec": False, "preemptions": pre}
    d = {k: b.get(k, 0) - a.get(k, 0) for k in KEYS}
    drafts = d[KEYS[0]]
    res = {"spec": True, "drafts": int(drafts), "draft_tokens": int(d[KEYS[1]]), "accepted": int(d[KEYS[2]]),
           "preemptions": pre}
    if drafts > 0:
        res["acceptance_length"] = round(1 + d[KEYS[2]] / drafts, 3)
        res["k_mean"] = round(d[KEYS[1]] / drafts, 3)
        res["per_pos"] = [round((b["pos"].get(p, 0) - a.get("pos", {}).get(p, 0)) / drafts, 3)
                          for p in sorted(b["pos"], key=int)]
    return res


def main():
    if sys.argv[1] == "snap":
        url = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--url" else "http://127.0.0.1:8125"
        print(json.dumps(snap(url)))
    else:
        a, b = (json.load(open(f)) for f in sys.argv[2:4])
        print(json.dumps(diff(a, b)))


if __name__ == "__main__":
    main()
