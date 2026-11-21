#!/usr/bin/env python3
"""Markdown table of a speculative-decoding sweep run by bench/run-configs.sh.

  python3 sweep_table.py logs/runcfg-<name>.log [...]

One row per (config, concurrency): output tok/s and TPOT (vllm-bench results under ~/work/edge0-bench/results/),
acceptance length, per-position acceptance (bench/specstat.py, unconditional a_i), scheduler preemptions and the
peak GPU memory of the bench window (GPUMEM), plus the largest 'Running: N reqs' the server logged for the config.
'step ms' = TPOT x acceptance length: the time of one decode (or draft + verify) step as one request sees it.
Standard library only.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import load  # noqa: E402

START = re.compile(r"^# \S+ start: (.*)$")
SPEC = re.compile(r"^SPECSTAT (\S+) c(\d+) (\{.*\})$")
MEM = re.compile(r"^GPUMEM (\S+) c(\d+) peak (\d+) MiB$")
RUN = re.compile(r"^max running: (\S+) (\d+) reqs")
KV = re.compile(r"^Available KV cache memory: ([0-9.]+) GiB")
SHORT = ("EDGE0_DEV=1 ", "")


def parse(path):
    cfgs, cur = {}, None
    order = []
    name = re.sub(r"^runcfg-|\.log$", "", os.path.basename(path))
    i = 0
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        m = START.match(line)
        if m:
            i += 1
            cur = f"{name}-{i}"
            cfgs[cur] = {"env": m.group(1).replace(*SHORT), "spec": {}, "mem": {}, "running": None, "kv": None}
            order.append(cur)
            continue
        m = KV.match(line)
        if m and cur:
            cfgs[cur]["kv"] = float(m.group(1))
        m = SPEC.match(line)
        if m and m.group(1) in cfgs:
            cfgs[m.group(1)]["spec"][int(m.group(2))] = json.loads(m.group(3))
        m = MEM.match(line)
        if m and m.group(1) in cfgs:
            cfgs[m.group(1)]["mem"][int(m.group(2))] = int(m.group(3))
        m = RUN.match(line)
        if m and m.group(1) in cfgs:
            cfgs[m.group(1)]["running"] = int(m.group(2))
    return order, cfgs


def main():
    print("| label | config | KV GiB | conc | out tok/s | TPOT ms | acc len | step ms | per-position acceptance | "
          "preempt | peak MiB | max running |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|")
    for path in sys.argv[1:]:
        order, cfgs = parse(path)
        for lab in order:
            c = cfgs[lab]
            res = load(lab)
            for conc in sorted(set(res) | set(c["spec"])):
                r = res.get(conc)
                s = c["spec"].get(conc, {})
                tput = f"{r['output_throughput']:.1f}" if r else "fail"
                tpot = f"{r['mean_tpot_ms']:.1f}" if r else ""
                al = s.get("acceptance_length", "")
                pp = "/".join(f"{p:.2f}" for p in s.get("per_pos", []))
                step = f"{r['mean_tpot_ms'] * (al or 1):.1f}" if r else ""   # one decode/verify step per request
                print(f"| {lab} | `{c['env']}` | {c['kv'] or ''} | {conc} | {tput} | {tpot} | {al} | {step} | {pp} | "
                      f"{s.get('preemptions', '')} | {c['mem'].get(conc, '')} | {c['running'] or ''} |")


if __name__ == "__main__":
    main()
