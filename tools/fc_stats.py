#!/usr/bin/env python3
"""Per-pass table of a bench/fc-ab.sh (or fc-probe.sh) log: throughput, TPOT, acceptance, and the expert-tier
statistics of the pass (EDGE0_TIER lines, else EDGE0_TRACE=bw call-time lines; adaptive-epoch swaps).

  python3 tools/fc_stats.py logs/fcab-<name>.log [...]      -> one row per pass, then the A / B means of c1 passes
"""
import json
import re
import sys
from collections import defaultdict

ROW = re.compile(r"^(\S+)\s+(\d+)\s+(\d+)/(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$")
TIER = re.compile(r"\[(\S+)\] edge0 tier: ([\d.]+)s (\d+) steps \| per layer-step: cold ([\d.]+) hot ([\d.]+) distinct "
                  r"experts, routes from VRAM ([\d.]+)% \| swaps ([\d.]+) / step .*\| host ([\d.]+) \+ ([\d.]+) ms"
                  r".*steps since prefill: (.*)$")
TRACE = re.compile(r"\[(\S+)\] edge0 trace bw \(map at call time, decode calls\): per layer-call cold ([\d.]+) hot "
                   r"([\d.]+) distinct experts, routes from VRAM ([\d.]+)% \((\d+) layer-calls\)")
ADAPT = re.compile(r"\[(\S+)\] edge0 adapt: epoch \d+ .* (\d+) swaps \(total")
CPU = re.compile(r"\[(\S+)\] edge0 cpu-experts: (\d+) jobs, ([\d.]+) experts")


def parse(path):
    rows, spec, tier, trace, adapt, cpu = {}, {}, defaultdict(list), defaultdict(list), defaultdict(int), {}
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        m = ROW.match(line)
        if m:
            rows[(m.group(1), int(m.group(2)))] = (float(m.group(5)), float(m.group(6)), int(m.group(7)))
            continue
        if line.startswith("SPECSTAT "):
            lab, js = line.split(" ", 2)[1:]
            try:
                spec[lab] = json.loads(js)
            except ValueError:
                pass
            continue
        m = TIER.search(line)
        if m:
            tier[m.group(1)].append([float(x) for x in m.group(3, 4, 5, 6, 7, 8, 9)] + [m.group(10)])
            continue
        m = TRACE.search(line)
        if m:
            trace[m.group(1)].append([float(x) for x in m.group(2, 3, 4, 5)])
            continue
        m = ADAPT.search(line)
        if m:
            adapt[m.group(1)] += int(m.group(2))
            continue
        m = CPU.search(line)
        if m:
            cpu.setdefault(m.group(1), []).append(float(m.group(3)))
    return rows, spec, tier, trace, adapt, cpu


def wmean(vals, wi, ci):
    w = sum(v[wi] for v in vals)
    return sum(v[ci] * v[wi] for v in vals) / w if w else float("nan")


def main():
    for path in sys.argv[1:]:
        rows, spec, tier, trace, adapt, cpu = parse(path)
        print(f"== {path}")
        print(f"{'label':30s} {'c':>2s} {'tok/s':>7s} {'TPOT':>6s} {'acc':>5s} {'cold/L':>7s} {'hot/L':>6s} {'VRAM%':>6s} "
              f"{'swp/st':>7s} {'host ms':>8s} {'cpuexp':>6s}  curve")
        groups = defaultdict(list)
        for (lab, c), (tps, tpot, _ttft) in sorted(rows.items()):
            if lab.endswith("-warm"):
                continue
            sp = spec.get(lab, {})
            acc = sp.get("acceptance_length", float("nan"))
            curve = ""
            if tier.get(lab):
                t = tier[lab]
                cold, hot, vram, swp = (wmean(t, 0, i) for i in (1, 2, 3, 4))
                host = wmean(t, 0, 5) + wmean(t, 0, 6)
                curve = t[-1][7] if len(t) else ""
            elif trace.get(lab):
                t = trace[lab]
                cold, hot, vram = (wmean(t, 3, i) for i in (0, 1, 2))
                swp, host = float("nan"), float("nan")
            else:
                cold = hot = vram = swp = host = float("nan")
            cx = sum(cpu.get(lab, [])) / max(len(cpu.get(lab, [])), 1) if cpu.get(lab) else float("nan")
            print(f"{lab:30s} {c:>2d} {tps:7.1f} {tpot:6.1f} {acc:5.2f} {cold:7.2f} {hot:6.2f} {vram:6.1f} {swp:7.1f} "
                  f"{host:8.2f} {cx:6.1f}  {curve[:90]}")
            letter = re.sub(r"\d+$", "", lab.rsplit("-", 1)[-1]) if c == 1 and not lab.endswith("-c8") else None
            if letter:
                groups[letter].append(tps)
        for k, v in sorted(groups.items()):
            print(f"  c1 {k}: mean {sum(v) / len(v):.1f} tok/s over {len(v)} passes: {', '.join(f'{x:.1f}' for x in v)}")
        if len(groups) == 2:
            a, b = (sum(v) / len(v) for _, v in sorted(groups.items()))
            print(f"  B vs A: {100 * (b / a - 1):+.1f}%")


if __name__ == "__main__":
    main()
