#!/usr/bin/env python3
"""Summarize vllm-bench results: one row per (label, concurrency); several labels = side by side.

  python3 summarize.py <label|dir> [...]          (labels are looked up in ~/work/edge0-bench/results/)
  python3 summarize.py --md vllm-base             markdown table
"""
import glob
import json
import os
import sys

ROOT = os.path.expanduser("~/work/edge0-bench/results")


def load(label):
    d = label if os.path.isdir(label) else os.path.join(ROOT, label)
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            r = json.load(open(f))
        except (OSError, ValueError):
            continue
        c = int(r.get("max_concurrency") or 0)
        rows[c] = r            # last run per concurrency wins
    return rows


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    md = "--md" in sys.argv
    hdr = ["label", "conc", "ok", "out tok/s", "TPOT ms", "med TTFT ms", "p99 TTFT ms", "out len"]
    if md:
        print("| " + " | ".join(hdr) + " |")
        print("|" + "|".join("---:" if i else "---" for i in range(len(hdr))) + "|")
    else:
        print("{:28s} {:>4s} {:>7s} {:>9s} {:>8s} {:>11s} {:>11s} {:>7s}".format(*hdr))
    for lab in args:
        for c, r in sorted(load(lab).items()):
            out_len = r["total_output_tokens"] / max(r["completed"], 1)
            vals = [os.path.basename(lab), c, f"{r['completed']}/{r['num_prompts']}", f"{r['output_throughput']:.1f}",
                    f"{r['mean_tpot_ms']:.1f}", f"{r['median_ttft_ms']:.0f}", f"{r['p99_ttft_ms']:.0f}", f"{out_len:.0f}"]
            if md:
                print("| " + " | ".join(str(v) for v in vals) + " |")
            else:
                print("{:28s} {:>4} {:>7s} {:>9s} {:>8s} {:>11s} {:>11s} {:>7s}".format(*vals))


if __name__ == "__main__":
    main()
