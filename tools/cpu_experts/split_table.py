"""Markdown table of a bench_cpu_split.py JSON (docs/TUNING-LOG.md "CPU cold experts").

  python3 split_table.py results/cpu-experts/split-HHMM.json [threads=8] [tiles=1:2]
"""
import json
import sys


def main():
    rows = json.load(open(sys.argv[1]))
    th = sys.argv[2] if len(sys.argv) > 2 else "8"
    tl = sys.argv[3] if len(sys.argv) > 3 else "1:2"
    cases = sorted({(r["T"], r["C"]) for r in rows})

    def get(label, t, c):
        for r in rows:
            if r["label"] == label and r["T"] == t and r["C"] == c:
                return r
        return None

    base = {c: get("gpu-only", *c)["us_med"] for c in cases}
    print("| config | " + " | ".join(f"T={t} C={c}" for t, c in cases) + " |")
    print("|---|" + "---:|" * len(cases))
    print("| GPU only | " + " | ".join(f"{base[c]:.0f}" for c in cases) + " |")
    labels = sorted({r["label"] for r in rows if r["label"].startswith(f"th={th} t={tl} ")})
    for lab in labels:
        vals = []
        for c in cases:
            r = get(lab, *c)
            vals.append(f"{r['us_med']:.0f} ({100 * (r['us_med'] / base[c] - 1):+.0f}%)" if r else "-")
        print(f"| {lab} | " + " | ".join(vals) + " |")
    print()
    print("| case | CPU experts | CPU us/job | us per CPU expert | GPU-side us | GPU experts | us per GPU expert |"
          " GPU waited us |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for lab in labels:
        for c in cases:
            r = get(lab, *c)
            if not r or r.get("cpu_experts", 0) == 0:
                continue
            g = c[1] - r["cpu_experts"]
            print(f"| {lab} T={c[0]} C={c[1]} | {r['cpu_experts']:.0f} | {r['cpu_us']:.0f} | "
                  f"{r['cpu_us'] / r['cpu_experts']:.1f} | {r['gpu_side_us']:.0f} | {g:.0f} | "
                  f"{r['gpu_side_us'] / max(g, 1):.1f} | {r['wait_us']:.0f} |")


if __name__ == "__main__":
    main()
