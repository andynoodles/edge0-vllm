#!/usr/bin/env python3
"""Per-request c1 summary of vllm-bench runs saved with --save-detailed (TUNING-LOG "Single-stream" C1).

  python3 tools/c1_compare.py <label> [...] [--metrics DIR]     (labels under ~/work/edge0-bench/results/)

Per label: output tok/s (vllm-bench: all output tokens / pass duration, prefill included), mean reply length, vllm-bench's
mean TPOT, and from the per-request data: TTFT to the first content token and the decode TPOT from there to the end
((latency - first content) / (tokens - 1)). Strata's server sends an empty role chunk at once (vllm-bench TTFT ~1 ms,
so its TPOT then includes the prefill); vLLM sends its first chunk with the first token. "First content" is therefore
ttft + itls[0] when ttft < 20 ms, else ttft. With --metrics DIR and a Strata label, the engine's own first-token time
(GET /metrics, requests[].first_engine_token_s, the label's last N requests) is printed as well. Standard library only.
"""
import glob
import json
import os
import statistics as st
import sys

ROOT = os.path.expanduser("~/work/edge0-bench/results")


def load(label):
    files = sorted(glob.glob(os.path.join(ROOT, label, "*.json")))
    if not files:
        return None
    return json.load(open(files[-1]))


def first_content(ttft, itls):
    return ttft + itls[0] if ttft < 0.02 and itls else ttft


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mdir = None
    if "--metrics" in sys.argv:
        mdir = sys.argv[sys.argv.index("--metrics") + 1]
        args = [a for a in args if a != mdir]
    hdr = ("label", "n", "tok/s", "reply", "TPOT(vb)", "TPOT dec", "TTFT med", "TTFT mean", "engine TTFT")
    print("{:22s} {:>3s} {:>7s} {:>6s} {:>9s} {:>9s} {:>9s} {:>10s} {:>22s}".format(*hdr))
    for lab in args:
        d = load(lab)
        if d is None:
            print(f"{lab:22s} (no result)")
            continue
        n = d["completed"]
        reply = d["total_output_tokens"] / max(n, 1)
        tf, dec = [], []
        if "ttfts" in d:
            for ttft, itls, lat, ol in zip(d["ttfts"], d["itls"], d["latencies"], d["output_lens"]):
                t0 = first_content(ttft, itls)
                tf.append(t0 * 1e3)
                if ol > 1:
                    dec.append((lat - t0) / (ol - 1) * 1e3)
        eng = ""
        if mdir:
            mf = os.path.join(mdir, f"{lab}.metrics.json")
            if os.path.isfile(mf):
                try:
                    reqs = json.load(open(mf)).get("requests") or []
                    last = reqs[:n]          # newest first
                    v = [r["first_engine_token_s"] * 1e3 for r in last if r.get("first_engine_token_s") is not None]
                    reused = sum(r.get("reused") or 0 for r in last) / max(sum(r.get("prompt_tokens") or 0 for r in last), 1)
                    eng = f"{st.mean(v):.0f} ms, {reused:.0%} reused" if v else ""
                except (ValueError, KeyError, TypeError):
                    eng = "?"
        print("{:22s} {:>3d} {:>7.1f} {:>6.0f} {:>9.2f} {:>9s} {:>9s} {:>10s} {:>22s}".format(
            lab, n, d["output_throughput"], reply, d["mean_tpot_ms"],
            f"{st.mean(dec):.2f}" if dec else "-", f"{st.median(tf):.0f}" if tf else "-",
            f"{st.mean(tf):.0f}" if tf else "-", eng))


if __name__ == "__main__":
    main()
