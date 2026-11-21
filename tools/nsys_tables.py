#!/usr/bin/env python3
"""Markdown tables for docs/PROFILE.md from ``nsys_breakdown.py steps --json`` summaries.

  nsys_tables.py <profiles dir> [nospec|spec]
"""
import json
import sys

ROWS = [
    ("Attention: GDN kernels (36 layers)", ["attn_gdn"]),
    ("Attention: QSA indexer + sparse attention (12 layers)", ["attn_qsa"]),
    ("Dense GEMMs: attention projections (GDN qkvz/ba/out, QSA qkv/o)", ["attn_proj"]),
    ("Dense: router gate (bf16, fp32 logits)", ["router_gate"]),
    ("Latent D (Q8, 2560 -> 8x1792)", ["latent_D"]),
    ("Latent U (Q8, 8x1792 -> 2560)", ["latent_U"]),
    ("Shared expert (Q8 gate_up + down, sigmoid gate)", ["shared_expert"]),
    ("Hyper-connections + norms (incl. their bf16 GEMMs)", ["hc_norm"]),
    ("MoE routing (top-10, renormalize, touch counter)", ["moe_route"]),
    ("MoE route gather + combine", ["moe_gather", "moe_combine"]),
    ("Hot-expert Marlin (VRAM), part not hidden under the cold branch", ["hot_marlin"]),
    ("**Cold-expert Marlin (zero-copy over PCIe)**", ["cold_marlin"]),
    ("Final HC/norm + lm_head + sampler", ["lm_head_sample"]),
    ("Drafter (DFlash2 forward, inside propose)", ["drafter"]),
    ("Rejection sampling", ["spec_reject"]),
    ("Input prep / other GPU kernels / PLE GPU kernels", ["other_gpu", "ple_gpu"]),
    ("memcpy H2D + D2H + memset (PLE rows, ids)", ["memcpy_h2d", "memcpy_d2h", "memset"]),
]
IDLE = [
    ("GPU idle: PLE row reads (host preadv after a full stream sync)", ["ple.read"]),
    ("GPU idle: graph launch (cudaGraphLaunch; node tracing inflates it)", ["graph.full", "graph.pw"]),
    ("GPU idle: rest (input prep, Python, scheduler, sync)", None),
]
SETS = {
    "nospec": [("c1", "nospec/c1-k1"), ("c8", "nospec/c8-k2"), ("c32", "nospec/c32-k3")],
    "spec": [("no-spec c1", "nospec11/c1-k1"), ("DFlash2 k=3 c1", "dflash3/c1-k1"),
             ("no-spec c8", "nospec11/c8-k2"), ("DFlash2 k=3 c8", "dflash3/c8-k2")],
    "graph": [("c1", "graph/c1-k1"), ("c8", "graph/c8-k2"), ("c32", "graph/c32-k3")],
}


def main():
    base = sys.argv[1]
    which = sys.argv[2] if len(sys.argv) > 2 else "nospec"
    cols = [(n, json.load(open(f"{base}/{p}.json"))["summary"]) for n, p in SETS[which]]

    def cell(v, s):
        return f"{v:.2f} | {100 * v / s['period_ms']:.1f}"

    print("| per decode step | " + " | ".join(f"{n} ms | %" for n, _ in cols) + " |")
    print("|---|" + "---:|---:|" * len(cols))
    for name, keys in ROWS:
        vals = [sum(s["excl_ms"].get(k, 0) for k in keys) for _, s in cols]
        if all(v < 0.005 for v in vals):
            continue
        print(f"| {name} | " + " | ".join(cell(v, s) for v, (_, s) in zip(vals, cols)) + " |")
    for name, keys in IDLE:
        if keys is None:
            vals = [s["idle_ms"] - sum(s["gap_ms"].get(k, 0) for k in ("ple.read", "graph.full", "graph.pw"))
                    for _, s in cols]
        else:
            vals = [sum(s["gap_ms"].get(k, 0) for k in keys) for _, s in cols]
        print(f"| {name} | " + " | ".join(cell(v, s) for v, (_, s) in zip(vals, cols)) + " |")
    print("| **GPU idle total** | " + " | ".join(cell(s["idle_ms"], s) for _, s in cols) + " |")
    print("| **step period (GPU, mean)** | " + " | ".join(f"**{s['period_ms']:.2f}** | 100" for _, s in cols) + " |")
    print()
    for n, s in cols:
        print(f"{n}: {s['steps']} steps, median {s['period_med_ms']:.2f}, busy {s['busy_ms']:.2f}, "
              f"hot sum {s['sums_ms'].get('hot_marlin', 0):.2f}, cold sum {s['sums_ms'].get('cold_marlin', 0):.2f}, "
              f"kernels {s['kernels']:.0f}, host {({k: round(v, 3) for k, v in s['host_ms'].items()})}, "
              f"api {({k: round(v, 3) for k, v in s['api_ms'].items()})}")


if __name__ == "__main__":
    main()
