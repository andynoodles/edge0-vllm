#!/usr/bin/env python3
"""How many of the 8 latent D/U routing groups a decode step actually uses (single-stream dense lever).

  python3 group_usage.py --trace '<dir>/routes-*.npz' --from-ts T0 --to-ts T1 --labels <labels.bin> [--w 1,2,4]

Per layer and step width w (consecutive c1 decode tokens = one verify window of w positions), the distinct groups
used by the window's routes. D rows / U columns needed per step = distinct groups x 1792 (of 8 x 1792).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_tier import L, load_c1  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--trace", required=True)
ap.add_argument("--from-ts", type=float, default=0)
ap.add_argument("--to-ts", type=float, default=1e12)
ap.add_argument("--labels", required=True)
ap.add_argument("--w", default="1,2,4")
a = ap.parse_args()
lab = np.fromfile(a.labels, dtype=np.int32).reshape(48, 512)
toks = load_c1(a.trace, a.from_ts, a.to_ts)          # [L, T, 10]
T = toks.shape[1]
print(f"{T} c1 decode tokens")
for w in [int(x) for x in a.w.split(",")]:
    n = T // w
    per_layer = []
    hist = np.zeros(9)
    for l in range(L):
        g = lab[l][toks[l, :n * w].reshape(n, w * 10)]          # [n, w*10]
        d = np.array([len(np.unique(r)) for r in g])
        per_layer.append(d.mean())
        hist += np.bincount(d, minlength=9)[:9]
    per_layer = np.array(per_layer)
    print(f"w={w}: distinct groups per layer-step mean {per_layer.mean():.2f} of 8 "
          f"(D/U bytes {per_layer.mean() / 8:.0%}); layer min {per_layer.min():.2f} max {per_layer.max():.2f}; "
          "histogram 0..8 " + " ".join(f"{x / hist.sum():.3f}" for x in hist))
