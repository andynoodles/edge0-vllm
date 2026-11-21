"""Hot/cold expert placement: which experts of each layer live in VRAM (DESIGN.md 3.3).

A placement maps, per layer, physical slots to logical experts: ``order[i]`` is the logical expert stored in slot
``i``; slots ``< n_hot`` are in VRAM, the rest in pinned host memory. The router is never permuted: the latent MoE op
translates logical ids through per-tier expert maps, so routing stays bit-exact.

Sources:
  profile        top-H (layer, expert) pairs of Strata's profile.bin ranking (Strata's ``--expert-cache auto`` order)
  identity       control run: logical order, H split evenly over the layers
  plan:<json>    planner / adaptive-cache output, format ``edge0-placement-v1``

Pure numpy; no torch.
"""
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import SLOT_BYTES, hot_slots_for_gib

PLAN_FORMAT = "edge0-placement-v1"


@dataclass
class LayerPlacement:
    order: list[int]        # physical slot -> logical expert
    n_hot: int              # slots [0, n_hot) are in VRAM

    @property
    def num_experts(self) -> int:
        return len(self.order)

    @property
    def n_cold(self) -> int:
        return len(self.order) - self.n_hot

    def slot_of(self) -> list[int]:
        inv = [-1] * len(self.order)
        for s, e in enumerate(self.order):
            inv[e] = s
        return inv

    def hot_map(self) -> list[int]:
        """logical expert -> hot slot index, or -1 (the Marlin ``expert_map`` of the hot call)."""
        m = [-1] * len(self.order)
        for s in range(self.n_hot):
            m[self.order[s]] = s
        return m

    def cold_map(self) -> list[int]:
        """logical expert -> cold slot index, or -1 (the Marlin ``expert_map`` of the cold call)."""
        m = [-1] * len(self.order)
        for s in range(self.n_hot, len(self.order)):
            m[self.order[s]] = s - self.n_hot
        return m

    def validate(self, num_experts: int, where: str = "") -> None:
        if sorted(self.order) != list(range(num_experts)):
            raise ValueError(f"placement{where}: order is not a permutation of 0..{num_experts - 1}")
        if not 0 <= self.n_hot <= num_experts:
            raise ValueError(f"placement{where}: n_hot={self.n_hot} outside [0, {num_experts}]")


# ---------------------------------------------------------------------------------------------- profile.bin
def read_profile(path) -> dict:
    """Strata profile.bin: 'STRP', <5I version n_layers n_expert slots n_ranked>, n_ranked x <HH layer expert>
    (best first), then int32 table [n_layers][n_expert] = rank (MODEL-SPEC 5.7)."""
    blob = Path(path).read_bytes()
    if blob[:4] != b"STRP":
        raise ValueError(f"{path}: not a Strata profile (magic {blob[:4]!r})")
    ver, nl, ne, slots, n = struct.unpack_from("<5I", blob, 4)
    need = 24 + 4 * n + 4 * nl * ne
    if len(blob) < need:
        raise ValueError(f"{path}: {len(blob)} B, expected >= {need}")
    ranked = np.frombuffer(blob, "<u2", count=2 * n, offset=24).reshape(n, 2).astype(np.int64)
    table = np.frombuffer(blob, "<i4", count=nl * ne, offset=24 + 4 * n).reshape(nl, ne)
    return {"version": ver, "n_layers": nl, "n_expert": ne, "slots": slots, "n_ranked": n,
            "ranked": ranked, "table": table}


def plan_from_ranking(ranked: np.ndarray, num_layers: int, num_experts: int, total_hot: int
                      ) -> dict[int, LayerPlacement]:
    """Top-``total_hot`` pairs of a best-first (layer, expert) ranking. Per layer: the chosen experts by rank, then
    the rest by rank, then any expert the ranking never mentions in logical order. Pairs of layers >= num_layers
    are skipped (the L4 mini checkpoint uses the first 4 layers of the full profile)."""
    rank_lists: list[list[int]] = [[] for _ in range(num_layers)]
    seen = np.zeros((num_layers, num_experts), dtype=bool)
    for layer, expert in ranked.tolist():
        if layer < num_layers and expert < num_experts and not seen[layer, expert]:
            seen[layer, expert] = True
            rank_lists[layer].append(expert)
    n_hot = [0] * num_layers
    budget = min(int(total_hot), num_layers * num_experts)
    # Walk the global ranking (restricted to our layers) and take the first `budget` pairs.
    taken = 0
    if budget > 0:
        for layer, expert in ranked.tolist():
            if taken >= budget:
                break
            if layer < num_layers and expert < num_experts:
                n_hot[layer] += 1
                taken += 1
    # Pairs the ranking never lists (not expected for Strata's full ranking) fill the remaining budget evenly.
    out = {}
    for layer in range(num_layers):
        order = rank_lists[layer] + [e for e in range(num_experts) if not seen[layer, e]]
        out[layer] = LayerPlacement(order=order, n_hot=min(n_hot[layer], num_experts))
    missing = budget - sum(p.n_hot for p in out.values())
    layer = 0
    while missing > 0:
        p = out[layer % num_layers]
        if p.n_hot < num_experts:
            p.n_hot += 1
            missing -= 1
        layer += 1
    return out


def plan_identity(num_layers: int, num_experts: int, total_hot: int) -> dict[int, LayerPlacement]:
    """Control placement: logical order, ``total_hot`` split evenly (the first layers get the remainder)."""
    total_hot = min(int(total_hot), num_layers * num_experts)
    base, extra = divmod(total_hot, num_layers)
    return {l: LayerPlacement(order=list(range(num_experts)), n_hot=min(num_experts, base + (1 if l < extra else 0)))
            for l in range(num_layers)}


def load_plan(path, num_layers: int, num_experts: int) -> dict[int, LayerPlacement]:
    d = json.loads(Path(path).read_text())
    if d.get("format") != PLAN_FORMAT:
        raise ValueError(f"{path}: format {d.get('format')!r}, expected {PLAN_FORMAT}")
    if int(d.get("num_experts", num_experts)) != num_experts:
        raise ValueError(f"{path}: num_experts {d.get('num_experts')} != {num_experts}")
    if int(d.get("slot_bytes", SLOT_BYTES)) != SLOT_BYTES:
        raise ValueError(f"{path}: slot_bytes {d.get('slot_bytes')} != {SLOT_BYTES}")
    layers = d["layers"]
    out = {}
    for layer in range(num_layers):
        entry = layers.get(str(layer))
        if entry is None:
            raise ValueError(f"{path}: no entry for layer {layer}")
        out[layer] = LayerPlacement(order=[int(e) for e in entry["order"]], n_hot=int(entry["n_hot"]))
        out[layer].validate(num_experts, f" {path} layer {layer}")
    return out


def save_plan(path, placements: dict[int, LayerPlacement], source: str) -> None:
    d = {"format": PLAN_FORMAT, "num_layers": len(placements),
         "num_experts": next(iter(placements.values())).num_experts if placements else 0,
         "slot_bytes": SLOT_BYTES, "source": source,
         "layers": {str(l): {"order": list(p.order), "n_hot": p.n_hot} for l, p in sorted(placements.items())}}
    Path(path).write_text(json.dumps(d))


def placement_digest(placements: dict[int, LayerPlacement]) -> str:
    h = hashlib.sha256()
    for layer in sorted(placements):
        p = placements[layer]
        h.update(np.asarray(p.order, np.int32).tobytes())
        h.update(int(p.n_hot).to_bytes(4, "little"))
    return h.hexdigest()[:12]


def resolve_placement(placement: str, expert_vram_gib: float | None, ckpt_dir, profile_file: str,
                      num_layers: int, num_experts: int) -> tuple[dict[int, LayerPlacement], str]:
    """Build the per-layer placement for the EDGE0_PLACEMENT / EDGE0_EXPERT_VRAM_GIB settings.

    Returns (placements, description). A plan file brings its own n_hot; it must fit the VRAM budget when one is
    given. Without EDGE0_EXPERT_VRAM_GIB the plugin refuses to guess (an accidental all-cold run pins 44 GiB)."""
    if placement.startswith("plan:"):
        path = placement[len("plan:"):]
        plans = load_plan(path, num_layers, num_experts)
        total = sum(p.n_hot for p in plans.values())
        if expert_vram_gib is not None and total > hot_slots_for_gib(expert_vram_gib):
            raise ValueError(f"plan {path}: {total} hot slots ({total * SLOT_BYTES / 2**30:.2f} GiB) exceed "
                             f"EDGE0_EXPERT_VRAM_GIB={expert_vram_gib}")
        return plans, f"plan:{path} ({total} hot)"
    if expert_vram_gib is None:
        raise ValueError("EDGE0_EXPERT_VRAM_GIB is not set: give the VRAM budget for hot experts in GiB "
                         "(0 = every expert in host memory, fallback F0); serve.sh sets it")
    total = hot_slots_for_gib(expert_vram_gib)
    if placement == "identity":
        return plan_identity(num_layers, num_experts, total), f"identity H={total}"
    prof_path = Path(ckpt_dir) / profile_file
    prof = read_profile(prof_path)
    if prof["n_expert"] != num_experts or prof["n_layers"] < num_layers:
        raise ValueError(f"{prof_path}: {prof['n_layers']} layers x {prof['n_expert']} experts does not cover "
                         f"{num_layers} x {num_experts}")
    plans = plan_from_ranking(prof["ranked"], num_layers, num_experts, total)
    return plans, f"profile {prof_path.name} H={min(total, num_layers * num_experts)}"
