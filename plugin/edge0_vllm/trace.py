"""Expert-traffic instruments (DESIGN.md 4.2), ``EDGE0_TRACE=bw,routes``.

``bw``: per layer, a float32 [E] touch counter (+1 per distinct expert per MoE call, TidalMoE's ``count=touch``) and
a call counter, updated inside the op with no host sync (CUDA-graph safe). Every ``EDGE0_TRACE_FLUSH_S`` the engine
thread (``flush`` is called from ``compute_logits``, which runs eagerly once per step) reads and zeroes them and
logs distinct hot / cold experts per call and the implied cold bytes per step (x 1.85 MB). The same counters are the
input of the phase-3 adaptive cache.

``routes``: per layer, a GPU ring of logical expert ids [cap, K] int16 plus a device cursor; ``flush`` appends the
new rows to ``<trace_dir>/routes-<pid>.npz`` shards for offline placement planning (L2). A second ring records, for
every MoE call of layer 0, ``(rows, real tokens)``, so the planner can split the layer arrays back into engine steps
(npz key ``calls``; padding rows are -1 in the layer arrays).

Padding rows of CUDA-graph-padded batches are excluded when vLLM's forward context exposes ``is_padding``
(``VLLM_MOE_SKIP_PADDING``, default on). Dummy/profile runs before serving are counted too; the first flush after
start-up therefore over-reports and should be ignored.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from .config import SLOT_BYTES
from .kernels import TIER_STATS, tier_count, touch_count


def _padding_mask(num_tokens: int) -> torch.Tensor | None:
    try:
        from vllm.forward_context import get_forward_context, is_forward_context_available
        if not is_forward_context_available():
            return None
        pad = getattr(get_forward_context(), "is_padding", None)
        return None if pad is None else pad[:num_tokens]
    except Exception:  # noqa: BLE001
        return None


class ExpertTrace:
    def __init__(self, modes: frozenset, num_layers: int, num_experts: int, top_k: int, device: torch.device,
                 flush_s: float, ring_tokens: int, trace_dir: str, logger):
        self.modes = modes
        self.num_layers, self.num_experts, self.top_k = num_layers, num_experts, top_k
        self.device = device
        self.flush_s = flush_s
        self.log = logger
        self.touch = torch.zeros((num_layers, num_experts), dtype=torch.float32, device=device)
        self.calls = torch.zeros(num_layers, dtype=torch.int64, device=device)
        self.tokens = torch.zeros(num_layers, dtype=torch.int64, device=device)
        # hot / cold distinct experts and routes against the expert map at call time (kernels.tier_count); the
        # touch counter above is classified with the map at flush time, which a fast-changing tier makes stale
        self.callstats = torch.zeros((num_layers, num_experts + TIER_STATS), dtype=torch.float32, device=device)
        self.stats_max_tokens = 128          # decode-sized calls only (the model sets max_decode_tokens)
        self.ring = None
        if "routes" in modes:
            self.cap = int(ring_tokens)
            self.ring = torch.full((num_layers, self.cap, top_k), -1, dtype=torch.int16, device=device)
            self.cursor = torch.zeros(num_layers, dtype=torch.int64, device=device)
            self._read = np.zeros(num_layers, dtype=np.int64)
            self.call_cap = 1 << 16
            self.call_ring = torch.zeros((self.call_cap, 2), dtype=torch.int64, device=device)
            self.call_cursor = torch.zeros(1, dtype=torch.int64, device=device)
            self._call_read = 0
            os.makedirs(trace_dir, exist_ok=True)
            self.trace_path = os.path.join(trace_dir, f"routes-{os.getpid()}-{int(time.time())}")
            self._shard = 0
        self._last = time.monotonic()
        self.hot_masks: dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------------------------------ in the op
    def record(self, layer: int, ids: torch.Tensor) -> None:
        """ids: [T, K] int32 logical expert ids of one MoE call (graph-safe)."""
        t = ids.shape[0]
        pad = _padding_mask(t)
        if pad is not None:
            ids = torch.where(pad.view(-1, 1), torch.full_like(ids, -1), ids)
            n_real = t - pad.to(torch.int64).sum()
        else:
            n_real = torch.full((), t, dtype=torch.int64, device=ids.device)
        touch_count(ids, self.touch[layer], touch=True)
        hm = self.hot_masks.get(layer)
        if hm is not None and "bw" in self.modes and t <= self.stats_max_tokens:
            tier_count(ids, self.callstats[layer], hm)
        self.calls[layer] += 1
        self.tokens[layer] += n_real
        if self.ring is not None and t:
            pos = (self.cursor[layer] + torch.arange(t, device=ids.device)) % self.cap
            self.ring[layer].index_copy_(0, pos, ids.to(torch.int16))
            self.cursor[layer] += t
            if layer == 0:
                rec = torch.stack([torch.full((), t, dtype=torch.int64, device=ids.device), n_real.to(torch.int64)])
                self.call_ring.index_copy_(0, self.call_cursor % self.call_cap, rec.view(1, 2))
                self.call_cursor += 1

    # ------------------------------------------------------------------------------------------ engine thread
    def register_pool(self, layer: int, hot_map: torch.Tensor) -> None:
        self.hot_masks[layer] = hot_map

    def maybe_flush(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last < self.flush_s:
            return
        dt, self._last = now - self._last, now
        if "bw" in self.modes:
            self._flush_bw(dt)
        if self.ring is not None:
            self._flush_routes()

    def _flush_bw(self, dt: float) -> None:
        touch = self.touch.clone()
        calls = self.calls.clone()
        tokens = self.tokens.clone()
        cs = self.callstats[:, self.num_experts:self.num_experts + 5].sum(0).cpu().numpy()
        self.touch.zero_()
        self.calls.zero_()
        self.tokens.zero_()
        self.callstats.zero_()
        calls_h = calls.cpu().numpy()
        if calls_h.max(initial=0) == 0:
            return
        hot_t = np.zeros(self.num_layers)
        cold_t = np.zeros(self.num_layers)
        for layer, hot_map in self.hot_masks.items():
            is_hot = (hot_map >= 0).to(torch.float32)
            hot_t[layer] = float((touch[layer] * is_hot).sum())
            cold_t[layer] = float((touch[layer] * (1.0 - is_hot)).sum())
        steps = max(1, int(calls_h.max()))
        cold_per_step = cold_t.sum() / steps
        self.log.info(
            "edge0 trace bw: %.1fs %d steps %d tok | per step: hot %.0f cold %.0f distinct experts, cold %.1f MB "
            "(%.2f GB/s if all in %.1fs) | worst layers cold/step: %s",
            dt, steps, int(tokens.max().item()), hot_t.sum() / steps, cold_per_step,
            cold_per_step * SLOT_BYTES / 1e6, cold_t.sum() * SLOT_BYTES / 1e9 / max(dt, 1e-9), dt,
            ", ".join(f"L{l}:{cold_t[l] / steps:.0f}" for l in np.argsort(-cold_t)[:6]))
        if cs[4] > 0:
            self.log.info("edge0 trace bw (map at call time, decode calls): per layer-call cold %.2f hot %.2f "
                          "distinct experts, routes from VRAM %.1f%% (%d layer-calls)", cs[0] / cs[4], cs[1] / cs[4],
                          100.0 * cs[3] / max(cs[2] + cs[3], 1.0), int(cs[4]))

    def _flush_routes(self) -> None:
        cur = self.cursor.cpu().numpy()
        rows, lost = [], 0
        for layer in range(self.num_layers):
            new = int(cur[layer] - self._read[layer])
            if new <= 0:
                rows.append(np.zeros((0, self.top_k), np.int16))
                continue
            if new > self.cap:
                lost += new - self.cap
                new = self.cap
            start = int(cur[layer] - new) % self.cap
            idx = (np.arange(new) + start) % self.cap
            rows.append(self.ring[layer][torch.from_numpy(idx).to(self.device)].cpu().numpy())
            self._read[layer] = cur[layer]
        ccur = int(self.call_cursor.item())
        new_calls = min(ccur - self._call_read, self.call_cap)
        cidx = (np.arange(new_calls) + (ccur - new_calls)) % self.call_cap
        calls = self.call_ring[torch.from_numpy(cidx).to(self.device)].cpu().numpy() if new_calls > 0 else \
            np.zeros((0, 2), np.int64)
        self._call_read = ccur
        if any(r.shape[0] for r in rows):
            path = f"{self.trace_path}-{self._shard:05d}.npz"
            np.savez_compressed(path, calls=calls, lost=np.array([lost]),
                                **{f"layer{l}": r for l, r in enumerate(rows)})
            self._shard += 1
            self.log.info("edge0 trace routes: wrote %s (%d rows layer 0, %d lost to ring overflow)", path,
                          rows[0].shape[0], lost)


class PredictTrace:
    """``EDGE0_TRACE=predict``: would cross-layer prefetch work? Layer l+1's router (``gate``) is applied to layer l's
    MoE input (pre-gating, TidalMoE EXPERT-CACHE-ANALYSIS design C / Fate); its top-k' per token is compared at layer
    l+1 with the experts actually routed to. Per layer and k' in ``KS``: route-level hits (all routes, cold routes)
    and step-level coverage of the distinct cold experts by the union of the predicted cold experts (what a
    prefetch of that union would have saved) plus the over-fetch. Graph-safe counters, logged every flush."""

    KS = (10, 16, 24, 32)

    def __init__(self, blocks, max_tokens: int, device, flush_s: float, logger, out_dir: str):
        self.blocks = {b.layer_idx: b for b in blocks}
        self.L = max(self.blocks) + 1
        self.kmax = max(self.KS)
        self.max_tokens = max_tokens
        self.buf = torch.full((self.L, max_tokens, self.kmax), -1, dtype=torch.int32, device=device)
        nk = len(self.KS)
        # route[l, i, 0/1] = hits (all / cold routes) with k' = KS[i]; route[l, nk, 0/1] = routes (all / cold)
        self.route = torch.zeros((self.L, nk + 1, 2), dtype=torch.float64, device=device)
        # step[l, i, 0/1/2] = distinct cold touched / predicted cold (union) / covered, for k' = KS[i]
        self.step = torch.zeros((self.L, nk, 3), dtype=torch.float64, device=device)
        self.flush_s = flush_s
        self.log = logger
        self.out_dir = out_dir
        self._last = time.monotonic()

    def stash_from(self, layer: int, x: torch.Tensor) -> None:
        nxt = self.blocks.get(layer + 1)
        t = x.shape[0]
        if nxt is None or t > self.max_tokens:
            return
        lg = nxt.gate(x)
        if isinstance(lg, tuple):
            lg = lg[0]
        ids = torch.topk(lg.float(), self.kmax, dim=-1).indices.to(torch.int32)
        self.buf[layer + 1, :t].copy_(ids)

    def score(self, layer: int, ids: torch.Tensor, hot_map: torch.Tensor) -> None:
        t = ids.shape[0]
        if layer == 0 or layer >= self.L or t > self.max_tokens or t == 0:
            return
        pred = self.buf[layer, :t]                                            # [t, kmax]
        pad = _padding_mask(t)
        valid = torch.ones((t, 1), dtype=torch.bool, device=ids.device) if pad is None else ~pad.view(-1, 1)
        e = hot_map.numel()
        cold = (hot_map[ids.long()] < 0) & valid                              # [t, 10]
        match = ids.unsqueeze(-1) == pred.unsqueeze(1)                         # [t, 10, kmax]
        pcold = (hot_map[pred.long().clamp(0, e - 1)] < 0) & valid             # [t, kmax]
        touched = torch.zeros(e + 1, dtype=torch.float32, device=ids.device)
        touched.index_fill_(0, torch.where(cold, ids.long(), e).reshape(-1), 1.0)
        touched = touched[:e]
        nk = len(self.KS)
        for i, k in enumerate(self.KS):
            hit = match[..., :k].any(-1) & valid
            self.route[layer, i, 0] += hit.sum()
            self.route[layer, i, 1] += (hit & cold).sum()
            pu = torch.zeros(e + 1, dtype=torch.float32, device=ids.device)
            pu.index_fill_(0, torch.where(pcold[:, :k], pred[:, :k].long(), e).reshape(-1), 1.0)
            pu = pu[:e]
            self.step[layer, i, 0] += touched.sum()
            self.step[layer, i, 1] += pu.sum()
            self.step[layer, i, 2] += (pu * touched).sum()
        self.route[layer, nk, 0] += valid.expand_as(cold).sum()
        self.route[layer, nk, 1] += cold.sum()

    def maybe_flush(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last < self.flush_s:
            return
        self._last = now
        r = self.route.cpu().numpy()
        s = self.step.cpu().numpy()
        self.route.zero_()
        self.step.zero_()
        nk = len(self.KS)
        if r[:, nk, 0].sum() == 0:
            return
        parts = []
        for i, k in enumerate(self.KS):
            all_rec = r[:, i, 0].sum() / max(r[:, nk, 0].sum(), 1)
            cold_rec = r[:, i, 1].sum() / max(r[:, nk, 1].sum(), 1)
            cov = s[:, i, 2].sum() / max(s[:, i, 0].sum(), 1)
            over = s[:, i, 1].sum() / max(s[:, i, 0].sum(), 1)
            parts.append(f"k'={k}: route recall {all_rec * 100:.0f}% (cold {cold_rec * 100:.0f}%), step cold coverage "
                         f"{cov * 100:.0f}% fetching {over:.2f}x the touched cold")
        rec = r[1:, 0, 1] / np.maximum(r[1:, nk, 1], 1)
        worst = np.argsort(rec)[:4] + 1
        self.log.info("edge0 trace predict: %s | worst layers (k'=10 cold recall): %s", "; ".join(parts),
                      ", ".join(f"L{l}:{r[l, 0, 1] / max(r[l, nk, 1], 1) * 100:.0f}%" for l in worst))
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            np.savez(os.path.join(self.out_dir, f"predict-{os.getpid()}-{int(now)}.npz"), route=r, step=s,
                     ks=np.array(self.KS))
        except OSError:
            pass
