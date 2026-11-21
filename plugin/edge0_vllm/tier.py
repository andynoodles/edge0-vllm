"""Request-reactive expert tier (``EDGE0_TIER=1``): Strata's adaptive tier, without a host sync and without copies on
the critical path (docs/TUNING-LOG.md "Single-stream", docs/SINGLE-STREAM-GAP.md lever 1).

Why: at concurrency 1 the static plan + the 3 s adaptive epochs leave ~8 cold experts per layer per verify step;
Strata's tier (+1 per routed entry, decay 0.7 per round, swap while candidate >= victim + 1.5, <= 96 swaps every 4
windows) follows the current request within tens of tokens and halves that (replay: ``tools/sim_tier.py``).

Layout (``ExpertPool`` with ``n_free``): every layer with both tiers gets ``n_free`` empty VRAM slots after its
``n_hot`` hot slots (taken out of the plan's hot budget, ``demote``) and as many spare slots after its cold slots in
the pinned host pool. The expert maps of all layers are rows of one ``[2, L, E]`` int32 tensor and the Marlin global
scales of all slots one ``[2, N]`` fp32 tensor (``TierStorage``), so one kernel updates them for every layer.

Per step (``on_step`` from ``compute_logits``, i.e. while the GPU runs this step's forward):

1. In the op (``record``, graph-safe): ``kernels.tier_count`` adds 1 per routed entry of decode-sized calls into a
   ``[L, E + 8]`` counter and the call's distinct cold / hot experts and cold / hot routes *at call time*.
2. The counter is copied into a pinned ring buffer on the main stream after the forward and zeroed; the host folds a
   finished buffer into ``usage`` (``usage = usage * decay + counts``) one step later, after polling its event, so
   it never waits for the GPU.
3. Decision (numpy): per layer the top non-resident experts by usage (>= ``min_use``) against the least-used
   resident ones (never the padding experts), pairs while ``cand >= victim + gain`` and while the layer has a free
   VRAM slot; at most ``swaps`` pairs per step, largest gain first.
4. Copies with the copy engines, on two side streams: H2D of each incoming expert's four Marlin arrays from its host
   slot into a free VRAM slot, D2H of each victim's arrays from its VRAM slot into a spare host slot (the victim
   stays readable where it is), then the global scales of the two new slots. Nothing reads a free / spare slot, so
   the copies overlap this step's forward with no hazard.
5. The flip: the main stream waits for both copy streams and rewrites the maps of every pair (incoming: hot = free
   slot, cold = -1; victim: hot = -1, cold = spare slot) in one ``index_copy_``. It sits after this step's forward
   (``compute_logits`` runs after the forward is enqueued), so the next step sees the new placement; the victim's old
   VRAM slot and the incoming expert's old host slot become free / spare for later rounds, whose copies wait for
   this flip's event. Addresses never change, so captured CUDA graphs stay valid.

So a decision uses the counts of the steps before the current one and its experts are resident from the next step
(Strata: blocking swaps, resident one window earlier; the replay puts the cost of that lag at ~1%).

EDGE0_CPU_EXPERTS reads cold experts by host slot through ``cold_map`` and the cold global scales, which the flip
and the copies keep current: a swapped-in expert is never sent to the CPU, an evicted one is computed from its new
host slot. The per-step statistics (routes served from VRAM, cold experts per layer-step, a reaction curve by
steps since the last prefill) are logged every ``EDGE0_TIER_LOG_S``.

``EDGE0_TIER_EARLY=1`` (round 3, docs/SINGLE-STREAM-GAP.md section 12): a swap is resident one step earlier and its
copies run while the PCIe link is otherwise idle. ``on_step`` only snapshots the step's counts. The decision moves to
the start of the *next* step's input staging (``model_state.stage_checkpoint_rows``, wrapped: ``pre_sync`` before
vLLM's stream sync, ``post_sync`` after it): the host waits for the snapshot event of the forward that just ran (the
stream sync would wait longer: LM head, sampling, drafter), folds the counts, decides, and issues the copies right
away. They run during the LM head, the sampler and the DFlash2 drafter (~2 ms at c1, no cold-expert reads), not
during the next forward's first layers. The flip (scales, then maps) is enqueued on the copy stream behind both
copy streams, and ``post_sync`` makes the main stream wait for it before the next forward's launch. So the counts of
step N decide placement from step N+1 on (served tier: from N+2). ``EDGE0_TIER_COPY=batch`` issues each direction's
copies in one ``cudaMemcpyBatchAsync`` (host cost ~1 call instead of 4 per expert).

``EDGE0_TIER_SHADOW=1`` (default: on with EARLY): on this box an H2D and a D2H copy do not overlap (host DRAM is the
shared limit: 62 MB each way take 1.07 ms alone, 2.0 ms together; ``tools/tier_copy_bench.py``), so every victim's
write-back costs as much as the swap itself. A swapped-in expert therefore keeps its host slot as a *shadow* (its
bytes stay valid there while it is hot: nothing writes it) and is evicted back to it with no D2H. Only a victim that
was never in host memory (hot in the plan) is written back, into a spare host slot or, when none is left, into the
shadow of the most used hot expert (that one loses its shadow). ``EDGE0_TIER_SPARE`` adds spare host slots per layer.
"""
from __future__ import annotations

import collections
import time
from dataclasses import dataclass

import numpy as np
import torch

from .config import S2_BYTES, S13_BYTES, TOP_K, W2_BYTES, W13_BYTES
from .kernels import TIER_STATS, tier_count
from .placement import LayerPlacement
from .trace import _padding_mask

_RING = 4
_CURVE = (4, 8, 16, 32, 64, 128, 256, 1 << 30)     # reaction curve bins: steps since the last prefill (upper bounds)
_ARR_BYTES = (W13_BYTES, W2_BYTES, S13_BYTES, S2_BYTES)


# ---------------------------------------------------------------------------------------------- layout
def free_slots(p: LayerPlacement, n_free: int) -> int:
    """Free VRAM slots of a layer (= spare host slots): only layers with both tiers swap."""
    return int(n_free) if (p.n_hot and p.n_cold) else 0


def demote(placements: dict[int, LayerPlacement], n_free: int) -> tuple[dict[int, LayerPlacement], int]:
    """Take ``n_free`` slots per layer out of the plan's hot budget (its least valuable hot experts become cold), so
    hot + free slots = the plan's VRAM. Keeps >= TOP_K hot experts (the padding rows' targets). Returns the new
    placements and the number of demoted experts."""
    out, moved = {}, 0
    for layer, p in placements.items():
        k = min(int(n_free), max(0, p.n_hot - TOP_K)) if p.n_cold else 0
        out[layer] = LayerPlacement(order=list(p.order), n_hot=p.n_hot - k)
        moved += k
    return out, moved


@dataclass
class _Views:
    hot_map: torch.Tensor
    cold_map: torch.Tensor
    hot_g13: torch.Tensor
    hot_g2: torch.Tensor
    cold_g13: torch.Tensor
    cold_g2: torch.Tensor


class TierStorage:
    """Expert maps ``[2, L, E]`` (0 = hot slot, 1 = cold slot, -1 = not in that tier) and Marlin global scales
    ``[2, Nh + Nc]`` (0 = g13, 1 = g2; layer l's hot slots at ``off_h[l]``, its cold slots at ``Nh + off_c[l]``)."""

    def __init__(self, placements: dict[int, LayerPlacement], n_free: int, device, spare_extra: int = 0):
        self.L = max(placements) + 1
        self.E = next(iter(placements.values())).num_experts
        self.n_free = {l: free_slots(p, n_free) for l, p in placements.items()}
        # spare host slots: n_free, plus EDGE0_TIER_SPARE more (room for shadow copies, tier.py EARLY)
        self.n_spare = {l: (f + int(spare_extra)) if f else 0 for l, f in self.n_free.items()}
        self.n_h = {l: p.n_hot + self.n_free[l] for l, p in placements.items()}
        self.n_c = {l: p.n_cold + self.n_spare[l] for l, p in placements.items()}
        self.off_h, self.off_c = {}, {}
        oh = oc = 0
        for l in range(self.L):
            self.off_h[l], self.off_c[l] = oh, oc
            oh += self.n_h.get(l, 0)
            oc += self.n_c.get(l, 0)
        self.Nh, self.Nc = oh, oc
        self.maps = torch.full((2, self.L, self.E), -1, dtype=torch.int32, device=device)
        self.g = torch.zeros((2, self.Nh + self.Nc), dtype=torch.float32, device=device)

    def views(self, layer: int) -> _Views:
        h0, hn = self.off_h[layer], self.n_h[layer]
        c0, cn = self.Nh + self.off_c[layer], self.n_c[layer]
        return _Views(self.maps[0, layer], self.maps[1, layer], self.g[0, h0:h0 + hn], self.g[1, h0:h0 + hn],
                      self.g[0, c0:c0 + cn], self.g[1, c0:c0 + cn])


def storage(rt, device) -> TierStorage:
    """The process-wide storage (created by the first latent MoE block, on its device)."""
    st = getattr(rt, "tier_storage", None)
    if st is None:
        st = TierStorage(rt.placements, rt.settings.tier_free, device, getattr(rt.settings, "tier_spare", 0))
        rt.tier_storage = st
    return st


# ---------------------------------------------------------------------------------------------- copies
class _Copier:
    """Slot-to-slot copies of one expert's four Marlin arrays, VRAM <-> pinned host, on a given stream. ``cudart``:
    raw ``cudaMemcpyAsync`` (~2 us of host time per call); ``torch``: ``Tensor.copy_(non_blocking=True)``; ``batch``:
    a whole round's copies of one direction in one ``cudaMemcpyBatchAsync`` (``many``; per-expert calls as cudart)."""

    def __init__(self, mode: str):
        self.mode = mode
        self.rt = None
        if mode in ("cudart", "batch"):
            try:
                from cuda.bindings import runtime as cudart
                self.rt = cudart
                self.kind = cudart.cudaMemcpyKind.cudaMemcpyDefault
            except Exception:  # noqa: BLE001
                self.mode = "torch"
        self.attr = None
        if self.mode == "batch":
            try:
                a = self.rt.cudaMemcpyAttributes()
                a.srcAccessOrder = self.rt.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderStream
                self.attr = a
            except Exception:  # noqa: BLE001
                self.mode = "cudart"

    def many(self, dsts: list[int], srcs: list[int], sizes: list[int], stream) -> None:
        """Raw pointer copies (cudart / batch modes), in stream order on ``stream``."""
        if not dsts:
            return
        sh = stream.cuda_stream
        if self.mode == "batch":
            err = self.rt.cudaMemcpyBatchAsync(dsts, srcs, sizes, len(dsts), [self.attr], [0], 1, sh)
            if int(err[0]) == 0:
                return
            self.mode = "cudart"             # fall back for good (logged by the tier's statistics line)
        for d, s, n in zip(dsts, srcs, sizes):
            err = self.rt.cudaMemcpyAsync(d, s, n, self.kind, sh)
            if int(err[0]):
                raise RuntimeError(f"edge0 tier: cudaMemcpyAsync failed: {err[0]}")

    @staticmethod
    def ptrs(pool):
        hot = [t.data_ptr() for t in (pool.hot.w13, pool.hot.w2, pool.hot.s13, pool.hot.s2)]
        cold = [t.data_ptr() for t in pool.cold.cpu]
        return hot, cold

    def h2d(self, pool, ptrs, hot_slot: int, cold_slot: int, stream) -> None:
        if self.mode in ("cudart", "batch"):
            hot, cold = ptrs
            sh = stream.cuda_stream
            for i, nb in enumerate(_ARR_BYTES):
                err = self.rt.cudaMemcpyAsync(hot[i] + hot_slot * nb, cold[i] + cold_slot * nb, nb, self.kind, sh)
                if int(err[0]):
                    raise RuntimeError(f"edge0 tier: cudaMemcpyAsync H2D failed: {err[0]}")
            return
        with torch.cuda.stream(stream):
            for th, tc in zip((pool.hot.w13, pool.hot.w2, pool.hot.s13, pool.hot.s2), pool.cold.cpu):
                th[hot_slot].copy_(tc[cold_slot], non_blocking=True)

    def d2h(self, pool, ptrs, hot_slot: int, cold_slot: int, stream) -> None:
        if self.mode in ("cudart", "batch"):
            hot, cold = ptrs
            sh = stream.cuda_stream
            for i, nb in enumerate(_ARR_BYTES):
                err = self.rt.cudaMemcpyAsync(cold[i] + cold_slot * nb, hot[i] + hot_slot * nb, nb, self.kind, sh)
                if int(err[0]):
                    raise RuntimeError(f"edge0 tier: cudaMemcpyAsync D2H failed: {err[0]}")
            return
        with torch.cuda.stream(stream):
            for th, tc in zip((pool.hot.w13, pool.hot.w2, pool.hot.s13, pool.hot.s2), pool.cold.cpu):
                tc[cold_slot].copy_(th[hot_slot], non_blocking=True)


# ---------------------------------------------------------------------------------------------- the tier
class ReactiveTier:
    def __init__(self, blocks, settings, device, log, max_tokens: int, storage: TierStorage | None = None):
        self.log = log
        self.blocks = sorted(blocks, key=lambda b: b.layer_idx)
        self.pools = {b.layer_idx: b.pool for b in self.blocks}
        self.st: TierStorage = storage if storage is not None else _runtime_storage()
        self.L, self.E = self.st.L, self.st.E
        L, E = self.L, self.E
        s = settings
        self.max_swaps = int(s.tier_swaps)
        self.decay = float(s.tier_decay)
        self.gain = float(s.tier_gain)
        self.min_use = float(s.tier_min_use)
        self.max_tokens = int(s.tier_max_tokens) or int(max_tokens)
        self.log_s = float(s.tier_log_s)
        self.first_layer = self.blocks[0].layer_idx if self.blocks else 0
        self.device = torch.device(device)
        # ---- in-op counter: [L, E + TIER_STATS] (per-entry counts, then cold_d, hot_d, cold_r, hot_r, calls, prefills)
        self.counts = torch.zeros((L, E + TIER_STATS), dtype=torch.float32, device=device)
        # ---- host mirrors of the placement
        self.is_hot = np.zeros((L, E), dtype=bool)
        self.hot_slot = np.full((L, E), -1, dtype=np.int64)
        self.cold_slot = np.full((L, E), -1, dtype=np.int64)
        self.pinned = np.zeros((L, E), dtype=bool)
        self.active = np.zeros(L, dtype=bool)
        self.free_v: list[collections.deque] = [collections.deque() for _ in range(L)]
        self.spare_h: list[collections.deque] = [collections.deque() for _ in range(L)]
        self.g = np.zeros((L, E, 2), dtype=np.float32)
        self.ptrs = {}
        for l, p in self.pools.items():
            for slot, e in enumerate(p.order):
                if slot < p.n_hot:
                    self.is_hot[l, e] = True
                    self.hot_slot[l, e] = slot
                else:
                    self.cold_slot[l, e] = slot - p.n_hot
            if p.pad_ids is not None:
                self.pinned[l, p.order[:TOP_K]] = True
            self.free_v[l].extend(range(p.n_hot, p.n_hot + p.n_free))
            self.spare_h[l].extend(range(p.n_cold, p.n_cold + p.n_spare))
            self.active[l] = p.n_free > 0
            if p.g_logical is not None:
                self.g[l] = p.g_logical.numpy()
            if not p.meta:
                self.ptrs[l] = _Copier.ptrs(p)
        self.usage = np.zeros((L, E), dtype=np.float64)
        self.K = max([p.n_free for p in self.pools.values()] + [1])
        # ---- streams, events, pinned ring buffers
        cuda = self.device.type == "cuda"
        self.h2d = torch.cuda.Stream(device=self.device) if cuda else None
        self.d2h = torch.cuda.Stream(device=self.device) if cuda else None
        self.copier = _Copier(s.tier_copy)
        self.cnt_host = [torch.zeros((L, E + TIER_STATS), dtype=torch.float32, pin_memory=cuda)
                         for _ in range(_RING)]
        self.cnt_ev = [None] * _RING
        self.cnt_q: collections.deque = collections.deque()    # filled ring slots, oldest first
        P = max(self.max_swaps, 1)
        self.P = P
        # list buffer per round: map idx int64 [4P] | g idx int64 [2P] | map values int32 [4P] | g values fp32 [2, 2P]
        self._lay = (0, 32 * P, 48 * P, 64 * P, 80 * P)
        nbytes = 80 * P
        self.lst_host = [torch.zeros(nbytes, dtype=torch.uint8, pin_memory=cuda) for _ in range(_RING)]
        self.lst_dev = [torch.zeros(nbytes, dtype=torch.uint8, device=device) for _ in range(_RING)]
        self.lst_ev = [None] * _RING
        self.round = 0
        self.flip_ev = None
        self.started = False
        # ---- EDGE0_TIER_EARLY: decide at the next step's staging, flip before its launch (module docstring)
        self.early = bool(getattr(s, "tier_early", False)) and cuda
        self.flip_wait = None                      # flip event the main stream must wait for before the next launch
        self.stall_q: collections.deque = collections.deque()   # (before, after) timing events around that wait
        self.pre_calls = 0
        # ---- EDGE0_TIER_SHADOW: host slot of a hot expert whose bytes are still valid there (-1 = none)
        self.shadow_on = bool(getattr(s, "tier_shadow", False))
        self.wb_cost = float(getattr(s, "tier_wb_cost", 0.0))
        self.shadow = np.full((L, E), -1, dtype=np.int64)
        self.fresh = np.zeros((L, E), dtype=bool)          # shadows made in the current round (scratch)
        # ---- EDGE0_TIER_GPU_DECIDE (off: measured slower, R3): the decision rule runs on a side stream after the
        # forward (usage folded in on the GPU), so the host only reads the pair list when it wakes up
        self.gpu_decide = (bool(getattr(s, "tier_gpu_decide", False)) and self.early and self.wb_cost == 0
                           and self.max_swaps > 0)
        self.dec_ev = None
        if self.gpu_decide:
            self.u_dev = torch.zeros((L, E), dtype=torch.float32, device=device)
            self.pinned_dev = torch.from_numpy(self.pinned).to(device)
            self.active_dev = torch.from_numpy(self.active).to(device)
            cap = [len(self.free_v[l]) if self.shadow_on else min(len(self.free_v[l]), len(self.spare_h[l]))
                   for l in range(L)]
            self.cap_dev = torch.tensor(cap, dtype=torch.int64, device=device)
            self.ar_k = torch.arange(self.K, device=device)
            self.kk = min(self.P, L * self.K)
            self.dec_dev = torch.zeros((5, self.P), dtype=torch.float32, device=device)
            self.dec_host = torch.zeros((5, self.P), dtype=torch.float32, pin_memory=True)
            self.dec_stream = torch.cuda.Stream(device=self.device)
        self.stolen = 0
        # ---- statistics
        self.since_prefill = 1 << 20
        self._reset_stats()
        self.t_log = time.monotonic()
        self.total_swaps = 0
        n_dem = sum(int(self.active[l]) for l in range(L))
        log.info("edge0 tier: reactive expert tier on %d layers: %d free VRAM slots / layer (%.2f GiB, from the hot "
                 "budget), <= %d swaps / step, decay %.3f / step, gain %.2f, min usage %.1f, counting calls <= %d "
                 "tokens, copies via %s%s; shadow copies %s, %d spare host slots", n_dem, self.K,
                 sum(p.n_free for p in self.pools.values()) * 1_935_360 / 2**30, self.max_swaps, self.decay,
                 self.gain, self.min_use, self.max_tokens, self.copier.mode,
                 ", EARLY (decided at the next step's staging, flipped before its launch)" if self.early else "",
                 "on" if self.shadow_on else "off", sum(p.n_spare for p in self.pools.values()))
        if self.early:
            install_stage_hook(self, log)

    # ------------------------------------------------------------------------------------------ in the op
    def count_row(self, layer: int, t: int) -> torch.Tensor | None:
        """``EDGE0_ROUTE_FUSE``: the counter row ``kernels.route_post`` adds this call's routes to, or None when the
        call is not counted (prefill-sized: the request-boundary mark is set here, as ``record`` does)."""
        if t == 0:
            return None
        row = self.counts[layer]
        if t > self.max_tokens:
            if layer == self.first_layer:
                row[self.E + 5] += 1.0
            return None
        return row

    def record(self, layer: int, ids: torch.Tensor) -> None:
        t = ids.shape[0]
        if t == 0:
            return
        row = self.counts[layer]
        if t > self.max_tokens:                   # static per captured size -> graph-safe branch
            if layer == self.first_layer:
                row[self.E + 5] += 1.0           # a prefill chunk: request boundary for the reaction curve
            return
        pad = _padding_mask(t)
        if pad is not None:
            ids = torch.where(pad.view(-1, 1), torch.full_like(ids, -1), ids)
        tier_count(ids, row, self.pools[layer].hot_map)

    # ------------------------------------------------------------------------------------------ engine thread
    def on_step(self) -> None:
        if self.device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            return
        main = torch.cuda.current_stream(self.device)
        if not self.started:                       # drop counts of profiling / warm-up / graph capture
            self.counts.zero_()
            self.started = True
            return
        if self.early and self.pre_calls == 0 and self.n_calls >= 64 and len(self.cnt_q) >= _RING - 1:
            self.early = False                     # the staging hook never ran (no file-gather PLE): decide here
            self.log.warning("edge0 tier: EDGE0_TIER_EARLY: the staging hook was not called in %d steps; deciding "
                             "at compute_logits as without EARLY", self.n_calls)
        t0 = time.perf_counter()
        t1 = t0
        if not self.early:
            fresh = False
            while self.cnt_q and self.cnt_ev[self.cnt_q[0]].query():
                self._consume(self.cnt_q.popleft())
                fresh = True
            if not fresh:
                self.n_stale += 1
            pairs = self._decide() if (fresh and self.max_swaps > 0) else []
            t1 = time.perf_counter()
            if pairs:
                self._issue(pairs, main)
        t2 = time.perf_counter()
        slot = self.round_cnt_slot()
        if slot is not None:                       # snapshot this step's counts (enqueued after its forward)
            self.cnt_host[slot].copy_(self.counts, non_blocking=True)
            if self.early and self.gpu_decide:     # usage = counts + decay * usage, on the GPU
                torch.add(self.counts[:, :self.E], self.u_dev, alpha=self.decay, out=self.u_dev)
            self.counts.zero_()
            ev = torch.cuda.Event()
            ev.record(main)
            self.cnt_ev[slot] = ev
            self.cnt_q.append(slot)
            if self.early and self.gpu_decide:
                self._decide_gpu(ev)
        self.t_decide += t1 - t0
        self.t_issue += t2 - t1
        self.n_calls += 1
        now = time.monotonic()
        if now - self.t_log >= self.log_s:
            self._log(now - self.t_log)
            self.t_log = now

    def round_cnt_slot(self):
        used = set(self.cnt_q)
        for k in range(_RING):
            if k not in used:
                return k
        return None

    def _consume(self, slot: int) -> None:
        c = self.cnt_host[slot].numpy()
        E = self.E
        self.usage *= self.decay
        self.usage += c[:, :E]
        st = c[:, E:E + 6].sum(axis=0)             # cold_d, hot_d, cold_r, hot_r, calls, prefills (all layers)
        calls0 = float(c[self.first_layer, E + 4])
        if c[self.first_layer, E + 5] > 0:
            self.since_prefill = 0
        if calls0 > 0:
            self.s_steps += calls0
            self.s_stats += st[:4]
            self.s_layer_calls += st[4]
            b = next(i for i, x in enumerate(_CURVE) if self.since_prefill < x)
            self.curve[b, 0] += st[0]
            self.curve[b, 1] += st[4]
            self.since_prefill += 1

    def _decide(self) -> list[tuple[float, int, int, int]]:
        K = self.K
        u = self.usage
        cand = np.where(~self.is_hot & (u >= self.min_use), u, -np.inf)
        cand[~self.active] = -np.inf
        vict = np.where(self.is_hot & ~self.pinned, u, np.inf)
        if self.shadow_on and self.wb_cost > 0:    # a victim without a shadow costs a write-back: evict it later
            vict = vict + np.where(self.shadow < 0, self.wb_cost, 0.0)
        ci = np.argpartition(-cand, K - 1, axis=1)[:, :K]
        cv = np.take_along_axis(cand, ci, 1)
        o = np.argsort(-cv, axis=1, kind="stable")
        ci, cv = np.take_along_axis(ci, o, 1), np.take_along_axis(cv, o, 1)
        vi = np.argpartition(vict, K - 1, axis=1)[:, :K]
        vv = np.take_along_axis(vict, vi, 1)
        o = np.argsort(vv, axis=1, kind="stable")
        vi, vv = np.take_along_axis(vi, o, 1), np.take_along_axis(vv, o, 1)
        with np.errstate(invalid="ignore"):
            ok = np.isfinite(cv) & np.isfinite(vv) & (cv - vv >= self.gain)
        ok = np.cumprod(ok, axis=1).astype(bool)                       # Strata: pair while the gain holds
        if self.shadow_on:      # a host slot is always found (the victim's shadow, a spare one, or a stolen shadow)
            cap = np.array([len(self.free_v[l]) for l in range(self.L)])
        else:
            cap = np.array([min(len(self.free_v[l]), len(self.spare_h[l])) for l in range(self.L)])
        ok &= np.arange(K)[None, :] < cap[:, None]
        ls, js = np.nonzero(ok)
        if ls.size == 0:
            return []
        gains = (cv - vv)[ls, js]
        top = np.argsort(-gains, kind="stable")[:self.max_swaps]
        self.cap_hits += int(ls.size > self.max_swaps)
        return [(float(gains[i]), int(ls[i]), int(ci[ls[i], js[i]]), int(vi[ls[i], js[i]])) for i in top]

    def _assign(self, pairs, on_h2d=None):
        """Take a free VRAM slot and a host slot per pair and update the host mirrors. The victim's host slot is its
        shadow copy when it has one (``EDGE0_TIER_SHADOW``: written when it was swapped in, untouched since, so no
        write-back), else a spare one (or another hot expert's shadow, stolen) that the D2H fills. ``on_h2d([(l, f,
        c)])`` is called as soon as the incoming experts' slots are known (EARLY issues their copies there, before
        the victims' bookkeeping). Returns the ring slot and [(l, x, v, f, s, h, c, d2h)], or None (round skipped:
        the ring slot's last upload is still queued)."""
        b = self.round % _RING
        ev = self.lst_ev[b]
        if ev is not None and not ev.query():      # that ring slot's list upload is still queued: skip a round
            self.skipped += 1
            return None
        self.round += 1
        pre = [(l, x, v, self.free_v[l].popleft(), int(self.cold_slot[l, x])) for (_g, l, x, v) in pairs]
        if on_h2d is not None:
            on_h2d([(l, f, c) for (l, x, v, f, c) in pre])
        out = []
        fresh = self.fresh
        for (l, x, v, f, c) in pre:
            h = int(self.hot_slot[l, v])
            d2h = True
            if self.shadow_on and self.shadow[l, v] >= 0:
                s, d2h = int(self.shadow[l, v]), False
                self.shadow[l, v] = -1
            elif self.spare_h[l]:
                s = self.spare_h[l].popleft()
            else:                                  # steal the shadow of the most used hot expert (evicted last)
                ok = (self.shadow[l] >= 0) & ~fresh[l]          # not a shadow made this round (its H2D reads it)
                ok[v] = False
                cand = np.flatnonzero(ok)
                if cand.size == 0:                 # cannot happen while pairs per layer <= spare slots; drop the pair
                    self.free_v[l].appendleft(f)   # (its H2D, if issued, only wrote this still-free slot)
                    continue
                y = int(cand[np.argmax(self.usage[l, cand])])
                s = int(self.shadow[l, y])
                self.shadow[l, y] = -1
                self.stolen += 1
            self.is_hot[l, x], self.is_hot[l, v] = True, False
            self.hot_slot[l, x], self.hot_slot[l, v] = f, -1
            self.cold_slot[l, x], self.cold_slot[l, v] = -1, s
            if self.shadow_on:
                self.shadow[l, x] = c              # x's host slot keeps its bytes while x is hot
                fresh[l, x] = True
            out.append((l, x, v, f, s, h, c, d2h))
        for (l, x, v, f, c) in pre:
            fresh[l, x] = False
        if not out:
            return None
        return b, out

    def _fill(self, b, out) -> int:
        """The round's list buffer (map / scale indices and values) in ring slot ``b``'s pinned host buffer."""
        n = len(out)
        P, E, st = self.P, self.E, self.st
        o0, o1, o2, o3, _ = self._lay
        hb = self.lst_host[b]
        map_idx = hb[o0:o1].view(torch.int64).numpy()        # [4P]: hot X, hot V, cold X, cold V (flat [2, L, E])
        g_idx = hb[o1:o2].view(torch.int64).numpy()          # [2P]: X's new hot slot, V's new cold slot (flat g)
        map_val = hb[o2:o3].view(torch.int32).numpy()        # [4P]
        g_val = hb[o3:].view(torch.float32).numpy().reshape(2, 2 * P)
        LE = self.L * E
        for i, (l, x, v, f, s, h, c, _d) in enumerate(out):
            map_idx[i] = l * E + x
            map_idx[n + i] = l * E + v
            map_idx[2 * n + i] = LE + l * E + x
            map_idx[3 * n + i] = LE + l * E + v
            map_val[i], map_val[n + i], map_val[2 * n + i], map_val[3 * n + i] = f, -1, -1, s
            g_idx[i] = st.off_h[l] + f
            g_idx[n + i] = st.Nh + st.off_c[l] + s
            g_val[:, i] = self.g[l, x]
            g_val[:, n + i] = self.g[l, v]
        return n

    def _release(self, out) -> None:
        """After the flip is enqueued: the victims' VRAM slots become free; without shadows the incoming experts'
        host slots become spare (later rounds' copies into them wait for this flip's event)."""
        for (l, x, v, f, s, h, c, d2h) in out:
            self.free_v[l].append(h)
            if not self.shadow_on:
                self.spare_h[l].append(c)
            self.n_d2h += int(d2h)
        self.n_swaps += len(out)
        self.total_swaps += len(out)

    def _issue(self, pairs, main) -> None:
        r = self._assign(pairs)
        if r is None:
            return
        b, out = r
        n = self._fill(b, out)
        P, st = self.P, self.st
        o0, o1, o2, o3, _ = self._lay
        hb = self.lst_host[b]
        db = self.lst_dev[b]
        for strm in (self.h2d, self.d2h):
            if self.flip_ev is not None:            # slots freed by the last flip are reused only after it ran
                strm.wait_event(self.flip_ev)
        with torch.cuda.stream(self.h2d):
            db.copy_(hb, non_blocking=True)
        lev = torch.cuda.Event()
        lev.record(self.h2d)
        self.lst_ev[b] = lev
        for (l, x, v, f, s, h, c, d2h) in out:
            p = self.pools[l]
            self.copier.h2d(p, self.ptrs[l], f, c, self.h2d)
            if d2h:
                self.copier.d2h(p, self.ptrs[l], h, s, self.d2h)
        gi = db[o1:o2].view(torch.int64)
        gv = db[o3:].view(torch.float32).view(2, 2 * P)
        with torch.cuda.stream(self.h2d):
            st.g[0].index_copy_(0, gi[:2 * n], gv[0, :2 * n])
            st.g[1].index_copy_(0, gi[:2 * n], gv[1, :2 * n])
        eh, ed = torch.cuda.Event(), torch.cuda.Event()
        eh.record(self.h2d)
        ed.record(self.d2h)
        main.wait_event(eh)
        main.wait_event(ed)
        mi = db[o0:o1].view(torch.int64)
        mv = db[o2:o3].view(torch.int32)
        st.maps.view(-1).index_copy_(0, mi[:4 * n], mv[:4 * n])
        fe = torch.cuda.Event()
        fe.record(main)
        self.flip_ev = fe
        self._release(out)

    # ------------------------------------------------------------------------------------------ EDGE0_TIER_EARLY
    def pre_sync(self) -> None:
        """Before vLLM's stream sync in the next step's staging: wait for the snapshot of the forward that just ran
        (its LM head, sampler and drafter are still running), fold it in, decide, issue the copies and the flip."""
        if not (self.early and self.started) or torch.cuda.is_current_stream_capturing():
            return
        self.pre_calls += 1
        if not self.cnt_q:
            return
        t0 = time.perf_counter()
        while self.cnt_q:
            slot = self.cnt_q[0]
            self.cnt_ev[slot].synchronize()        # releases the GIL; returns when that forward has finished
            self._consume(self.cnt_q.popleft())
        t1 = time.perf_counter()
        if self.gpu_decide:
            pairs = self._read_gpu_decision()
        else:
            pairs = self._decide() if self.max_swaps > 0 else []
        t2 = time.perf_counter()
        if pairs:
            self._issue_early(pairs)
        t3 = time.perf_counter()
        self.t_wait += t1 - t0
        self.t_decide += t2 - t1
        self.t_issue += t3 - t2

    def _decide_gpu(self, fwd_ev) -> None:
        """``_decide``'s rule on the GPU (side stream, after the forward): top-K non-resident experts with usage >=
        min_use against the K least used non-pinned residents per layer, paired while cand >= victim + gain and
        within the layer's free slots, then the ``max_swaps`` largest gains over all layers. Ties may pick other
        (equal-usage) experts than numpy's stable sorts."""
        ds = self.dec_stream
        ds.wait_event(fwd_ev)
        K, kk = self.K, self.kk
        inf = float("inf")
        with torch.cuda.stream(ds):
            u = self.u_dev
            hot = self.st.maps[0] >= 0
            cand = torch.where(~hot & (u >= self.min_use) & self.active_dev[:, None], u, -inf)
            vict = torch.where(hot & ~self.pinned_dev, u, inf)
            cv, ci = cand.topk(K, dim=1)
            nv, vi = vict.neg().topk(K, dim=1)
            vv = nv.neg()
            ok = torch.isfinite(cv) & torch.isfinite(vv) & ((cv - vv) >= self.gain)
            ok = ok.to(torch.int32).cumprod(dim=1).bool() & (self.ar_k[None, :] < self.cap_dev[:, None])
            g = torch.where(ok, cv - vv, -inf).view(-1)
            gt, gi = g.topk(kk)
            out = self.dec_dev
            out[0, :kk] = gt
            out[1, :kk] = torch.div(gi, K, rounding_mode="floor").float()
            out[2, :kk] = ci.view(-1)[gi].float()
            out[3, :kk] = vi.view(-1)[gi].float()
            out[4, 0] = ok.sum().float()
            self.dec_host.copy_(out, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(ds)
        self.dec_ev = ev

    def _read_gpu_decision(self) -> list[tuple[float, int, int, int]]:
        ev = self.dec_ev
        if ev is None:
            return []
        self.dec_ev = None
        ev.synchronize()
        d = self.dec_host.numpy()
        self.cap_hits += int(d[4, 0] > self.max_swaps)
        g = d[0, :self.kk]
        n = int(np.isfinite(g).sum())              # topk is sorted: the finite gains come first
        return [(float(g[i]), int(d[1, i]), int(d[2, i]), int(d[3, i])) for i in range(n)]

    def post_sync(self) -> None:
        """After the staging's stream sync: the next forward (launched next on this stream) waits for the flip."""
        fe = self.flip_wait
        if fe is None:
            return
        self.flip_wait = None
        main = torch.cuda.current_stream(self.device)
        a, z = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(main)
        main.wait_event(fe)
        z.record(main)
        self.stall_q.append((a, z))
        while self.stall_q and self.stall_q[0][1].query():
            a, z = self.stall_q.popleft()
            ms = a.elapsed_time(z)
            self.n_stall += 1
            self.t_stall += ms * 1e-3
            self.n_stalled += int(ms > 0.02)

    def _issue_early(self, pairs) -> None:
        """Copies + flip of a round decided after its forward finished: nothing reads the free / spare slots and no
        forward runs until ``post_sync``'s wait, so the flip only has to follow both copy streams. The H2D batch is
        issued as soon as the incoming experts' slots are known; the victims' bookkeeping, the write-backs and the
        list buffer follow while it runs."""
        if self.flip_ev is not None:                # slots freed by the last flip are reused only after it ran
            self.d2h.wait_event(self.flip_ev)       # (the h2d stream ran that flip itself)
        torch_mode = self.copier.mode == "torch"

        def h2d(items):
            if torch_mode:
                for (l, f, c) in items:
                    self.copier.h2d(self.pools[l], self.ptrs[l], f, c, self.h2d)
                return
            hd, hs, hn = [], [], []
            for (l, f, c) in items:
                hot, cold = self.ptrs[l]
                for i, nb in enumerate(_ARR_BYTES):
                    hd.append(hot[i] + f * nb)
                    hs.append(cold[i] + c * nb)
                    hn.append(nb)
            self.copier.many(hd, hs, hn, self.h2d)

        r = self._assign(pairs, on_h2d=h2d)
        if r is None:
            return
        b, out = r
        P, st = self.P, self.st
        o0, o1, o2, o3, _ = self._lay
        if torch_mode:
            for (l, x, v, f, s, h, c, d2h) in out:
                if d2h:
                    self.copier.d2h(self.pools[l], self.ptrs[l], h, s, self.d2h)
        else:
            dd, ds, dn = [], [], []
            for (l, x, v, f, s, h, c, d2h) in out:
                if d2h:
                    hot, cold = self.ptrs[l]
                    for i, nb in enumerate(_ARR_BYTES):
                        dd.append(cold[i] + s * nb)
                        ds.append(hot[i] + h * nb)
                        dn.append(nb)
            self.copier.many(dd, ds, dn, self.d2h)
        n = self._fill(b, out)
        hb = self.lst_host[b]
        db = self.lst_dev[b]
        with torch.cuda.stream(self.d2h):          # the list rides the other stream: not queued behind the H2D batch
            db.copy_(hb, non_blocking=True)
        lev = torch.cuda.Event()
        lev.record(self.d2h)
        self.lst_ev[b] = lev
        self.h2d.wait_event(lev)
        gi = db[o1:o2].view(torch.int64)
        gv = db[o3:].view(torch.float32).view(2, 2 * P)
        mi = db[o0:o1].view(torch.int64)
        mv = db[o2:o3].view(torch.int32)
        with torch.cuda.stream(self.h2d):
            st.g[0].index_copy_(0, gi[:2 * n], gv[0, :2 * n])
            st.g[1].index_copy_(0, gi[:2 * n], gv[1, :2 * n])
            st.maps.view(-1).index_copy_(0, mi[:4 * n], mv[:4 * n])
        fe = torch.cuda.Event()
        fe.record(self.h2d)
        self.flip_ev = fe
        self.flip_wait = fe
        self._release(out)

    # ------------------------------------------------------------------------------------------ statistics
    def _reset_stats(self):
        self.s_steps = 0.0
        self.s_layer_calls = 0.0
        self.s_stats = np.zeros(4)
        self.curve = np.zeros((len(_CURVE), 2))
        self.n_swaps = 0
        self.n_calls = 0
        self.n_stale = 0
        self.cap_hits = 0
        self.skipped = 0
        self.t_decide = 0.0
        self.t_issue = 0.0
        self.t_wait = 0.0
        self.n_d2h = 0
        self.n_stall = 0
        self.n_stalled = 0
        self.t_stall = 0.0

    def _log(self, dt: float) -> None:
        if self.n_calls == 0:
            return
        steps = max(self.s_steps, 1.0)
        lc = max(self.s_layer_calls, 1.0)
        cold_d, hot_d, cold_r, hot_r = self.s_stats
        names = ("<4", "<8", "<16", "<32", "<64", "<128", "<256", ">=256")
        curve = " ".join(f"{nm}:{c / max(k, 1):.2f}" for nm, (c, k) in zip(names, self.curve) if k > 0)
        self.log.info(
            "edge0 tier: %.1fs %d steps | per layer-step: cold %.2f hot %.2f distinct experts, routes from VRAM "
            "%.1f%% | swaps %.1f / step (total %d, cap reached %d, stale counts %d, skipped %d) | host %.2f + %.2f "
            "ms / step (decide + issue) | cold / layer-step by steps since prefill: %s",
            dt, int(self.s_steps), cold_d / lc, hot_d / lc, 100.0 * hot_r / max(hot_r + cold_r, 1.0),
            self.n_swaps / max(self.n_calls, 1), self.total_swaps, self.cap_hits, self.n_stale, self.skipped,
            1e3 * self.t_decide / self.n_calls, 1e3 * self.t_issue / self.n_calls, curve)
        if self.early:
            self.log.info(
                "edge0 tier: EARLY: host waited %.2f ms / step for the forward's counts (hidden: the stream sync "
                "waits longer); next forward waited for the flip %.0f us / flip on the GPU (%d of %d flips > 20 us); "
                "copies via %s; write-backs %.1f / step (%d of %d swaps; shadows %s, %d held, %d stolen)",
                1e3 * self.t_wait / self.n_calls, 1e6 * self.t_stall / max(self.n_stall, 1), self.n_stalled,
                self.n_stall, self.copier.mode, self.n_d2h / max(self.n_calls, 1), self.n_d2h, self.n_swaps,
                "on" if self.shadow_on else "off", int((self.shadow >= 0).sum()), self.stolen)
        self._reset_stats()


_EARLY_TIER: "ReactiveTier | None" = None


def install_stage_hook(tier: "ReactiveTier", log=None) -> bool:
    """EDGE0_TIER_EARLY: wrap ``model_state.stage_checkpoint_rows`` (vLLM's, or EDGE0_PLE_DEFER's replacement; both
    end in the stream sync that waits for the previous step) with ``pre_sync`` / ``post_sync``. False when the module
    is missing (then ``on_step`` falls back to deciding at compute_logits)."""
    global _EARLY_TIER
    _EARLY_TIER = tier
    try:
        import vllm.models.qwen4_exp.nvidia.model_state as ms
    except Exception:  # noqa: BLE001
        return False
    inner = ms.stage_checkpoint_rows
    if getattr(inner, "_edge0_tier", False):
        return True

    def stage_checkpoint_rows(*args, **kwargs):
        t = _EARLY_TIER
        if t is not None:
            t.pre_sync()
        out = inner(*args, **kwargs)
        if t is not None:
            t.post_sync()
        return out

    stage_checkpoint_rows._edge0_tier = True
    stage_checkpoint_rows._edge0_defer = getattr(inner, "_edge0_defer", False)
    ms.stage_checkpoint_rows = stage_checkpoint_rows
    if log is not None:
        log.info("edge0 tier: EARLY: staging hook installed around %s", getattr(inner, "__qualname__", inner))
    return True


def _runtime_storage() -> TierStorage:
    from . import runtime
    st = getattr(runtime.current(), "tier_storage", None)
    if st is None:
        raise RuntimeError("edge0 tier: no TierStorage (EDGE0_TIER must be set when the model is built)")
    return st


def check_consistency(tier: ReactiveTier) -> list[str]:
    """Debug / tests: host mirrors vs the device maps (syncs)."""
    errs = []
    maps = tier.st.maps.cpu().numpy()
    for l, p in tier.pools.items():
        hm, cm = maps[0, l], maps[1, l]
        for e in range(tier.E):
            hs, cs = int(tier.hot_slot[l, e]), int(tier.cold_slot[l, e])
            if (hs >= 0) == (cs >= 0):
                errs.append(f"L{l} e{e}: hot {hs} cold {cs}")
            if hm[e] != hs or cm[e] != cs:
                errs.append(f"L{l} e{e}: device {hm[e]}/{cm[e]} host {hs}/{cs}")
        used_h = set(int(x) for x in tier.hot_slot[l] if x >= 0)
        if used_h & set(tier.free_v[l]) or len(used_h) + len(tier.free_v[l]) != p.n_hot + p.n_free:
            errs.append(f"L{l}: hot slots {len(used_h)} + free {len(tier.free_v[l])} != {p.n_hot + p.n_free}")
        used_c = set(int(x) for x in tier.cold_slot[l] if x >= 0)
        sh = [int(x) for x in tier.shadow[l] if x >= 0]
        if any(tier.shadow[l, e] >= 0 and not tier.is_hot[l, e] for e in range(tier.E)):
            errs.append(f"L{l}: a cold expert holds a shadow slot")
        if (used_c & set(tier.spare_h[l]) or used_c & set(sh) or set(sh) & set(tier.spare_h[l]) or len(set(sh)) != len(sh)
                or len(used_c) + len(tier.spare_h[l]) + len(sh) != p.n_cold + p.n_spare):
            errs.append(f"L{l}: cold slots {len(used_c)} + spare {len(tier.spare_h[l])} + shadows {len(sh)} != "
                        f"{p.n_cold + p.n_spare}")
    return errs


