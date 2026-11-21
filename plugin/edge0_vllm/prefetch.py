"""Cross-layer expert prefetch (``EDGE0_PREFETCH=S``): overlap the next layer's cold-expert transfers with the current
layer's compute (TidalMoE EXPERT-CACHE-ANALYSIS design C, rebuilt for PCIe and the Edge0 pools).

Each layer's hot tier gets ``S`` extra VRAM *staging* slots after its ``n_hot`` static slots. At layer l-1 (in
``Edge0LatentMoeBlock.forward``, before the D projection) layer l's router is applied to layer l-1's MoE input
(pre-gating). Per expert the largest predicted probability over the step's tokens is taken, hot experts are masked
out, and the top ``S`` cold experts with probability >= ``EDGE0_PREFETCH_MIN_P`` are chosen. Then, all on the GPU:

* ``stage_map[l]`` (logical -> staging slot or -1), and the effective expert maps of layer l's two Marlin calls:
  ``hot_eff = where(staged, n_hot + i, hot_map)``, ``cold_eff = where(staged, -1, cold_map)``;
* an SM gather kernel on a dedicated low-priority stream copies the chosen experts' Marlin rows (w13, w2, s13, s2,
  1.85 MB each) from the pinned host pool into the staging slots, plus their global scales.

Layer l's op waits for that stream before its hot call; its cold call (side stream) reads only the experts that
were not staged. Wrong predictions only cost PCIe time; misses are read zero-copy exactly as before, so results do
not depend on the predictor. The transfer runs while layer l-1's MoE tail and layer l's attention keep the GPU busy
and the PCIe link would otherwise idle (at concurrency 1 about half of each layer).

Capture rules: breakable CUDA graphs end a capture segment at every attention op in PIECEWISE mode, so a fork that
is joined one layer later is only legal in FULL graphs (uniform decode) and eagerly; in PIECEWISE steps nothing is
prefetched and the previous staging stays valid (its slots still hold the experts its map names). Adaptive swaps
reset the staging (``ExpertPool.reset_stage``).
"""
from __future__ import annotations

import torch

from .config import S2_BYTES, S13_BYTES, W2_BYTES, W13_BYTES

_WORDS = (W13_BYTES // 4, W2_BYTES // 4, S13_BYTES // 4, S2_BYTES // 4)      # int32 words per slot and tensor
_BLOCK = 512                                                                    # words per task (2 KB)
_CHUNKS = tuple(w // _BLOCK for w in _WORDS)                                    # 560, 280, 70, 35
assert all(w % _BLOCK == 0 for w in _WORDS)

try:
    from vllm.triton_utils import tl, triton
    _HAS_TRITON = True
except Exception:  # noqa: BLE001
    _HAS_TRITON = False

if _HAS_TRITON:
    @triton.jit
    def _stage_copy_kernel(s0, s1, s2, s3, d0, d1, d2, d3, src_slot, dst_slot, valid, n_tasks,
                           W0: tl.constexpr, W1: tl.constexpr, W2: tl.constexpr, W3: tl.constexpr,
                           C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr, C3: tl.constexpr,
                           BLOCK: tl.constexpr):
        """Grid-stride over (staged expert i, 2 KB chunk c) tasks; chunk c walks w13, w2, s13, s2 rows in turn."""
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        per = C0 + C1 + C2 + C3
        offs = tl.arange(0, BLOCK)
        for task in range(pid, n_tasks, nprog):
            i = task // per
            c = task % per
            ok = tl.load(valid + i) != 0
            if ok:
                src = tl.load(src_slot + i).to(tl.int64)
                dst = tl.load(dst_slot + i).to(tl.int64)
                if c < C0:
                    v = tl.load(s0 + src * W0 + c * BLOCK + offs)
                    tl.store(d0 + dst * W0 + c * BLOCK + offs, v)
                elif c < C0 + C1:
                    cc = c - C0
                    v = tl.load(s1 + src * W1 + cc * BLOCK + offs)
                    tl.store(d1 + dst * W1 + cc * BLOCK + offs, v)
                elif c < C0 + C1 + C2:
                    cc = c - C0 - C1
                    v = tl.load(s2 + src * W2 + cc * BLOCK + offs)
                    tl.store(d2 + dst * W2 + cc * BLOCK + offs, v)
                else:
                    cc = c - C0 - C1 - C2
                    v = tl.load(s3 + src * W3 + cc * BLOCK + offs)
                    tl.store(d3 + dst * W3 + cc * BLOCK + offs, v)


def _words(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int32) if t.dtype == torch.int32 else t.view(torch.uint8).view(torch.int32)


def stage_copy(pool, src_slot: torch.Tensor, dst_slot: torch.Tensor, valid: torch.Tensor, ctas: int) -> None:
    """Copy cold slots ``src_slot[i]`` into hot-tier slots ``dst_slot[i]`` where ``valid[i]`` (current stream)."""
    hot, cold = pool.hot, pool.cold
    n = src_slot.numel()
    if _HAS_TRITON and src_slot.is_cuda:
        _stage_copy_kernel[(ctas,)](
            _words(cold.w13), _words(cold.w2), _words(cold.s13), _words(cold.s2),
            _words(hot.w13), _words(hot.w2), _words(hot.s13), _words(hot.s2),
            src_slot, dst_slot, valid, n * sum(_CHUNKS),
            W0=_WORDS[0], W1=_WORDS[1], W2=_WORDS[2], W3=_WORDS[3],
            C0=_CHUNKS[0], C1=_CHUNKS[1], C2=_CHUNKS[2], C3=_CHUNKS[3], BLOCK=_BLOCK, num_warps=4)
    else:
        for i in range(n):
            if bool(valid[i]):
                s, d = int(src_slot[i]), int(dst_slot[i])
                for a, b in zip((hot.w13, hot.w2, hot.s13, hot.s2), (cold.w13, cold.w2, cold.s13, cold.s2)):
                    a[d].copy_(b[s])
    # global scales live in VRAM for both tiers: masked gather/scatter
    s = src_slot.long().clamp(0, max(cold.n - 1, 0))
    d = dst_slot.long()                                    # distinct staging slots: no duplicate writes
    for gh, gc in ((hot.g13, cold.g13), (hot.g2, cold.g2)):
        gh.index_copy_(0, d, torch.where(valid, gc[s], gh[d]))


class Prefetcher:
    """Process-wide prefetch state: one low-priority stream, the step's per-layer predictions."""

    def __init__(self, blocks, slots: int, min_p: float, ctas: int, device, log, keep: bool = True,
                 max_tokens: int = 64):
        self.by_layer = {b.layer_idx: b for b in blocks}
        self.S = slots
        self.min_p = min_p
        self.ctas = ctas
        self.keep = keep
        self.max_tokens = max_tokens
        self.xbuf = None
        self.step = torch.zeros((), dtype=torch.float32, device=device)
        lo, _hi = torch.cuda.Stream.priority_range() if torch.cuda.is_available() else (0, 0)
        self.stream = torch.cuda.Stream(device=device, priority=lo) if torch.cuda.is_available() else None
        self.issued = torch.zeros(len(self.by_layer) + 1, dtype=torch.float32, device=device)   # staged experts
        log.info("edge0 prefetch: %d staging slots per layer (%.2f GiB VRAM), min p %.3f, copy kernel %d CTAs, %s",
                 slots, slots * len(blocks) * 1_935_360 / 2**30, min_p, ctas,
                 "sticky (resident experts keep serving hits)" if keep else "fresh per step")

    @staticmethod
    def _mode_allows() -> bool:
        try:
            from vllm.config import CUDAGraphMode
            from vllm.forward_context import get_forward_context, is_forward_context_available
            if not is_forward_context_available():
                return True
            mode = get_forward_context().cudagraph_runtime_mode
            return mode in (CUDAGraphMode.FULL, CUDAGraphMode.NONE)
        except Exception:  # noqa: BLE001
            return True

    def predict_and_copy(self, layer: int, x: torch.Tensor) -> None:
        """At layer ``layer`` (= l-1): stage layer l's predicted cold experts.

        Only one small copy runs on the main stream (x into a persistent buffer: the main stream may reuse x's memory
        before the prefetch stream is joined). The router GEMV, top-k, the slot bookkeeping, the map updates and the
        copies all run on the low-priority prefetch stream, off the critical path (measured: ~20 small kernels per
        layer cost +2.8 ms per token at c1 when they ran on the main stream). Decode-sized calls only."""
        nxt = self.by_layer.get(layer + 1)
        t = x.shape[0]
        if nxt is None or nxt.pool.n_stage == 0 or t == 0 or t > self.max_tokens or not self._mode_allows():
            return
        pool = nxt.pool
        if self.xbuf is None or self.xbuf.dtype != x.dtype:
            self.xbuf = torch.empty((self.max_tokens, x.shape[1]), dtype=x.dtype, device=x.device)
        xb = self.xbuf[:t]
        xb.copy_(x)
        main = torch.cuda.current_stream()
        self.stream.wait_stream(main)
        with torch.cuda.stream(self.stream):
            lg = nxt.gate(xb)
            if isinstance(lg, tuple):
                lg = lg[0]
            p = torch.softmax(lg.float(), dim=-1).amax(dim=0)                     # [E]
            p = torch.where(pool.hot_map >= 0, torch.full_like(p, -1.0), p)
            sc, cand = torch.topk(p, self.S)                                      # [S]
            want = sc >= self.min_p
            if self.keep:
                cand, dst, valid = self._sticky(pool, cand, want)
            else:
                dst, valid = self._fresh(pool, cand, want)
            staged = pool.stage_map >= 0
            torch.where(staged, pool.stage_map, pool.hot_map, out=pool.hot_eff)
            torch.where(staged, torch.full_like(pool.cold_map, -1), pool.cold_map, out=pool.cold_eff)
            src = pool.cold_map[cand.long()]
            valid = valid & (src >= 0)
            self.issued[layer + 1] += valid.sum()
            pool.pf_src.copy_(src)
            pool.pf_dst.copy_(dst)
            pool.pf_valid.copy_(valid)
            stage_copy(pool, pool.pf_src, pool.pf_dst, pool.pf_valid, self.ctas)
        pool.pf_pending = self.stream          # joined by latent_moe._join_prefetch before layer l uses the maps

    def _fresh(self, pool, cand, want):
        """keep=0: the staging holds exactly this step's predictions (all copied)."""
        e = pool.num_experts
        dst = torch.arange(pool.n_hot, pool.n_hot + self.S, dtype=torch.int32, device=cand.device)
        pool.stage_map_ext.fill_(-1)
        pool.stage_map_ext.index_copy_(0, torch.where(want, cand, torch.full_like(cand, e)), dst)
        return dst, want

    def _sticky(self, pool, cand, want):
        """keep=1 (default): staged experts stay resident and keep serving hits until a newly predicted expert needs
        their slot; slots of experts predicted again are kept, the others are reused oldest first. Only predicted
        experts that are not resident are copied. Fixed-size tensor ops only (CUDA-graph safe)."""
        e, s, n_hot = pool.num_experts, self.S, pool.n_hot
        dev = cand.device
        self.step += 1
        cand_slot = pool.stage_map[cand]                                     # absolute slot or -1
        resident = want & (cand_slot >= 0)
        new = want & (cand_slot < 0)
        keep = torch.zeros(s + 1, dtype=torch.bool, device=dev)
        keep.index_fill_(0, torch.where(resident, (cand_slot - n_hot).long(), torch.full_like(cand, s)), True)
        keep = keep[:s]
        key = torch.where(keep, torch.full_like(pool.stage_time, float("inf")), pool.stage_time)
        free_order = torch.argsort(key, stable=True)                          # free slots, oldest first
        new_order = torch.argsort((~new).to(torch.int8), stable=True)        # new candidates first
        cand_new = cand[new_order]
        is_new = new[new_order]
        slot = free_order                                                     # j-th new candidate -> j-th free slot
        dummy_e = torch.full_like(cand_new, e)
        old = pool.stage_owner[slot]
        pool.stage_map_ext.index_fill_(0, torch.where(is_new & (old >= 0), old, dummy_e), -1)      # evict
        pool.stage_map_ext.index_copy_(0, torch.where(is_new, cand_new, dummy_e),
                                       (slot + n_hot).to(torch.int32))                             # admit
        dummy_s = torch.full_like(slot, s)
        pool.stage_owner_ext.index_copy_(0, torch.where(is_new, slot, dummy_s), cand_new)
        ar = torch.arange(s, device=dev)
        now = self.step.expand(s).contiguous()           # tensor-valued index_fill_ syncs (.item()): copy instead
        pool.stage_time_ext.index_copy_(0, torch.where(keep, ar, dummy_s), now)                    # predicted again
        pool.stage_time_ext.index_copy_(0, torch.where(is_new, slot, dummy_s), now)                # newly staged
        return cand_new, (slot + n_hot).to(torch.int32), is_new
