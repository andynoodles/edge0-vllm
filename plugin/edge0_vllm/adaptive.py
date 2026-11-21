"""Adaptive expert cache (``EDGE0_ADAPT=1``): TidalMoE's ``_AdaptiveCache`` (``plugin/moe_placement/__init__.py``
``_AdaptiveCache``) ported to the Edge0 hot/cold pools.

Counting (inside the op, CUDA-graph safe, no host sync): per layer a float32 [512] counter gets +1 for every
distinct expert a decode call touches (TidalMoE ``count=touch``). Calls with more than ``EDGE0_ADAPT_MAX_TOKENS``
rows (prefill chunks; static per captured graph size) are not counted, so the hot set follows decode traffic
(TidalMoE EXPERT-CACHE-ANALYSIS #3). Padding rows of graph-padded batches are masked.

Epochs (engine thread, from ``compute_logits`` once per step): every ``EDGE0_ADAPT_EPOCH_S`` seconds with at least
``EDGE0_ADAPT_MIN_STEPS`` counted steps, the counters are read and zeroed and folded into an exponential moving
score (half-life ``EDGE0_ADAPT_HALFLIFE`` epochs, raw counts). Per layer the hottest cold experts are paired with
the coldest hot ones; a pair swaps when ``x > y (1 + margin)`` and the Poisson significance test
``x - y > z sqrt(x + y + 1)`` passes (TidalMoE policy "z"). At most ``EDGE0_ADAPT_SWAPS`` pairs per epoch, largest
gain first, executed with ``ExpertPool.swap_many`` on the current stream: in place, addresses unchanged, so the
captured CUDA graphs stay valid. Per-layer budgets (n_hot) never change.

``EDGE0_ADAPT_SAVE=<path>`` writes the current placement as an ``edge0-placement-v1`` plan every
``EDGE0_ADAPT_SAVE_EVERY`` epochs; ``EDGE0_PLACEMENT=plan:<path>`` resumes from it.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from .kernels import touch_count
from .placement import LayerPlacement, save_plan
from .trace import _padding_mask


class AdaptiveCache:
    def __init__(self, blocks, device, log, env=None, max_tokens_default: int = 64):
        env = os.environ if env is None else env
        self.log = log
        self.blocks = sorted(blocks, key=lambda b: b.layer_idx)
        self.pools = [b.pool for b in self.blocks]
        self.L = len(self.pools)
        self.E = self.pools[0].num_experts if self.pools else 512
        self.epoch_s = float(env.get("EDGE0_ADAPT_EPOCH_S", "3") or 3)
        self.max_swaps = int(env.get("EDGE0_ADAPT_SWAPS", "64") or 64)
        self.half_life = float(env.get("EDGE0_ADAPT_HALFLIFE", "10") or 10)
        self.margin = float(env.get("EDGE0_ADAPT_MARGIN", "0.25") or 0.25)
        self.z = float(env.get("EDGE0_ADAPT_Z", "3") or 3)
        self.min_steps = int(env.get("EDGE0_ADAPT_MIN_STEPS", "30") or 30)
        self.max_tokens = int(env.get("EDGE0_ADAPT_MAX_TOKENS", str(max_tokens_default)) or max_tokens_default)
        self.save_path = env.get("EDGE0_ADAPT_SAVE", "")
        self.save_every = int(env.get("EDGE0_ADAPT_SAVE_EVERY", "100") or 100)
        self.counts = torch.zeros((self.L, self.E), dtype=torch.float32, device=device)
        self.counted = torch.zeros(1, dtype=torch.float32, device=device)     # counted calls of layer 0
        self.layer_row = {b.layer_idx: i for i, b in enumerate(self.blocks)}
        self.score = None
        self.started = False
        self.t_last = time.monotonic()
        self.epoch = 0
        self.total_swaps = 0
        self.saved_swaps = 0
        log.info("edge0 adapt: adaptive cache on %d layers: epoch %.1fs, <= %d swaps/epoch, half-life %.0f epochs, "
                 "margin %.2f, z %.1f, counting decode calls <= %d tokens, min %d steps/epoch%s", self.L,
                 self.epoch_s, self.max_swaps, self.half_life, self.margin, self.z, self.max_tokens, self.min_steps,
                 f", saving to {self.save_path}" if self.save_path else "")

    # ------------------------------------------------------------------------------------------ in the op
    def record(self, layer: int, ids: torch.Tensor) -> None:
        t = ids.shape[0]
        if t == 0 or t > self.max_tokens:          # static per captured size -> graph-safe branch
            return
        pad = _padding_mask(t)
        if pad is not None:
            ids = torch.where(pad.view(-1, 1), torch.full_like(ids, -1), ids)
        row = self.layer_row[layer]
        touch_count(ids, self.counts[row], touch=True)
        if row == 0:
            self.counted += 1.0

    # ------------------------------------------------------------------------------------------ engine thread
    def on_step(self) -> None:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        now = time.monotonic()
        if not self.started:                       # drop counts from profiling / warm-up / graph capture
            self.counts.zero_()
            self.counted.zero_()
            self.started, self.t_last, self._t_epoch, self._idle = True, now, now, 0
            return
        if now - self.t_last >= self.epoch_s:
            self._epoch(now)

    def _cold_mask(self) -> np.ndarray:
        m = np.ones((self.L, self.E), dtype=bool)
        for i, p in enumerate(self.pools):
            m[i, p.order[:p.n_hot]] = False
        return m

    def _epoch(self, now: float) -> None:
        t0 = time.monotonic()
        steps = int(self.counted.item())                       # syncs the stream once per epoch
        if steps < self.min_steps:
            self.t_last = now                      # keep accumulating; look again one epoch later
            self._idle += 1
            return
        c = self.counts.cpu().numpy().astype(np.float64)
        self.counts.zero_()
        self.counted.zero_()
        dt = now - self._t_epoch
        self.t_last = self._t_epoch = now
        self.epoch += 1
        beta = 0.5 ** (1.0 / self.half_life)
        self.score = c if self.score is None else beta * self.score + c
        cold = self._cold_mask()
        tot = c.sum()
        cold_share = float((c * cold).sum() / max(tot, 1e-9))
        pairs = []
        for li, p in enumerate(self.pools):
            if p.n_hot == 0 or p.n_cold == 0:
                continue
            sc = self.score[li]
            cs = np.where(cold[li])[0]
            hs = np.where(~cold[li])[0]
            cs = cs[np.argsort(-sc[cs], kind="stable")]
            hs = hs[np.argsort(sc[hs], kind="stable")]
            for e_cold, e_hot in zip(cs, hs):
                x, y = sc[e_cold], sc[e_hot]
                if x > y * (1 + self.margin) and (x - y) > self.z * np.sqrt(x + y + 1.0):
                    pairs.append((x - y, li, int(e_hot), int(e_cold)))
                else:
                    break
        pairs.sort(reverse=True)
        todo = pairs[:self.max_swaps]
        by_layer: dict[int, list] = {}
        for _g, li, e_hot, e_cold in todo:
            by_layer.setdefault(li, []).append((e_hot, e_cold))
        for li, prs in by_layer.items():
            p = self.pools[li]
            slot = {e: s for s, e in enumerate(p.order)}
            p.swap_many([slot[eh] for eh, _ in prs], [slot[ec] - p.n_hot for _, ec in prs])
        self.total_swaps += len(todo)
        cold = self._cold_mask()
        pred = float((self.score * cold).sum() / max(self.score.sum(), 1e-9))
        if todo or self.epoch <= 10 or self.epoch % 20 == 0:
            self.log.info("edge0 adapt: epoch %d (%.1fs, %d counted steps): cold share of decode touches %.1f%%, "
                          "%d candidates, %d swaps (total %d) in %.1f ms; predicted cold share now %.1f%%",
                          self.epoch, dt, steps, cold_share * 100, len(pairs), len(todo), self.total_swaps,
                          (time.monotonic() - t0) * 1e3, pred * 100)
        if self.save_path and self.epoch % self.save_every == 0 and self.total_swaps != self.saved_swaps:
            self.save()

    def save(self) -> None:
        try:
            d = os.path.dirname(self.save_path) or "."
            os.makedirs(d, exist_ok=True)
            tmp = f"{self.save_path}.tmp{os.getpid()}"
            plans = {b.layer_idx: LayerPlacement(order=list(b.pool.order), n_hot=b.pool.n_hot) for b in self.blocks}
            save_plan(tmp, plans, f"adaptive epoch {self.epoch}, {self.total_swaps} swaps")
            os.replace(tmp, self.save_path)
            self.saved_swaps = self.total_swaps
            self.log.info("edge0 adapt: saved placement to %s (epoch %d)", self.save_path, self.epoch)
        except OSError as e:
            self.log.warning("edge0 adapt: cannot save placement to %s: %s", self.save_path, e)
