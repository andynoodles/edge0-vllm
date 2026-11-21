"""MTP drafter on the 32 GB card: build it without its own copies of the vocabulary matrices.

vLLM's ``Qwen4ExpMTP`` (``vllm.models.qwen4_exp.nvidia.mtp``) constructs its own ``embed_tokens``
(``VocabParallelEmbedding``) and ``lm_head`` (``ParallelLMHead``), 248,320 x 2,560 bf16 = 1.18 GiB each, loads the
target's ``model.embed_tokens`` / ``lm_head`` checkpoint tensors into them, runs its quant post-processing, and only
then (``spec_decode/eagle/utils.py`` ``load_eagle_model``) replaces both with the target's modules. On this card the
target leaves ~8 GiB free, and the 2.36 GiB of throw-away copies plus the Marlin repack of the 512 MTP experts ran
out of memory (TUNING-LOG.md section 8.1).

``install()`` (called by ``Edge0ForCausalLM.__init__`` when ``speculative_config.method == "mtp"``) patches the
module so that:

1. while ``Qwen4ExpMTP`` is being built, its ``VocabParallelEmbedding`` / ``ParallelLMHead`` are parameter-less
   placeholders;
2. ``Qwen4ExpMTP.load_weights`` first plugs in the target's own ``embed_tokens`` and ``lm_head`` modules (the same
   objects ``load_eagle_model`` would share afterwards; ``embed_tokens`` lives in pinned host memory with
   ``EDGE0_EMBED_HOST=1``), skips the checkpoint's copies of them (already loaded into the target), and reports
   their names as loaded.

Numerics are unchanged: the drafter ends up with exactly the modules vLLM would have shared.
"""
from __future__ import annotations

import functools
import weakref

import torch.nn as nn

_TARGET: "weakref.ReferenceType | None" = None
_INSTALLED = False


class _VocabPlaceholder(nn.Module):
    """Stands in for the drafter's own vocab matrix until ``load_weights`` plugs in the target's module."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, *args, **kwargs):  # pragma: no cover - replaced before use
        raise RuntimeError("edge0: MTP vocab placeholder used before the target's module was shared")


def set_target(model) -> None:
    global _TARGET
    _TARGET = weakref.ref(model)


def install(log) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    import vllm.models.qwen4_exp.nvidia.mtp as mtp

    cls = mtp.Qwen4ExpMTP
    orig_init = cls.__init__
    orig_load = cls.load_weights

    # functools.wraps keeps the signature visible to inspect.signature (vLLM's initialize_model picks the
    # new-style "vllm_config, prefix" call from it).
    @functools.wraps(orig_init)
    def __init__(self, *, vllm_config, prefix: str = ""):
        saved = (mtp.VocabParallelEmbedding, mtp.ParallelLMHead)
        mtp.VocabParallelEmbedding = _VocabPlaceholder
        mtp.ParallelLMHead = _VocabPlaceholder
        try:
            orig_init(self, vllm_config=vllm_config, prefix=prefix)
        finally:
            mtp.VocabParallelEmbedding, mtp.ParallelLMHead = saved

    @functools.wraps(orig_load)
    def load_weights(self, weights):
        target = _TARGET() if _TARGET is not None else None
        if target is None:
            raise RuntimeError("edge0: MTP drafter built without an Edge0 target model")
        self.model.embed_tokens = target.model.embed_tokens
        self.lm_head = target.lm_head
        skipped: set[str] = set()

        def keep():
            for name, w in weights:
                new = mtp._remap_mtp_weight_name(name)
                if new is not None and (new.startswith("model.embed_tokens.") or new.startswith("lm_head.")):
                    skipped.add(new)
                    continue
                yield name, w

        loaded = set(orig_load(self, keep()) or ())
        shared = {n for n, _ in self.named_parameters()
                  if n.startswith("model.embed_tokens.") or n.startswith("lm_head.")}
        log.info("edge0: MTP drafter shares the target's embed_tokens and lm_head from construction (skipped %d "
                 "checkpoint tensors; saves 2.36 GiB of transient VRAM)", len(skipped))
        return loaded | shared

    cls.__init__ = __init__
    cls.load_weights = load_weights
    _INSTALLED = True


_DFLASH_INSTALLED = False


def install_dflash(log) -> None:
    """``EDGE0_DFLASH_SHARE_VOCAB=1``: the DFlash2 drafter (``qwen3_dflash.DFlashQwen3ForCausalLM`` and its
    ``DFlashQwen3Model``) builds its own BF16 ``embed_tokens`` and ``lm_head`` (248,320 x 2,560 = 1.18 GiB each) and
    ``spec_decode/dflash/utils.py`` ``load_dflash_model`` replaces both with the target's right after loading (the
    Qwen3.8 DFlash2 checkpoint ships neither and sets no ``has_own_*`` flag). Those 2.36 GiB are a load transient that
    the CUDA caching allocator keeps, and at 17.0 GiB of hot experts the constructor ran out of memory (TUNING-LOG
    "Single-stream" R2 item 4). Here they are parameter-less placeholders while the drafter is built when nothing will
    be loaded into them (no ``has_own_embed_tokens`` / ``has_own_lm_head``); vLLM's sharing then installs the target's
    modules exactly as before, so the drafter runs with the same modules and numerics."""
    global _DFLASH_INSTALLED
    if _DFLASH_INSTALLED:
        return
    import vllm.model_executor.models.qwen3_dflash as qd

    cls = qd.DFlashQwen3ForCausalLM
    orig_init = cls.__init__

    @functools.wraps(orig_init)
    def __init__(self, *, vllm_config, prefix: str = ""):
        cfg = vllm_config.speculative_config.draft_model_config.hf_config
        own = getattr(cfg, "has_own_embed_tokens", False) or getattr(cfg, "has_own_lm_head", False)
        if own:
            return orig_init(self, vllm_config=vllm_config, prefix=prefix)
        saved = (qd.VocabParallelEmbedding, qd.ParallelLMHead)
        qd.VocabParallelEmbedding = _VocabPlaceholder
        qd.ParallelLMHead = _VocabPlaceholder
        try:
            orig_init(self, vllm_config=vllm_config, prefix=prefix)
        finally:
            qd.VocabParallelEmbedding, qd.ParallelLMHead = saved
        log.info("edge0: DFlash2 drafter built without its own embed_tokens / lm_head (the target's are shared after "
                 "loading; saves 2.36 GiB of transient VRAM)")

    cls.__init__ = __init__
    _DFLASH_INSTALLED = True
