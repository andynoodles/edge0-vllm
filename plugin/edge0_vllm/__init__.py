"""edge0-vllm: serve Edge0-145B-NVFP4 with vLLM (``nvcr.io/nvidia/vllm:26.11-py3``, vLLM 0.33).

Loaded as a vLLM general plugin (entry point group ``vllm.general_plugins``, ``edge0 = edge0_vllm:register``);
``register()`` runs once in every vLLM process (API server, engine core, workers, registry inspection
subprocess). It must not initialize CUDA. It:

1. registers ``Edge0ForCausalLM`` lazily (``edge0_vllm.model:Edge0ForCausalLM``);
2. registers the ``edge0`` quantization config (``edge0_vllm.quant``);
3. adds ``Edge0ForCausalLM`` wherever vLLM keys Qwen4Exp behaviour on the architecture name:
   ``MODELS_CONFIG_MAP`` (Qwen4Exp hybrid-cache contract + config checks), the default breakable-CUDA-graph set,
   and the Engram n-gram layer table (``--engram-config`` needs ``model_has_engram_layers``).

See edge0/docs/DESIGN.md (contract) and edge0/docs/PLUGIN.md (what was built and measured).
"""
from __future__ import annotations

ARCH = "Edge0ForCausalLM"
_REGISTERED = False


def _edge0_config_class(base_cls):
    """Qwen4Exp's config check (``Qwen4ExpForConditionalGenerationConfig.verify_and_update_config``) accepts only
    the MTP and n-gram proposers, because stock Qwen4Exp cannot export auxiliary hidden states. Edge0ForCausalLM
    can (the DFlash2 drafter's completed-layer features, ``model.py``), so ``method="dflash"`` is passed through:
    that check is the only place the Qwen4Exp verification reads the speculative config."""

    class Edge0ForCausalLMConfig(base_cls):
        @staticmethod
        def verify_and_update_config(vllm_config) -> None:
            sc = vllm_config.speculative_config
            if sc is None or sc.method != "dflash":
                return base_cls.verify_and_update_config(vllm_config)
            vllm_config.speculative_config = None
            try:
                base_cls.verify_and_update_config(vllm_config)
            finally:
                vllm_config.speculative_config = sc

    return Edge0ForCausalLMConfig


def _patch_arch_tables(log) -> None:
    import vllm.config.engram as engram
    import vllm.config.vllm as vcfg
    from vllm.model_executor.models import config as mcfg

    base = "Qwen4ExpForCausalLM"
    if ARCH not in mcfg.MODELS_CONFIG_MAP:
        mcfg.MODELS_CONFIG_MAP[ARCH] = _edge0_config_class(mcfg.MODELS_CONFIG_MAP[base])
    if ARCH not in vcfg.DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES and \
            base in vcfg.DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES:
        vcfg.DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES = frozenset(
            vcfg.DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES | {ARCH})
        cached = getattr(vcfg, "default_breakable_cudagraph_architectures", None)
        if cached is not None and hasattr(cached, "cache_clear"):
            cached.cache_clear()
    if ARCH not in engram._NGRAM_LAYER_FIELDS:
        engram._NGRAM_LAYER_FIELDS[ARCH] = engram._NGRAM_LAYER_FIELDS[base]


def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from vllm.logger import init_logger
    from vllm.model_executor.models import ModelRegistry

    log = init_logger("vllm.edge0")
    ModelRegistry.register_model(ARCH, "edge0_vllm.model:Edge0ForCausalLM")
    from . import quant  # noqa: F401  (registers --quantization edge0)
    _patch_arch_tables(log)
    _REGISTERED = True
    log.debug("edge0: registered %s and quantization 'edge0'", ARCH)
