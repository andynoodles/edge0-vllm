"""``--quantization edge0``: ModelOpt MIXED_PRECISION semantics plus the Q8_0 -> BF16 load path (DESIGN.md 2.4).

config.json carries ``quantization_config = {"quant_method": "edge0", "dense_format": "ggml_q8_0",
"quantized_layers": {<module>: {"quant_algo": "Q8_0", ...}, "mtp.layers.0.mlp.experts": {"quant_algo":
"W4A16_NVFP4", ...}}}`` and **no** top-level ``quant_algo``. Per prefix:

* Q8_0 linears, ``EDGE0_DENSE=bf16``: a stock ``UnquantizedLinearMethod`` whose ``create_weights`` wraps the
  parameter's weight loader (``q8.wrap_weight_loader``). It is an *instance* of the exact stock class on purpose:
  ``enable_qwen4_exp_low_latency_gemm`` only converts layers whose method ``type(...) is UnquantizedLinearMethod``
  to the sm_120 skinny-GEMM plans, and the wrapper lives on the parameter, not on the method.
* Q8_0 linears, ``EDGE0_DENSE=q8``: the parent's ``ModelOptBlockQuantLinearMethod("q8_0")`` (b12x, sm_12x).
* ``W4A16_NVFP4`` routed experts (the MTP layer's stock FusedMoE): the parent's ``ModelOptNvFp4FusedMoE`` -> Marlin.
* Everything else (HC, router, GDN a/b, PLE kv, QSA qkv + indexer, embedding, LM head): unquantized BF16.

The main model's routed experts never consult the quant config: ``Edge0LatentMoeBlock`` owns their storage.
"""
from __future__ import annotations

from typing import Any

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.modelopt import ModelOptMixedPrecisionConfig

from . import q8
from .config import get_settings

logger = init_logger("vllm.edge0.quant")

QUANT_NAME = "edge0"


def make_q8_dequant_linear_method() -> UnquantizedLinearMethod:
    """A stock UnquantizedLinearMethod whose weights accept Q8_0 blocks (dequantized while loading)."""
    method = UnquantizedLinearMethod()
    base_create = method.create_weights

    def create_weights(layer, input_size_per_partition, output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        extra_weight_attrs["weight_loader"] = q8.wrap_weight_loader(extra_weight_attrs["weight_loader"])
        return base_create(layer, input_size_per_partition, output_partition_sizes, input_size, output_size,
                           params_dtype, **extra_weight_attrs)

    method.create_weights = create_weights
    method.edge0_q8_dequant = True
    return method


@register_quantization_config(QUANT_NAME)
class Edge0QuantConfig(ModelOptMixedPrecisionConfig):
    """ModelOpt MIXED_PRECISION with the ``edge0`` name and the Q8_0 -> BF16 dequant-on-load path."""

    def get_name(self):  # type: ignore[override]
        return QUANT_NAME

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if isinstance(hf_quant_cfg, dict) and str(hf_quant_cfg.get("quant_method", "")).lower() == QUANT_NAME:
            return QUANT_NAME
        return None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Edge0QuantConfig":
        if "quant_algo" in config:
            # ModelOpt's own override would treat the checkpoint as modelopt_* (DESIGN.md 1.2 rules).
            raise ValueError("edge0 quantization_config must not carry a top-level 'quant_algo'")
        if not config.get("quantized_layers"):
            raise ValueError("edge0 quantization_config needs a non-empty 'quantized_layers' map")
        bad = {k: v.get("quant_algo") for k, v in config["quantized_layers"].items()
               if str(v.get("quant_algo", "")).upper() not in ("Q8_0", "W4A16_NVFP4")}
        if bad:
            raise ValueError(f"edge0 quantized_layers: unsupported quant_algo {bad}")
        return cls._from_config(quant_method="MIXED_PRECISION", kv_cache_quant_method=None, exclude_modules=[],
                                original_config=config, group_size=16)

    def get_quant_method(self, layer, prefix: str):
        if isinstance(layer, LinearBase) and (get_settings().dense == "bf16" or _mtp_dense_bf16(prefix)):
            if self._resolve_quant_algo(prefix) == "Q8_0":
                return make_q8_dequant_linear_method()
        return super().get_quant_method(layer, prefix)


def _mtp_dense_bf16(prefix: str) -> bool:
    """The MTP drafter's Q8_0 linears (~0.1 GB) are dequantized to BF16 even with ``EDGE0_DENSE=q8``: its
    ``fc_hidden`` runs on the 4 hyper-connection streams as rows (4 x max_num_batched_tokens), more than the b12x
    Q8_0 plans' row capacity (max_num_batched_tokens; "packed execution rows 16384 exceed capacity 4096" in the
    profile run). ``EDGE0_MTP_DENSE=q8`` keeps them Q8_0 (needs max_num_batched_tokens >= 4 x the drafter's rows)."""
    import os
    if (os.environ.get("EDGE0_MTP_DENSE", "bf16").strip().lower() or "bf16") != "bf16":
        return False
    head = prefix.split(".", 1)[0]
    return head == "mtp" or ".mtp." in f".{prefix}" or prefix.startswith("model.mtp")
