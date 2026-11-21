"""Probe what the nvcr.io/nvidia/vllm image provides (no GPU needed).

Run inside a throwaway container, e.g. on the 5090 box:
  docker run --rm --memory 3g --cpus 2 -v /home/andychen/work/edge0-vllm:/workspace/edge0 \
      --entrypoint bash nvcr.io/nvidia/vllm:26.11-py3 \
      -c 'nice -n 19 python -I /workspace/edge0/tools/vllm_base_probe.py'
Prints versions, model-registry entries, NVFP4 MoE kernel availability and plugin entry points.
"""
import importlib
import importlib.metadata as md
import os
import sys


def line(k, v):
    print(f"{k:48s} {v}")


def try_(fn, default="ERR"):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return f"{default}: {type(e).__name__}: {str(e)[:160]}"


print("== versions")
for pkg in ("vllm", "torch", "flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache",
            "flashinfer-jit-cache-sm120f", "triton", "triton_kernels", "nvidia-cutlass-dsl", "transformers",
            "cuda-python", "cuda-bindings", "nixl", "safetensors", "fastsafetensors", "gguf", "b12x",
            "humming", "deep_gemm", "lmcache", "tokenspeed-triton", "cuda-tile"):
    line(pkg, try_(lambda: md.version(pkg), "missing"))

import torch  # noqa: E402

line("torch.version.cuda", torch.version.cuda)
line("torch.cuda arch list", try_(lambda: torch.cuda.get_arch_list()))
line("torch.backends.cudnn.version", try_(lambda: torch.backends.cudnn.version()))
line("python", sys.version.split()[0])

import vllm  # noqa: E402

line("vllm.__version__", vllm.__version__)
line("vllm path", os.path.dirname(vllm.__file__))
line("vllm commit", try_(lambda: importlib.import_module("vllm._version").__commit_id__))
import flashinfer  # noqa: E402

line("flashinfer.__version__", flashinfer.__version__)
line("flashinfer path", os.path.dirname(flashinfer.__file__))
import triton  # noqa: E402

line("triton.__version__", triton.__version__)
line("cutlass (DSL) import", try_(lambda: importlib.import_module("cutlass").__file__))

print("== optional kernel packages")
for mod in ("b12x", "b12x.moe.fused_moe", "humming", "deep_gemm", "vllm.third_party.deep_gemm",
            "flashinfer.fused_moe", "flashinfer.cute_dsl", "flashinfer.cutile", "triton_kernels",
            "vllm.vllm_flash_attn", "vllm._flashmla_C", "vllm._qutlass_C"):
    line(mod, try_(lambda: importlib.import_module(mod).__name__ and "ok", "missing"))
fm = try_(lambda: importlib.import_module("flashinfer.fused_moe"))
if not isinstance(fm, str):
    for attr in ("cutlass_fused_moe", "trtllm_fp4_block_scale_moe", "trtllm_fp4_block_scale_routed_moe",
                 "b12x_fused_moe", "cutile_fused_moe", "RoutingMethodType"):
        line(f"flashinfer.fused_moe.{attr}", hasattr(fm, attr))

print("== vllm.utils.flashinfer has_* (import level, no device)")
fu = try_(lambda: importlib.import_module("vllm.utils.flashinfer"))
if not isinstance(fu, str):
    for name in sorted(n for n in dir(fu) if n.startswith("has_flashinfer")):
        line(name, try_(lambda: getattr(fu, name)()))

print("== vllm custom ops (torch.ops._C / _moe_C)")
try:
    import vllm._custom_ops  # noqa: F401
    for ns in ("_C", "_moe_C"):
        opsns = getattr(torch.ops, ns)
        names = [n for n in ("cutlass_fp4_group_mm", "cutlass_scaled_fp4_mm", "scaled_fp4_quant",
                             "scaled_fp4_experts_quant", "cutlass_moe_mm", "get_cutlass_moe_mm_data",
                             "shuffle_rows", "moe_wna16_marlin_gemm", "gptq_marlin_moe_repack",
                             "cutlass_group_gemm_supported", "topk_softmax", "moe_align_block_size")
                 if try_(lambda: getattr(opsns, n), "x") and not isinstance(try_(lambda: getattr(opsns, n), "x"), str)]
        line(f"torch.ops.{ns} has", names)
except Exception as e:  # noqa: BLE001
    line("vllm._custom_ops", f"ERR {e}")

print("== model registry")
from vllm import ModelRegistry  # noqa: E402

archs = ModelRegistry.get_supported_archs()
line("#archs", len(archs))
line("Qwen4Exp*", [a for a in archs if "Qwen4" in a])
line("Qwen3Next*/3_5", [a for a in archs if "Qwen3Next" in a or "Qwen3_5" in a])

print("== entry points")
for group in ("vllm.general_plugins", "vllm.platform_plugins", "vllm.io_processor_plugins", "vllm.stat_logger_plugins"):
    eps = md.entry_points(group=group)
    line(group, [f"{e.name}={e.value}" for e in eps])

print("== speculative methods")
sc = try_(lambda: importlib.import_module("vllm.config.speculative"))
if not isinstance(sc, str):
    for n in ("SpeculativeMethod", "MTPModelTypes", "EagleModelTypes", "NgramModelTypes"):
        v = getattr(sc, n, None)
        if v is not None:
            line(n, try_(lambda: list(getattr(v, "__args__", ()) or v)))
print("== done")
