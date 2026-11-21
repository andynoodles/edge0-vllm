"""NVFP4 expert tensors: ModelOpt checkpoint layout -> Marlin W4A16 layout, one expert at a time (DESIGN.md 2.6).

This is ``prepare_nvfp4_moe_layer_for_marlin`` (vllm ``marlin_utils_fp4.py``) split per expert, so experts can be
written straight into whichever pool (VRAM or host) their slot lives in, with no whole-layer GPU staging. The scale
rescaling factor that Marlin's S0E5M3 scale format needs is computed once per layer and per tensor over **all**
512 experts (exactly what vLLM does for a layer), so an expert's processed scales are valid in either pool and swaps
between pools stay exact.

No padding is needed: K = 1792 is a multiple of 128 and N = 640 a multiple of 64.

The pure-torch helpers at the bottom (``dequant_modelopt_nvfp4``) decode the checkpoint layout directly; they back
the F1 reference path and the CPU tests.
"""
from __future__ import annotations

import torch

from .config import EXPERT_FF, LATENT_DIM, NVFP4_GROUP

K = LATENT_DIM      # w13 input width / w2 output width
N = EXPERT_FF       # expert intermediate size
PARAM_DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------------------------- Marlin (GPU)
def repack_w13(w13_u8: torch.Tensor) -> torch.Tensor:
    """One expert's ModelOpt ``uint8 [2N, K/2]`` -> Marlin ``int32 [K/16, 2N*2]`` = [112, 2560] (on its device)."""
    import vllm._custom_ops as ops
    qweight = w13_u8.contiguous().view(torch.int32).T.contiguous()      # [K/8, 2N]
    return ops.gptq_marlin_repack(b_q_weight=qweight, size_k=K, size_n=2 * N, num_bits=4, is_a_8bit=False)


def repack_w2(w2_u8: torch.Tensor) -> torch.Tensor:
    """One expert's ModelOpt ``uint8 [K, N/2]`` -> Marlin ``int32 [N/16, K*2]`` = [40, 3584]."""
    import vllm._custom_ops as ops
    qweight = w2_u8.contiguous().view(torch.int32).T.contiguous()       # [N/8, K]
    return ops.gptq_marlin_repack(b_q_weight=qweight, size_k=N, size_n=K, num_bits=4, is_a_8bit=False)


def scale_factor(scales_e4m3_all: torch.Tensor, device: torch.device | str | None = None, chunk: int = 32) -> float:
    """Layer-wide Marlin rescale factor over every expert's E4M3 block scales: vLLM's ``_nvfp4_compute_scale_factor``
    on the bf16 cast of the whole stacked tensor, computed chunk by chunk (round 2, agent exceed-2).

    vLLM's version materialises the whole layer as fp32 plus a boolean mask and a masked copy (~1.6 GiB of transient
    VRAM for a w13 scale tensor), which capped the expert VRAM the load can leave. It only needs the largest positive
    value: max over chunks of ``where(x > 0, x, 0)`` (NaN and non-positive entries drop out, as with vLLM's ``> 0``
    mask), then the same float32 tensor ops (``448 * 2^7 / max``, log2, floor, exp2), so the factor is identical."""
    dev = torch.device(device) if device is not None else scales_e4m3_all.device
    m = None
    for c in scales_e4m3_all.split(chunk, dim=0):
        x = c.to(dev).to(PARAM_DTYPE).float()
        cm = torch.where(x > 0, x, torch.zeros_like(x)).max()
        m = cm if m is None else torch.maximum(m, cm)
    if m is None:
        return 1.0
    max_val = m * (2 ** 7)
    if not bool(max_val > 0):
        return 1.0
    if bool(max_val < 448 * (2 ** 7)):
        return float((448 * (2 ** 7) / max_val).log2().floor().exp2().item())
    return 1.0


def process_scales(scales_e4m3: torch.Tensor, size_k: int, size_n: int, factor: float) -> torch.Tensor:
    """A chunk of experts' ModelOpt scales ``e4m3 [E, size_n, size_k/16]`` -> Marlin ``fp8 [E, size_k/16, size_n]``.

    Same ops as vLLM's per-expert loop (``marlin_permute_scales`` then ``nvfp4_marlin_process_scales`` with the given
    factor); both are row-local on a [rows, size_n] view, so a chunk of experts is processed in one go."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import nvfp4_marlin_process_scales
    e = scales_e4m3.shape[0]
    s = scales_e4m3.to(PARAM_DTYPE).transpose(1, 2).contiguous()          # [E, K/16, N]
    s = s.reshape(e * (size_k // NVFP4_GROUP), size_n)
    s = marlin_permute_scales(s=s, size_k=size_k, size_n=size_n, group_size=NVFP4_GROUP, is_a_8bit=False)
    s, _ = nvfp4_marlin_process_scales(s, scale_factor=factor, a_dtype=PARAM_DTYPE)
    return s.reshape(e, size_k // NVFP4_GROUP, size_n)


def process_global_scale(scale_2: torch.Tensor, factor: float) -> torch.Tensor:
    """ModelOpt ``weight_scale_2`` (fp32 per expert) -> Marlin global scale (vLLM: process, then / factor)."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import nvfp4_marlin_process_global_scale
    return (nvfp4_marlin_process_global_scale(scale_2.to(torch.float32), PARAM_DTYPE) / factor).contiguous()


# ---------------------------------------------------------------------------------------------- pure torch
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def e2m1_table(device=None) -> torch.Tensor:
    return torch.tensor(_E2M1, dtype=torch.float32, device=device)


def unpack_codes(w_u8: torch.Tensor) -> torch.Tensor:
    """ModelOpt packing: byte m holds element 2m in the low nibble and 2m+1 in the high nibble.
    ``uint8 [..., K/2]`` -> ``int64 codes [..., K]``."""
    lo = (w_u8 & 0x0F).to(torch.int64)
    hi = (w_u8 >> 4).to(torch.int64)
    return torch.stack([lo, hi], dim=-1).reshape(*w_u8.shape[:-1], w_u8.shape[-1] * 2)


def dequant_modelopt_nvfp4(w_u8: torch.Tensor, scale_e4m3: torch.Tensor, scale_2) -> torch.Tensor:
    """``value = E2M1(code) * e4m3(block scale) * scale_2`` -> float32 ``[rows, K]`` (DESIGN.md 1.5)."""
    codes = unpack_codes(w_u8)
    vals = e2m1_table(w_u8.device)[codes]
    sc = scale_e4m3.to(torch.float32).repeat_interleave(NVFP4_GROUP, dim=-1)
    return vals * sc * torch.as_tensor(scale_2, dtype=torch.float32, device=w_u8.device)


def expert_forward_reference(x: torch.Tensor, w13_u8, s13, g13: float, w2_u8, s2, g2: float) -> torch.Tensor:
    """One expert on rows ``x [r, K]`` in float32: down(silu(gate x) * (up x)); w13 rows 0..N-1 = gate."""
    w13 = dequant_modelopt_nvfp4(w13_u8, s13, g13)              # [2N, K]
    w2 = dequant_modelopt_nvfp4(w2_u8, s2, g2)                  # [K, N]
    h = x.to(torch.float32) @ w13.T
    gate, up = h[:, :N], h[:, N:]
    return (torch.nn.functional.silu(gate) * up) @ w2.T
