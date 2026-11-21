"""GGUF Q8_0 blocks <-> torch (DESIGN.md 1.5 "Q8", 2.4).

A Q8 checkpoint tensor is ``uint8 [N, K/32, 34]``: per 32 inputs one ``block_q8_0`` = little-endian fp16 ``d``
then 32 x int8 ``q``; ``value = d * q``. At ``EDGE0_DENSE=bf16`` the weight loader of every Q8_0 linear layer is
wrapped so the incoming blocks are dequantized (fp32 math, bf16 result, round to nearest even) on the parameter's
device and handed to vLLM's own loader, which then slices fused shards as usual. No uint8 copy stays resident.
"""
from __future__ import annotations

import functools

import torch

Q8_BLOCK = 32
Q8_BLOCK_BYTES = 34
_ROWS_PER_CHUNK = 4096      # bounds the fp32 temporary to 4096 x K x 4 B (42 MB at K = 2560)

# Q8_0 block tensors dequantized by a wrapped loader in this process (loadcheck.LoadAudit checks that every Q8_0
# checkpoint tensor of the main model went through one).
DEQUANT_TENSORS = 0


def is_q8_blocks(t: torch.Tensor) -> bool:
    return t.dtype == torch.uint8 and t.dim() == 3 and t.shape[-1] == Q8_BLOCK_BYTES


def dequant_q8_0(blocks: torch.Tensor, out_dtype: torch.dtype = torch.bfloat16,
                 device: torch.device | str | None = None) -> torch.Tensor:
    """``uint8 [N, nb, 34]`` -> ``out_dtype [N, nb * 32]`` on ``device`` (default: the input's device)."""
    if not is_q8_blocks(blocks):
        raise ValueError(f"not a Q8_0 block tensor: {blocks.dtype} {tuple(blocks.shape)}")
    n, nb, _ = blocks.shape
    device = torch.device(device) if device is not None else blocks.device
    if device.type == "meta" or blocks.device.type == "meta":
        return torch.empty((n, nb * Q8_BLOCK), dtype=out_dtype, device="meta")
    out = torch.empty((n, nb * Q8_BLOCK), dtype=out_dtype, device=device)
    for r0 in range(0, n, _ROWS_PER_CHUNK):
        r1 = min(n, r0 + _ROWS_PER_CHUNK)
        b = blocks[r0:r1].to(device, non_blocking=False).contiguous()
        d = b[..., :2].contiguous().view(torch.float16).to(torch.float32)          # [r, nb, 1]
        q = b[..., 2:].contiguous().view(torch.int8).to(torch.float32)             # [r, nb, 32]
        out[r0:r1] = (q * d).reshape(r1 - r0, nb * Q8_BLOCK).to(out_dtype)
    return out


def quant_q8_0(w: torch.Tensor) -> torch.Tensor:
    """Reference quantizer (tests only): float [N, K] -> uint8 [N, K/32, 34], ggml's rounding (d = amax/127)."""
    n, k = w.shape
    assert k % Q8_BLOCK == 0
    x = w.detach().float().reshape(n, k // Q8_BLOCK, Q8_BLOCK)
    amax = x.abs().amax(dim=-1, keepdim=True)
    d = (amax / 127.0).to(torch.float16)
    inv = torch.where(d.float() > 0, 1.0 / d.float(), torch.zeros_like(amax))
    q = torch.round(x * inv).clamp(-127, 127).to(torch.int8)
    out = torch.empty((n, k // Q8_BLOCK, Q8_BLOCK_BYTES), dtype=torch.uint8)
    out[..., :2] = d.view(torch.uint8).reshape(n, k // Q8_BLOCK, 2)
    out[..., 2:] = q.view(torch.uint8)
    return out


def wrap_weight_loader(loader):
    """Wrap a vLLM ``weight_loader(param, loaded_weight, *args)`` so Q8_0 block tensors arrive dequantized.

    The stacked-name mapper tags each tensor with ``shard_id`` (and fused linears read it back), so the tag is
    copied onto the dequantized tensor."""
    if getattr(loader, "_edge0_q8", False):
        return loader

    @functools.wraps(loader)
    def q8_dequant_loader(param, loaded_weight, *args, **kwargs):
        if is_q8_blocks(loaded_weight) and param.dtype != torch.uint8:
            global DEQUANT_TENSORS
            DEQUANT_TENSORS += 1
            shard_id = getattr(loaded_weight, "shard_id", None)
            loaded_weight = dequant_q8_0(loaded_weight, param.dtype, device=param.device)
            if shard_id is not None:
                loaded_weight.shard_id = shard_id
        return loader(param, loaded_weight, *args, **kwargs)

    q8_dequant_loader._edge0_q8 = True
    return q8_dequant_loader
