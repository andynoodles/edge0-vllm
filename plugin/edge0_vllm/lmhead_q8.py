"""``EDGE0_LMHEAD=q8``: serve the LM head as Q8_0 through vLLM's b12x block-quant linear method.

Edge0's ``output.weight`` is Q8_0 in the GGUF (Strata multiplies by those Q8 blocks); the converter stored it BF16
(DESIGN.md D3: ``Qwen4ExpMTP`` builds its own BF16 head from the same names). At c1 the head is the largest single
GEMV of a step: 1.27 GB of BF16 read by the target's logits (0.76 ms) and again by the DFlash2 drafter, which shares
the target's ``lm_head`` module (0.75 ms). After loading, the BF16 rows are re-quantized to Q8_0 with ggml's rule
(per 32 inputs ``d = fp16(amax / 127)`` refined over its fp16 neighbours, ``q = round(w / d)``): this recovers the
original Q8 values up to rare +-1 steps, so the head stays as close to Strata's as the BF16 copy was. The packed weight
(0.675 GB) replaces the BF16 one (frees ~0.6 GiB for the KV pool), and the module's quant method becomes
``ModelOptBlockQuantLinearMethod("q8_0")`` (b12x; with ``EDGE0_Q8_GEMV=1`` small calls take q8gemv), so the target
and every module sharing the head (the DFlash2 drafter) read half the bytes.
"""
from __future__ import annotations

import torch

_CHUNK_ROWS = 16384
HEAD_ROWS = 512         # rows per b12x head call (plan capacity); larger logits calls are chunked


@torch.no_grad()
def quantize_q8_0_blocks(w: torch.Tensor) -> torch.Tensor:
    """bf16/fp32 [N, K] -> uint8 [N, K/32, 34] GGUF Q8_0 blocks (fp16 d, int8 q), ggml's rounding.

    ``w`` is expected to be bf16(q * d) of original Q8_0 blocks. ggml's d = fp16(amax / 127) of the original values,
    and bf16 rounding moves amax by up to 2^-9, i.e. d by up to ~2 fp16 steps; so the fp16 neighbours d0-3..d0+3
    of d0 = fp16(amax / 127) are tried and the one whose w / d is closest to integers wins. On synthetic Q8 blocks
    this recovers 99.8% of the q values exactly and 56% of the d exactly (the others are a neighbouring fp16 value
    that fits the BF16 data equally well); a plain d0 recovers 98.8% / 12.5%."""
    n, k = w.shape
    kb = k // 32
    out = torch.empty((n, kb, 34), dtype=torch.uint8, device=w.device)
    for r0 in range(0, n, _CHUNK_ROWS):
        r1 = min(n, r0 + _CHUNK_ROWS)
        x = w[r0:r1].float().view(r1 - r0, kb, 32)
        amax = x.abs().amax(dim=-1, keepdim=True)
        d0 = (amax / 127.0).to(torch.float16)
        best_d, best_err = d0, None
        for j in (0, -1, 1, -2, 2, -3, 3):
            cand = (d0.view(torch.int16) + j).view(torch.float16) if j else d0
            cf = cand.float()
            ok = cf > 0
            r = x / torch.where(ok, cf, torch.ones_like(cf))
            err = (r - torch.round(r)).abs().amax(dim=-1, keepdim=True)
            err = torch.where(ok & (r.abs().amax(dim=-1, keepdim=True) <= 127.49), err, torch.full_like(err, 9.0))
            if best_err is None:
                best_err = err
            else:
                better = err < best_err
                best_d = torch.where(better, cand, best_d)
                best_err = torch.where(better, err, best_err)
        d = best_d
        df = d.float()
        inv = torch.where(df > 0, 1.0 / df, torch.zeros_like(df))
        q = torch.round(x * inv).clamp_(-127, 127).to(torch.int8)
        out[r0:r1, :, :2] = d.view(torch.uint8).view(r1 - r0, kb, 2)
        out[r0:r1, :, 2:] = q.view(torch.uint8)
    return out


def _gemv_rows() -> int:
    import os
    v = os.environ.get("EDGE0_LMHEAD_GEMV_ROWS", "").strip()
    return int(v) if v else 16


GEMV_ROWS = _gemv_rows()


def convert(model, log) -> bool:
    """Swap ``model.lm_head`` to a b12x Q8_0 head. Returns False (head unchanged) when not applicable."""
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.layers.quantization.modelopt_block_quant import ModelOptBlockQuantLinearMethod
    from vllm.model_executor.utils import replace_parameter
    from vllm.utils.b12x import get_b12x_blockscaled

    head = getattr(model, "lm_head", None)
    w = getattr(head, "weight", None)
    api = get_b12x_blockscaled()
    if head is None or w is None or w.dim() != 2 or w.dtype != torch.bfloat16 or w.device.type != "cuda" \
            or api is None or w.shape[0] % 128 or w.shape[1] % 32:
        log.warning("edge0: EDGE0_LMHEAD=q8 not applicable (lm_head %s); BF16 head kept",
                    None if w is None else (tuple(w.shape), w.dtype, w.device))
        return False
    n, k = w.shape
    blocks = quantize_q8_0_blocks(w.data)
    packed = api.pack_weight(blocks, recipe="q8_0")
    del blocks

    class _Q8HeadMethod(ModelOptBlockQuantLinearMethod):
        """b12x Q8_0 head whose plan covers HEAD_ROWS rows: vLLM's b12x warmup runs a layer at every warmup token
        count up to max_num_batched_tokens, i.e. a [4096, 248320] logits tensor (1.9 GiB) after the KV cache is
        allocated (OOM). Logits rows per step are few (sampled positions, drafts); larger calls (prompt logprobs)
        are chunked."""

        def apply(self, layer, x, bias=None):
            x2 = x.reshape(-1, x.shape[-1])
            if bias is None and x.dtype == torch.bfloat16 and GEMV_ROWS:
                # opaque op: <= GEMV_ROWS rows (target verify rows, the DFlash2 drafter's 16 candidate rows) take
                # the Triton GEMV (413 us vs b12x 435-638 us), larger calls b12x
                # (row chunking for b12x inside the op: no shape branch here, which torch.compile would guard on)
                w = layer.b12x_block_weight
                out = torch.ops.vllm.edge0_q8_linear(x2, w.values, w.metadata, int(layer.b12x_block_plan.handle),
                                                      int(w.out_features), int(w.in_features), GEMV_ROWS, HEAD_ROWS)
                return out.view(*x.shape[:-1], w.out_features)
            if x2.shape[0] <= HEAD_ROWS:
                return super().apply(layer, x, bias)
            outs = [super(_Q8HeadMethod, self).apply(layer, x2[r:r + HEAD_ROWS], bias)
                    for r in range(0, x2.shape[0], HEAD_ROWS)]
            return torch.cat(outs, 0).view(*x.shape[:-1], -1)

        def get_b12x_warmup_unit(self, layer, token_counts, output_dtype):
            return super().get_b12x_warmup_unit(layer, tuple(t for t in token_counts if t <= HEAD_ROWS),
                                                output_dtype)

    method = _Q8HeadMethod(codec="q8_0")
    config = get_current_vllm_config()
    sizes = config.compilation_config.cudagraph_capture_sizes or []
    cap = HEAD_ROWS
    query = api.BlockscaledQuery(recipe="q8_0", num_tokens=cap, in_features=k, padded_in_features=k, out_features=n,
                                 activation_mode="a16")
    plan = api.plan_regimes(query, exact_m=tuple(sorted({m for m in sizes if 0 < m < cap})))
    from . import q8gemv
    q8gemv._register_ops()
    head.quant_method = method
    head.b12x_block_weight = packed
    head.b12x_block_plan = plan
    head.b12x_warmup_provider = method
    replace_parameter(head, "weight", torch.empty(0, dtype=torch.uint8, device=w.device))
    del w
    torch.cuda.empty_cache()
    log.info("edge0: lm_head served as Q8_0 via b12x (%d x %d, %.2f GB instead of %.2f GB BF16; EDGE0_LMHEAD=q8)",
             n, k, n * k * 34 / 32 / 1e9, n * k * 2 / 1e9)
    return True


# ---------------------------------------------------------------------------------------------- QSA qkv (+ indexer)
def edge0_qsa_qkv(x: torch.Tensor, values: torch.Tensor, bases: torch.Tensor, plan_handle: int, n_q8: int, k: int,
                  w_bf16: torch.Tensor) -> torch.Tensor:
    """[x @ Wq8^T | x @ Wbf16^T]: the QSA projection's q/gate, k, v rows as Q8_0 (q8gemv for small calls, b12x
    otherwise), its indexer rows (BF16 in the GGUF) as BF16."""
    from . import q8gemv
    a = q8gemv.edge0_q8_linear(x, values, bases, plan_handle, n_q8, k)
    b = torch.nn.functional.linear(x, w_bf16)
    return torch.cat([a, b], dim=1)


def edge0_qsa_qkv_fake(x, values, bases, plan_handle: int, n_q8: int, k: int, w_bf16) -> torch.Tensor:
    return x.new_empty((x.shape[0], n_q8 + w_bf16.shape[0]))


def convert_qsa_qkv(model, log) -> int:
    """``EDGE0_QSA_Q8=1``: every QSA layer's fused ``qkv_proj`` (q/gate, k, v, then the indexer's q/k rows; BF16 in the
    checkpoint, 13,952 x 2,560 = 71 MB, 47 us per decode call) is split: the q/gate/k/v rows, which are Q8_0 in the
    GGUF (MODEL-SPEC 7), are re-quantized like the LM head and served by q8gemv / b12x; the indexer rows (BF16 in
    the GGUF) stay BF16. Returns the number of layers converted."""
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.layers.quantization.modelopt_block_quant import ModelOptBlockQuantLinearMethod
    from vllm.model_executor.utils import replace_parameter
    from vllm.utils.b12x import get_b12x_blockscaled
    from vllm.utils.torch_utils import direct_register_custom_op

    from . import q8gemv

    api = get_b12x_blockscaled()
    q8gemv._register_ops()
    try:
        torch.ops.vllm.edge0_qsa_qkv  # noqa: B018
    except (AttributeError, RuntimeError):
        direct_register_custom_op(op_name="edge0_qsa_qkv", op_func=edge0_qsa_qkv, mutates_args=[],
                                  fake_impl=edge0_qsa_qkv_fake)
    config = get_current_vllm_config()
    sched = config.scheduler_config
    sizes = config.compilation_config.cudagraph_capture_sizes or []
    cap = max(sched.max_num_batched_tokens, sched.max_num_scheduled_tokens or 0, *sizes)

    class _QsaQ8Method(ModelOptBlockQuantLinearMethod):
        def apply(self, layer, x, bias=None):
            w = layer.b12x_block_weight
            x2 = x.reshape(-1, x.shape[-1])
            out = torch.ops.vllm.edge0_qsa_qkv(x2, w.values, w.metadata, int(layer.b12x_block_plan.handle),
                                               int(w.out_features), int(w.in_features), layer.edge0_bf16_rows)
            return out.view(*x.shape[:-1], out.shape[-1])

    n_done = 0
    for name, mod in model.named_modules():
        if not name.endswith("self_attn.qkv_proj") or not getattr(mod, "output_sizes", None):
            continue
        w = getattr(mod, "weight", None)
        sizes_o = list(mod.output_sizes)
        if w is None or w.dtype != torch.bfloat16 or w.dim() != 2 or len(sizes_o) != 4 or api is None:
            continue
        n_q8 = sum(sizes_o[:3])
        if n_q8 % 128 or w.shape[1] % 32 or w.shape[0] != sum(sizes_o):
            continue
        blocks = quantize_q8_0_blocks(w.data[:n_q8])
        packed = api.pack_weight(blocks, recipe="q8_0")
        del blocks
        query = api.BlockscaledQuery(recipe="q8_0", num_tokens=cap, in_features=w.shape[1],
                                     padded_in_features=w.shape[1], out_features=n_q8, activation_mode="a16")
        plan = api.plan_regimes(query, exact_m=tuple(sorted({m for m in sizes if 0 < m < cap})))
        method = _QsaQ8Method(codec="q8_0")
        mod.edge0_bf16_rows = w.data[n_q8:].clone()
        mod.quant_method = method
        mod.b12x_block_weight = packed
        mod.b12x_block_plan = plan
        mod.b12x_warmup_provider = method
        replace_parameter(mod, "weight", torch.empty(0, dtype=torch.uint8, device=w.device))
        del w
        n_done += 1
    torch.cuda.empty_cache()
    log.info("edge0: %d QSA qkv projections: q/k/v rows as Q8_0, indexer rows BF16 (EDGE0_QSA_Q8=1)", n_done)
    return n_done
