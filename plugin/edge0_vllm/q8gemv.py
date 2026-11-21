"""Small-M Q8_0 GEMV over b12x's packed weights, with optional latent-group skipping (single-stream dense path).

b12x (``ModelOptBlockQuantLinearMethod("q8_0")``) keeps a Q8_0 linear as ``BlockQuantLinearWeight``:
``values`` = int8 payload tiles in memory order [N/128, K/32, 128, 32] and ``metadata`` = fp16 block bases [K/32, N]
(N % 128 == 0 for every Edge0 Q8 linear). This module reads those same tensors (no extra VRAM) with a Triton GEMV
for decode-sized calls (M <= ``max_m`` tokens), computing per 32-input block ``sum_j q*x`` in fp32 and scaling by
the block base, fp32 accumulation (the same math as b12x's A16 kernels, another summation order).

Group skipping (``EDGE0_DU_MASK=1``, latent D / U): a decode step's tokens route to experts of only some of the 8
latent groups (c1: 4.4 groups per token, 6.5 per 4-token verify step; ``tools/group_usage.py``). D's output rows
and U's input columns of the other groups are never used (the gather reads only xl[t, group(e)] of routed experts,
and z is zero for unused groups), so the kernel skips them. Every program decides from the step's route ids on the
GPU (no host sync, CUDA-graph safe):
  mode 1 (D, groups along N): a program whose rows belong to an unused group writes nothing;
  mode 2 (U, groups along K): split-K by group, an unused group's split writes zeros to its partial.
"""
from __future__ import annotations

import torch

import triton
import triton.language as tl

from .config import LATENT_DIM, LATENT_GROUPS

TILE_N = 128          # b12x Q8_0 payload tile rows (N % 128 == 0)


_PDL = None
_TUNE: dict = {}        # launch-parameter overrides (tools/q8gemv_sweep.py: UNROLL 4, stages 3, warps 4 are best)


def _pdl(x: torch.Tensor) -> bool:
    """PDL launches (as vLLM's own HC kernels) on GPUs that support them; EDGE0_Q8_PDL=0 turns them off."""
    global _PDL
    if _PDL is None:
        import os
        _PDL = False
        if x.is_cuda and os.environ.get("EDGE0_Q8_PDL", "1") != "0":
            try:
                from vllm.platforms import current_platform
                _PDL = bool(current_platform.is_arch_support_pdl())
            except Exception:  # noqa: BLE001
                _PDL = False
    return _PDL


def _pow2(v: int) -> int:
    return 1 << max(0, int(v) - 1).bit_length()


@triton.jit
def _q8_gemv_kernel(x_ptr, q_ptr, d_ptr, out_ptr, ids_ptr, grp_ptr,
                    M, N, KB, KB_PER_SPLIT, N_IDS, stride_xm,
                    MASK_MODE: tl.constexpr, GROUP_N: tl.constexpr, GROUP_KB: tl.constexpr,
                    M_PAD: tl.constexpr, BLOCK_N: tl.constexpr, IDS_PAD: tl.constexpr,
                    PARTIAL: tl.constexpr, USE_DOT: tl.constexpr, UNROLL: tl.constexpr, launch_pdl: tl.constexpr):
    if launch_pdl:      # programmatic dependent launch: start while the producer of x / ids finishes
        tl.extra.cuda.gdc_wait()
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    n0 = pid_n * BLOCK_N
    kb0 = pid_k * KB_PER_SPLIT
    offs_m = tl.arange(0, M_PAD)
    offs_n = tl.arange(0, BLOCK_N)
    offs_j = tl.arange(0, 32)
    active = True
    if MASK_MODE != 0:
        offs_i = tl.arange(0, IDS_PAD)
        im = offs_i < N_IDS
        e = tl.load(ids_ptr + offs_i, mask=im, other=0)
        g = tl.load(grp_ptr + e, mask=im, other=-1)
        if MASK_MODE == 1:
            my_g = n0 // GROUP_N
        else:
            my_g = kb0 // GROUP_KB
        active = tl.sum((g == my_g).to(tl.int32), axis=0) > 0
    acc = tl.zeros([M_PAD, BLOCK_N], dtype=tl.float32)
    if active:
        tile = n0 // 128
        r0 = n0 % 128
        mrow = offs_m[:, None] < M
        q_base = q_ptr + ((tile * KB) * 128 + r0 + offs_n[:, None]) * 32 + offs_j[None, :]
        x_base = x_ptr + offs_m[:, None] * stride_xm + offs_j[None, :]
        for kbi in range(kb0, kb0 + KB_PER_SPLIT, UNROLL):
            for u in tl.static_range(UNROLL):
                kb = kbi + u
                q = tl.load(q_base + kb * 4096)                                    # [BLOCK_N, 32] int8
                d = tl.load(d_ptr + kb * N + n0 + offs_n).to(tl.float32)          # [BLOCK_N]
                x = tl.load(x_base + kb * 32, mask=mrow, other=0.0)                # [M_PAD, 32] bf16
                if USE_DOT:
                    s = tl.dot(x, tl.trans(q.to(tl.bfloat16)), out_dtype=tl.float32)   # [M_PAD, BLOCK_N]
                else:
                    s = tl.sum(q.to(tl.float32)[None, :, :] * x.to(tl.float32)[:, None, :], axis=2)
                acc += s * d[None, :]
    if PARTIAL:
        o = out_ptr + (pid_k * M + offs_m[:, None]) * N + n0 + offs_n[None, :]
        if launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()
        tl.store(o, acc, mask=offs_m[:, None] < M)
    else:
        if launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()
        if active:
            o = out_ptr + offs_m[:, None] * N + n0 + offs_n[None, :]
            tl.store(o, acc.to(out_ptr.dtype.element_ty), mask=offs_m[:, None] < M)


EPI_NONE, EPI_ADD, EPI_SILU_MUL, EPI_GATED_ADD = 0, 1, 2, 3


@triton.jit
def _splitk_sum_kernel(p_ptr, out_ptr, add_ptr, xg_ptr, wg_ptr, M, N, KG, stride_xg,
                       SPLIT: tl.constexpr, EPI: tl.constexpr, BLOCK: tl.constexpr, KG_BLOCK: tl.constexpr,
                       launch_pdl: tl.constexpr):
    """Sum of the split-K partials [SPLIT, M, N] fp32 of one row (program axis 1) with an epilogue:
    EPI 0: y = s; 1: y = add + s; 2 (SwiGLU of a merged gate|up weight): y[:, j] = silu(s[:, j]) * s[:, j + N/2];
    3 (sigmoid-gated shared expert added to the routed output): y = add + sigmoid(xg . wg) * s."""
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    row = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EPI == 2:
        NO = N // 2
    else:
        NO = N
    msk = offs < NO
    base = p_ptr + row * N + offs
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(base + s * M * N, mask=msk, other=0.0)
    if EPI == 2:
        up = tl.zeros([BLOCK], dtype=tl.float32)
        for s in tl.static_range(SPLIT):
            up += tl.load(base + s * M * N + NO, mask=msk, other=0.0)
        acc = acc / (1.0 + tl.exp(-acc)) * up
    if EPI == 1:
        acc += tl.load(add_ptr + row * N + offs, mask=msk, other=0.0).to(tl.float32)
    if EPI == 3:
        ko = tl.arange(0, KG_BLOCK)
        gs = tl.zeros([KG_BLOCK], dtype=tl.float32)
        for k0 in range(0, KG, KG_BLOCK):
            km = (k0 + ko) < KG
            xv = tl.load(xg_ptr + row * stride_xg + k0 + ko, mask=km, other=0.0).to(tl.float32)
            wv = tl.load(wg_ptr + k0 + ko, mask=km, other=0.0).to(tl.float32)
            gs += xv * wv
        g = tl.sum(gs, axis=0)
        acc = tl.load(add_ptr + row * N + offs, mask=msk, other=0.0).to(tl.float32) + acc / (1.0 + tl.exp(-g))
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(out_ptr + row * NO + offs, acc.to(out_ptr.dtype.element_ty), mask=msk)


def _cfg(m: int, n: int, kb: int, mask_mode: int):
    """(BLOCK_N, SPLIT_K) per shape. Targets >= ~400 programs of >= ~16 KB each; split-K only along whole groups
    for mode 2."""
    if mask_mode == 2:
        return 32, LATENT_GROUPS
    block_n = 32
    programs = n // block_n
    split = 1
    while programs * split < 400 and kb // (split * 2) >= 8 and kb % (split * 2) == 0:
        split *= 2
    return block_n, split


def q8_gemv(x: torch.Tensor, values: torch.Tensor, bases: torch.Tensor, n: int, k: int,
            ids: torch.Tensor | None = None, expert_group: torch.Tensor | None = None, mask_mode: int = 0,
            add: torch.Tensor | None = None, use_dot: bool = True, block_n: int = 0, split: int = 0,
            epi: int = -1, xg: torch.Tensor | None = None, wg: torch.Tensor | None = None,
            unroll: int = 0, warps: int = 0, stages: int = 0) -> torch.Tensor:
    """y [M, n] bf16 = x [M, k] bf16 @ W^T for a b12x-packed Q8_0 weight (``values`` [n, k] bytes in tile order,
    ``bases`` [k/32, n] fp16 or the uint8 [k/32, n, 2] metadata). mask_mode 1 / 2: latent D / U group skipping by
    the step's route ``ids`` [M, top_k] and ``expert_group`` [512]; ``add``: [M, n] added in the split-K sum.
    ``epi`` (_splitk_sum_kernel): EPI_SILU_MUL returns [M, n/2]; EPI_GATED_ADD needs ``add``, ``xg`` [M, KG] and the
    gate weight ``wg`` [KG]."""
    if epi < 0:
        epi = EPI_ADD if add is not None else EPI_NONE
    m = x.shape[0]
    kb = k // 32
    assert x.stride(-1) == 1 and n % TILE_N == 0 and k % 32 == 0
    if bases.dtype == torch.uint8:
        bases = bases.view(torch.float16).view(kb, n)
    q = values.view(torch.int8)
    bn, sp = _cfg(m, n, kb, mask_mode)
    bn = block_n or bn
    sp = split or sp
    un = unroll or next(u for u in (_TUNE.get("unroll", 4), 2, 1) if (kb // sp) % u == 0)
    assert kb % sp == 0 and n % bn == 0 and (kb // sp) % un == 0
    m_pad = 16 if use_dot else max(2, _pow2(m))
    if mask_mode:
        assert ids is not None and expert_group is not None
        n_ids = ids.numel()
        ids_t, grp_t = ids, expert_group
    else:
        n_ids = 0
        ids_t = grp_t = bases          # unused pointers
    ids_pad = max(16, _pow2(max(n_ids, 1)))
    partial = sp > 1 or epi != EPI_NONE
    out = torch.empty((m, n // 2 if epi == EPI_SILU_MUL else n), dtype=torch.bfloat16, device=x.device)
    if partial:
        part = torch.empty((sp, m, n), dtype=torch.float32, device=x.device)
        dst = part
    else:
        dst = out
    _q8_gemv_kernel[(n // bn, sp)](
        x, q, bases, dst, ids_t, grp_t, m, n, kb, kb // sp, n_ids, x.stride(0),
        MASK_MODE=mask_mode, GROUP_N=LATENT_DIM, GROUP_KB=LATENT_DIM // 32, M_PAD=m_pad, BLOCK_N=bn,
        IDS_PAD=ids_pad, PARTIAL=partial, USE_DOT=use_dot, UNROLL=un, num_warps=warps or _TUNE.get("warps", 4),
        num_stages=stages or _TUNE.get("stages", 3), launch_pdl=_pdl(x))
    if partial:
        no = out.shape[1]
        blk = 512
        if epi == EPI_GATED_ADD:
            assert add is not None and xg is not None and wg is not None and xg.stride(-1) == 1
            kg, xg_t, wg_t, sxg = xg.shape[1], xg, wg, xg.stride(0)
        else:
            kg, xg_t, wg_t, sxg = 0, out, out, 0
        _splitk_sum_kernel[((no + blk - 1) // blk, m)](part, out, add if add is not None else out, xg_t, wg_t, m, n,
                                                       kg, sxg, SPLIT=sp, EPI=epi, BLOCK=blk, KG_BLOCK=512,
                                                       num_warps=4, launch_pdl=_pdl(x))
    return out


def shared_expert_add(x: torch.Tensor, routed: torch.Tensor, gu, down, wg: torch.Tensor) -> torch.Tensor:
    """routed + sigmoid(x . wg) * down(silu(gate(x)) * up(x)) for b12x-packed Q8_0 ``gu`` (merged gate|up) and
    ``down`` (BlockQuantLinearWeight): 4 kernels (2 GEMV + 2 split-K sums with the epilogues)."""
    h = q8_gemv(x, gu.values, gu.metadata, gu.out_features, gu.in_features, epi=EPI_SILU_MUL)
    return q8_gemv(h, down.values, down.metadata, down.out_features, down.in_features, epi=EPI_GATED_ADD,
                   add=routed, xg=x, wg=wg)


# ---------------------------------------------------------------------------------------------- serving hooks
def _env_int(name: str, default: int) -> int:
    import os
    v = os.environ.get(name, "").strip()
    return int(v) if v else default


Q8_GEMV = _env_int("EDGE0_Q8_GEMV", 0)              # 1 = small-M Q8_0 linears through q8_gemv (b12x otherwise)
DU_MASK = _env_int("EDGE0_DU_MASK", 0)              # 1 = latent D / U skip the step's unused routing groups
SHARED_FUSE = _env_int("EDGE0_SHARED_FUSE", 0)      # 1 = shared expert + residual add as 4 fused kernels
MAX_M = _env_int("EDGE0_Q8_GEMV_MAX_M", 8)          # largest call (tokens) served by q8_gemv


def _b12x_mm(x, values, bases, plan_handle: int, n: int, k: int):
    from b12x.preparation.types import plan_from_handle

    from vllm.utils.b12x import get_b12x_blockscaled
    api = get_b12x_blockscaled()
    w = api.BlockQuantLinearWeight(values, bases, k, n, "q8_0")
    return api.mm(x.contiguous(), w, plan=plan_from_handle(plan_handle))


def edge0_q8_linear(x: torch.Tensor, values: torch.Tensor, bases: torch.Tensor, plan_handle: int, n: int,
                    k: int, max_m: int = 0, chunk: int = 0) -> torch.Tensor:
    """Opaque Q8_0 linear: calls of <= ``max_m`` (default MAX_M) rows take q8_gemv, larger ones b12x (in row chunks
    of ``chunk`` when set: the b12x plan's capacity)."""
    m = x.shape[0]
    if 0 < m <= (max_m or MAX_M) and x.stride(-1) == 1:
        return q8_gemv(x, values, bases, n, k)
    if chunk and m > chunk:
        return torch.cat([_b12x_mm(x[r:r + chunk], values, bases, plan_handle, n, k) for r in range(0, m, chunk)])
    return _b12x_mm(x, values, bases, plan_handle, n, k)


def edge0_q8_linear_fake(x, values, bases, plan_handle: int, n: int, k: int, max_m: int = 0,
                         chunk: int = 0) -> torch.Tensor:
    return x.new_empty((x.shape[0], n))


def install_linear_patch(logger=None) -> None:
    """EDGE0_Q8_GEMV=1: ``ModelOptBlockQuantLinearMethod.apply`` (b12x Q8_0) goes through the opaque op
    ``vllm::edge0_q8_linear``, which serves calls of <= MAX_M tokens with q8_gemv and the rest with b12x."""
    from vllm.model_executor.layers.quantization.modelopt_block_quant import ModelOptBlockQuantLinearMethod as M
    if getattr(M.apply, "_edge0_q8", False):
        return
    _register_ops()
    orig = M.apply

    def apply(self, layer, x, bias=None):
        w = getattr(layer, "b12x_block_weight", None)
        if (bias is not None or self.codec != "q8_0" or w is None or x.dtype != torch.bfloat16
                or w.out_features % TILE_N or w.in_features % 32):
            return orig(self, layer, x, bias)
        x2 = x.reshape(-1, x.shape[-1])
        out = torch.ops.vllm.edge0_q8_linear(x2, w.values, w.metadata, int(layer.b12x_block_plan.handle),
                                              int(w.out_features), int(w.in_features))
        return out.view(*x.shape[:-1], w.out_features)

    apply._edge0_q8 = True
    M.apply = apply
    if logger is not None:
        logger.info("edge0: Q8_0 linears with <= %d tokens use the Triton GEMV (EDGE0_Q8_GEMV=1)", MAX_M)


def _register_ops() -> None:
    from vllm.utils.torch_utils import direct_register_custom_op
    try:
        torch.ops.vllm.edge0_q8_linear  # noqa: B018
        return
    except (AttributeError, RuntimeError):
        pass
    direct_register_custom_op(op_name="edge0_q8_linear", op_func=edge0_q8_linear, mutates_args=[],
                              fake_impl=edge0_q8_linear_fake)
