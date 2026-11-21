"""Small kernels of the latent MoE op (DESIGN.md 2.5 steps 3, 5, 6). Triton on CUDA, torch everywhere else.

* ``gather_routes``: route-as-token. Row ``r = t*K + j`` of the output is ``xl[t, g*L:(g+1)*L]`` with
  ``g = expert_group[ids[t, j]]``: each of a token's K routes becomes one Marlin input row of latent width L.
* ``combine_routes``: ``z[t, g, :] = sum_{j : group(ids[t,j]) == g, j ascending} Y[t*K + j]``, where ``Y`` is the
  hot output when the route's expert is hot (``hot_map[e] >= 0``) and the cold output otherwise. Fixed summation
  order, fp32 accumulation, no atomics: deterministic. It never reads a row of the other tier, so it does not rely
  on Marlin leaving skipped rows zero.
* ``touch_count``: TidalMoE's counter (``plugin/moe_placement/__init__.py`` ``_count_kernel``): adds 1 per distinct
  expert touched by one MoE call (``touch``) or per selection, into a float32 [E] row. Graph-safe (no host sync).
* ``tier_count``: ``touch_count(touch=False)`` plus the call's hot / cold statistics against the expert map *at call
  time* (EDGE0_TIER, EDGE0_TRACE=bw): row ``[E + TIER_STATS]`` gets +1 per routed entry in ``[:E]`` and, in
  ``[E:]``, distinct cold experts, distinct hot experts, cold routes, hot routes, calls. Graph-safe.
"""
from __future__ import annotations

import torch

try:  # vLLM ships triton; CPU-only test environments may not
    from vllm.triton_utils import tl, triton
    _HAS_TRITON = True
except Exception:  # noqa: BLE001
    _HAS_TRITON = False


def _use_triton(t: torch.Tensor) -> bool:
    return _HAS_TRITON and t.is_cuda


# ---------------------------------------------------------------------------------------------- gather
if _HAS_TRITON:
    @triton.jit
    def _gather_routes_kernel(xl_ptr, ids_ptr, grp_ptr, out_ptr, stride_xl, stride_out, num_experts,
                              TOPK: tl.constexpr, L: tl.constexpr, BLOCK: tl.constexpr):
        r = tl.program_id(0)
        t = r // TOPK
        e = tl.load(ids_ptr + r)
        e = tl.where((e >= 0) & (e < num_experts), e, 0)
        g = tl.load(grp_ptr + e)
        offs = tl.arange(0, BLOCK)
        m = offs < L
        v = tl.load(xl_ptr + t.to(tl.int64) * stride_xl + g * L + offs, mask=m, other=0.0)
        tl.store(out_ptr + r.to(tl.int64) * stride_out + offs, v, mask=m)

    @triton.jit
    def _combine_routes_kernel(yh_ptr, yc_ptr, ids_ptr, grp_ptr, hot_ptr, z_ptr, stride_y, stride_z, num_experts,
                               ycpu_ptr, cpu_ptr,
                               TOPK: tl.constexpr, L: tl.constexpr, BLOCK: tl.constexpr,
                               HAS_HOT: tl.constexpr, HAS_COLD: tl.constexpr, HAS_CPU: tl.constexpr):
        t = tl.program_id(0)
        g = tl.program_id(1)
        offs = tl.arange(0, BLOCK)
        m = offs < L
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for j in tl.static_range(TOPK):
            r = t * TOPK + j
            e = tl.load(ids_ptr + r)
            ok = (e >= 0) & (e < num_experts)
            e = tl.where(ok, e, 0)
            sel = ok & (tl.load(grp_ptr + e) == g)
            is_hot = tl.load(hot_ptr + e) >= 0
            row = r.to(tl.int64) * stride_y
            if HAS_HOT:
                vh = tl.load(yh_ptr + row + offs, mask=m & sel & is_hot, other=0.0)
                acc += vh.to(tl.float32)
            if HAS_CPU:
                # EDGE0_CPU_EXPERTS: rows of the CPU's experts come from mapped host memory (uncached loads)
                is_cpu = tl.load(cpu_ptr + e) != 0
                vp = tl.load(ycpu_ptr + row + offs, mask=m & sel & (is_hot == 0) & is_cpu, other=0.0,
                             cache_modifier=".cv")
                acc += vp.to(tl.float32)
                cold_sel = sel & (is_hot == 0) & (is_cpu == 0)
            else:
                cold_sel = sel & (is_hot == 0)
            if HAS_COLD:
                vc = tl.load(yc_ptr + row + offs, mask=m & cold_sel, other=0.0)
                acc += vc.to(tl.float32)
        tl.store(z_ptr + t.to(tl.int64) * stride_z + g * L + offs, acc.to(z_ptr.dtype.element_ty), mask=m)

    @triton.jit(do_not_specialize=["n_ids"])
    def _touch_count_kernel(ids_ptr, n_ids, cnt_ptr, n_e, TOUCH: tl.constexpr, BLOCK: tl.constexpr,
                            NBINS: tl.constexpr):
        acc = tl.zeros([NBINS], dtype=tl.int32)
        for start in range(0, n_ids, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            m = offs < n_ids
            ids = tl.load(ids_ptr + offs, mask=m, other=0).to(tl.int32)
            ok = m & (ids >= 0) & (ids < n_e)
            acc += tl.histogram(tl.where(ok, ids, 0), NBINS, mask=ok)
        b = tl.arange(0, NBINS)
        bm = b < n_e
        if TOUCH:
            inc = (acc > 0).to(tl.float32)
        else:
            inc = acc.to(tl.float32)
        old = tl.load(cnt_ptr + b, mask=bm, other=0.0)
        tl.store(cnt_ptr + b, old + inc, mask=bm)


if _HAS_TRITON:
    @triton.jit(do_not_specialize=["n_ids"])
    def _tier_count_kernel(ids_ptr, n_ids, row_ptr, hot_ptr, n_e, BLOCK: tl.constexpr, NBINS: tl.constexpr):
        acc = tl.zeros([NBINS], dtype=tl.int32)
        for start in range(0, n_ids, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            m = offs < n_ids
            ids = tl.load(ids_ptr + offs, mask=m, other=0).to(tl.int32)
            ok = m & (ids >= 0) & (ids < n_e)
            acc += tl.histogram(tl.where(ok, ids, 0), NBINS, mask=ok)
        b = tl.arange(0, NBINS)
        bm = b < n_e
        hot = tl.load(hot_ptr + b, mask=bm, other=-1) >= 0
        old = tl.load(row_ptr + b, mask=bm, other=0.0)
        tl.store(row_ptr + b, old + acc.to(tl.float32), mask=bm)
        touched = (acc > 0) & bm
        cold_d = tl.sum(tl.where(touched & (hot == 0), 1.0, 0.0), axis=0)
        hot_d = tl.sum(tl.where(touched & hot, 1.0, 0.0), axis=0)
        cold_r = tl.sum(tl.where(bm & (hot == 0), acc.to(tl.float32), 0.0), axis=0)
        hot_r = tl.sum(tl.where(bm & hot, acc.to(tl.float32), 0.0), axis=0)
        s = row_ptr + n_e
        tl.store(s + 0, tl.load(s + 0) + cold_d)
        tl.store(s + 1, tl.load(s + 1) + hot_d)
        tl.store(s + 2, tl.load(s + 2) + cold_r)
        tl.store(s + 3, tl.load(s + 3) + hot_r)
        tl.store(s + 4, tl.load(s + 4) + 1.0)


if _HAS_TRITON:
    @triton.jit
    def _route_post_kernel(ids_ptr, w_ptr, pad_ptr, padids_ptr, ids_out_ptr, w_out_ptr, row_ptr, hot_ptr, n_ids, n_e,
                           K: tl.constexpr, BLOCK: tl.constexpr, NBINS: tl.constexpr, HAS_PAD: tl.constexpr,
                           SKIP_PAD: tl.constexpr, COUNT: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        m = offs < n_ids
        t = offs // K
        ids = tl.load(ids_ptr + offs, mask=m, other=-1).to(tl.int32)
        w = tl.load(w_ptr + offs, mask=m, other=0.0)
        if HAS_PAD:
            pad = tl.load(pad_ptr + t, mask=m, other=0) != 0
        else:
            pad = offs < 0
        if SKIP_PAD:
            pid = tl.load(padids_ptr + offs % K, mask=m, other=0)
            tl.store(ids_out_ptr + offs, tl.where(pad, pid, ids), mask=m)
            tl.store(w_out_ptr + offs, tl.where(pad, 0.0, w), mask=m)
        else:
            tl.store(ids_out_ptr + offs, ids, mask=m)
            tl.store(w_out_ptr + offs, w, mask=m)
        if COUNT:     # = tier_count over the call's ids with padding rows ignored
            ok = m & (pad == 0) & (ids >= 0) & (ids < n_e)
            acc = tl.histogram(tl.where(ok, ids, 0), NBINS, mask=ok)
            b = tl.arange(0, NBINS)
            bm = b < n_e
            hot = tl.load(hot_ptr + b, mask=bm, other=-1) >= 0
            old = tl.load(row_ptr + b, mask=bm, other=0.0)
            tl.store(row_ptr + b, old + acc.to(tl.float32), mask=bm)
            touched = (acc > 0) & bm
            cold_d = tl.sum(tl.where(touched & (hot == 0), 1.0, 0.0), axis=0)
            hot_d = tl.sum(tl.where(touched & hot, 1.0, 0.0), axis=0)
            cold_r = tl.sum(tl.where(bm & (hot == 0), acc.to(tl.float32), 0.0), axis=0)
            hot_r = tl.sum(tl.where(bm & hot, acc.to(tl.float32), 0.0), axis=0)
            s = row_ptr + n_e
            tl.store(s + 0, tl.load(s + 0) + cold_d)
            tl.store(s + 1, tl.load(s + 1) + hot_d)
            tl.store(s + 2, tl.load(s + 2) + cold_r)
            tl.store(s + 3, tl.load(s + 3) + hot_r)
            tl.store(s + 4, tl.load(s + 4) + 1.0)


ROUTE_POST_MAX_IDS = 1024


def route_post(ids: torch.Tensor, w: torch.Tensor, pad: torch.Tensor | None, pad_ids: torch.Tensor | None,
               row: torch.Tensor | None, hot_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``EDGE0_ROUTE_FUSE``: the routing tail of one decode-sized MoE call in one kernel (round 2, agent exceed-2):
    padding rows (``pad [T]`` bool, the forward context's ``is_padding``) rerouted to ``pad_ids [1, K]`` with weight 0
    (``skip_pad``, when ``pad_ids`` is given) and, when ``row`` is given, ``tier_count`` of the call's non-padding
    routes into it. Same results as ``torch.where`` x 2 + ``zeros_like`` and the tier's ``full_like`` + ``where`` +
    ``tier_count`` (six kernels). Returns new (w, ids); ``T * K <= ROUTE_POST_MAX_IDS``."""
    t, k = ids.shape
    n = t * k
    assert n <= ROUTE_POST_MAX_IDS and ids.is_contiguous() and w.is_contiguous()
    ids_out = torch.empty_like(ids)
    w_out = torch.empty_like(w)
    n_e = hot_map.numel()
    has_pad = pad is not None
    skip = has_pad and pad_ids is not None
    count = row is not None
    padv = pad.view(torch.uint8) if has_pad and pad.dtype == torch.bool else pad
    _route_post_kernel[(1,)](ids, w, padv if has_pad else ids, pad_ids if skip else ids, ids_out, w_out,
                             row if count else w, hot_map, n, n_e, K=k,
                             BLOCK=max(32, triton.next_power_of_2(n)), NBINS=1 << max(n_e - 1, 1).bit_length(),
                             HAS_PAD=has_pad, SKIP_PAD=skip, COUNT=count, num_warps=4)
    return w_out, ids_out


TIER_STATS = 8      # columns after the E per-expert counts: cold_d, hot_d, cold_r, hot_r, calls, prefills, 2 spare


def tier_count(ids: torch.Tensor, row: torch.Tensor, hot_map: torch.Tensor) -> None:
    """``row [E + TIER_STATS] float32``: ``row[:E] += routed entries per expert``; ``row[E:E+5] +=`` (distinct cold,
    distinct hot, cold routes, hot routes, 1) of this call, hot = ``hot_map[e] >= 0`` now. Ids < 0 are ignored."""
    flat = ids.reshape(-1)
    n_e = hot_map.numel()
    if flat.numel() == 0:
        return
    if _use_triton(flat):
        flat = flat.contiguous()
        nbins = 1 << max(n_e - 1, 1).bit_length()
        _tier_count_kernel[(1,)](flat, flat.numel(), row, hot_map, n_e, BLOCK=1024, NBINS=nbins, num_warps=4)
        return
    valid = flat[(flat >= 0) & (flat < n_e)].long()
    hist = torch.bincount(valid, minlength=n_e).to(torch.float32)
    hot = hot_map >= 0
    touched = hist > 0
    row[:n_e] += hist
    row[n_e + 0] += (touched & ~hot).sum().to(torch.float32)
    row[n_e + 1] += (touched & hot).sum().to(torch.float32)
    row[n_e + 2] += hist[~hot].sum()
    row[n_e + 3] += hist[hot].sum()
    row[n_e + 4] += 1.0


def gather_routes(xl: torch.Tensor, ids: torch.Tensor, expert_group: torch.Tensor, latent_dim: int,
                  out: torch.Tensor | None = None) -> torch.Tensor:
    """``xl [T, G*L]``, ``ids [T, K]`` (logical expert ids), ``expert_group [E]`` -> ``x_routes [T*K, L]``."""
    t, k = ids.shape
    r = t * k
    if out is None:
        out = torch.empty((r, latent_dim), dtype=xl.dtype, device=xl.device)
    else:
        out = out[:r]
    if r == 0:
        return out
    if _use_triton(xl):
        assert xl.stride(1) == 1 and out.stride(1) == 1
        _gather_routes_kernel[(r,)](xl, ids.contiguous(), expert_group, out, xl.stride(0), out.stride(0),
                                    expert_group.numel(), TOPK=k, L=latent_dim,
                                    BLOCK=triton.next_power_of_2(latent_dim), num_warps=4)
        return out
    grp = expert_group[ids.long().clamp(0, expert_group.numel() - 1)].long()            # [T, K]
    src = xl.view(t, -1, latent_dim)                                                     # [T, G, L]
    out.copy_(src[torch.arange(t, device=xl.device).repeat_interleave(k), grp.reshape(-1)])
    return out


def combine_routes(y_hot: torch.Tensor | None, y_cold: torch.Tensor | None, ids: torch.Tensor,
                   expert_group: torch.Tensor, hot_map: torch.Tensor, num_groups: int, latent_dim: int,
                   out_dtype: torch.dtype, out: torch.Tensor | None = None, y_cpu: torch.Tensor | None = None,
                   cpu_sel: torch.Tensor | None = None) -> torch.Tensor:
    """Deterministic per-(token, group) sum of the routes' expert outputs -> ``z [T, G*L]`` (into ``out``).
    ``y_cpu`` / ``cpu_sel`` (EDGE0_CPU_EXPERTS): rows of cold experts with ``cpu_sel[e] != 0`` come from ``y_cpu``
    (route-indexed, mapped host memory written by the CPU pool) instead of ``y_cold``."""
    t, k = ids.shape
    ref = y_hot if y_hot is not None else y_cold
    device = ids.device
    z = out if out is not None else torch.empty((t, num_groups * latent_dim), dtype=out_dtype, device=device)
    assert z.shape == (t, num_groups * latent_dim) and z.stride(1) == 1
    if t == 0:
        return z
    if ref is None:
        return z.zero_()
    if _use_triton(ref):
        stride = ref.stride(0)
        if y_hot is not None and y_cold is not None:
            assert y_hot.stride(0) == y_cold.stride(0)
        has_cpu = y_cpu is not None
        if has_cpu:
            assert y_cpu.stride(0) == stride and cpu_sel is not None
        _combine_routes_kernel[(t, num_groups)](
            y_hot if y_hot is not None else ref, y_cold if y_cold is not None else ref, ids.contiguous(),
            expert_group, hot_map, z, stride, z.stride(0), expert_group.numel(),
            y_cpu if has_cpu else ref, cpu_sel if has_cpu else hot_map, TOPK=k, L=latent_dim,
            BLOCK=triton.next_power_of_2(latent_dim), HAS_HOT=y_hot is not None, HAS_COLD=y_cold is not None,
            HAS_CPU=has_cpu, num_warps=4)
        return z
    if y_cpu is not None:
        raise NotImplementedError("combine_routes with y_cpu needs CUDA/Triton")
    ids_l = ids.long().clamp(0, expert_group.numel() - 1)
    is_hot = (hot_map[ids_l] >= 0).reshape(-1, 1)                                        # [T*K, 1]
    if y_hot is None:
        y = y_cold.float()
    elif y_cold is None:
        y = y_hot.float()
    else:
        y = torch.where(is_hot, y_hot.float(), y_cold.float())
    y = y.view(t, k, latent_dim)
    grp = expert_group[ids_l].long()                                                     # [T, K]
    acc = torch.zeros((t * num_groups, latent_dim), dtype=torch.float32, device=device)
    base = torch.arange(t, device=device) * num_groups
    for j in range(k):                       # ascending j: same summation order as the Triton kernel
        acc.index_add_(0, base + grp[:, j], y[:, j])
    z.copy_(acc.view(t, num_groups * latent_dim).to(out_dtype))
    return z


def touch_count(ids: torch.Tensor, counts: torch.Tensor, touch: bool = True) -> None:
    """``counts [E] float32 += 1`` per distinct expert in ``ids`` (``touch``) or per selection."""
    flat = ids.reshape(-1)
    if flat.numel() == 0:
        return
    n_e = counts.numel()
    if _use_triton(flat):
        flat = flat.contiguous()
        nbins = 1 << max(n_e - 1, 1).bit_length()
        _touch_count_kernel[(1,)](flat, flat.numel(), counts, n_e, TOUCH=touch, BLOCK=1024, NBINS=nbins,
                                  num_warps=4)
        return
    valid = flat[(flat >= 0) & (flat < n_e)].long()
    hist = torch.bincount(valid, minlength=n_e).to(torch.float32)
    counts += (hist > 0).to(torch.float32) if touch else hist


def route_topk_softmax(logits: torch.Tensor, top_k: int, renormalize: bool = True
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax over all experts, top-k, renormalize (DESIGN.md 2.5 step 1) -> (weights fp32 [T,K], ids int32 [T,K]).

    CUDA: vLLM's ``topk_softmax`` op (the one FusedMoE uses), called without a padding mask so every row gets valid
    ids. Elsewhere: the torch equivalent (ties broken by torch.topk)."""
    t = logits.shape[0]
    if logits.is_cuda:
        import vllm._custom_ops as ops
        w = torch.empty((t, top_k), dtype=torch.float32, device=logits.device)
        ids = torch.empty((t, top_k), dtype=torch.int32, device=logits.device)
        tok = torch.empty((t, top_k), dtype=torch.int32, device=logits.device)
        ops.topk_softmax(w, ids, tok, logits.float().contiguous(), renormalize, None, None)
        return w, ids
    p = torch.softmax(logits.float(), dim=-1)
    w, ids = torch.topk(p, top_k, dim=-1)
    if renormalize:
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(2.0 ** -14)
    return w.float(), ids.to(torch.int32)
