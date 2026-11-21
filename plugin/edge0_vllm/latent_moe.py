"""``Edge0LatentMoeBlock``: Edge0's latent MoE as a drop-in for ``Qwen4ExpSparseMoeBlock`` (DESIGN.md 2.5).

    x [T, 2560] --gate (fp32 logits)--------------------------------------------------.
       |--latent_down (D_cat, 2560 -> 8 x 1792)--> xl --vllm::edge0_latent_moe(xl, logits)--> z [T, 8 x 1792]
       |                                                                                    |
       |                                                            latent_up (U_cat, 14336 -> 2560)
       '--shared_expert (sigmoid-gated SwiGLU 2560 -> 640 -> 2560) ------------------------ + --> out [T, 2560]

``vllm::edge0_latent_moe`` (opaque to torch.compile, captured in CUDA graphs):
 1. softmax top-10 + renormalize on the fp32 router logits (vLLM ``topk_softmax``; the router is never permuted);
 2. route-as-token gather: every (token, route) becomes one row ``xl[t, group(e)]`` of width 1792;
 3. hot and cold experts as two Marlin W4A16 NVFP4 MoE calls with ``topk=1`` (logical id -> slot through each
    tier's ``expert_map``; the route weight is applied in GEMM2). With ``EDGE0_SPLIT=1`` the cold call runs on a side
    stream, so the PCIe-bound cold reads overlap the VRAM-bound hot GEMMs;
 4. deterministic combine of the route outputs per (token, group) -> z.

All Marlin workspaces and caches are preallocated once per process (layers run one after the other) for
``max_num_batched_tokens x 10`` routes; the hot and cold calls never share a Marlin workspace.
"""
import torch
from torch import nn

from vllm.model_executor.layers.fused_moe import GateLinear
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.torch_utils import direct_register_custom_op

try:  # torch >= 2.11: hoisted opaque layer names avoid per-layer recompiles (same as vLLM's own MoE op)
    from vllm.utils.torch_utils import LayerNameType, _encode_layer_name, _resolve_layer_name
except ImportError:  # pragma: no cover
    LayerNameType = str

    def _encode_layer_name(name):
        return name

    def _resolve_layer_name(name):
        return name

from . import kernels
from . import marlin_nvfp4 as mn
from . import q8gemv
from . import runtime
from .nvtx import rng as _nvtx
from .config import EXPERT_FF, LATENT_DIM, LATENT_GROUPS, NUM_EXPERTS, TOP_K
from .expert_pool import ExpertPool
from .trace import _padding_mask

_BLOCKS: dict = {}
# Per-tier Marlin overrides {"hot"|"cold": (block_size_m or 0 = auto, blocks_per_sm or -1, min tokens)}; set by
# set_tier_tuning from EDGE0_HOT_BSM / EDGE0_COLD_BSM / EDGE0_COLD_BLOCKS_PER_SM / EDGE0_BSM_MIN_TOKENS (absent =
# stock fused_marlin_moe path). The M block applies only to calls with more than ``min tokens`` tokens (static per
# captured graph size). Why: Marlin reads an expert's weights once per M block of its routes, so with vLLM's
# auto block of 8 a cold expert that gets 64 routes in a prefill chunk is read 8 times over PCIe (measured 3.9 ms vs
# 0.40 ms for 10 such experts, tools/bench_expert_io.py "reread"); one 64-row block reads it once.
_TIER_TUNING: dict = {}


def set_tier_tuning(name: str, bsm: int = 0, blocks_per_sm: int = -1, min_tokens: int = 0) -> None:
    if bsm or blocks_per_sm > 0:
        _TIER_TUNING[name] = (int(bsm), int(blocks_per_sm), int(min_tokens))
    else:
        _TIER_TUNING.pop(name, None)


# ---------------------------------------------------------------------------------------------- workspace
class _MoeWorkspace:
    """Process-wide buffers of the op, sized for ``max_routes`` = max tokens per call x top_k."""

    def __init__(self, max_routes: int, device: torch.device, dtype: torch.dtype):
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (MARLIN_MAX_BLOCKS_PER_SM,
                                                                                marlin_make_workspace_new)
        L, N = LATENT_DIM, EXPERT_FF
        self.max_routes = max_routes
        self.device = device
        self.x_routes = torch.empty((max_routes, L), dtype=dtype, device=device)
        self.c13 = {t: torch.empty(max_routes * max(2 * N, L), dtype=dtype, device=device) for t in ("hot", "cold")}
        self.c2 = {t: torch.empty(max_routes * N, dtype=dtype, device=device) for t in ("hot", "cold")}
        self.locks = {t: marlin_make_workspace_new(device, MARLIN_MAX_BLOCKS_PER_SM) for t in ("hot", "cold")}
        self.side_stream = torch.cuda.Stream(device=device)
        self.all_hot = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=device)

    @property
    def nbytes(self) -> int:
        n = self.x_routes.numel() * self.x_routes.element_size()
        for d in (self.c13, self.c2):
            n += sum(t.numel() * t.element_size() for t in d.values())
        return n


_WS: _MoeWorkspace | None = None


def _workspace(rt, device, dtype) -> _MoeWorkspace:
    global _WS
    if _WS is None:
        tokens = rt.settings.moe_chunk_tokens or rt.max_num_batched_tokens
        _WS = _MoeWorkspace(tokens * rt.top_k, device, dtype)
        rt.log.info("edge0: MoE workspace for %d tokens x %d routes per call: %.0f MB", tokens, rt.top_k,
                    _WS.nbytes / 2**20)
    return _WS


def _keep_route_rows(moe_output, output, topk_ids, expert_map):
    """``moe_sum`` override: keep the per-route rows ([R, 1, K] view of the tier's cache); combine reduces them."""
    return moe_output.view(-1, moe_output.shape[-1])


def _auto_bsm(routes: int) -> int:
    """vLLM's ``fused_marlin_moe`` M-block choice for our calls (topk=1, M scaled to the tier: R / 512 per expert)."""
    for bsm in (8, 16, 32, 48, 64):
        if routes / NUM_EXPERTS / bsm < 0.9:
            return bsm
    return 64


def marlin_tier_direct(tier, expert_map, x_routes, topk_ids, topk_w, ws: _MoeWorkspace, name: str,
                       bsm: int = 0, blocks_per_sm: int = -1) -> torch.Tensor:
    """``fused_marlin_moe`` for one tier, unrolled so the M-block size and the CTA count can be chosen per tier
    (``EDGE0_{HOT,COLD}_BSM`` / ``EDGE0_COLD_BLOCKS_PER_SM``). Same kernels, same arguments otherwise: align ->
    GEMM1 (w13) -> SiLU*up -> GEMM2 (w2, route weight applied). Returns the per-route rows ``[R, L]``."""
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    from vllm.scalar_type import scalar_types
    r, k = x_routes.shape
    n = EXPERT_FF
    bsm = bsm or _auto_bsm(r)
    sorted_ids, expert_ids, ntpp = moe_align_block_size(topk_ids, bsm, NUM_EXPERTS, expert_map,
                                                         ignore_invalid_experts=True)
    c1 = ws.c13[name][:r * 2 * n].view(r, 2 * n)
    c3 = ws.c13[name][:r * k].view(r, k)
    c2 = ws.c2[name][:r * n].view(r, n)
    qt = scalar_types.float4_e2m1f
    ops.moe_wna16_marlin_gemm(x_routes, c1, tier.w13, None, tier.s13, None, tier.g13, None, ws.locks[name],
                              sorted_ids, expert_ids, ntpp, topk_w, moe_block_size=bsm, top_k=1,
                              mul_topk_weights=False, b_q_type=qt, size_m=r, size_n=2 * n, size_k=k,
                              use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False,
                              blocks_per_sm=blocks_per_sm)
    torch.ops._C.silu_and_mul(c2, c1)
    ops.moe_wna16_marlin_gemm(c2, c3, tier.w2, None, tier.s2, None, tier.g2, None, ws.locks[name],
                              sorted_ids, expert_ids, ntpp, topk_w, moe_block_size=bsm, top_k=1,
                              mul_topk_weights=True, b_q_type=qt, size_m=r, size_n=k, size_k=n,
                              use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False,
                              blocks_per_sm=blocks_per_sm)
    return c3


def _marlin_tier(tier, expert_map, x_routes, topk_ids, topk_w, ws: _MoeWorkspace, name: str) -> torch.Tensor:
    tune = _TIER_TUNING.get(name)
    if tune is not None:
        bsm, bps, min_tokens = tune
        if x_routes.shape[0] // TOP_K <= min_tokens:
            bsm = 0
        if bsm or bps > 0:
            return marlin_tier_direct(tier, expert_map, x_routes, topk_ids, topk_w, ws, name, bsm, bps)
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    from vllm.scalar_type import scalar_types
    return fused_marlin_moe(
        hidden_states=x_routes, w1=tier.w13, w2=tier.w2, bias1=None, bias2=None,
        w1_scale=tier.s13, w2_scale=tier.s2, topk_weights=topk_w, topk_ids=topk_ids,
        quant_type_id=scalar_types.float4_e2m1f.id, global_num_experts=NUM_EXPERTS, expert_map=expert_map,
        global_scale1=tier.g13, global_scale2=tier.g2, workspace=ws.locks[name],
        intermediate_cache13=ws.c13[name], intermediate_cache2=ws.c2[name],
        moe_sum=_keep_route_rows, output=x_routes, mul_topk_weights_in_gemm2=True)


# ---------------------------------------------------------------------------------------------- block
class _MoeConfigStub:
    skip_final_all_reduce = False


class _ExpertsStub:
    """What Qwen4Exp's EPLB bookkeeping (``set_moe_parameters``) reads from ``block.experts`` while the Edge0 block is
    swapped in. Not an nn.Module, so it adds no children / parameters."""

    def __init__(self, block):
        self.moe_config = _MoeConfigStub()
        self.layer_name = block.prefix

    def update_expert_map(self):
        pass


class Edge0LatentMoeBlock(nn.Module):
    def __init__(self, vllm_config, prefix: str = "", reduce_results: bool = True) -> None:
        super().__init__()
        rt = runtime.current()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.prefix = prefix
        self.layer_idx = extract_layer_index(prefix)
        self.hidden_size = int(config.hidden_size)
        self.num_experts = int(config.num_experts)
        self.top_k = int(config.num_experts_per_tok)
        if (self.num_experts, self.top_k, int(config.moe_intermediate_size)) != (NUM_EXPERTS, TOP_K, EXPERT_FF):
            raise ValueError(f"{prefix}: Edge0 latent MoE expects 512 experts, top-10, ff 640; got "
                             f"{self.num_experts}, {self.top_k}, {config.moe_intermediate_size}")
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise NotImplementedError("Edge0LatentMoeBlock supports tensor_parallel_size=1 only")
        H, GL = self.hidden_size, LATENT_GROUPS * LATENT_DIM

        self.gate = GateLinear(H, self.num_experts, out_dtype=torch.float32, prefix=f"{prefix}.gate")
        self.shared_expert_gate = ReplicatedLinear(H, 1, bias=False, quant_config=None,
                                                   prefix=f"{prefix}.shared_expert_gate")
        self.shared_expert = Qwen3NextMLP(hidden_size=H, intermediate_size=int(config.shared_expert_intermediate_size),
                                          hidden_act=config.hidden_act, quant_config=quant_config,
                                          reduce_results=False, expert_gate=self.shared_expert_gate,
                                          prefix=f"{prefix}.shared_expert")
        self.latent_down = ReplicatedLinear(H, GL, bias=False, quant_config=quant_config,
                                            prefix=f"{prefix}.latent_down")
        self.latent_up = ReplicatedLinear(GL, H, bias=False, quant_config=quant_config,
                                          prefix=f"{prefix}.latent_up")
        # D/U routing group of every logical expert (labels.bin); loaded from the checkpoint.
        self.register_buffer("expert_group", torch.zeros(self.num_experts, dtype=torch.int32), persistent=True)
        device = self.expert_group.device
        tier_st = None
        if rt.settings.tier:          # EDGE0_TIER: free / spare slots and shared map storage (tier.py)
            from . import tier as _tier
            tier_st = _tier.storage(rt, device)
        self.pool = ExpertPool(self.layer_idx, rt.placements[self.layer_idx], device, rt.settings.cold_alloc,
                               keep_reference=rt.keeps_reference(self.layer_idx),
                               n_stage=rt.settings.prefetch if self.layer_idx > 0 else 0,
                               n_free=tier_st.n_free[self.layer_idx] if tier_st is not None else 0,
                               storage=tier_st)
        self._rt = rt

        # Attributes Qwen4Exp's model code reads from its MoE block while this class is swapped in (DESIGN.md 2.3).
        self.n_routed_experts = self.n_logical_experts = self.num_experts
        self.n_physical_experts = self.n_local_physical_experts = self.num_experts
        self.n_redundant_experts = 0
        self.n_shared_experts = 1
        self.is_fused_shared_expert_enabled = False
        self.replicate_shared_expert = False
        self.is_sequence_parallel = False
        self.experts = _ExpertsStub(self)

        cc = vllm_config.compilation_config
        if prefix in cc.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        cc.static_forward_context[prefix] = self
        _BLOCKS[prefix] = self
        rt.blocks.append(self)

    # ------------------------------------------------------------------------------------------ forward
    def forward(self, hidden_states: torch.Tensor, already_sequence_parallel: bool = False) -> torch.Tensor:
        x = hidden_states.view(-1, self.hidden_size)
        with _nvtx("edge0.gate"):
            logits = self.gate(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        if self._rt.predict is not None or self._rt.prefetch is not None:
            # EDGE0_TRACE=predict: score next-layer pre-gating; EDGE0_PREFETCH: stage the next layer's experts
            torch.ops.vllm.edge0_moe_predict(x, self.expert_group, _encode_layer_name(self.prefix))
        if q8gemv.DU_MASK:
            # EDGE0_DU_MASK=1: route first, then D / U only over the step's used latent groups (one opaque op)
            out = torch.ops.vllm.edge0_latent_moe_du(x, logits, _encode_layer_name(self.prefix))
        else:
            with _nvtx("edge0.D"):
                xl, _ = self.latent_down(x)
            with _nvtx("edge0.moe"):
                z = torch.ops.vllm.edge0_latent_moe(xl, logits, _encode_layer_name(self.prefix))
            with _nvtx("edge0.U"):
                out, _ = self.latent_up(z)
        with _nvtx("edge0.shared"):
            if q8gemv.SHARED_FUSE:     # EDGE0_SHARED_FUSE=1: shared expert + add in 4 kernels (q8gemv)
                out = torch.ops.vllm.edge0_shared_add(x, out, _encode_layer_name(self.prefix))
            else:
                out = out + self.shared_expert(x)
        return out.view(hidden_states.shape)

    def shared_add(self, x: torch.Tensor, routed: torch.Tensor) -> torch.Tensor:
        """The body of ``vllm::edge0_shared_add``: routed + shared_expert(x), fused for calls <= q8gemv.MAX_M."""
        se = self.shared_expert
        gu = getattr(se.gate_up_proj, "b12x_block_weight", None)
        dn = getattr(se.down_proj, "b12x_block_weight", None)
        if not (0 < x.shape[0] <= q8gemv.MAX_M) or gu is None or dn is None:
            return routed + se(x)
        return q8gemv.shared_expert_add(x.contiguous(), routed.contiguous(), gu, dn,
                                        self.shared_expert_gate.weight.view(-1))

    def forward_experts(self, xl: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """The body of ``vllm::edge0_latent_moe``: xl [T, 8*1792], fp32 logits [T, 512] -> z [T, 8*1792]."""
        if xl.shape[0] == 0:
            return torch.empty_like(xl)
        w, ids = self.route(logits)
        return self.experts_routed(xl, ids, w)

    def forward_du(self, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """The body of ``vllm::edge0_latent_moe_du`` (EDGE0_DU_MASK=1): x [T, 2560] -> U(experts(D(x))) [T, 2560].
        Calls of <= q8gemv.MAX_M tokens route first and run D / U with q8gemv over the used latent groups only;
        larger calls (prefill) take the b12x D / U modules unchanged."""
        t = x.shape[0]
        dw = getattr(self.latent_down, "b12x_block_weight", None)
        uw = getattr(self.latent_up, "b12x_block_weight", None)
        if t == 0 or t > q8gemv.MAX_M or dw is None or uw is None:
            xl, _ = self.latent_down(x)
            z = self.forward_experts(xl, logits)
            out, _ = self.latent_up(z)
            return out
        w, ids = self.route(logits)
        ids_c = ids.contiguous()
        with _nvtx("edge0.D"):
            xl = q8gemv.q8_gemv(x.contiguous(), dw.values, dw.metadata, dw.out_features, dw.in_features, ids=ids_c,
                                expert_group=self.expert_group, mask_mode=1)
        z = self.experts_routed(xl, ids, w)
        with _nvtx("edge0.U"):
            return q8gemv.q8_gemv(z, uw.values, uw.metadata, uw.out_features, uw.in_features, ids=ids_c,
                                  expert_group=self.expert_group, mask_mode=2)

    def route(self, logits: torch.Tensor):
        """Top-10 routing of the op, padding rows rerouted, trace / adapt / predict hooks: (w [T, K], ids [T, K])."""
        t = logits.shape[0]
        nv_route = _nvtx("edge0.route")
        nv_route.__enter__()
        w, ids = kernels.route_topk_softmax(logits, self.top_k, renormalize=True)
        rt = self._rt
        if (rt.settings.route_fuse and ids.is_cuda and t * self.top_k <= kernels.ROUTE_POST_MAX_IDS
                and rt.trace is None and rt.adapt is None and rt.predict is None):
            # EDGE0_ROUTE_FUSE: padding reroute + tier count in one kernel (decode-sized calls)
            pad = _padding_mask(t)
            if pad is not None and pad.shape[0] != t:
                pad = None
            pad_ids = self.pool.pad_ids if rt.settings.skip_pad else None
            row = rt.tier.count_row(self.layer_idx, t) if rt.tier is not None else None
            if pad is not None or row is not None:
                w, ids = kernels.route_post(ids, w, pad, pad_ids, row, self.pool.hot_map)
            nv_route.__exit__(None, None, None)
            return w, ids
        if rt.settings.skip_pad and self.pool.pad_ids is not None:
            # Padding rows of a CUDA-graph-padded batch (stale inputs, outputs discarded) would otherwise route like
            # real tokens and pull up to 10 cold experts each over PCIe; send them to hot experts with weight 0.
            pad = _padding_mask(t)
            if pad is not None and pad.shape[0] == t:
                pm = pad.view(-1, 1)
                ids = torch.where(pm, self.pool.pad_ids, ids)
                w = torch.where(pm, torch.zeros_like(w), w)
        if rt.trace is not None:
            rt.trace.record(self.layer_idx, ids)
        if rt.adapt is not None:
            rt.adapt.record(self.layer_idx, ids)
        if rt.tier is not None:
            rt.tier.record(self.layer_idx, ids)
        if rt.predict is not None:
            rt.predict.score(self.layer_idx, ids, self.pool.hot_map)
        nv_route.__exit__(None, None, None)
        return w, ids

    def experts_routed(self, xl: torch.Tensor, ids: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Gather, hot / cold (/ CPU) experts and combine for routed tokens: z [T, 8*1792]."""
        t = xl.shape[0]
        z = torch.empty_like(xl)
        rt = self._rt
        if self.pool.ref is not None and rt.settings.moe_impl == "torch":
            return self._forward_torch(xl, ids, w, z)
        ws = _workspace(rt, xl.device, xl.dtype)
        step = ws.max_routes // self.top_k
        for t0 in range(0, t, step):
            t1 = min(t, t0 + step)
            self._forward_marlin(xl[t0:t1], ids[t0:t1], w[t0:t1], ws, z[t0:t1])
        return z

    def _forward_marlin(self, xl, ids, w, ws: _MoeWorkspace, z_out) -> None:
        latent_experts_marlin(self.pool, self.expert_group, xl, ids, w, ws, z_out, split=self._rt.settings.split,
                              cpu=self._rt.cpu_experts)

    def _forward_torch(self, xl, ids, w, z) -> torch.Tensor:
        """F1: eager per-expert reference from the ModelOpt copy (needs --enforce-eager: host syncs)."""
        return latent_experts_torch(self.pool.ref, self.expert_group, xl, ids, w, z)

    # ------------------------------------------------------------------------------------------ loading
    def load_expert_tensor(self, name: str, tensor: torch.Tensor) -> None:
        self.pool.load_tensor(name, tensor)

    def finalize(self) -> dict:
        stats = self.pool.finalize()
        g = self.expert_group
        if g.device.type != "meta":
            lo, hi = int(g.min()), int(g.max())
            if lo < 0 or hi >= LATENT_GROUPS:
                raise ValueError(f"{self.prefix}.expert_group outside 0..{LATENT_GROUPS - 1}: [{lo}, {hi}]")
        return stats


# ---------------------------------------------------------------------------------------------- op bodies
def latent_experts_marlin(pool: ExpertPool, expert_group: torch.Tensor, xl: torch.Tensor, ids: torch.Tensor,
                          w: torch.Tensor, ws: _MoeWorkspace, z_out: torch.Tensor, split: bool = True,
                          cpu=None) -> None:
    """Steps 2-4 of the op for one chunk: route-as-token gather, hot / cold Marlin calls, deterministic combine.
    ``ids`` / ``w``: [T, K] logical expert ids (int32) / route weights (fp32); writes ``z_out`` [T, 8*1792].
    ``cpu`` (EDGE0_CPU_EXPERTS, cpu_experts.CpuExperts): for calls of its token range, part of the cold experts is
    computed by the CPU pool (dispatch -> gather + doorbell -> Marlin without them -> wait -> combine)."""
    k = ids.shape[1]
    r = ids.shape[0] * k
    if cpu is not None and not (pool.n_cold and cpu.applies(ids.shape[0])):
        cpu = None
    if cpu is not None:
        with _nvtx("edge0.cpu_dispatch"):
            cold_map_cpu = cpu.dispatch(ids, w, pool)
            x_routes = cpu.gather(xl, ids, expert_group, ws.x_routes)
    else:
        with _nvtx("edge0.gather"):
            x_routes = kernels.gather_routes(xl, ids, expert_group, LATENT_DIM, out=ws.x_routes)
    topk_ids = ids.reshape(r, 1)
    topk_w = w.reshape(r, 1)
    y_hot = y_cold = None
    n_stage = getattr(pool, "n_stage", 0)
    if n_stage:
        _join_prefetch(pool)     # the prefetch stream wrote this layer's maps and staging rows
    hot_map, cold_map = pool.maps() if n_stage else (pool.hot_map, pool.cold_map)
    if cpu is not None:
        cold_map = cold_map_cpu          # the cold map minus this call's CPU experts
    has_hot = bool(pool.n_hot or n_stage)
    if has_hot and pool.n_cold and split:
        main = torch.cuda.current_stream()
        side = ws.side_stream
        side.wait_stream(main)
        with torch.cuda.stream(side), _nvtx("edge0.cold"):
            y_cold = _marlin_tier(pool.cold, cold_map, x_routes, topk_ids, topk_w, ws, "cold")
        with _nvtx("edge0.hot"):
            y_hot = _marlin_tier(pool.hot, hot_map, x_routes, topk_ids, topk_w, ws, "hot")
        main.wait_stream(side)
    else:
        if has_hot:
            with _nvtx("edge0.hot"):
                y_hot = _marlin_tier(pool.hot, hot_map, x_routes, topk_ids, topk_w, ws, "hot")
        if pool.n_cold:
            with _nvtx("edge0.cold"):
                y_cold = _marlin_tier(pool.cold, cold_map, x_routes, topk_ids, topk_w, ws, "cold")
    if cpu is not None:
        with _nvtx("edge0.cpu_wait"):
            cpu.wait()
    with _nvtx("edge0.combine"):
        kernels.combine_routes(y_hot, y_cold, ids, expert_group, hot_map, LATENT_GROUPS, LATENT_DIM,
                               z_out.dtype, out=z_out, y_cpu=cpu.y if cpu is not None else None,
                               cpu_sel=cpu.cpu_sel if cpu is not None else None)


def _join_prefetch(pool) -> None:
    """EDGE0_PREFETCH: the hot call reads the staging slots, so wait for this step's staging copy (if any)."""
    s = getattr(pool, "pf_pending", None)
    if s is not None:
        torch.cuda.current_stream().wait_stream(s)
        pool.pf_pending = None


def latent_experts_torch(ref, expert_group: torch.Tensor, xl: torch.Tensor, ids: torch.Tensor, w: torch.Tensor,
                         z_out: torch.Tensor) -> torch.Tensor:
    """F1 reference: per distinct expert, fp32 SwiGLU from the ModelOpt-layout copy, then the same combine."""
    r = ids.numel()
    x_routes = kernels.gather_routes(xl, ids, expert_group, LATENT_DIM)
    y = torch.zeros((r, LATENT_DIM), dtype=torch.float32, device=xl.device)
    flat = ids.reshape(-1).long()
    wf = w.reshape(-1)
    dev = xl.device
    for e in torch.unique(flat).tolist():
        rows = (flat == e).nonzero().squeeze(1)
        out = mn.expert_forward_reference(
            x_routes[rows], ref.w13[e].to(dev), ref.w13_scale[e].to(dev), float(ref.w13_scale_2[e, 0]),
            ref.w2[e].to(dev), ref.w2_scale[e].to(dev), float(ref.w2_scale_2[e]))
        y[rows] = out * wf[rows, None]
    all_hot = torch.zeros(expert_group.numel(), dtype=torch.int32, device=dev)
    return kernels.combine_routes(y, None, ids, expert_group, all_hot, LATENT_GROUPS, LATENT_DIM, z_out.dtype,
                                  out=z_out)


# ---------------------------------------------------------------------------------------------- custom op
def _lookup(layer_name: str) -> Edge0LatentMoeBlock:
    try:
        from vllm.forward_context import get_forward_context, is_forward_context_available
        if is_forward_context_available():
            block = get_forward_context().no_compile_layers.get(layer_name)
            if block is not None:
                return block
    except Exception:  # noqa: BLE001
        pass
    return _BLOCKS[layer_name]


def edge0_latent_moe(xl: torch.Tensor, router_logits: torch.Tensor, layer_name: LayerNameType) -> torch.Tensor:
    return _lookup(_resolve_layer_name(layer_name)).forward_experts(xl, router_logits)


def edge0_latent_moe_fake(xl: torch.Tensor, router_logits: torch.Tensor, layer_name: LayerNameType) -> torch.Tensor:
    return torch.empty_like(xl)


def edge0_latent_moe_du(x: torch.Tensor, router_logits: torch.Tensor, layer_name: LayerNameType) -> torch.Tensor:
    return _lookup(_resolve_layer_name(layer_name)).forward_du(x, router_logits)


def edge0_latent_moe_du_fake(x: torch.Tensor, router_logits: torch.Tensor, layer_name: LayerNameType) -> torch.Tensor:
    return torch.empty_like(x)


def edge0_shared_add(x: torch.Tensor, routed: torch.Tensor, layer_name: LayerNameType) -> torch.Tensor:
    return _lookup(_resolve_layer_name(layer_name)).shared_add(x, routed)


def edge0_shared_add_fake(x: torch.Tensor, routed: torch.Tensor, layer_name: LayerNameType) -> torch.Tensor:
    return torch.empty_like(routed)


def edge0_moe_predict(x: torch.Tensor, expert_group: torch.Tensor, layer_name: LayerNameType) -> None:
    """EDGE0_TRACE=predict: layer l+1's router applied to layer l's MoE input; top-k stashed for scoring at l+1.
    Declared as mutating ``expert_group`` (it does not change it) so the compiler keeps the call."""
    blk = _lookup(_resolve_layer_name(layer_name))
    pr = blk._rt.predict
    if pr is not None:
        pr.stash_from(blk.layer_idx, x)
    pf = blk._rt.prefetch
    if pf is not None:
        pf.predict_and_copy(blk.layer_idx, x)


def edge0_moe_predict_fake(x: torch.Tensor, expert_group: torch.Tensor, layer_name: LayerNameType) -> None:
    return None


def _register_op() -> None:
    try:
        torch.ops.vllm.edge0_latent_moe  # noqa: B018
        return
    except (AttributeError, RuntimeError):
        pass
    direct_register_custom_op(op_name="edge0_latent_moe", op_func=edge0_latent_moe, mutates_args=[],
                              fake_impl=edge0_latent_moe_fake)
    direct_register_custom_op(op_name="edge0_moe_predict", op_func=edge0_moe_predict,
                              mutates_args=["expert_group"], fake_impl=edge0_moe_predict_fake)
    direct_register_custom_op(op_name="edge0_latent_moe_du", op_func=edge0_latent_moe_du, mutates_args=[],
                              fake_impl=edge0_latent_moe_du_fake)
    direct_register_custom_op(op_name="edge0_shared_add", op_func=edge0_shared_add, mutates_args=[],
                              fake_impl=edge0_shared_add_fake)


_register_op()
