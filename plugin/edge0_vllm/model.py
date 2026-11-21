"""``Edge0ForCausalLM`` = vLLM's ``Qwen4ExpForCausalLM`` with one change: the MoE block of the main decoder layers is
``Edge0LatentMoeBlock`` (DESIGN.md D1, 2.3). Hyper-connections, GDN, QSA + indexer, PLE, logits, MTP hooks, hybrid
state, scheduler, CUDA graphs and torch.compile are vLLM's own code.
"""
import contextlib
import os
import re
from collections.abc import Iterable

import torch

import vllm.models.qwen4_exp.nvidia.model as _qwen4_exp_model
from vllm.model_executor.models.interfaces import SupportsEagle3
from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpForCausalLM

from . import hostmem, loadcheck, ple, runtime
from .config import SLOT_BYTES
from .latent_moe import Edge0LatentMoeBlock

# Checkpoint names of the stacked routed-expert tensors of the main layers (never the mtp.* ones).
_EXPERT_RE = re.compile(r"^(?:model\.language_model\.|model\.)layers\.(\d+)\.mlp\.experts\.([A-Za-z0-9_]+)$")


@contextlib.contextmanager
def _edge0_moe_block():
    """``Qwen4ExpDecoderLayer.__init__``, ``Qwen4ExpModel.__init__`` and ``set_moe_parameters`` look the MoE class
    up in their module globals at call time; swap it while the target model is built (the MTP drafter, built later,
    keeps the stock block)."""
    orig = _qwen4_exp_model.Qwen4ExpSparseMoeBlock
    _qwen4_exp_model.Qwen4ExpSparseMoeBlock = Edge0LatentMoeBlock
    try:
        yield
    finally:
        _qwen4_exp_model.Qwen4ExpSparseMoeBlock = orig


def _init_nccl_early(log) -> None:
    """Create the default process group's NCCL communicator now, while the GPU is still empty.

    With ``TORCH_DIST_USE_NCCL2=0`` (serve.sh; torch 2.15's default nccl2 backend reserves ~3.4 GiB of VRAM at init)
    the stock ProcessGroupNCCL creates its communicator lazily at the first collective. vLLM's first collective is
    the ``barrier()`` in ``_drop_checkpoint_page_cache`` right after the weights are loaded, when the hot expert pools
    have filled the GPU, and NCCL then fails with "Failed to CUDA calloc async 4 bytes". A barrier here costs
    ~0.67 GiB (measured) instead."""
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()) or not torch.cuda.is_available():
        return
    try:
        if dist.get_backend() != "nccl":
            return
    except Exception:  # noqa: BLE001
        return
    free0 = torch.cuda.mem_get_info()[0]
    dist.barrier(device_ids=[torch.cuda.current_device()])
    torch.cuda.synchronize()
    log.info("edge0: NCCL communicator created before weight allocation (%.2f GiB)",
             (free0 - torch.cuda.mem_get_info()[0]) / 2**30)


class Edge0ForCausalLM(Qwen4ExpForCausalLM, SupportsEagle3):
    """``SupportsEagle3`` serves only the DFlash2 drafter (``--speculative-config '{"method": "dflash", ...}'``):
    vLLM's runner then sets auxiliary hidden-state layers (``set_aux_hidden_state_layers``) and expects ``forward``
    to return ``(hidden_states, [aux, ...])``. Without such a drafter nothing changes."""

    # Aux layer id L (vLLM's EAGLE-3 convention = DFlash ``target_layer_ids`` + 1) is the input of layer L, i.e. the
    # *completed* output of layer L-1. Qwen4Exp defers each layer's hyper-connection combine into the next layer, so
    # the capture runs that combine once more (``mlp_hyper_connection.combine``, out of place) and averages the 4 HC
    # streams: what the model card's SGLang patch (patches/qwen38-dflash2-capture.patch) feeds DFlash2.
    _edge0_aux_layers: tuple[int, ...] = ()

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        rt = runtime.setup(vllm_config)
        _init_nccl_early(rt.log)
        ple.install_patch(rt.settings, rt.log)
        from . import q8gemv
        if q8gemv.Q8_GEMV:
            q8gemv.install_linear_patch(rt.log)
        if q8gemv.DU_MASK:
            rt.log.info("edge0: latent D / U skip unused routing groups for calls <= %d tokens (EDGE0_DU_MASK=1)",
                        q8gemv.MAX_M)
        with _edge0_moe_block():
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Re-run outside the swap: no stock FusedMoE layer -> num_moe_layers = 0 (no EPLB / routed-expert capture).
        self.set_moe_parameters(self.model.layers)
        self.edge0 = rt
        self._edge0_post_done = False
        self._edge0_aux_out: list | None = None
        self._edge0_aux_hooks: list = []
        if rt.spec_method == "mtp" and os.environ.get("EDGE0_MTP_SHARE_VOCAB", "1") != "0":
            from . import draft
            draft.set_target(self)
            draft.install(rt.log)
        elif rt.spec_method in ("dflash", "dflash2") and os.environ.get("EDGE0_DFLASH_SHARE_VOCAB", "0").strip() == "1":
            from . import draft
            draft.install_dflash(rt.log)
        pools = [b.pool for b in rt.blocks]
        rt.log.info("edge0: built %d latent MoE layers: %d hot slots in VRAM (%.2f GiB), %d cold slots in host "
                    "memory (%.2f GiB, %s), dense=%s moe_impl=%s split=%s", len(pools),
                    sum(p.n_hot for p in pools), sum(p.vram_bytes for p in pools) / 2**30,
                    sum(p.n_cold for p in pools), sum(p.host_bytes for p in pools) / 2**30,
                    rt.settings.cold_alloc, rt.settings.dense, rt.settings.moe_impl, rt.settings.split)
        from . import nvtx      # EDGE0_NVTX (off by default): Nsight Systems ranges, docs/PROFILE.md
        if nvtx.ENABLED:
            nvtx.install_host_ranges(rt.log)
            nvtx.install_module_ranges(self, rt.log)

    # ------------------------------------------------------------------------------------------ loading
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Stacked expert tensors go to their layer's pool; everything else to vLLM's Qwen4Exp loader (which also
        drops ``mtp.*``)."""
        expert_names: set[str] = set()
        layers = self.model.layers
        # EDGE0_VERIFY_LOAD=1 only (off by default): Q8 invariant, dtype-warning summary, param-vs-checkpoint check.
        audit = loadcheck.LoadAudit(self.edge0) if self.edge0.settings.verify_load else None

        def non_expert():
            for name, tensor in weights:
                m = _EXPERT_RE.match(name)
                if m is None:
                    if audit is not None:
                        audit.see(name, tensor)
                    yield name, tensor
                    continue
                layer = int(m.group(1))
                mlp = getattr(layers[layer], "mlp", None) if layer < len(layers) else None
                if not isinstance(mlp, Edge0LatentMoeBlock):
                    raise ValueError(f"{name}: layer {layer} has no Edge0 latent MoE block")
                mlp.load_expert_tensor(m.group(2), tensor)
                expert_names.add(name)

        with audit if audit is not None else contextlib.nullcontext():
            loaded = set(super().load_weights(non_expert()) or ())
        if audit is not None:
            audit.finish()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()   # drop the Q8 dequant / repack transients before vLLM's next allocations
        if self.edge0.settings.verify_load:
            # Before any process_weights_after_loading transform (low-latency GEMM plans, embed -> host).
            loadcheck.verify_loaded(self, self.edge0.model_dir, self.edge0.log)
        return loaded | expert_names

    def process_weights_after_loading(self) -> None:
        """Model-level post-load hook (vLLM ``model_loader/utils.py``), before memory profiling and graph capture."""
        if self._edge0_post_done:
            return
        self._edge0_post_done = True
        rt = self.edge0
        stats = [b.finalize() for b in rt.blocks]
        mism = sum(s["gate_up_scale2_mismatch"] for s in stats)
        if mism:
            rt.log.warning("edge0: %d experts have w13_weight_scale_2 gate != up (contract says equal); gate used",
                           mism)
        if stats and stats[0]["factor13"] is not None:
            f13 = sorted({s["factor13"] for s in stats})
            f2 = sorted({s["factor2"] for s in stats})
            rt.log.info("edge0: Marlin NVFP4 scale factors w13 %s, w2 %s", f13, f2)
        if rt.settings.embed_host:
            self._embedding_to_host()
        import os
        if os.environ.get("EDGE0_LMHEAD", "bf16").strip().lower() == "q8":     # lmhead_q8.py (single-stream)
            from . import lmhead_q8
            lmhead_q8.convert(self, rt.log)
        if os.environ.get("EDGE0_QSA_Q8", "0").strip() == "1":     # QSA q/k/v rows as Q8_0 (lmhead_q8.py)
            from . import lmhead_q8
            lmhead_q8.convert_qsa_qkv(self, rt.log)
        if rt.settings.trace - {"predict"}:
            from .trace import ExpertTrace
            dev = rt.blocks[0].expert_group.device if rt.blocks else torch.device("cpu")
            rt.trace = ExpertTrace(rt.settings.trace, rt.num_layers, rt.num_experts, rt.top_k, dev,
                                   rt.settings.trace_flush_s, rt.settings.trace_ring_tokens, rt.settings.trace_dir,
                                   rt.log)
            rt.trace.stats_max_tokens = max(rt.max_decode_tokens, 16)
            for b in rt.blocks:
                rt.trace.register_pool(b.layer_idx, b.pool.hot_map)
        if "predict" in rt.settings.trace and rt.blocks:
            from .trace import PredictTrace
            dev = rt.blocks[0].expert_group.device
            rt.predict = PredictTrace(rt.blocks, max(rt.max_num_seqs, 64), dev, rt.settings.trace_flush_s, rt.log,
                                      rt.settings.trace_dir)
        if rt.settings.prefetch and rt.blocks:
            from .prefetch import Prefetcher
            dev = rt.blocks[0].expert_group.device
            rt.prefetch = Prefetcher(rt.blocks, rt.settings.prefetch, rt.settings.prefetch_min_p,
                                     rt.settings.prefetch_ctas, dev, rt.log, keep=rt.settings.prefetch_keep,
                                     max_tokens=max(rt.max_num_seqs, 64))
        from . import latent_moe
        latent_moe.set_tier_tuning("hot", rt.settings.hot_bsm, -1, rt.settings.bsm_min_tokens)
        latent_moe.set_tier_tuning("cold", rt.settings.cold_bsm, rt.settings.cold_blocks_per_sm,
                                   rt.settings.bsm_min_tokens)
        if rt.settings.tier and rt.blocks:
            from .tier import ReactiveTier
            dev = rt.blocks[0].expert_group.device
            rt.tier = ReactiveTier(rt.blocks, rt.settings, dev, rt.log, max(rt.max_decode_tokens, 16))
            if rt.settings.adapt:
                rt.log.info("edge0: EDGE0_TIER=1 supersedes EDGE0_ADAPT=1 (the epoch cache is off)")
        elif rt.settings.adapt and rt.blocks and any(b.pool.n_hot and b.pool.n_cold for b in rt.blocks):
            from .adaptive import AdaptiveCache
            dev = rt.blocks[0].expert_group.device
            # Count every pure-decode call, speculative verify steps included (max_num_seqs x (1 + k) rows).
            rt.adapt = AdaptiveCache(rt.blocks, dev, rt.log, max_tokens_default=max(rt.max_decode_tokens, 16))
        if rt.settings.cpu_experts and rt.blocks:
            from .cpu_experts import CpuExperts
            rt.cpu_experts = CpuExperts(rt.settings, rt.blocks[0].expert_group.device, rt.log)
            for b in rt.blocks:
                rt.cpu_experts.register(b.layer_idx, b.pool)
        if torch.cuda.is_available() and any(p.is_cuda for p in self.parameters()):
            free, total = torch.cuda.mem_get_info()
            rt.log.info("edge0: after load: torch allocated %.2f GiB, GPU free %.2f / %.2f GiB; hot experts %.2f GiB; "
                        "cold experts %.2f GiB pinned host", torch.cuda.memory_allocated() / 2**30, free / 2**30,
                        total / 2**30, rt.hot_slots * SLOT_BYTES / 2**30,
                        sum(b.pool.host_bytes for b in rt.blocks) / 2**30)

    def _embedding_to_host(self) -> None:
        """``model.embed_tokens.weight`` -> pinned mapped host memory (gather-only: ~T rows per step), freeing
        1.18 GiB of VRAM (TidalMoE ``MOE_HOST_PARAMS``)."""
        rt = self.edge0
        w = self.model.embed_tokens.weight
        if not w.is_cuda or hostmem.is_host_memory(w):
            return
        nbytes = w.numel() * w.element_size()
        buf = hostmem.alloc(nbytes, rt.settings.cold_alloc, w.device)
        new = buf.dev.view(w.dtype).view(w.shape)
        new.copy_(w.data)
        torch.cuda.synchronize()
        w.data = new
        torch.cuda.empty_cache()
        rt.log.info("edge0: embed_tokens %s -> pinned host memory (%.2f GiB)", tuple(w.shape), nbytes / 2**30)

    # ------------------------------------------------------------------------------------------ DFlash2 features
    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return (6, 16, 26, 36, 46)        # DFlash2 target_layer_ids [5, 15, 25, 35, 45] + 1

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        layers = tuple(sorted({int(x) for x in layers}))
        # EDGE0_AUX_OFFSET (default 0) captures the output of layer L-1+offset instead (semantics check only).
        off = int(os.environ.get("EDGE0_AUX_OFFSET", "0") or 0)
        for h in self._edge0_aux_hooks:
            h.remove()
        self._edge0_aux_hooks = []
        hc, hidden = int(self.config.hc_count), int(self.config.hidden_size)
        for L in layers:
            src = L - 1 + off
            if not 0 <= src < len(self.model.layers):
                raise ValueError(f"aux hidden state layer {L} (offset {off}) is outside the model's layers")
            self._edge0_aux_hooks.append(self.model.layers[src].register_forward_hook(self._make_aux_hook(hc, hidden)))
        self._edge0_aux_layers = layers
        self.edge0.log.info("edge0: drafter features = completed outputs of layers %s (aux ids %s), mean over %d HC "
                            "streams", [L - 1 + off for L in layers], list(layers), hc)

    def _make_aux_hook(self, hc: int, hidden: int):
        def hook(module, args, output):
            out = self._edge0_aux_out
            if out is None:
                return None
            hs, block_out, injection = output
            full = hs if block_out is None else module.mlp_hyper_connection.combine(hs, block_out, injection)
            out.append(full.unflatten(-1, (hc, hidden)).mean(dim=-2))
            return None
        return hook

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        if not self._edge0_aux_layers:
            return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
        self._edge0_aux_out = []
        try:
            hidden = super().forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
            aux = self._edge0_aux_out
        finally:
            self._edge0_aux_out = None
        if len(aux) != len(self._edge0_aux_layers):
            raise RuntimeError(f"edge0: captured {len(aux)} aux hidden states, expected {len(self._edge0_aux_layers)}")
        return hidden, aux

    # ------------------------------------------------------------------------------------------ per step
    def compute_logits(self, hidden_states: torch.Tensor):
        trace = self.edge0.trace
        if trace is not None:
            trace.maybe_flush()
        adapt = self.edge0.adapt
        if adapt is not None:
            adapt.on_step()
        tier = self.edge0.tier
        if tier is not None:
            tier.on_step()
        if self.edge0.predict is not None:
            self.edge0.predict.maybe_flush()
        cx = self.edge0.cpu_experts
        if cx is not None:
            cx.maybe_log()
        return super().compute_logits(hidden_states)
