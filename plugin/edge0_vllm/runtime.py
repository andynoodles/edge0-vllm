"""Per-process Edge0 state shared by the model and its 48 latent MoE blocks: settings, checkpoint spec, placement,
logger and (after loading) the expert trace. ``setup`` is called by ``Edge0ForCausalLM.__init__`` before the
decoder layers are built; the blocks read it with ``current()``.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from .config import Edge0CheckpointSpec, Edge0Settings, get_settings
from .placement import LayerPlacement, placement_digest, resolve_placement


def get_logger():
    """vLLM only wires handlers for the ``vllm`` logger tree; also mirror to EDGE0_LOG_DIR (TidalMoE-style)."""
    from vllm.logger import init_logger
    log = init_logger("vllm.edge0")
    if not getattr(log, "_edge0_file", False):
        log._edge0_file = True
        try:
            d = get_settings().log_dir
            os.makedirs(d, exist_ok=True)
            fh = logging.FileHandler(os.path.join(d, f"edge0-{os.getpid()}.log"))
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            log.addHandler(fh)
        except (OSError, ValueError):
            pass
    return log


@dataclass
class Edge0Runtime:
    settings: Edge0Settings
    spec: Edge0CheckpointSpec
    model_dir: str
    num_layers: int
    num_experts: int
    top_k: int
    max_num_batched_tokens: int
    placements: dict[int, LayerPlacement]
    placement_desc: str
    log: object
    trace: object | None = None
    adapt: object | None = None
    predict: object | None = None
    prefetch: object | None = None
    cpu_experts: object | None = None        # EDGE0_CPU_EXPERTS: cpu_experts.CpuExperts
    tier: object | None = None               # EDGE0_TIER: tier.ReactiveTier
    tier_storage: object | None = None       # EDGE0_TIER: tier.TierStorage (maps + global scales of all layers)
    max_num_seqs: int = 16
    num_spec_tokens: int = 0                 # speculative_config.num_speculative_tokens (0 = no speculative decoding)
    spec_method: str = ""
    blocks: list = field(default_factory=list)

    @property
    def max_decode_tokens(self) -> int:
        """Largest pure-decode MoE call: every running request verifies 1 + k tokens."""
        return self.max_num_seqs * (1 + self.num_spec_tokens)

    def keeps_reference(self, layer: int) -> bool:
        return self.settings.moe_impl == "torch" and layer < self.settings.ref_layers

    @property
    def hot_slots(self) -> int:
        return sum(p.n_hot for p in self.placements.values())


_CURRENT: Edge0Runtime | None = None


def setup(vllm_config) -> Edge0Runtime:
    global _CURRENT
    settings = get_settings()
    cfg = vllm_config.model_config.hf_text_config
    spec = Edge0CheckpointSpec.from_hf_config(cfg)
    model_dir = vllm_config.model_config.model
    num_layers = int(cfg.num_hidden_layers)
    num_experts = int(cfg.num_experts)
    placements, desc = resolve_placement(settings.placement, settings.expert_vram_gib, model_dir,
                                         spec.profile_file, num_layers, num_experts)
    log = get_logger()
    if settings.tier:      # EDGE0_TIER: free VRAM slots per layer come out of the plan's hot budget
        from .tier import demote
        placements, moved = demote(placements, settings.tier_free)
        desc += f", EDGE0_TIER: {moved} hot experts demoted to make {settings.tier_free} free slots per layer"
    sc = getattr(vllm_config, "speculative_config", None)
    _CURRENT = Edge0Runtime(
        num_spec_tokens=int(getattr(sc, "num_speculative_tokens", 0) or 0) if sc is not None else 0,
        spec_method=str(getattr(sc, "method", "") or "") if sc is not None else "",
        settings=settings, spec=spec, model_dir=model_dir, num_layers=num_layers, num_experts=num_experts,
        top_k=int(cfg.num_experts_per_tok),
        max_num_batched_tokens=int(vllm_config.scheduler_config.max_num_batched_tokens),
        max_num_seqs=int(getattr(vllm_config.scheduler_config, "max_num_seqs", 16) or 16),
        placements=placements, placement_desc=desc, log=log)
    log.info("edge0: settings %s", settings.describe())
    log.info("edge0: placement %s, %d hot slots (%.2f GiB VRAM), digest %s, per-layer n_hot min/max %d/%d",
             desc, _CURRENT.hot_slots, _CURRENT.hot_slots * 1_935_360 / 2**30, placement_digest(placements),
             min(p.n_hot for p in placements.values()), max(p.n_hot for p in placements.values()))
    return _CURRENT


def current() -> Edge0Runtime:
    if _CURRENT is None:
        raise RuntimeError("Edge0 runtime not set up: Edge0LatentMoeBlock must be built by Edge0ForCausalLM")
    return _CURRENT
