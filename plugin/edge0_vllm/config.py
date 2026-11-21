"""Edge0 runtime settings: the ``EDGE0_*`` environment knobs (DESIGN.md 2.9) and the checkpoint's ``edge0`` block.

Everything is read once per process, validated, and logged by the model on construction. Pure Python (no torch
import), so the settings can be unit-tested anywhere.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------------------------- model geometry
NUM_EXPERTS = 512
TOP_K = 10
HIDDEN = 2560
LATENT_DIM = 1792          # latent width of one routing group (expert input/output width)
LATENT_GROUPS = 8          # D/U projection groups per layer
EXPERT_FF = 640            # expert intermediate size
NVFP4_GROUP = 16           # inputs per E4M3 block scale

# Marlin layout of one expert ("slot"), DESIGN.md 2.6.
W13_SHAPE = (LATENT_DIM // 16, 2 * EXPERT_FF * 2)        # int32 [112, 2560]
W2_SHAPE = (EXPERT_FF // 16, LATENT_DIM * 2)             # int32 [40, 3584]
S13_SHAPE = (LATENT_DIM // NVFP4_GROUP, 2 * EXPERT_FF)   # fp8   [112, 1280]
S2_SHAPE = (EXPERT_FF // NVFP4_GROUP, LATENT_DIM)        # fp8   [40, 1792]
W13_BYTES = W13_SHAPE[0] * W13_SHAPE[1] * 4              # 1,146,880
W2_BYTES = W2_SHAPE[0] * W2_SHAPE[1] * 4                 # 573,440
S13_BYTES = S13_SHAPE[0] * S13_SHAPE[1]                  # 143,360
S2_BYTES = S2_SHAPE[0] * S2_SHAPE[1]                     # 71,680
SLOT_BYTES = W13_BYTES + W2_BYTES + S13_BYTES + S2_BYTES  # 1,935,360 B = 1.846 MiB
assert SLOT_BYTES == 1_935_360

# ModelOpt NVFP4 checkpoint layout of one layer's stacked experts (DESIGN.md 1.5), name -> (dtype, per-expert shape).
EXPERT_TENSORS = {
    "w13_weight": ("uint8", (2 * EXPERT_FF, LATENT_DIM // 2)),
    "w13_weight_scale": ("float8_e4m3fn", (2 * EXPERT_FF, LATENT_DIM // NVFP4_GROUP)),
    "w13_weight_scale_2": ("float32", (2,)),
    "w2_weight": ("uint8", (LATENT_DIM, EXPERT_FF // 2)),
    "w2_weight_scale": ("float8_e4m3fn", (LATENT_DIM, EXPERT_FF // NVFP4_GROUP)),
    "w2_weight_scale_2": ("float32", ()),
}

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off", "")


def hot_slots_for_gib(gib: float) -> int:
    """Number of expert slots that fit in ``gib`` GiB of VRAM (H in DESIGN.md 2.9)."""
    return int(float(gib) * (1 << 30) // SLOT_BYTES)


def _bool(env, name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(f"{name}={raw!r}: expected 0/1")


def _choice(env, name: str, default: str, choices: tuple[str, ...]) -> str:
    v = env.get(name, default).strip().lower() or default
    if v not in choices:
        raise ValueError(f"{name}={v!r}: expected one of {choices}")
    return v


@dataclass(frozen=True)
class Edge0Settings:
    """Resolved EDGE0_* knobs. Phase-3 knobs that are not implemented yet are rejected, never silently ignored."""

    dense: str = "bf16"                      # EDGE0_DENSE: bf16 (dequantize Q8_0 at load) | q8 (b12x native)
    expert_vram_gib: float | None = None     # EDGE0_EXPERT_VRAM_GIB: VRAM for hot expert slots; 0 = F0 (all cold)
    placement: str = "profile"               # EDGE0_PLACEMENT: profile | identity | plan:<json>
    cold_alloc: str = "host_register"        # EDGE0_COLD_ALLOC: host_register | host_alloc | torch_pinned
    split: bool = True                       # EDGE0_SPLIT: hot and cold Marlin calls on two streams
    moe_impl: str = "marlin"                 # EDGE0_MOE_IMPL: marlin | torch (F1 reference, eager only)
    ref_layers: int = 4                      # EDGE0_REF_LAYERS: layers that keep a ModelOpt copy for F1
    embed_host: bool = True                  # EDGE0_EMBED_HOST: embed_tokens in pinned mapped host memory
    trace: frozenset = field(default_factory=frozenset)   # EDGE0_TRACE: {"bw", "routes"}
    trace_flush_s: float = 10.0              # EDGE0_TRACE_FLUSH_S
    trace_ring_tokens: int = 65536           # EDGE0_TRACE_RING_TOKENS (routes ring per layer)
    trace_dir: str = "/root/.cache/edge0-vllm/traces"
    log_dir: str = "/root/.cache/edge0-vllm/logs"
    ple_io: str = "vllm"                     # EDGE0_PLE_IO: vllm (serial preadv) | threads
    ple_threads: int = 16                    # EDGE0_PLE_THREADS
    moe_chunk_tokens: int = 0                # EDGE0_MOE_CHUNK_TOKENS: 0 = max_num_batched_tokens (one call per tier)
    verify_load: bool = False                # EDGE0_VERIFY_LOAD: compare loaded params with a numpy decode (loadcheck)
    adapt: bool = False                      # EDGE0_ADAPT: adaptive hot/cold swaps (adaptive.py; EDGE0_ADAPT_* knobs)
    hot_bsm: int = 0                         # EDGE0_HOT_BSM: Marlin M-block of the hot call (0 = vLLM's choice)
    cold_bsm: int = 0                        # EDGE0_COLD_BSM: Marlin M-block of the cold call (0 = vLLM's choice)
    cold_blocks_per_sm: int = -1             # EDGE0_COLD_BLOCKS_PER_SM: CTA cap of the cold call (-1 = Marlin's)
    bsm_min_tokens: int = 0                  # EDGE0_BSM_MIN_TOKENS: the BSM overrides apply to calls above this size
    prefetch: int = 0                        # EDGE0_PREFETCH: staging slots per layer for next-layer prefetch (0 = off)
    prefetch_min_p: float = 0.02             # EDGE0_PREFETCH_MIN_P: stage only experts with predicted prob >= this
    prefetch_ctas: int = 32                  # EDGE0_PREFETCH_CTAS: CTAs of the staging copy kernel
    prefetch_keep: bool = True               # EDGE0_PREFETCH_KEEP: staged experts stay resident (LRU-like) across steps
    skip_pad: bool = True                    # EDGE0_SKIP_PAD: padding rows of graph-padded batches route to hot experts
    route_fuse: bool = False                 # EDGE0_ROUTE_FUSE: padding reroute + tier count of a decode call in one kernel
    # EDGE0_CPU_*: CPU computation of part of each call's cold experts (cpu_experts.py, docs/CPU-EXPERTS.md)
    cpu_experts: bool = False                # EDGE0_CPU_EXPERTS
    cpu_threads: int = 8                     # EDGE0_CPU_THREADS: pinned worker threads
    cpu_cores: str = ""                      # EDGE0_CPU_CORES: e.g. "2-13" (empty = physical cores 2..)
    cpu_frac: str = "0.55"                   # EDGE0_CPU_FRAC: share of a call's cold experts on the CPU, or "auto"
    cpu_frac_start: float = 0.5              # EDGE0_CPU_FRAC_START: first value of the auto split
    cpu_min_tokens: int = 2                  # EDGE0_CPU_MIN_TOKENS / _MAX_TOKENS: call sizes that use the CPU
    cpu_max_tokens: int = 256
    cpu_min_cold: int = 4                    # EDGE0_CPU_MIN_COLD: fewer cold experts in a call -> GPU only
    cpu_max_routes: int = 8                  # EDGE0_CPU_MAX_ROUTES: experts with more routes stay on the GPU
    cpu_max_experts: int = 512               # EDGE0_CPU_MAX_EXPERTS: cap per call
    cpu_spin_ms: float = 200.0               # EDGE0_CPU_SPIN_MS: workers busy-poll this long after the last job
    cpu_timeout_ms: float = 2000.0           # EDGE0_CPU_TIMEOUT_MS: the GPU gives up waiting (logged, never hangs)
    cpu_g1_pairs: int = 1                    # EDGE0_CPU_G1_PAIRS: (gate, up) n-tile pairs per CPU task (1/2/5/10)
    cpu_g2_tiles: int = 2                    # EDGE0_CPU_G2_TILES: w2 n-tiles per CPU task (1/2/4/7/14/28)
    # EDGE0_TIER*: request-reactive expert tier (tier.py, TUNING-LOG "Single-stream"); supersedes EDGE0_ADAPT
    tier: bool = False                       # EDGE0_TIER
    tier_free: int = 4                       # EDGE0_TIER_FREE: free VRAM slots per layer (taken from its hot budget)
    tier_swaps: int = 48                     # EDGE0_TIER_SWAPS: swaps per step, all layers (0 = count only)
    tier_decay: float = 0.915                # EDGE0_TIER_DECAY: usage decay per step (Strata: 0.7 per 4 windows)
    tier_gain: float = 1.5                   # EDGE0_TIER_GAIN: swap while candidate usage >= victim usage + gain
    tier_min_use: float = 2.0                # EDGE0_TIER_MIN_USE: candidates need at least this decayed usage
    tier_max_tokens: int = 0                 # EDGE0_TIER_MAX_TOKENS: count calls up to this many tokens (0 = decode)
    tier_log_s: float = 10.0                 # EDGE0_TIER_LOG_S: statistics line every N seconds
    tier_copy: str = "cudart"                # EDGE0_TIER_COPY: cudart (raw cudaMemcpyAsync) | torch (copy_) | batch
    tier_early: bool = False                 # EDGE0_TIER_EARLY: decide at the next step's staging, flip before its launch
    tier_shadow: bool = False                # EDGE0_TIER_SHADOW (default = EDGE0_TIER_EARLY): a swapped-in expert keeps
                                             # its host slot, so its eviction needs no D2H write-back
    tier_spare: int = 0                      # EDGE0_TIER_SPARE: spare host slots per layer beyond EDGE0_TIER_FREE
    tier_wb_cost: float = 0.0                # EDGE0_TIER_WB_COST: usage added to a victim without a shadow copy
    tier_gpu_decide: bool = False            # EDGE0_TIER_GPU_DECIDE: with EARLY, decide on the GPU (measured slower)

    @classmethod
    def from_env(cls, env=None) -> "Edge0Settings":
        env = os.environ if env is None else env
        gib_raw = env.get("EDGE0_EXPERT_VRAM_GIB", "").strip()
        gib = float(gib_raw) if gib_raw else None
        if gib is not None and gib < 0:
            raise ValueError(f"EDGE0_EXPERT_VRAM_GIB={gib_raw!r} must be >= 0")
        placement = env.get("EDGE0_PLACEMENT", "profile").strip() or "profile"
        if placement not in ("profile", "identity") and not placement.startswith("plan:"):
            raise ValueError(f"EDGE0_PLACEMENT={placement!r}: expected profile | identity | plan:<path>")
        cold_alloc = env.get("EDGE0_COLD_ALLOC", "host_register").strip().lower() or "host_register"
        if cold_alloc == "vmm_host_numa":
            raise NotImplementedError("EDGE0_COLD_ALLOC=vmm_host_numa is phase 3 (not implemented yet)")
        if cold_alloc not in ("host_register", "host_alloc", "torch_pinned"):
            raise ValueError(f"EDGE0_COLD_ALLOC={cold_alloc!r}: expected host_register | host_alloc | torch_pinned")
        trace_raw = env.get("EDGE0_TRACE", "0").strip().lower()
        trace = frozenset() if trace_raw in _FALSE else frozenset(t.strip() for t in trace_raw.split(",") if t.strip())
        bad = trace - {"bw", "routes", "predict"}
        if bad:
            raise ValueError(f"EDGE0_TRACE={trace_raw!r}: unknown {sorted(bad)} (expected bw, routes, predict)")
        # Phase-3 levers: refuse instead of silently ignoring them.
        if int(env.get("EDGE0_STREAM_MIN_TOKENS", "0") or 0) > 0:
            raise NotImplementedError("EDGE0_STREAM_MIN_TOKENS (prefill streaming ring) is phase 3")
        if (env.get("EDGE0_COLD_SCALES", "host").strip().lower() or "host") != "host":
            raise NotImplementedError("EDGE0_COLD_SCALES=vram is phase 3")
        s = cls(
            dense=_choice(env, "EDGE0_DENSE", "bf16", ("bf16", "q8")),
            expert_vram_gib=gib,
            placement=placement,
            cold_alloc=cold_alloc,
            split=_bool(env, "EDGE0_SPLIT", True),
            moe_impl=_choice(env, "EDGE0_MOE_IMPL", "marlin", ("marlin", "torch")),
            ref_layers=int(env.get("EDGE0_REF_LAYERS", "4") or 4),
            embed_host=_bool(env, "EDGE0_EMBED_HOST", True),
            trace=trace,
            trace_flush_s=float(env.get("EDGE0_TRACE_FLUSH_S", "10") or 10),
            trace_ring_tokens=int(env.get("EDGE0_TRACE_RING_TOKENS", "65536") or 65536),
            trace_dir=env.get("EDGE0_TRACE_DIR", "/root/.cache/edge0-vllm/traces"),
            log_dir=env.get("EDGE0_LOG_DIR", "/root/.cache/edge0-vllm/logs"),
            ple_io=_choice(env, "EDGE0_PLE_IO", "vllm", ("vllm", "threads")),
            ple_threads=int(env.get("EDGE0_PLE_THREADS", "16") or 16),
            moe_chunk_tokens=int(env.get("EDGE0_MOE_CHUNK_TOKENS", "0") or 0),
            verify_load=_bool(env, "EDGE0_VERIFY_LOAD", False),
            adapt=_bool(env, "EDGE0_ADAPT", False),
            hot_bsm=int(env.get("EDGE0_HOT_BSM", "0") or 0),
            cold_bsm=int(env.get("EDGE0_COLD_BSM", "0") or 0),
            cold_blocks_per_sm=int(env.get("EDGE0_COLD_BLOCKS_PER_SM", "-1") or -1),
            bsm_min_tokens=int(env.get("EDGE0_BSM_MIN_TOKENS", "0") or 0),
            prefetch=int(env.get("EDGE0_PREFETCH", "0") or 0),
            prefetch_min_p=float(env.get("EDGE0_PREFETCH_MIN_P", "0.02") or 0.02),
            prefetch_ctas=int(env.get("EDGE0_PREFETCH_CTAS", "32") or 32),
            prefetch_keep=_bool(env, "EDGE0_PREFETCH_KEEP", True),
            skip_pad=_bool(env, "EDGE0_SKIP_PAD", True),
            route_fuse=_bool(env, "EDGE0_ROUTE_FUSE", False),
            cpu_experts=_bool(env, "EDGE0_CPU_EXPERTS", False),
            cpu_threads=int(env.get("EDGE0_CPU_THREADS", "8") or 8),
            cpu_cores=env.get("EDGE0_CPU_CORES", "").strip(),
            cpu_frac=(env.get("EDGE0_CPU_FRAC", "0.55").strip().lower() or "0.55"),
            cpu_frac_start=float(env.get("EDGE0_CPU_FRAC_START", "0.5") or 0.5),
            cpu_min_tokens=int(env.get("EDGE0_CPU_MIN_TOKENS", "2") or 2),
            cpu_max_tokens=int(env.get("EDGE0_CPU_MAX_TOKENS", "256") or 256),
            cpu_min_cold=int(env.get("EDGE0_CPU_MIN_COLD", "4") or 4),
            cpu_max_routes=int(env.get("EDGE0_CPU_MAX_ROUTES", "8") or 8),
            cpu_max_experts=int(env.get("EDGE0_CPU_MAX_EXPERTS", "512") or 512),
            cpu_spin_ms=float(env.get("EDGE0_CPU_SPIN_MS", "200") or 200),
            cpu_timeout_ms=float(env.get("EDGE0_CPU_TIMEOUT_MS", "2000") or 2000),
            cpu_g1_pairs=int(env.get("EDGE0_CPU_G1_PAIRS", "1") or 1),
            cpu_g2_tiles=int(env.get("EDGE0_CPU_G2_TILES", "2") or 2),
            tier=_bool(env, "EDGE0_TIER", False),
            tier_free=int(env.get("EDGE0_TIER_FREE", "4") or 4),
            tier_swaps=int(env.get("EDGE0_TIER_SWAPS", "48") or 0),
            tier_decay=float(env.get("EDGE0_TIER_DECAY", "0.915") or 0.915),
            tier_gain=float(env.get("EDGE0_TIER_GAIN", "1.5") or 1.5),
            tier_min_use=float(env.get("EDGE0_TIER_MIN_USE", "2") or 2),
            tier_max_tokens=int(env.get("EDGE0_TIER_MAX_TOKENS", "0") or 0),
            tier_log_s=float(env.get("EDGE0_TIER_LOG_S", "10") or 10),
            tier_copy=_choice(env, "EDGE0_TIER_COPY", "cudart", ("cudart", "torch", "batch")),
            tier_early=_bool(env, "EDGE0_TIER_EARLY", False),
            tier_shadow=_bool(env, "EDGE0_TIER_SHADOW", _bool(env, "EDGE0_TIER_EARLY", False)),
            tier_spare=int(env.get("EDGE0_TIER_SPARE", "0") or 0),
            tier_wb_cost=float(env.get("EDGE0_TIER_WB_COST", "0") or 0),
            tier_gpu_decide=_bool(env, "EDGE0_TIER_GPU_DECIDE", False),
        )
        if s.cpu_frac != "auto" and not 0.0 <= float(s.cpu_frac) <= 1.0:
            raise ValueError(f"EDGE0_CPU_FRAC={s.cpu_frac!r}: expected a share in [0, 1] or 'auto'")
        if s.cpu_experts and (s.prefetch or s.moe_impl == "torch"):
            raise ValueError("EDGE0_CPU_EXPERTS=1 cannot be combined with EDGE0_PREFETCH or EDGE0_MOE_IMPL=torch")
        if s.tier and s.prefetch:
            raise ValueError("EDGE0_TIER=1 cannot be combined with EDGE0_PREFETCH (both use the slots after n_hot)")
        if s.tier and not (1 <= s.tier_free <= 64 and s.tier_swaps >= 0 and 0.0 < s.tier_decay <= 1.0):
            raise ValueError("EDGE0_TIER_FREE must be 1..64, EDGE0_TIER_SWAPS >= 0, EDGE0_TIER_DECAY in (0, 1]")
        if s.cpu_experts and not (1 <= s.cpu_min_tokens <= s.cpu_max_tokens and s.cpu_threads >= 1):
            raise ValueError("EDGE0_CPU_MIN_TOKENS <= EDGE0_CPU_MAX_TOKENS and EDGE0_CPU_THREADS >= 1 required")
        for nm, v in (("EDGE0_HOT_BSM", s.hot_bsm), ("EDGE0_COLD_BSM", s.cold_bsm)):
            if v not in (0, 8, 16, 32, 48, 64):
                raise ValueError(f"{nm}={v}: expected 0 (auto) or one of 8, 16, 32, 48, 64")
        if s.moe_chunk_tokens < 0 or s.ref_layers < 0 or s.ple_threads < 1:
            raise ValueError("EDGE0_MOE_CHUNK_TOKENS/EDGE0_REF_LAYERS must be >= 0 and EDGE0_PLE_THREADS >= 1")
        return s

    def describe(self) -> str:
        return " ".join(f"{k}={v}" for k, v in self.__dict__.items())


_SETTINGS: Edge0Settings | None = None


def get_settings() -> Edge0Settings:
    """Process-wide settings (parsed on first use)."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Edge0Settings.from_env()
    return _SETTINGS


def reset_settings() -> None:
    """Tests only: re-read the environment on the next get_settings()."""
    global _SETTINGS
    _SETTINGS = None


# ---------------------------------------------------------------------------------------------- checkpoint block
@dataclass(frozen=True)
class Edge0CheckpointSpec:
    """The ``edge0`` block of config.json (DESIGN.md 1.2)."""

    format: str
    latent_dim: int
    latent_groups: int
    expert_intermediate_size: int
    expert_format: str
    ple_file: str
    ple_data_offset: int
    ple_rows: int
    ple_row_bytes: int
    ple_scale: float
    profile_file: str
    draft_vocab_file: str | None = None
    source_dir: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Edge0CheckpointSpec":
        if not isinstance(d, dict):
            raise ValueError("config.json has no 'edge0' block: not an Edge0 vLLM checkpoint (DESIGN.md 1.2)")
        spec = cls(
            format=str(d["format"]),
            latent_dim=int(d["latent_dim"]),
            latent_groups=int(d["latent_groups"]),
            expert_intermediate_size=int(d["expert_intermediate_size"]),
            expert_format=str(d["expert_format"]),
            ple_file=str(d["ple_file"]),
            ple_data_offset=int(d["ple_data_offset"]),
            ple_rows=int(d["ple_rows"]),
            ple_row_bytes=int(d["ple_row_bytes"]),
            ple_scale=float(d["ple_scale"]),
            profile_file=str(d.get("profile_file", "profile.bin")),
            draft_vocab_file=d.get("draft_vocab_file"),
            source_dir=d.get("source_dir"),
        )
        if spec.format != "edge0-vllm-v1":
            raise ValueError(f"edge0.format={spec.format!r}: this plugin reads edge0-vllm-v1")
        if spec.expert_format != "modelopt_nvfp4_stacked":
            raise ValueError(f"edge0.expert_format={spec.expert_format!r}: expected modelopt_nvfp4_stacked")
        if (spec.latent_dim, spec.latent_groups, spec.expert_intermediate_size) != (LATENT_DIM, LATENT_GROUPS,
                                                                                    EXPERT_FF):
            raise ValueError(f"edge0 geometry {spec} does not match the plugin's constants "
                             f"({LATENT_DIM}, {LATENT_GROUPS}, {EXPERT_FF})")
        return spec

    @classmethod
    def from_hf_config(cls, hf_text_config) -> "Edge0CheckpointSpec":
        d = getattr(hf_text_config, "edge0", None)
        if d is None and hasattr(hf_text_config, "to_dict"):
            d = hf_text_config.to_dict().get("edge0")
        return cls.from_dict(d)

    @classmethod
    def from_checkpoint_dir(cls, ckpt_dir) -> "Edge0CheckpointSpec | None":
        p = Path(ckpt_dir) / "config.json"
        if not p.is_file():
            return None
        d = json.loads(p.read_text()).get("edge0")
        return None if d is None else cls.from_dict(d)
