"""Per-layer routed-expert storage: a hot pool in VRAM and a cold pool in pinned mapped host memory (DESIGN.md 2.6).

Both pools hold experts in Marlin W4A16 NVFP4 layout (``config.W13_SHAPE`` ... ``S2_SHAPE`` per slot, 1,935,360 B).
The pool never permutes the router: ``hot_map`` / ``cold_map`` translate *logical* expert ids into slot indices
of each tier (-1 = not in this tier); they are the ``expert_map`` of the two Marlin calls.

Loading (``load_tensor``): vLLM's iterator hands over each stacked ModelOpt tensor of the layer
(``w13_weight`` [512, 1280, 896] u8 ...) in logical order. Weights are repacked in GPU chunks of 64 experts and
written into their slots; block scales are processed with the layer-wide Marlin factor computed over all 512
experts when that tensor arrives; ``*_scale_2`` stay raw until ``finalize`` (which needs the factor). Peak
transient memory: one stacked tensor in RAM (<= 0.55 GiB, page cache) plus one 64-expert chunk on the GPU.

``swap(hot_slot, cold_slot)`` exchanges two experts between the tiers in place (addresses never change, so
captured CUDA graphs stay valid). It must run on the engine thread between steps.

``EDGE0_TIER`` (tier.py) adds ``n_free`` empty VRAM slots after the ``n_hot`` hot ones and as many spare host slots
after the ``n_cold`` cold ones; the reactive tier copies incoming experts into free slots and evicted ones into spare
slots, and keeps the expert maps and global scales in shared storage (``storage``) so one kernel updates all layers.

On the ``meta`` device nothing is allocated or computed; ``load_tensor`` only checks names, shapes and dtypes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from . import hostmem
from . import marlin_nvfp4 as mn
from .config import (EXPERT_FF, EXPERT_TENSORS, LATENT_DIM, S2_BYTES, S2_SHAPE, S13_BYTES, S13_SHAPE, SLOT_BYTES,
                     TOP_K, W2_BYTES, W2_SHAPE, W13_BYTES, W13_SHAPE)
from .placement import LayerPlacement

_CHUNK = 64
_FP8 = torch.float8_e4m3fn
_DTYPES = {"uint8": torch.uint8, "float8_e4m3fn": _FP8, "float32": torch.float32}


@dataclass
class Tier:
    """``n`` expert slots of one layer in Marlin layout. ``w13 ... s2`` are device-visible tensors (VRAM for the hot
    tier, UVA views of host memory for the cold tier); ``g13`` / ``g2`` (global scales) always live in VRAM."""

    n: int
    w13: torch.Tensor
    w2: torch.Tensor
    s13: torch.Tensor
    s2: torch.Tensor
    g13: torch.Tensor
    g2: torch.Tensor
    host: hostmem.HostBuffer | None = None
    cpu: tuple | None = None        # host tier: CPU views (w13, w2, s13, s2) of the same bytes (DMA copies)

    @property
    def nbytes(self) -> int:
        return self.n * SLOT_BYTES

    @classmethod
    def in_vram(cls, n: int, device: torch.device, g: tuple | None = None) -> "Tier":
        """``g``: optional (g13, g2) views of shared storage (EDGE0_TIER), else allocated here."""
        g13, g2 = g if g is not None else (torch.zeros(n, dtype=torch.float32, device=device),
                                           torch.zeros(n, dtype=torch.float32, device=device))
        assert g13.shape == (n,) and g2.shape == (n,)
        return cls(n=n,
                   w13=torch.zeros((n, *W13_SHAPE), dtype=torch.int32, device=device),
                   w2=torch.zeros((n, *W2_SHAPE), dtype=torch.int32, device=device),
                   s13=torch.zeros((n, *S13_SHAPE), dtype=_FP8, device=device),
                   s2=torch.zeros((n, *S2_SHAPE), dtype=_FP8, device=device),
                   g13=g13, g2=g2)

    @classmethod
    def in_host(cls, n: int, device: torch.device, backend: str, g: tuple | None = None) -> "Tier":
        buf = hostmem.alloc(n * SLOT_BYTES, backend, device)
        off = 0
        views, cpu = [], []
        for nbytes, dtype, shape in ((W13_BYTES, torch.int32, W13_SHAPE), (W2_BYTES, torch.int32, W2_SHAPE),
                                     (S13_BYTES, _FP8, S13_SHAPE), (S2_BYTES, _FP8, S2_SHAPE)):
            c, dev = buf.carve(off, n * nbytes, dtype, (n, *shape))
            views.append(dev)
            cpu.append(c)
            off += n * nbytes
        g13, g2 = g if g is not None else (torch.zeros(n, dtype=torch.float32, device=device),
                                           torch.zeros(n, dtype=torch.float32, device=device))
        assert g13.shape == (n,) and g2.shape == (n,)
        return cls(n=n, w13=views[0], w2=views[1], s13=views[2], s2=views[3], g13=g13, g2=g2, host=buf,
                   cpu=tuple(cpu))

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return self.w13, self.w2, self.s13, self.s2, self.g13, self.g2


@dataclass
class ModelOptCopy:
    """Checkpoint-layout copy of one layer's experts, kept in host RAM for the F1 torch reference."""
    w13: torch.Tensor | None = None
    w13_scale: torch.Tensor | None = None
    w13_scale_2: torch.Tensor | None = None
    w2: torch.Tensor | None = None
    w2_scale: torch.Tensor | None = None
    w2_scale_2: torch.Tensor | None = None


@dataclass
class _LoadState:
    seen: set = field(default_factory=set)
    f13: float | None = None
    f2: float | None = None
    raw_g13: torch.Tensor | None = None     # [E] fp32 (gate == up)
    raw_g2: torch.Tensor | None = None      # [E] fp32
    g13_mismatch: int = 0


class ExpertPool:
    def __init__(self, layer_idx: int, placement: LayerPlacement, device: torch.device, cold_alloc: str,
                 keep_reference: bool = False, n_stage: int = 0, n_free: int = 0, storage=None):
        placement.validate(placement.num_experts, f" layer {layer_idx}")
        self.layer_idx = layer_idx
        self.num_experts = placement.num_experts
        self.device = torch.device(device)
        self.meta = self.device.type == "meta"
        self.order = list(placement.order)
        self.n_hot = placement.n_hot
        self.n_cold = placement.n_cold
        # EDGE0_PREFETCH: n_stage extra VRAM slots after the static hot ones, refilled per step (prefetch.py)
        self.n_stage = int(n_stage) if self.n_cold else 0
        # EDGE0_TIER: n_free empty VRAM slots after the hot ones, n_spare = n_free spare host slots after the cold ones
        self.n_free = int(n_free) if (self.n_cold and self.n_hot) else 0
        self.n_spare = self.n_free
        if storage is not None and self.n_free:       # EDGE0_TIER_SPARE: extra spare host slots (shadow copies)
            self.n_spare = int(storage.n_spare.get(layer_idx, self.n_free))
        if self.n_free and self.n_stage:
            raise ValueError("EDGE0_TIER free slots and EDGE0_PREFETCH staging slots are exclusive")
        v = storage.views(layer_idx) if storage is not None else None
        self.hot = Tier.in_vram(self.n_hot + self.n_stage + self.n_free, self.device,
                                g=(v.hot_g13, v.hot_g2) if v is not None else None)
        self.cold = Tier.in_host(self.n_cold + self.n_spare, self.device, cold_alloc,
                                 g=(v.cold_g13, v.cold_g2) if v is not None else None)
        if v is not None:                    # shared [2, L, E] map storage: one kernel flips every layer's maps
            self.hot_map, self.cold_map = v.hot_map, v.cold_map
            self.hot_map.copy_(torch.tensor(placement.hot_map(), dtype=torch.int32))
            self.cold_map.copy_(torch.tensor(placement.cold_map(), dtype=torch.int32))
        else:
            self.hot_map = torch.tensor(placement.hot_map(), dtype=torch.int32, device=self.device)
            self.cold_map = torch.tensor(placement.cold_map(), dtype=torch.int32, device=self.device)
        self.g_logical = None                # [E, 2] fp32 (g13, g2) per logical expert, CPU (after finalize)
        # EDGE0_SKIP_PAD: TOP_K distinct *hot* logical experts that padding rows of CUDA-graph-padded batches are
        # routed to (weight 0), so padding never loads a cold expert over PCIe. Refreshed in place by swap_many.
        self.pad_ids = (torch.tensor([self.order[:TOP_K]], dtype=torch.int32, device=self.device)
                        if self.n_hot >= TOP_K and not self.meta else None)
        if self.n_stage:
            # effective maps of the two Marlin calls (= static maps + this step's staged experts)
            # *_ext have one extra dummy entry so masked scatters need no dynamic shapes
            self.stage_map_ext = torch.full((self.num_experts + 1,), -1, dtype=torch.int32, device=self.device)
            self.stage_map = self.stage_map_ext[:self.num_experts]          # logical -> absolute hot-tier slot
            self.stage_owner_ext = torch.full((self.n_stage + 1,), -1, dtype=torch.int64, device=self.device)
            self.stage_owner = self.stage_owner_ext[:self.n_stage]           # staging slot -> logical expert
            self.stage_time_ext = torch.zeros(self.n_stage + 1, dtype=torch.float32, device=self.device)
            self.stage_time = self.stage_time_ext[:self.n_stage]             # last step the slot was predicted
            self.hot_eff = self.hot_map.clone()
            self.cold_eff = self.cold_map.clone()
            self.pf_pending = None
            # copy list read by the prefetch stream (persistent: see Prefetcher.predict_and_copy)
            self.pf_src = torch.zeros(self.n_stage, dtype=torch.int32, device=self.device)
            self.pf_dst = torch.zeros(self.n_stage, dtype=torch.int32, device=self.device)
            self.pf_valid = torch.zeros(self.n_stage, dtype=torch.bool, device=self.device)
        # bool per logical expert: True if hot (used by traces / combine fallbacks)
        self.ref = ModelOptCopy() if keep_reference else None
        self._load = _LoadState()
        self.finalized = False

    # ------------------------------------------------------------------------------------------ bookkeeping
    @property
    def vram_bytes(self) -> int:
        return self.hot.nbytes

    @property
    def host_bytes(self) -> int:
        return self.cold.nbytes

    def slot_of(self, expert: int) -> tuple[str, int]:
        s = self.order.index(expert)
        return ("hot", s) if s < self.n_hot else ("cold", s - self.n_hot)

    def _dest(self, slot: int) -> tuple[Tier, int]:
        return (self.hot, slot) if slot < self.n_hot else (self.cold, slot - self.n_hot)

    def _write(self, attr: str, logical_ids: range, values: torch.Tensor) -> None:
        """Write per-expert ``values[i]`` (logical expert ``logical_ids[i]``) into its slot of either tier."""
        slot_of = self._slot_of_list()
        for i, e in enumerate(logical_ids):
            tier, idx = self._dest(slot_of[e])
            getattr(tier, attr)[idx].copy_(values[i])

    def _slot_of_list(self) -> list[int]:
        if getattr(self, "_slot_cache", None) is None:
            inv = [0] * self.num_experts
            for s, e in enumerate(self.order):
                inv[e] = s
            self._slot_cache = inv
        return self._slot_cache

    # ------------------------------------------------------------------------------------------ loading
    def load_tensor(self, name: str, tensor: torch.Tensor) -> None:
        """Load one stacked checkpoint tensor ``model.layers.{l}.mlp.experts.<name>`` (logical expert order)."""
        if name not in EXPERT_TENSORS:
            raise KeyError(f"layer {self.layer_idx}: unexpected expert tensor {name!r}")
        dtype_name, per_expert = EXPERT_TENSORS[name]
        want = (self.num_experts, *per_expert)
        if tuple(tensor.shape) != want or tensor.dtype != _DTYPES[dtype_name]:
            raise ValueError(f"layer {self.layer_idx} experts.{name}: got {tensor.dtype} {tuple(tensor.shape)}, "
                             f"expected {_DTYPES[dtype_name]} {want} (DESIGN.md 1.5)")
        if name in self._load.seen:
            raise ValueError(f"layer {self.layer_idx} experts.{name} loaded twice")
        self._load.seen.add(name)
        if self.ref is not None:
            setattr(self.ref, {"w13_weight": "w13", "w13_weight_scale": "w13_scale",
                               "w13_weight_scale_2": "w13_scale_2", "w2_weight": "w2", "w2_weight_scale": "w2_scale",
                               "w2_weight_scale_2": "w2_scale_2"}[name],
                    tensor.detach().to("cpu", copy=True) if not self.meta else tensor)
        if self.meta:
            return
        st = self._load
        if name == "w13_weight":
            self._load_weights(tensor, mn.repack_w13, "w13")
        elif name == "w2_weight":
            self._load_weights(tensor, mn.repack_w2, "w2")
        elif name == "w13_weight_scale":
            st.f13 = self._load_scales(tensor, LATENT_DIM, 2 * EXPERT_FF, "s13")
        elif name == "w2_weight_scale":
            st.f2 = self._load_scales(tensor, EXPERT_FF, LATENT_DIM, "s2")
        elif name == "w13_weight_scale_2":
            g = tensor.to(torch.float32)
            st.g13_mismatch = int((g[:, 0] != g[:, 1]).sum())
            st.raw_g13 = g[:, 0].contiguous().to(self.device)
        elif name == "w2_weight_scale_2":
            st.raw_g2 = tensor.to(torch.float32).contiguous().to(self.device)

    def _load_weights(self, stacked: torch.Tensor, repack, attr: str) -> None:
        for e0 in range(0, self.num_experts, _CHUNK):
            e1 = min(self.num_experts, e0 + _CHUNK)
            chunk = stacked[e0:e1].to(self.device, non_blocking=False)
            out = torch.stack([repack(chunk[i]) for i in range(e1 - e0)])
            self._write(attr, range(e0, e1), out)

    def _load_scales(self, stacked: torch.Tensor, size_k: int, size_n: int, attr: str) -> float:
        factor = mn.scale_factor(stacked, self.device)     # chunked: no layer-sized fp32 transient in VRAM
        for e0 in range(0, self.num_experts, _CHUNK):
            e1 = min(self.num_experts, e0 + _CHUNK)
            out = mn.process_scales(stacked[e0:e1].to(self.device), size_k, size_n, factor)
            self._write(attr, range(e0, e1), out)
        return factor

    def missing(self) -> list[str]:
        return sorted(set(EXPERT_TENSORS) - self._load.seen)

    def finalize(self) -> dict:
        """After every tensor of the layer arrived: global scales per slot. Returns load statistics."""
        missing = self.missing()
        if missing:
            raise RuntimeError(f"layer {self.layer_idx}: expert tensors never loaded: {missing}")
        st = self._load
        if not self.meta and not self.finalized:
            g13 = mn.process_global_scale(st.raw_g13, st.f13)
            g2 = mn.process_global_scale(st.raw_g2, st.f2)
            order = torch.tensor(self.order, dtype=torch.long, device=self.device)
            if self.n_hot:
                self.hot.g13[:self.n_hot].copy_(g13[order[:self.n_hot]])
                self.hot.g2[:self.n_hot].copy_(g2[order[:self.n_hot]])
            if self.n_cold:
                self.cold.g13[:self.n_cold].copy_(g13[order[self.n_hot:]])
                self.cold.g2[:self.n_cold].copy_(g2[order[self.n_hot:]])
            self.g_logical = torch.stack([g13, g2], dim=1).float().cpu()
        self.finalized = True
        return {"layer": self.layer_idx, "n_hot": self.n_hot, "n_cold": self.n_cold, "factor13": st.f13,
                "factor2": st.f2, "gate_up_scale2_mismatch": st.g13_mismatch}

    # ------------------------------------------------------------------------------------------ swaps
    @torch.no_grad()
    def swap(self, hot_slot: int, cold_slot: int) -> None:
        """Exchange the expert in hot slot ``hot_slot`` with the one in cold slot ``cold_slot`` (between steps)."""
        if not (0 <= hot_slot < self.n_hot and 0 <= cold_slot < self.n_cold):
            raise IndexError(f"swap({hot_slot}, {cold_slot}) outside hot {self.n_hot} / cold {self.n_cold}")
        for th, tc in zip(self.hot.tensors(), self.cold.tensors()):
            tmp = th[hot_slot].clone()
            th[hot_slot].copy_(tc[cold_slot])
            tc[cold_slot].copy_(tmp)
        e_hot, e_cold = self.order[hot_slot], self.order[self.n_hot + cold_slot]
        self.order[hot_slot], self.order[self.n_hot + cold_slot] = e_cold, e_hot
        self._slot_cache = None
        idx = torch.tensor([e_cold, e_hot], dtype=torch.long, device=self.device)
        self.hot_map.index_copy_(0, idx, torch.tensor([hot_slot, -1], dtype=torch.int32, device=self.device))
        self.cold_map.index_copy_(0, idx, torch.tensor([-1, cold_slot], dtype=torch.int32, device=self.device))
        self.reset_stage()

    def reset_stage(self) -> None:
        """Forget the staged experts (after a placement change the effective maps must follow the static ones)."""
        if self.n_stage:
            self.stage_map_ext.fill_(-1)
            self.stage_owner_ext.fill_(-1)
            self.stage_time_ext.zero_()
            self.hot_eff.copy_(self.hot_map)
            self.cold_eff.copy_(self.cold_map)

    def maps(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(hot, cold) expert maps of this step's Marlin calls (static maps unless prefetch staged experts)."""
        return (self.hot_eff, self.cold_eff) if self.n_stage else (self.hot_map, self.cold_map)

    @torch.no_grad()
    def swap_many(self, hot_slots: list[int], cold_slots: list[int]) -> None:
        """Batched ``swap``: exchange ``hot_slots[i]`` with ``cold_slots[i]`` for all i, queued on the current stream
        (index_select / index_copy_ kernels; the cold side is the UVA view of pinned host memory). Addresses never
        change, so captured CUDA graphs stay valid; the expert maps are updated in place. Engine thread, between
        steps only (the adaptive cache calls it from ``compute_logits``)."""
        if not hot_slots:
            return
        if len(set(hot_slots)) != len(hot_slots) or len(set(cold_slots)) != len(cold_slots):
            raise ValueError("swap_many: duplicate slots")
        if not (all(0 <= h < self.n_hot for h in hot_slots) and all(0 <= c < self.n_cold for c in cold_slots)):
            raise IndexError(f"swap_many outside hot {self.n_hot} / cold {self.n_cold}")
        dev = self.device
        hs = torch.tensor(hot_slots, dtype=torch.long).pin_memory().to(dev, non_blocking=True)
        cs = torch.tensor(cold_slots, dtype=torch.long).pin_memory().to(dev, non_blocking=True)
        for th, tc in zip(self.hot.tensors(), self.cold.tensors()):
            if th.dtype == _FP8:                  # no fp8 index_copy_ kernel: move the bytes
                th, tc = th.view(torch.uint8), tc.view(torch.uint8)
            a = th.index_select(0, hs)
            b = tc.index_select(0, cs)
            th.index_copy_(0, hs, b)
            tc.index_copy_(0, cs, a)
        new_hot, new_cold = [], []
        for h, c in zip(hot_slots, cold_slots):
            e_hot, e_cold = self.order[h], self.order[self.n_hot + c]
            self.order[h], self.order[self.n_hot + c] = e_cold, e_hot
            new_hot.append(e_cold)
            new_cold.append(e_hot)
        self._slot_cache = None
        idx = torch.tensor(new_hot + new_cold, dtype=torch.long).pin_memory().to(dev, non_blocking=True)
        n = len(hot_slots)
        hv = torch.tensor(list(hot_slots) + [-1] * n, dtype=torch.int32).pin_memory().to(dev, non_blocking=True)
        cv = torch.tensor([-1] * n + list(cold_slots), dtype=torch.int32).pin_memory().to(dev, non_blocking=True)
        self.hot_map.index_copy_(0, idx, hv)
        self.cold_map.index_copy_(0, idx, cv)
        if self.pad_ids is not None:     # slots 0..TOP_K-1 are hot slots: their experts stay hot
            self.pad_ids.copy_(torch.tensor([self.order[:TOP_K]], dtype=torch.int32).pin_memory()
                               .to(dev, non_blocking=True))
        self.reset_stage()

    def placement(self) -> LayerPlacement:
        return LayerPlacement(order=list(self.order), n_hot=self.n_hot)
