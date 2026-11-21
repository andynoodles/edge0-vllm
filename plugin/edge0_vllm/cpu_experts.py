"""``EDGE0_CPU_EXPERTS=1``: Strata-style CPU computation of part of every MoE call's cold experts (docs/CPU-EXPERTS.md).

The C++/CUDA extension (``csrc/``) is built with ``torch.utils.cpp_extension`` into a directory keyed by the
sources' hash: ``/opt/edge0/ext/<key>`` when the image prebuilt it (docker/Dockerfile), else
``$EDGE0_EXT_DIR`` (default ``/root/.cache/edge0-vllm/ext/<key>``, a host-mounted cache, so a dev-mode container
builds it once).

``CpuExperts`` owns the mailbox (pinned, mapped host memory), the VRAM state of the three GPU kernels and the CPU
worker pool. ``latent_moe.latent_experts_marlin`` calls ``dispatch`` -> ``gather`` -> (Marlin calls with
``cold_map_eff``) -> ``wait`` -> combine with ``y`` / ``cpu_sel`` for calls of ``min_tokens..max_tokens`` tokens.
"""
from __future__ import annotations

import hashlib
import os

import torch

from . import hostmem
from .config import LATENT_DIM, NUM_EXPERTS, S13_BYTES, TOP_K, W2_BYTES, W13_BYTES

_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
_SOURCES = ("cpu_experts.cpp", "cpu_experts_cuda.cu")
_HEADERS = ("mailbox.h",)
_CFLAGS = ["-O3", "-mavx512f", "-mavx512bw", "-mavx512vl", "-mavx512dq", "-mfma", "-mf16c",
           "-fno-strict-aliasing"]
_CUDA_FLAGS = ["-O3"]
_ARCH = "12.0"
_EXT = None


def ext_key() -> str:
    h = hashlib.sha1()
    for f in _SOURCES + _HEADERS:
        with open(os.path.join(_CSRC, f), "rb") as fh:
            h.update(fh.read())
    h.update(" ".join(_CFLAGS + _CUDA_FLAGS + [_ARCH, _CSRC]).encode())
    return h.hexdigest()[:16]


def _prebuilt() -> str:
    return os.path.join(os.environ.get("EDGE0_EXT_PREBUILT", "/opt/edge0/ext"), ext_key(), "edge0_cpu_experts.so")


def ext_dir() -> str:
    pre = _prebuilt()
    if os.path.isfile(pre):
        return os.path.dirname(pre)
    return os.path.join(os.environ.get("EDGE0_EXT_DIR", "/root/.cache/edge0-vllm/ext"), ext_key())


def load(verbose: bool = False, build_dir: str | None = None):
    """Import the extension: the image's prebuilt module for these exact sources, else build (first time) with
    ``torch.utils.cpp_extension`` into the cache directory."""
    global _EXT
    if _EXT is None and build_dir is None and os.path.isfile(_prebuilt()):
        import importlib.machinery
        import importlib.util
        so = _prebuilt()
        loader = importlib.machinery.ExtensionFileLoader("edge0_cpu_experts", so)
        spec = importlib.util.spec_from_file_location("edge0_cpu_experts", so, loader=loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        _EXT = mod
    if _EXT is None:
        from torch.utils.cpp_extension import load as _load
        d = build_dir or ext_dir()
        os.makedirs(d, exist_ok=True)
        old = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = _ARCH          # the image lists 7 archs; this box is sm_120 only
        try:
            _EXT = _load(name="edge0_cpu_experts", sources=[os.path.join(_CSRC, f) for f in _SOURCES],
                         extra_cflags=_CFLAGS, extra_cuda_cflags=_CUDA_FLAGS, extra_include_paths=[_CSRC],
                         build_directory=d, verbose=verbose)
        finally:
            if old is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = old
    return _EXT


def parse_cpus(spec: str, n: int) -> list[int]:
    """``EDGE0_CPU_CORES``: "2-13" / "2,3,8-11" (the first ``n`` are used); empty = physical cores 2.. (Linux numbers the
    first hardware thread of every core before the SMT siblings), wrapping onto core 1 and 0 if n > cores - 2."""
    if spec:
        out: list[int] = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-")
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(part))
        if len(out) < n:
            raise ValueError(f"EDGE0_CPU_CORES={spec!r} lists {len(out)} CPUs for {n} threads")
        return out[:n]
    cores = max(1, (os.cpu_count() or 2) // 2)
    order = list(range(2, cores)) + [1, 0]
    return [order[i % len(order)] for i in range(n)]


def slot_pointers(tier) -> tuple[int, int, int, int, int]:
    """Host addresses of a cold tier's four arrays (``Tier.in_host`` carves w13 | w2 | s13 | s2 back to back)."""
    n = tier.n
    base = tier.host.cpu.data_ptr()
    return (base, base + n * W13_BYTES, base + n * (W13_BYTES + W2_BYTES),
            base + n * (W13_BYTES + W2_BYTES + S13_BYTES), n)


class CpuExperts:
    def __init__(self, settings, device: torch.device, log, ext=None):
        self.ext = ext or load()
        self.log = log
        s = settings
        self.min_tokens = int(s.cpu_min_tokens)
        self.max_tokens = int(s.cpu_max_tokens)
        self.max_routes = self.max_tokens * TOP_K
        self.min_cold = int(s.cpu_min_cold)
        self.max_routes_e = int(s.cpu_max_routes)
        self.max_experts = int(s.cpu_max_experts)
        self.timeout_ns = int(s.cpu_timeout_ms * 1e6)
        lay = self.ext.layout(self.max_routes)
        self.lay = dict(zip(("tasks", "route_ids", "route_w", "x", "y", "total", "header"), lay))
        self.buf = hostmem.alloc(self.lay["total"], "host_register", device)
        self.mbox_dev = int(self.buf.dev.data_ptr())
        ny = self.max_routes * LATENT_DIM * 2
        self.y = self.buf.dev[self.lay["y"]:self.lay["y"] + ny].view(torch.bfloat16).view(self.max_routes, LATENT_DIM)
        self.state = torch.zeros(16, dtype=torch.int64, device=device)
        self.cpu_sel = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=device)
        self.cold_map_eff = torch.full((NUM_EXPERTS,), -1, dtype=torch.int32, device=device)
        self.cpus = parse_cpus(s.cpu_cores, int(s.cpu_threads))
        self.auto = s.cpu_frac == "auto"
        frac = float(s.cpu_frac_start if self.auto else s.cpu_frac)
        # A fixed split goes to the dispatch kernel as an argument (no mailbox read per call); auto: -1 = read the
        # mailbox's frac, which the pool updates.
        self.frac_arg = -1.0 if self.auto else frac
        self.pool = self.ext.Pool(int(self.buf.cpu.data_ptr()), self.max_routes, self.cpus, float(s.cpu_spin_ms))
        self.pool.set_frac(frac)
        self.pool.set_tiles(int(getattr(s, "cpu_g1_pairs", 1)), int(getattr(s, "cpu_g2_tiles", 2)))
        if self.auto:
            self.pool.set_auto(True, 0.05, 0.9)
        self.n_layers = 0
        self._last = None
        self._t_log = 0.0
        self.log_every_s = float(os.environ.get("EDGE0_CPU_LOG_S", "30") or 30)
        log.info("edge0 cpu-experts: %d workers on CPUs %s, frac %s, calls of %d..%d tokens, >= %d cold experts, "
                 "<= %d routes per CPU expert, mailbox %.1f MB", len(self.cpus), self.cpus,
                 "auto (start %.2f)" % frac if self.auto else "%.2f" % frac, self.min_tokens, self.max_tokens,
                 self.min_cold, self.max_routes_e, self.lay["total"] / 1e6)

    def register(self, layer: int, pool) -> None:
        if pool.n_cold and pool.cold.host is not None:
            self.pool.register_layer(layer, *slot_pointers(pool.cold))
            self.n_layers += 1

    def applies(self, tokens: int) -> bool:
        return self.min_tokens <= tokens <= self.max_tokens

    def dispatch(self, ids: torch.Tensor, w: torch.Tensor, pool) -> torch.Tensor:
        self.ext.dispatch(ids, w, pool.cold_map, pool.cold.g13, pool.cold.g2, pool.layer_idx, self.min_cold,
                          self.max_routes_e, self.max_experts, self.mbox_dev, self.max_routes, self.cold_map_eff,
                          self.cpu_sel, self.state, self.frac_arg)
        return self.cold_map_eff

    def gather(self, xl: torch.Tensor, ids: torch.Tensor, expert_group: torch.Tensor, out: torch.Tensor):
        out = out[:ids.numel()]
        self.ext.gather(xl, ids, expert_group, out, self.cpu_sel, self.mbox_dev, self.max_routes, self.state)
        return out

    def wait(self) -> None:
        self.ext.wait(self.state, self.mbox_dev, self.timeout_ns)

    def stats(self) -> dict:
        k = ("jobs", "experts", "routes", "cpu_s", "gpu_calls", "gpu_s", "wait_s", "timeouts", "frac", "g2_waits",
             "auto_updates")
        return dict(zip(k, self.pool.stats()))

    def maybe_log(self) -> None:
        import time
        t = time.monotonic()
        if t - self._t_log >= self.log_every_s:
            self._t_log = t
            self.log_stats()

    def log_stats(self) -> None:
        """Engine thread, every few seconds: deltas since the last call (no GPU sync: the totals are in mapped
        memory)."""
        st = self.stats()
        last, self._last = self._last, st
        if last is None:
            return
        d = {k: st[k] - last[k] for k in st}
        if d["jobs"] <= 0:
            return
        self.log.info("edge0 cpu-experts: %d jobs, %.1f experts / %.1f routes per job, CPU %.0f us per job, GPU side "
                      "%.0f us per call, GPU waited %.0f us per call, frac %.3f, timeouts %d",
                      d["jobs"], d["experts"] / d["jobs"], d["routes"] / d["jobs"], 1e6 * d["cpu_s"] / d["jobs"],
                      1e6 * d["gpu_s"] / max(1, d["gpu_calls"]), 1e6 * d["wait_s"] / max(1, d["gpu_calls"]),
                      st["frac"], int(st["timeouts"]))
