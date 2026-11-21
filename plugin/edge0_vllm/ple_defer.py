"""``EDGE0_PLE_DEFER=1``: PLE rows off the critical path (docs/SINGLE-STREAM-GAP.md round 1, TUNING-LOG "Single-stream").

vLLM's ``host_file_gather`` PLE (``ngram_embedding.stage_checkpoint_rows``, called from the model state's
``prepare_inputs``) computes the step's n-gram ids on the GPU, copies them to the host, **synchronizes the whole
stream** (the previous step's verify, sampling and drafter), reads every row with ``preadv`` and copies the rows to
device staging; only then is the step's CUDA graph launched. At c1 with DFlash2 k=3 that is ~64 random 160-byte NVMe
reads per step (the drive's IOPS ceiling makes them 1.3-1.6 ms) during which the GPU idles.

With this switch ``stage_checkpoint_rows`` no longer reads: a one-CTA kernel (``post_ids``) writes the step's ids
into pinned, device-mapped host memory and rings a doorbell, and a C++ reader thread (``csrc/ple_reader.cpp``; no
GIL, no CUDA call: it spins on the doorbell) reads the rows exactly as vLLM does (fadvise WILLNEED for all rows, then
pread each) into mapped memory and publishes the step's sequence number in a mapped flag. The main thread keeps
vLLM's stream sync (``EDGE0_PLE_DEFER_SYNC=1``, default; without it the host runs ahead and then blocks before its
output processing and scheduling, which costs more than the read: TUNING-LOG "Single-stream" R1) and launches the
step's graph right after it; the PLE embedding's forward (decoder layer 1, ``ple_layer_ids`` [2]) runs ``wait_copy``
(``csrc/ple_wait.cu``): spin until the flag reaches the step's number (written to device memory by a fill kernel
before the launch), then copy the rows into the same staging buffer vLLM's forward reads. So the launch, layer 0 and
the read overlap; the rows and their order are vLLM's (same ids, same bytes), so the outputs are unchanged.
``EDGE0_PLE_DEFER_WAKE=event`` wakes the reader with a D2H copy + ``cudaEventSynchronize`` instead of the doorbell.

The extension is built like ``cpu_experts``' (``torch.utils.cpp_extension``, keyed by the sources' hash; prebuilt
in the image under ``/opt/edge0/ext/<key>``). ``EDGE0_PLE_DEFER_TIMEOUT_MS`` (2000): a wait that long gives up
(stale rows, counted) instead of hanging; ``EDGE0_PLE_DEFER_CPU`` (-1): pin the reader thread to a CPU;
``EDGE0_PLE_LOG_S`` (30): stats log interval.
"""
from __future__ import annotations

import hashlib
import os
import time

import torch

from . import hostmem

_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
_SOURCES = ("ple_reader.cpp", "ple_wait.cu")
_CFLAGS = ["-O3"]
_CUDA_FLAGS = ["-O3"]
_ARCH = "12.0"
_EXT = None

ENABLED = os.environ.get("EDGE0_PLE_DEFER", "0").strip() == "1"
# How the reader learns that a step's ids are on the host: doorbell (a kernel writes them into mapped memory and
# rings a flag the reader spins on; no CUDA call from the reader thread) | event (D2H copy + cudaEventSynchronize)
WAKE = os.environ.get("EDGE0_PLE_DEFER_WAKE", "doorbell").strip().lower()
# 1 (default): keep the stream sync of vLLM's staging (the host waits for the previous step there); 0: no sync
SYNC = os.environ.get("EDGE0_PLE_DEFER_SYNC", "1").strip() != "0"


def ext_key() -> str:
    h = hashlib.sha1()
    for f in _SOURCES:
        with open(os.path.join(_CSRC, f), "rb") as fh:
            h.update(fh.read())
    h.update(" ".join(_CFLAGS + _CUDA_FLAGS + [_ARCH, _CSRC, "ple"]).encode())
    return h.hexdigest()[:16]


def _prebuilt() -> str:
    return os.path.join(os.environ.get("EDGE0_EXT_PREBUILT", "/opt/edge0/ext"), ext_key(), "edge0_ple.so")


def load(verbose: bool = False, build_dir: str | None = None):
    global _EXT
    if _EXT is None and build_dir is None and os.path.isfile(_prebuilt()):
        import importlib.machinery
        import importlib.util
        so = _prebuilt()
        loader = importlib.machinery.ExtensionFileLoader("edge0_ple", so)
        spec = importlib.util.spec_from_file_location("edge0_ple", so, loader=loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        _EXT = mod
    if _EXT is None:
        from torch.utils.cpp_extension import load as _load
        d = build_dir or os.path.join(os.environ.get("EDGE0_EXT_DIR", "/root/.cache/edge0-vllm/ext"), ext_key())
        os.makedirs(d, exist_ok=True)
        old = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = _ARCH
        try:
            _EXT = _load(name="edge0_ple", sources=[os.path.join(_CSRC, f) for f in _SOURCES],
                         extra_cflags=_CFLAGS, extra_cuda_cflags=_CUDA_FLAGS, build_directory=d, verbose=verbose)
        finally:
            if old is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = old
    return _EXT


class _State:
    """Process-wide: the step sequence number (host) and its device copy read by every wait_copy kernel."""

    def __init__(self, device):
        self.seq = 0
        self.expect = torch.zeros(1, dtype=torch.int64, device=device)
        self.readers: list = []
        self.t_log = time.monotonic()
        self.last = None


_STATE: _State | None = None
_LOG = None


class _Reader:
    """One per file-gather PLE module: mapped rows buffer + flag + the C++ reader thread."""

    def __init__(self, emb, ext, device, timeout_ms: float, cpu: int):
        plane = emb._planes.get("weight")
        if plane is None:
            raise RuntimeError("EDGE0_PLE_DEFER: PLE plane not bound")
        if getattr(emb, "_nvfp4", False) or len(emb._suffixes) != 1:
            raise NotImplementedError("EDGE0_PLE_DEFER supports the fp8 / bf16 PLE table only (no NVFP4 scales)")
        self.row_bytes = int(plane.row_bytes)
        host_rows = emb._host_rows["weight"]
        self.max_ids = int(host_rows.shape[0])
        rows_bytes = self.max_ids * self.row_bytes
        off_flag = (rows_bytes + 255) // 256 * 256          # [rows | flag | stats | ids flag | ids]
        off_stats = off_flag + 256
        off_idsflag = off_stats + 256
        off_ids = off_idsflag + 256
        self.buf = hostmem.alloc(off_ids + self.max_ids * 8 + 256, "host_register", device)
        self.buf.cpu.zero_()
        base_cpu, base_dev = int(self.buf.cpu.data_ptr()), int(self.buf.dev.data_ptr())
        self.rows_dev = self.buf.dev[:rows_bytes]
        self.flag_cpu = self.buf.cpu[off_flag:off_flag + 8].view(torch.int64)
        self.flag_dev = base_dev + off_flag
        self.stats_cpu = self.buf.cpu[off_stats:off_stats + 32].view(torch.int64)
        self.stats_dev = base_dev + off_stats
        self.doorbell = WAKE == "doorbell"
        if self.doorbell:          # the GPU writes the ids straight into mapped memory, then rings the ids flag
            ids_ptr, idsflag_ptr = base_cpu + off_ids, base_cpu + off_idsflag
            self.ids_dev, self.idsflag_dev = base_dev + off_ids, base_dev + off_idsflag
        else:                      # vLLM's pinned _host_ids (D2H copy) + a CUDA event the reader synchronizes on
            if not emb._host_ids.is_pinned():
                raise RuntimeError("EDGE0_PLE_DEFER: PLE host ids are not pinned")
            ids_ptr, idsflag_ptr = int(emb._host_ids.data_ptr()), 0
        self.timeout_ns = int(timeout_ms * 1e6)
        self.ext = ext
        self.r = ext.Reader([int(x) for x in plane.fds.tolist()], [int(x) for x in plane.bases.tolist()],
                            int(plane.rows_per_shard), self.row_bytes, int(emb.org_vocab_size), ids_ptr, base_cpu,
                            base_cpu + off_flag, self.max_ids, int(device.index or 0), int(cpu), idsflag_ptr)

    def stats(self) -> dict:
        k = ("jobs", "ids", "uniq", "read_s", "evwait_s", "lag_s", "errors", "maxq")
        d = dict(zip(k, self.r.stats()))
        g = self.stats_cpu.tolist()
        d.update(gpu_calls=g[0], gpu_wait_s=g[1] * 1e-9, gpu_timeouts=g[2], gpu_waited=g[3])
        return d


def _forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    """``Qwen4ExpPLEFileGatherEmbedding.forward`` with the rows taken from the reader's mapped buffer."""
    num_tokens = input_ids.shape[0]
    num_ids = num_tokens * self._num_heads
    staging = self._staging["weight"]
    rd = _reader(self)
    if rd is not None and num_ids:
        rd.ext.wait_copy(rd.rows_dev, staging, num_ids * rd.row_bytes, rd.flag_dev, _STATE.expect, rd.timeout_ns,
                         rd.stats_dev)
    rows = staging[:num_ids].view(self.weight.dtype)
    return rows.view(num_tokens, -1)


def _reader(emb):
    """The module's reader, created on first use (after the table is bound: before the profile run and the graph
    capture, so every captured graph contains the wait-copy kernel). None for a dummy-weight load."""
    global _STATE
    rd = getattr(emb, "_edge0_reader", None)
    if rd is not None or not emb._planes:
        return rd
    dev = emb._staging["weight"].device
    if _STATE is None:
        _STATE = _State(dev)
    rd = _Reader(emb, load(), dev, float(os.environ.get("EDGE0_PLE_DEFER_TIMEOUT_MS", "2000") or 2000),
                 int(os.environ.get("EDGE0_PLE_DEFER_CPU", "-1") or -1))
    emb._edge0_reader = rd
    _STATE.readers.append(rd)
    if _LOG is not None:
        _LOG.info("edge0: PLE rows deferred (EDGE0_PLE_DEFER=1): layer %d, %d ids max, %d B rows, reader thread "
                  "+ in-graph wait", emb.layer_index, rd.max_ids, rd.row_bytes)
    return rd


def stage_deferred(modules, input_ids, query_start_loc, ngram_context) -> None:
    """Replacement of ``stage_checkpoint_rows``: no stream sync, no read on the main thread."""
    for module in modules:
        if _reader(module.ngram_embedding) is None:
            raise RuntimeError("EDGE0_PLE_DEFER: PLE table not bound")
    num_tokens = input_ids.shape[0]
    st = _STATE
    st.seq += 1
    for module in modules:
        ids = module.compute_ngram_ids(input_ids, query_start_loc, ngram_context)
        emb = module.ngram_embedding
        rd = emb._edge0_reader
        if rd.doorbell:
            rd.ext.post_ids(ids.reshape(-1).contiguous(), rd.ids_dev, rd.idsflag_dev, st.seq)
        else:
            emb._host_ids[:num_tokens].copy_(ids, non_blocking=True)
    stream = torch.cuda.current_stream().cuda_stream
    for module in modules:
        emb = module.ngram_embedding
        emb._edge0_reader.r.submit(stream, num_tokens * emb._num_heads, st.seq)
    st.expect.fill_(st.seq)
    if SYNC:
        # Keep vLLM's host rhythm: wait here for the previous step (as stage_checkpoint_rows does), so the host work
        # left after the wait is only this step's launch; the rows are read meanwhile (the reader started on the
        # GPU's doorbell) and waited for inside the graph. Without the sync the host blocks later, before its
        # output processing and scheduling, which then sit on the critical path (round-1 nsys, g1defer).
        torch.cuda.current_stream().synchronize()
    _maybe_log()


def _maybe_log() -> None:
    st = _STATE
    t = time.monotonic()
    every = float(os.environ.get("EDGE0_PLE_LOG_S", "30") or 30)
    if t - st.t_log < every or not st.readers or _LOG is None:
        return
    st.t_log = t
    cur = st.readers[0].stats()
    last, st.last = st.last, cur
    if last is None:
        return
    d = {k: cur[k] - last[k] for k in cur}
    if d["jobs"] <= 0:
        return
    j = d["jobs"]
    _LOG.info("edge0 ple-defer: %d steps, %.1f ids / %.1f rows per step, read %.0f us, event wait %.0f us, "
              "submit->ready %.0f us per step; GPU waited in %.0f%% of %d calls, %.0f us per call; timeouts %d, "
              "errors %d, max queue %d", j, d["ids"] / j, d["uniq"] / j, 1e6 * d["read_s"] / j,
              1e6 * d["evwait_s"] / j, 1e6 * d["lag_s"] / j, 100.0 * d["gpu_waited"] / max(1, d["gpu_calls"]),
              d["gpu_calls"], 1e6 * d["gpu_wait_s"] / max(1, d["gpu_calls"]), int(cur["gpu_timeouts"]),
              int(cur["errors"]), int(cur["maxq"]))


def install(log=None) -> None:
    """Patch vLLM's file-gather PLE (engine-core process, before the model state is built)."""
    global _LOG
    if not ENABLED:
        return
    _LOG = log
    import vllm.models.qwen4_exp.nvidia.model_state as ms
    import vllm.models.qwen4_exp.nvidia.ngram_embedding as ne
    if getattr(ms.stage_checkpoint_rows, "_edge0_defer", False):
        return
    stage_deferred._edge0_defer = True
    ms.stage_checkpoint_rows = stage_deferred
    cls = ne.Qwen4ExpPLEFileGatherEmbedding
    cls.forward = _forward
    bind = cls.bind_storage_after_loading

    def bind_storage_after_loading(self):
        bind(self)
        if torch.cuda.is_available() and self._staging["weight"].is_cuda:
            _reader(self)

    cls.bind_storage_after_loading = bind_storage_after_loading
    load()           # build / import the extension now, not in the first step
    if log is not None:
        log.info("edge0: EDGE0_PLE_DEFER=1: PLE rows read by a reader thread, waited for inside the graph "
                 "(extension %s)", ext_key())
