"""NVTX ranges for Nsight Systems (``EDGE0_NVTX``, off by default; docs/PROFILE.md).

``EDGE0_NVTX=1``:
  * MoE op parts (CPU push/pop around the launches; nsys projects them onto the kernels they launch):
    ``edge0.gate``, ``edge0.D``, ``edge0.moe`` (route, gather, hot, cold, combine inside), ``edge0.U``,
    ``edge0.shared``. Inside a captured CUDA graph these ranges exist only at capture time (graph replays carry no
    NVTX), so they label kernels in eager runs; ``tools/nsys_breakdown.py`` uses an eager calibration capture to
    map the kernel signatures of a graph replay to these parts.
  * Host phases that run outside the graphs every engine step (always visible): runner ``prepare_inputs`` /
    ``execute_model`` / ``sample_tokens``, graph replays (``graph.full`` / ``graph.pw``), PLE row staging
    (``ple.stage`` incl. its stream sync, ``ple.read`` = the preadv loop), adaptive-cache epochs, scheduler
    ``schedule`` / ``update_from_output``, speculative ``propose`` / rejection sampling.
``EDGE0_NVTX=2``: additionally one range per ``nn.Module`` forward (qualified module name), for eager calibration
runs (vLLM's own ``--enable-layerwise-nvtx-tracing`` is not wired into the V2 model runner).

Disabled (default): every helper is a no-op and nothing is patched.
"""
from __future__ import annotations

import contextlib
import functools
import os

LEVEL = int(os.environ.get("EDGE0_NVTX", "0") or 0)
ENABLED = LEVEL > 0
_NULL = contextlib.nullcontext()

if ENABLED:
    import torch

    _push = torch.cuda.nvtx.range_push
    _pop = torch.cuda.nvtx.range_pop

    class _Range:
        __slots__ = ("name",)

        def __init__(self, name: str):
            self.name = name

        def __enter__(self):
            _push(self.name)
            return self

        def __exit__(self, *exc):
            _pop()
            return False

    def rng(name: str):
        return _Range(name)
else:
    def rng(name: str):  # noqa: ARG001
        return _NULL


def _wrap(owner, attr: str, label: str, log=None) -> bool:
    fn = getattr(owner, attr, None)
    if fn is None or getattr(fn, "_edge0_nvtx", False):
        return False

    @functools.wraps(fn)
    def wrapped(*a, **kw):
        _push(label)
        try:
            return fn(*a, **kw)
        finally:
            _pop()

    wrapped._edge0_nvtx = True
    setattr(owner, attr, wrapped)
    return True


def install_host_ranges(log=None) -> None:
    """Patch vLLM host-side phases with NVTX ranges (engine-core process, after the model is built)."""
    if not ENABLED:
        return
    import importlib
    done = []
    targets = [
        ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner",
         [("execute_model", "runner.execute_model"), ("prepare_inputs", "runner.prepare_inputs"),
          ("sample_tokens", "runner.sample_tokens"), ("sample", "runner.sample"),
          ("postprocess_sampled", "runner.postprocess")]),
        ("vllm.v1.worker.gpu.cudagraph_utils", "ModelCudaGraphManager",
         [("run_fullgraph", "graph.full"), ("run_pw_graph", "graph.pw")]),
        ("vllm.v1.worker.gpu.cudagraph_utils", "CudaGraphManager",
         [("run_fullgraph", "graph.full"), ("run_pw_graph", "graph.pw")]),
        ("vllm.models.qwen4_exp.nvidia.model_state", "Qwen4ExpModelState",
         [("prepare_inputs", "model_state.prepare_inputs")]),
        ("vllm.models.qwen4_exp.nvidia.ngram_embedding", "_FilePlane", [("read", "ple.read")]),
        ("vllm.v1.core.sched.scheduler", "Scheduler",
         [("schedule", "sched.schedule"), ("update_from_output", "sched.update_from_output")]),
        ("vllm.v1.worker.gpu.spec_decode.dflash.speculator", "DFlashSpeculator", [("propose", "spec.propose")]),
        ("vllm.v1.worker.gpu.spec_decode.rejection_sampler", "RejectionSampler", [("__call__", "spec.reject")]),
    ]
    for mod_name, cls_name, methods in targets:
        try:
            cls = getattr(importlib.import_module(mod_name), cls_name)
        except Exception:  # noqa: BLE001
            continue
        for attr, label in methods:
            if attr in cls.__dict__ and _wrap(cls, attr, label):
                done.append(f"{cls_name}.{attr}")
    # PLE staging: model_state imported the function by name.
    try:
        ms = importlib.import_module("vllm.models.qwen4_exp.nvidia.model_state")
        if _wrap(ms, "stage_checkpoint_rows", "ple.stage"):
            done.append("stage_checkpoint_rows")
    except Exception:  # noqa: BLE001
        pass
    try:
        from . import adaptive
        if _wrap(adaptive.AdaptiveCache, "_epoch", "edge0.adapt.epoch"):
            done.append("AdaptiveCache._epoch")
    except Exception:  # noqa: BLE001
        pass
    if log is not None:
        log.info("edge0: EDGE0_NVTX=%d host ranges on %s", LEVEL, ", ".join(done))


def install_module_ranges(model, log=None) -> None:
    """EDGE0_NVTX=2: one NVTX range per module forward (eager calibration runs only)."""
    if LEVEL < 2:
        return
    n = 0
    for name, mod in model.named_modules():
        if not name:
            continue
        label = f"mod:{name}"

        def pre(m, args, _label=label):
            _push(_label)

        def post(m, args, out):
            _pop()

        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)
        n += 1
    if log is not None:
        log.info("edge0: EDGE0_NVTX=2 module ranges on %d modules", n)
