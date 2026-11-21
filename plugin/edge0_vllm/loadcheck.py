"""Load-time audit of the checkpoint -> parameter path (docs/BRINGUP-LOG.md, "load warnings").

vLLM's ``AutoWeightsLoader`` logs ``Attempted to load weight X with dtype A into parameter with dtype B`` for every
tensor whose dtype differs from its parameter, *before* it calls the parameter's ``weight_loader``. On this
checkpoint that is 449 lines per load, in two expected classes:

* ``uint8 -> bfloat16`` on Q8_0 linears (192 non-fused modules; the 168 fused shards go through the stacked-name
  path, which does not log). The parameter's loader is ``q8.wrap_weight_loader``, which dequantizes the blocks
  before vLLM's own loader runs. If a Q8_0 tensor ever reached an unwrapped loader its raw bytes would be garbage
  (vLLM's loaders also reject the [N, K/32, 34] shape), so ``LoadAudit`` checks the invariant
  "Q8_0 tensors fed to vLLM == tensors dequantized by the wrapper" on every load and aborts if it fails.
* ``float32 / float16 -> bfloat16`` on parameters that stock vLLM itself declares bf16: hc_norm
  (``HyperConnectionConfig(params_dtype=torch.bfloat16)``), GDN ``dt_bias`` / ``conv1d`` / ``norm``, the
  ``GemmaRMSNorm`` q/k norms and indexer layernorms, the PLE norms and PLE conv1d. (``A_log`` is declared fp32 by
  vLLM and loads without narrowing.) The narrowing is vLLM's design, not something the plugin does.

Everything here runs only with ``EDGE0_VERIFY_LOAD=1`` (default off: the load path is unchanged). ``LoadAudit``
then counts both classes into summary lines; any other dtype mismatch still passes through as a WARNING.

``verify_loaded`` (``EDGE0_VERIFY_LOAD=1``) compares the loaded parameters, right after ``load_weights`` and before
any ``process_weights_after_loading`` transform, with an independent numpy decode of the checkpoint bytes
(``convert/strata_formats.py``): every Q8_0 tensor (all rows for layers 0, 3 and 47, 8 rows elsewhere; exact bf16
equality expected), every narrowed F32/F16 tensor (exact equality with the round-to-nearest-even bf16 of the
source, plus the narrowing error), the F32 tensors vLLM keeps in fp32, and sampled BF16 rows of embed_tokens,
lm_head and the routers. A mismatch fails the load; the report is written as JSON next to the plugin log.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import sys
import time
from collections import Counter

import numpy as np
import torch

from . import q8

_VLLM_LOADER_LOGGER = "vllm.model_executor.models.utils"
_MSG = "Attempted to load weight "
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
# Parameters stock vLLM declares bf16 while the checkpoint (like Strata) stores F32 / F16.
_NARROW_RE = re.compile(r"(?:^|\.)(hc_norm\.weight|linear_attn\.norm\.weight|linear_attn\.dt_bias|"
                        r"linear_attn\.conv1d\.weight|q_norm\.weight|k_norm\.weight|indexer\.[qk]_layernorm\.weight|"
                        r"ple\.norm_(?:key|query|conv)\.weight|ple\.conv1d\.weight)$")
# Checkpoint tensor -> (fused parameter, first output shard) for vLLM's stacked-name mapping (Qwen3.5-style GDN).
_FUSED = {
    "linear_attn.in_proj_qkv.weight": ("linear_attn.in_proj_qkvz.weight", 0),
    "linear_attn.in_proj_z.weight": ("linear_attn.in_proj_qkvz.weight", 3),
    "mlp.shared_expert.gate_proj.weight": ("mlp.shared_expert.gate_up_proj.weight", 0),
    "mlp.shared_expert.up_proj.weight": ("mlp.shared_expert.gate_up_proj.weight", 1),
}
_FULL_LAYERS = (0, 3, 47)
_SAMPLE_ROWS = 8
_CHUNK_ROWS = 2048


def _kind(name: str) -> str:
    return _LAYER_RE.sub(".layers.N.", "." + name).lstrip(".")


class _DtypeWarningFilter(logging.Filter):
    def __init__(self, q8_mode: bool):
        super().__init__()
        self.q8_mode = q8_mode
        self.counts: Counter = Counter()

    def filter(self, record: logging.LogRecord) -> bool:
        if not str(record.msg).startswith(_MSG):
            return True
        args = record.args
        if not isinstance(args, tuple) or len(args) != 3:
            return True
        name, src, dst = str(args[0]), args[1], args[2]
        if self.q8_mode and src == torch.uint8 and dst == torch.bfloat16:
            self.counts[("uint8->bf16 (Q8_0, dequantized by the edge0 loader)", _kind(name))] += 1
            return False
        if src in (torch.float32, torch.float16) and dst == torch.bfloat16:
            m = _NARROW_RE.search(name)
            if m:
                self.counts[(f"{str(src).replace('torch.', '')}->bf16 (stock vLLM bf16 param)", m.group(1))] += 1
                return False
        return True


class LoadAudit:
    """Context manager around ``Qwen4ExpForCausalLM.load_weights`` (see module doc)."""

    def __init__(self, rt):
        self.rt = rt
        self.q8_mode = rt.settings.dense == "bf16"
        self.filter = _DtypeWarningFilter(self.q8_mode)
        self.q8_in = 0
        self.q8_names: list[str] = []
        self._dq0 = 0
        self.t0 = 0.0

    def __enter__(self):
        self._dq0 = q8.DEQUANT_TENSORS
        self.t0 = time.time()
        if self.filter is not None:
            logging.getLogger(_VLLM_LOADER_LOGGER).addFilter(self.filter)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.filter is not None:
            logging.getLogger(_VLLM_LOADER_LOGGER).removeFilter(self.filter)
        return False

    def see(self, name: str, tensor: torch.Tensor) -> None:
        """Called for every non-expert checkpoint tensor handed to vLLM's loader."""
        if not name.startswith("mtp.") and q8.is_q8_blocks(tensor):
            self.q8_in += 1
            self.q8_names.append(name)

    def finish(self) -> None:
        log = self.rt.log
        dequantized = q8.DEQUANT_TENSORS - self._dq0
        if self.q8_mode and dequantized != self.q8_in:
            raise RuntimeError(f"edge0: {self.q8_in} Q8_0 tensors were fed to vLLM's loader but {dequantized} were "
                               "dequantized by the edge0 loader: some Q8_0 tensor reached an unwrapped weight_loader")
        log.info("edge0: Q8_0 load path: %d Q8_0 tensors in, %d dequantized to bf16 by the edge0 loader (%.0f s)",
                 self.q8_in, dequantized, time.time() - self.t0)
        if self.filter is not None and self.filter.counts:
            by_class: dict[str, list[str]] = {}
            for (cls, kind), n in sorted(self.filter.counts.items()):
                by_class.setdefault(cls, []).append(f"{kind} x{n}")
            for cls, kinds in by_class.items():
                n = sum(int(k.rsplit("x", 1)[1]) for k in kinds)
                log.info("edge0: %d expected vLLM dtype-mismatch warnings suppressed, %s: %s", n, cls,
                         ", ".join(kinds))


# ------------------------------------------------------------------------------------------------ verification
def _strata_formats(log):
    path = os.environ.get("EDGE0_VERIFY_FORMATS", "/workspace/edge0/convert/strata_formats.py")
    spec = importlib.util.spec_from_file_location("edge0_strata_formats", path)
    if spec is None or not os.path.exists(path):
        raise FileNotFoundError(f"EDGE0_VERIFY_LOAD needs {path} (edge0/convert/strata_formats.py)")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # its @dataclass definitions look the module up while executing
    spec.loader.exec_module(mod)
    log.info("edge0: verify: reference decoder %s", path)
    return mod


def _rows_to_check(name: str, n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    m = _LAYER_RE.search("." + name)
    if m is None or int(m.group(1)) in _FULL_LAYERS:
        return [(r, min(n, r + _CHUNK_ROWS)) for r in range(0, n, _CHUNK_ROWS)]
    rows = sorted({0, n - 1, *rng.choice(n, size=min(n, _SAMPLE_ROWS - 2), replace=False).tolist()})
    return [(r, r + 1) for r in rows]


def verify_loaded(model: torch.nn.Module, model_dir: str, log) -> dict:
    from safetensors import safe_open

    sf = _strata_formats(log)
    t0 = time.time()
    params = dict(model.named_parameters())
    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    names = sorted(n for n in index if not n.startswith("mtp.") and ".mlp.experts." not in n)
    rng = np.random.default_rng(0)
    rep = {"q8": Counter(), "narrow": {}, "fp32": Counter(), "bf16": Counter(), "unmapped": [], "failures": []}
    q8_rows = 0
    handles: dict[str, object] = {}

    def fail(msg):
        rep["failures"].append(msg)
        log.error("edge0: verify: %s", msg)

    def target(name):
        """checkpoint name -> (param, row offset) or (None, 0)."""
        if name in params:
            return params[name], 0
        for suffix, (psuffix, shard) in _FUSED.items():
            if name.endswith(suffix):
                pname = name[: -len(suffix)] + psuffix
                p = params.get(pname)
                if p is None:
                    return None, 0
                mod = model.get_submodule(pname.rsplit(".", 1)[0])
                return p, int(sum(mod.output_sizes[:shard]))
        return None, 0

    for name in names:
        fn = index[name]
        f = handles.get(fn)
        if f is None:
            f = handles[fn] = safe_open(os.path.join(model_dir, fn), framework="pt", device="cpu")
        sl = f.get_slice(name)
        dt, shape = sl.get_dtype(), list(sl.get_shape())
        p, off = target(name)
        is_q8 = dt == "U8" and len(shape) == 3 and shape[-1] == 34
        if p is None:
            if is_q8:
                fail(f"{name}: Q8_0 checkpoint tensor has no parameter")
            elif dt in ("F32", "F16"):
                rep["unmapped"].append(name)
            continue
        kind = _kind(name)
        if is_q8:
            n, nb = shape[0], shape[1]
            pv = p.data.view(p.shape[0], -1)
            if off + n > pv.shape[0] or pv.shape[1] != nb * 32 or p.dtype != torch.bfloat16:
                fail(f"{name}: Q8 [{n},{nb},34] does not fit param {tuple(p.shape)} {p.dtype} at row {off}")
                continue
            worst = 0.0
            for r0, r1 in _rows_to_check(name, n, rng):
                raw = sl[r0:r1].numpy()
                ref = torch.from_numpy(sf.deq_q8_0(raw).reshape(r1 - r0, nb * 32)).to(torch.bfloat16)
                got = pv[off + r0: off + r1].cpu()
                if not torch.equal(got, ref):
                    d = (got.float() - ref.float()).abs()
                    worst = max(worst, float(d.max()))
                    fail(f"{name} rows {r0}:{r1} (param rows +{off}): {int((d > 0).sum())} elements differ, "
                         f"max |diff| {float(d.max()):.3g}; param sample {got.flatten()[:4].tolist()} "
                         f"ref {ref.flatten()[:4].tolist()}")
                    break
                q8_rows += r1 - r0
            rep["q8"][kind] += 1
        elif dt in ("F32", "F16"):
            src = sl[:].float() if dt == "F16" else sl[:]
            if p.numel() != src.numel():
                fail(f"{name}: {dt} {shape} vs param {tuple(p.shape)}")
                continue
            got = p.data.reshape(-1).cpu()
            if p.dtype == torch.float32:
                if not torch.equal(got, src.reshape(-1).float()):
                    fail(f"{name}: fp32 param differs from the checkpoint")
                rep["fp32"][kind] += 1
                continue
            ref = src.reshape(-1).to(p.dtype)
            if not torch.equal(got, ref):
                d = (got.float() - ref.float()).abs()
                fail(f"{name}: {p.dtype} param != round({dt}) ({int((d > 0).sum())} elements, max |diff| "
                     f"{float(d.max()):.3g})")
                continue
            s32 = src.reshape(-1).float()
            err = (ref.float() - s32).abs()
            amax = float(s32.abs().max()) or 1.0
            nz = s32.abs() > 0
            rel = float((err[nz] / s32.abs()[nz]).max()) if bool(nz.any()) else 0.0
            k = rep["narrow"].setdefault(f"{dt}->{str(p.dtype).replace('torch.', '')} {kind}",
                                         {"tensors": 0, "max_err_over_amax": 0.0, "max_rel_err": 0.0})
            k["tensors"] += 1
            k["max_err_over_amax"] = max(k["max_err_over_amax"], float(err.max()) / amax)
            k["max_rel_err"] = max(k["max_rel_err"], rel)
        elif dt == "BF16" and (name in ("model.embed_tokens.weight", "lm_head.weight") or name.endswith("mlp.gate.weight")):
            n = shape[0]
            rows = sorted({0, n - 1, *rng.choice(n, size=14, replace=False).tolist()})
            for r in rows:
                if not torch.equal(p.data[r].cpu(), sl[r:r + 1][0].to(p.dtype)):
                    fail(f"{name}: bf16 row {r} differs")
                    break
            rep["bf16"][kind] += 1

    rep["q8_rows_checked"] = q8_rows
    rep["seconds"] = round(time.time() - t0, 1)
    out = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in rep.items()}
    log.info("edge0: verify: Q8_0 %d tensors (%d rows, layers %s complete) exact vs strata_formats.deq_q8_0; "
             "narrowed %d tensors exact vs bf16 RNE of the source; fp32 %d; bf16 %d; unmapped %d; failures %d (%.0f s)",
             sum(rep["q8"].values()), q8_rows, list(_FULL_LAYERS), sum(v["tensors"] for v in rep["narrow"].values()),
             sum(rep["fp32"].values()), sum(rep["bf16"].values()), len(rep["unmapped"]), len(rep["failures"]),
             rep["seconds"])
    for k, v in sorted(rep["narrow"].items()):
        log.info("edge0: verify: narrowing %s: %d tensors, max |err|/amax %.2e, max rel err %.2e", k, v["tensors"],
                 v["max_err_over_amax"], v["max_rel_err"])
    if rep["unmapped"]:
        log.info("edge0: verify: unmapped F32/F16/Q8 checkpoint tensors (not compared): %s", rep["unmapped"][:20])
    try:
        d = os.environ.get("EDGE0_LOG_DIR", "/root/.cache/edge0-vllm/logs")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"verify-load-{time.strftime('%Y%m%d-%H%M%S')}.json")
        json.dump(out, open(path, "w"), indent=1)
        log.info("edge0: verify: report %s", path)
    except OSError:
        pass
    if rep["failures"]:
        raise RuntimeError(f"edge0: EDGE0_VERIFY_LOAD found {len(rep['failures'])} mismatches: {rep['failures'][:5]}")
    return out
