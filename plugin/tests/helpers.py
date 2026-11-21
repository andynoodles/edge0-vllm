"""Shared test helpers: paths, the converter contract's ggml -> ModelOpt NVFP4 repack (DESIGN.md 1.5), and access to
the numpy oracle ``edge0/convert/strata_formats.py``.

Environment (defaults match the container mounts of serve.sh / the task's docker run flags):
  EDGE0_SRC_MODEL   source model folder   (/models/Edge0-145B-NVFP4)
  EDGE0_TREE        edge0 tree            (/workspace/edge0)
  EDGE0_CKPT        converted checkpoint  (unset = tests that need it skip)
Real-data tests read small slices only (one expert record = 1.9 MB, a few D/U rows).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

SRC_MODEL = Path(os.environ.get("EDGE0_SRC_MODEL", "/models/Edge0-145B-NVFP4"))
TREE = Path(os.environ.get("EDGE0_TREE", "/workspace/edge0"))
CKPT = os.environ.get("EDGE0_CKPT")


def have_src_model() -> bool:
    return (SRC_MODEL / "experts.bin").is_file()


def strata_formats():
    """Import edge0/convert/strata_formats.py (the oracle) or return None."""
    conv = TREE / "convert"
    if not (conv / "strata_formats.py").is_file():
        here = Path(__file__).resolve().parents[2] / "convert"
        conv = here if (here / "strata_formats.py").is_file() else None
    if conv is None:
        return None
    if str(conv) not in sys.path:
        sys.path.insert(0, str(conv))
    import strata_formats  # noqa: E402
    return strata_formats


# ---------------------------------------------------------------------------------------------- contract repack
def ggml_nvfp4_rows_to_modelopt(raw: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """ggml ``block_nvfp4`` rows (``[rows, k/64*36]`` uint8) -> ModelOpt (``codes uint8 [rows, k/2]``,
    ``scales uint8 [rows, k/16]`` = E4M3 bit patterns), exactly the DESIGN.md 1.5 formula."""
    rows = raw.shape[0]
    nb = k // 64
    b = raw.reshape(rows, nb, 36)
    scales = b[:, :, :4].reshape(rows, nb * 4).copy()
    qs = b[:, :, 4:].reshape(rows, nb, 4, 8)
    codes = np.concatenate([qs & 15, qs >> 4], axis=3).reshape(rows, k)       # j<8: low nibble of qs[j]; j>=8: high
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    return packed, scales


def expert_record_to_modelopt(blob: np.ndarray, sf) -> dict:
    """One 1,935,376-byte experts.bin record -> the contract's per-expert ModelOpt tensors (numpy)."""
    L = sf.BlobLayout(40, 40)
    gate = blob[:L.up_off].reshape(sf.N_FF, L.gu_row)
    up = blob[L.up_off:L.down_off].reshape(sf.N_FF, L.gu_row)
    down = blob[L.down_off:L.tail_off].reshape(sf.ELL, L.d_row)
    tail = np.frombuffer(blob[L.tail_off:L.tail_off + 16].tobytes(), np.float32)
    w13, s13 = ggml_nvfp4_rows_to_modelopt(np.concatenate([gate, up]), sf.ELL)
    w2, s2 = ggml_nvfp4_rows_to_modelopt(down, sf.N_FF)
    return {"w13": w13, "s13": s13, "g13": np.array([tail[0], tail[1]], np.float32),
            "w2": w2, "s2": s2, "g2": np.float32(tail[2])}


def synthetic_ggml_nvfp4_rows(rng: np.random.Generator, rows: int, k: int) -> np.ndarray:
    """Random valid ggml NVFP4 rows: scales are E4M3 patterns without bit 7 and != 0x7F."""
    nb = k // 64
    b = rng.integers(0, 256, size=(rows, nb, 36), dtype=np.uint8)
    sc = rng.integers(1, 0x7F, size=(rows, nb, 4), dtype=np.uint8)
    b[:, :, :4] = sc
    return b.reshape(rows, nb * 36)


def to_torch_fp8(u8: np.ndarray):
    import torch
    return torch.from_numpy(np.ascontiguousarray(u8)).view(torch.float8_e4m3fn)


def init_single_process_dist(vllm_config=None) -> None:
    """vLLM's TP/PP/EP groups for one CPU process (gloo), so vLLM layers can be constructed in tests."""
    from vllm.distributed import parallel_state as ps
    if ps._TP is not None:
        return
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    from vllm.config import VllmConfig, set_current_vllm_config
    with set_current_vllm_config(vllm_config or VllmConfig()):
        ps.init_distributed_environment(world_size=1, rank=0, distributed_init_method=f"tcp://127.0.0.1:{port}",
                                        local_rank=0, backend="gloo")
        ps.initialize_model_parallel(1, 1)


def make_vllm_config(ckpt: str, **overrides):
    """A real VllmConfig for a converted checkpoint (needs CUDA for vLLM's platform), as ``vllm serve`` builds it."""
    import edge0_vllm
    edge0_vllm.register()
    from vllm.engine.arg_utils import EngineArgs
    kw = dict(model=ckpt, quantization="edge0", dtype="bfloat16", max_model_len=4096, max_num_seqs=4,
              max_num_batched_tokens=512, enforce_eager=True, skip_tokenizer_init=True, gpu_memory_utilization=0.5,
              engram_config={"cpu_offload": False, "host_file_gather": True})
    kw.update(overrides)
    return EngineArgs(**kw).create_engine_config()
