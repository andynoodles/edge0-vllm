"""PLE table served in place from ``ple-fp8.gguf`` (DESIGN.md 2.7).

vLLM's ``host_file_gather`` backend (``--engram-config '{"cpu_offload": false, "host_file_gather": true}'``)
computes each step's n-gram ids, reads every distinct 160-byte row with ``preadv`` and copies the rows into fixed
device staging buffers; the forward pass only reads staging (CUDA-graph safe), the 51.2 GB table is never
allocated. It locates the table among the checkpoint's safetensors shards via
``vllm.models.qwen4_exp.nvidia.ngram_embedding.discover_table_layout`` (a module global looked up at call time).
``install_patch`` replaces that function: for the ``weight`` plane of an Edge0 checkpoint it returns one shard =
the GGUF tensor (offset ``edge0.ple_data_offset``), after checking the file; everything else goes to the original.

``EDGE0_PLE_IO=threads``: the serial ``_FilePlane.read`` is split over ``EDGE0_PLE_THREADS`` threads (``os.preadv``
releases the GIL); each thread runs vLLM's own read (fadvise + preadv) on a contiguous slice of the sorted ids.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import Edge0CheckpointSpec

_GGUF_MAGIC = b"GGUF"


def ple_table_location(ckpt_dir) -> tuple[str, Edge0CheckpointSpec] | None:
    """(absolute path of the GGUF table, spec) for an Edge0 checkpoint directory, else None. Validates the file."""
    spec = Edge0CheckpointSpec.from_checkpoint_dir(ckpt_dir)
    if spec is None:
        return None
    path = Path(ckpt_dir) / spec.ple_file
    if not path.is_file():
        raise FileNotFoundError(f"PLE table {path} (config.json edge0.ple_file) is missing")
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic != _GGUF_MAGIC:
        raise ValueError(f"{path}: not a GGUF file (magic {magic!r})")
    want = spec.ple_data_offset + spec.ple_rows * spec.ple_row_bytes
    size = path.stat().st_size
    if size != want:
        raise ValueError(f"{path}: {size} B, expected offset {spec.ple_data_offset} + {spec.ple_rows} rows x "
                         f"{spec.ple_row_bytes} B = {want}")
    return str(path.resolve()), spec


def make_discover_table_layout(orig, table_layout_cls, shard_location_cls, logger=None):
    def discover_table_layout(files, layer_index, num_rows, embedding_dim, dtype, split_parts, suffix="weight"):
        dirs = sorted({os.path.dirname(os.path.abspath(f)) for f in files})
        loc = None
        for d in dirs:
            loc = ple_table_location(d)
            if loc is not None:
                break
        if loc is None or suffix != "weight":
            return orig(files, layer_index, num_rows, embedding_dim, dtype, split_parts, suffix=suffix)
        path, spec = loc
        row_bytes = int(embedding_dim) * dtype.itemsize
        if row_bytes != spec.ple_row_bytes:
            raise ValueError(f"PLE row of {embedding_dim} x {dtype} = {row_bytes} B, checkpoint says "
                             f"{spec.ple_row_bytes} B")
        if num_rows > spec.ple_rows:
            raise ValueError(f"PLE layer {layer_index} needs {num_rows} rows, {path} has {spec.ple_rows}")
        if logger is not None:
            logger.info("edge0: PLE layer %d table = %s @ %d (%d rows x %d B, in place)", layer_index, path,
                        spec.ple_data_offset, spec.ple_rows, row_bytes)
        return table_layout_cls(num_rows=num_rows, row_bytes=row_bytes, rows_per_shard=spec.ple_rows,
                                shards=(shard_location_cls(path, spec.ple_data_offset, spec.ple_rows),))

    discover_table_layout._edge0_patched = True
    return discover_table_layout


def install_patch(settings=None, logger=None) -> None:
    import vllm.models.qwen4_exp.nvidia.ngram_embedding as ne
    from vllm.models.qwen4_exp.nvidia.ple_pageable import ShardLocation, TableLayout

    if not getattr(ne.discover_table_layout, "_edge0_patched", False):
        ne.discover_table_layout = make_discover_table_layout(ne.discover_table_layout, TableLayout, ShardLocation,
                                                              logger)
    if settings is not None and settings.ple_io == "threads":
        _install_threaded_reader(ne, settings.ple_threads, logger)
    from . import ple_defer         # EDGE0_PLE_DEFER=1: rows read by a reader thread, waited for inside the graph
    if ple_defer.ENABLED:
        ple_defer.install(logger)


def _install_threaded_reader(ne, threads: int, logger=None) -> None:
    plane = ne._FilePlane
    if getattr(plane.read, "_edge0_threads", False):
        return
    serial = plane.read
    pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="edge0-ple")
    min_rows = 64

    def read(self, ids, out):
        n = len(ids)
        if n < 2 * min_rows:
            return serial(self, ids, out)
        parts = min(threads, n // min_rows)
        bounds = [n * i // parts for i in range(parts + 1)]
        futs = [pool.submit(serial, self, ids[a:b], out[a:b]) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
        for f in futs:
            f.result()

    read._edge0_threads = True
    plane.read = read
    if logger is not None:
        logger.info("edge0: PLE row reader uses %d threads", threads)
