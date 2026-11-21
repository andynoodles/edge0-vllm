#!/usr/bin/env python3
"""edge0/convert/st_io.py - low-memory IO helpers for the Edge0 converter (numpy only).

- ShardWriter: a minimal safetensors writer.  The header (names, dtypes, shapes, offsets) is computed up front from
  the contract specs, so each tensor can be streamed in pieces, in any interleaving across tensors (each tensor is
  written front to back).  It writes `<name>.partial`, fsyncs, drops the written pages from the page cache and renames
  it, so a finished shard is always complete and a crash leaves only a `.partial` behind (resumability).
- read_header / ShardReader: parse a safetensors header and read tensor slices with pread (no torch needed).
- SrcFile: read-only mmap of a (multi-GB) source file; every read copies a slice out and immediately drops those pages
  (madvise DONTNEED + posix_fadvise DONTNEED), so the converter's footprint stays one chunk, not the file.
- Throttle: optional MB/s cap on reads+writes (used while the Strata benchmark shares the NVMe).
- Codecs: float32 -> bf16 (round to nearest even), float -> E4M3FN (nearest, ties to even), strict E4M3FN decode.
"""
from __future__ import annotations

import json
import mmap
import os
import struct
import time

import numpy as np

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "U8": 1, "F8_E4M3": 1}
NP_DTYPE = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "I32": np.int32, "U8": np.uint8,
            "F8_E4M3": np.uint8}
PAGE = mmap.PAGESIZE
_HAS_FADVISE = hasattr(os, "posix_fadvise")


def fadvise_dontneed(fd: int, off: int = 0, length: int = 0) -> None:
    if _HAS_FADVISE:
        try:
            os.posix_fadvise(fd, off, length, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass


class Throttle:
    """Average-rate limiter over all IO bytes reported to it (0 = unlimited)."""

    def __init__(self, mbps: float = 0.0):
        self.rate = float(mbps) * 1e6
        self.t0 = time.monotonic()
        self.bytes = 0

    def account(self, n: int) -> None:
        self.bytes += int(n)
        if self.rate > 0:
            ahead = self.bytes / self.rate - (time.monotonic() - self.t0)
            if ahead > 0.005:
                time.sleep(ahead)


NO_THROTTLE = Throttle(0)


# ============================================================================== source files
class SrcFile:
    def __init__(self, path, throttle: Throttle = NO_THROTTLE):
        self.path = str(path)
        self.fd = os.open(self.path, os.O_RDONLY)
        self.size = os.fstat(self.fd).st_size
        self.mm = mmap.mmap(self.fd, 0, prot=mmap.PROT_READ) if self.size else None
        self.throttle = throttle
        self.bytes_read = 0

    def read(self, off: int, n: int) -> np.ndarray:
        """Copy bytes [off, off+n) into a fresh uint8 array, then drop the pages from this process and the cache."""
        assert 0 <= off and off + n <= self.size, (self.path, off, n, self.size)
        a = np.frombuffer(self.mm, dtype=np.uint8, count=n, offset=off).copy()
        self.drop(off, n)
        self.bytes_read += n
        self.throttle.account(n)
        return a

    def drop(self, off: int, n: int) -> None:
        start = off - off % PAGE
        length = off + n - start
        try:
            self.mm.madvise(mmap.MADV_DONTNEED, start, min(length, self.size - start))
        except (AttributeError, OSError, ValueError):
            pass
        fadvise_dontneed(self.fd, start, length)

    def close(self) -> None:
        if self.mm is not None:
            self.mm.close()
            self.mm = None
        os.close(self.fd)


# ============================================================================== safetensors writer
def build_header(specs, metadata: dict | None = None) -> tuple[bytes, dict, int]:
    """specs: objects with .name .dtype .shape .nbytes, in file byte order.  Returns (header bytes incl. the 8-byte
    length prefix, {name: absolute file offset}, total file size)."""
    hdr = {"__metadata__": metadata or {"format": "pt"}}
    rel, off = {}, 0
    for s in specs:
        assert s.name not in hdr, s.name
        hdr[s.name] = {"dtype": s.dtype, "shape": list(s.shape), "data_offsets": [off, off + s.nbytes]}
        rel[s.name] = off
        off += s.nbytes
    raw = json.dumps(hdr, separators=(",", ":"), ensure_ascii=True).encode()
    raw += b" " * ((-len(raw)) % 8)
    base = 8 + len(raw)
    return struct.pack("<Q", len(raw)) + raw, {k: base + v for k, v in rel.items()}, base + off


class ShardWriter:
    def __init__(self, path, specs, throttle: Throttle = NO_THROTTLE, sync_every: int = 256 << 20,
                 metadata: dict | None = None):
        self.path = str(path)
        self.tmp = self.path + ".partial"
        self.specs = {s.name: s for s in specs}
        header, self.abs_off, self.size = build_header(specs, metadata)
        self.cursor = {s.name: 0 for s in specs}
        self.throttle = throttle
        self.sync_every = sync_every
        self.unsynced = 0
        self.bytes_written = 0
        self.fd = os.open(self.tmp, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        os.ftruncate(self.fd, self.size)
        self._pwrite(header, 0)

    def _pwrite(self, data, off: int) -> None:
        mv = memoryview(data).cast("B")
        while len(mv):
            n = os.pwrite(self.fd, mv, off)
            mv, off = mv[n:], off + n
            self.bytes_written += n
            self.unsynced += n
            self.throttle.account(n)
        if self.unsynced >= self.sync_every:
            (getattr(os, "fdatasync", None) or os.fsync)(self.fd)          # macOS has no fdatasync
            fadvise_dontneed(self.fd)
            self.unsynced = 0

    def write(self, name: str, arr) -> None:
        """Append `arr` (any numpy array / bytes; raw bytes are written) to tensor `name`."""
        a = np.ascontiguousarray(arr)
        n = a.nbytes
        s = self.specs[name]
        c = self.cursor[name]
        if c + n > s.nbytes:
            raise ValueError("%s: write of %d B at %d overflows %d B" % (name, n, c, s.nbytes))
        self._pwrite(a.view(np.uint8).reshape(-1), self.abs_off[name] + c)
        self.cursor[name] = c + n

    def missing(self) -> list[str]:
        return [k for k, s in self.specs.items() if self.cursor[k] != s.nbytes]

    def finish(self) -> None:
        miss = self.missing()
        if miss:
            raise ValueError("%s: incomplete tensors %s" % (self.path, ["%s %d/%d" % (k, self.cursor[k],
                                                                         self.specs[k].nbytes) for k in miss[:5]]))
        os.fsync(self.fd)
        fadvise_dontneed(self.fd)
        os.close(self.fd)
        self.fd = -1
        os.replace(self.tmp, self.path)
        dfd = os.open(os.path.dirname(os.path.abspath(self.path)), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)

    def abort(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


# ============================================================================== safetensors reader
def read_header(path) -> tuple[dict, int]:
    """(header dict without __metadata__, data start offset)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 100 << 20:
            raise ValueError("%s: header of %d B" % (path, n))
        raw = f.read(n)
    hdr = json.loads(raw)
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


class ShardReader:
    def __init__(self, path):
        self.path = str(path)
        self.header, self.base = read_header(self.path)
        self.fd = os.open(self.path, os.O_RDONLY)

    def info(self, name):
        return self.header[name]

    def raw(self, name: str, start: int = 0, length: int | None = None) -> np.ndarray:
        h = self.header[name]
        b, e = h["data_offsets"]
        length = (e - b - start) if length is None else length
        assert 0 <= start and start + length <= e - b
        buf = os.pread(self.fd, length, self.base + b + start)
        assert len(buf) == length
        fadvise_dontneed(self.fd, self.base + b + start, length)
        return np.frombuffer(buf, np.uint8)

    def tensor(self, name: str) -> np.ndarray:
        h = self.header[name]
        return self.raw(name).view(NP_DTYPE[h["dtype"]]).reshape(h["shape"])

    def rows(self, name: str, row0: int, nrows: int) -> np.ndarray:
        """Rows [row0, row0+nrows) along dim 0."""
        h = self.header[name]
        shape = h["shape"]
        row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * DTYPE_BYTES[h["dtype"]]
        a = self.raw(name, row0 * row_bytes, nrows * row_bytes).view(NP_DTYPE[h["dtype"]])
        return a.reshape([nrows] + list(shape[1:]))

    def close(self):
        os.close(self.fd)


# ============================================================================== codecs
def bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16 bit patterns (uint16), round to nearest even.  Inputs must be finite."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("non-finite value in bf16 conversion")
    u = x.view(np.uint32)
    return ((u + (np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1)))) >> np.uint32(16)).astype(np.uint16)


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(bits).view(np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def _e4m3fn_strict() -> np.ndarray:
    """E4M3FN decode as torch.float8_e4m3fn does it: 0x7F / 0xFF are NaN (no inf), subnormal m/8 * 2^-6."""
    t = np.zeros(256, np.float32)
    for x in range(256):
        s, e, m = x >> 7, (x >> 3) & 15, x & 7
        if e == 15 and m == 7:
            f = np.nan
        elif e == 0:
            f = (m / 8.0) * 2.0 ** -6
        else:
            f = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        t[x] = -f if s else f
    return t


E4M3FN_STRICT = _e4m3fn_strict()
_E4M3_POS = E4M3FN_STRICT[:0x7F].astype(np.float64)        # codes 0x00..0x7E, strictly increasing, max 448


def e4m3_encode(x: np.ndarray) -> np.ndarray:
    """Non-negative float -> nearest E4M3FN code (ties to even mantissa), saturating at 448 (0x7E)."""
    x = np.asarray(x, np.float64)
    assert (x >= 0).all() or not x.size
    hi = np.clip(np.searchsorted(_E4M3_POS, x, side="left"), 0, 0x7E)
    lo = np.clip(hi - 1, 0, 0x7E)
    d_hi = np.abs(_E4M3_POS[hi] - x)
    d_lo = np.abs(x - _E4M3_POS[lo])
    pick_lo = (d_lo < d_hi) | ((d_lo == d_hi) & (lo % 2 == 0))
    return np.where(pick_lo, lo, hi).astype(np.uint8)


E2M1_GRID = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
E2M1_VALUES = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def e2m1_encode(q: np.ndarray) -> np.ndarray:
    """float -> nearest E2M1 code (sign bit 3), ties to the even code, saturating at |6|."""
    q = np.asarray(q, np.float32)
    a = np.abs(q)
    dist = np.abs(a[..., None] - E2M1_GRID)                    # (..., 8)
    idx = np.argmin(dist, axis=-1)                              # first minimum = lower code on ties
    nxt = np.minimum(idx + 1, 7)
    tie_up = (idx % 2 == 1) & (idx < 7) & (np.take_along_axis(dist, nxt[..., None], -1)[..., 0] ==
                                           np.take_along_axis(dist, idx[..., None], -1)[..., 0])
    idx = np.where(tie_up, nxt, idx).astype(np.uint8)
    return idx | (np.signbit(q) & (idx > 0)).astype(np.uint8) << np.uint8(3)
