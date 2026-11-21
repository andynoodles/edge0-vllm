#!/usr/bin/env python3
"""PLE row-read latency per decode step: vLLM's serial reader vs parallel readers (single-stream lever 4).

  python3 ple_io_bench.py <ple-fp8.gguf> [--rows 16,64] [--trials 40] [--threads 16]

Rows are random (n-gram hashes are uniform over the 320M-row table), so most miss the page cache, as in serving.
Modes: serial = vLLM's ``_FilePlane.read`` (fadvise WILLNEED for every run, then one preadv per row);
threads = the same per-row preadv spread over a thread pool, one task per row (EDGE0_PLE_IO=threads with
EDGE0_PLE_MIN_ROWS=1); fadvise_only = serial preadv after fadvise, with a 300 us gap (does readahead overlap?).
"""
import argparse
import mmap
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROW = 160
OFF = 352
NROWS = 320001536


def read_serial(fd, ids, out, fadvise=True, gap_s=0.0):
    start = OFF + ids * ROW
    page = mmap.PAGESIZE
    if fadvise:
        for s in start.tolist():
            lo = s // page
            hi = (s + ROW - 1) // page
            os.posix_fadvise(fd, lo * page, (hi - lo + 1) * page, os.POSIX_FADV_WILLNEED)
    if gap_s:
        time.sleep(gap_s)
    for row, off in zip(out, start.tolist()):
        os.preadv(fd, [row], off)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--rows", default="16,64")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    fd = os.open(a.path, os.O_RDONLY)
    fd_rand = os.open(a.path, os.O_RDONLY)
    os.posix_fadvise(fd_rand, 0, 0, os.POSIX_FADV_RANDOM)
    fd_direct = os.open(a.path, os.O_RDONLY | os.O_DIRECT)
    page = mmap.PAGESIZE
    bufs = [mmap.mmap(-1, 2 * page) for _ in range(256)]
    pool = ThreadPoolExecutor(max_workers=a.threads)
    rng = np.random.default_rng(int(time.time()))

    def read_threads(ids, out):
        start = (OFF + ids * ROW).tolist()
        futs = [pool.submit(os.preadv, fd, [out[i]], start[i]) for i in range(len(ids))]
        for f in futs:
            f.result()

    def read_threads_chunked(ids, out):
        n = len(ids)
        parts = min(a.threads, n)
        b = [n * i // parts for i in range(parts + 1)]
        futs = [pool.submit(read_serial, fd, ids[x:y], out[x:y], False) for x, y in zip(b[:-1], b[1:]) if y > x]
        for f in futs:
            f.result()

    def read_direct(ids, out, threaded):
        start = (OFF + ids * ROW).tolist()

        def one(i):
            s = start[i]
            lo = s // page * page
            n = 2 * page if (s + ROW - 1) // page != s // page else page
            mv = memoryview(bufs[i])[:n]
            os.preadv(fd_direct, [mv], lo)
            out[i] = np.frombuffer(bufs[i], np.uint8, ROW, s - lo)
        if threaded:
            for f in [pool.submit(one, i) for i in range(len(ids))]:
                f.result()
        else:
            for i in range(len(ids)):
                one(i)

    modes = {
        "serial(vllm)": lambda ids, out: read_serial(fd, ids, out),
        "threads/row": read_threads,
        "threads/chunk": read_threads_chunked,
        "fadvise+300us": lambda ids, out: read_serial(fd, ids, out, True, 0.0003),
        "random+fadv": lambda ids, out: read_serial(fd_rand, ids, out),
        "random,nofadv": lambda ids, out: read_serial(fd_rand, ids, out, False),
        "random/threads": lambda ids, out: [f.result() for f in [pool.submit(os.preadv, fd_rand, [out[i]], int(OFF + ids[i] * ROW)) for i in range(len(ids))]],
        "direct serial": lambda ids, out: read_direct(ids, out, False),
        "direct threads": lambda ids, out: read_direct(ids, out, True),
    }
    for n in [int(x) for x in a.rows.split(",")]:
        out = np.empty((n, ROW), np.uint8)
        res = {m: [] for m in modes}
        for t in range(a.trials):
            for m, fn in modes.items():
                ids = np.sort(rng.integers(0, NROWS, n))
                t0 = time.perf_counter()
                fn(ids, out)
                res[m].append((time.perf_counter() - t0) * 1e3)
        for m, v in res.items():
            print(f"rows {n:3d} {m:15s} median {statistics.median(v):6.3f} ms  mean {statistics.mean(v):6.3f}  "
                  f"p90 {sorted(v)[int(0.9 * len(v))]:6.3f}")


if __name__ == "__main__":
    main()
