"""EDGE0_PLE_DEFER (ple_defer.py, csrc/ple_reader.cpp, csrc/ple_wait.cu): the reader thread reads the rows of a step
after its ids' D2H copy (CUDA event), the in-graph wait-copy kernel waits for the step's flag and copies them.
GPU only: rows equal to a direct read of the file, steps enqueued back to back without a host sync, a GPU that is
still busy when the step is submitted, a captured CUDA graph replayed for new steps, duplicate ids, and the
timeout path (a step that is never submitted)."""
import os
import tempfile
import types

import numpy as np
import pytest
import torch

_GPU = torch.cuda.is_available()
ROW = 160


def _table(n_rows: int, base: int = 352):
    fd, path = tempfile.mkstemp(prefix="ple-test-")
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, size=(n_rows, ROW), dtype=np.uint8)
    with os.fdopen(fd, "wb") as f:
        f.write(b"\0" * base)
        f.write(data.tobytes())
    return path, data, base


def _emb(path, base, n_rows, max_tokens, heads, dev):
    plane = types.SimpleNamespace(row_bytes=ROW, rows_per_shard=n_rows, fds=np.array([os.open(path, os.O_RDONLY)]),
                                  bases=np.array([base]))
    max_ids = max_tokens * heads
    return types.SimpleNamespace(
        _planes={"weight": plane}, _nvfp4=False, _suffixes=("weight",), _num_heads=heads, org_vocab_size=n_rows,
        layer_index=1, _host_ids=torch.empty((max_tokens, heads), dtype=torch.int64, pin_memory=True),
        _host_rows={"weight": torch.empty((max_ids, ROW), dtype=torch.uint8, pin_memory=True)},
        _staging={"weight": torch.zeros((max_ids, ROW), dtype=torch.uint8, device=dev)},
        weight=torch.empty(0, ROW, dtype=torch.uint8))


def _post(rd, emb, ids_dev, nt, seq):
    if rd.doorbell:
        rd.ext.post_ids(ids_dev.reshape(-1).contiguous(), rd.ids_dev, rd.idsflag_dev, seq)
    else:
        emb._host_ids[:nt].copy_(ids_dev, non_blocking=True)


@pytest.mark.skipif(not _GPU, reason="needs a GPU")
@pytest.mark.parametrize("wake", ["doorbell", "event"])
def test_deferred_rows_match_file(wake, monkeypatch):
    from edge0_vllm import ple_defer as pd
    monkeypatch.setattr(pd, "WAKE", wake)
    monkeypatch.setattr(pd, "_STATE", None)
    dev = torch.device("cuda", 0)
    n_rows, heads, max_tokens = 5000, 16, 64
    path, data, base = _table(n_rows)
    try:
        emb = _emb(path, base, n_rows, max_tokens, heads, dev)
        rd = pd._reader(emb)
        assert rd is not None and pd._STATE is not None
        st = pd._STATE
        g = torch.Generator(device="cpu").manual_seed(1)
        outs, wants, n_ids, n_uniq = [], [], 0, 0
        # 12 steps enqueued without a host sync; the GPU sleeps before each ids copy (still busy at submit time)
        for step in range(12):
            nt = [4, 1, 7, 64][step % 4]
            ids = torch.randint(0, n_rows, (nt, heads), generator=g)
            if step % 3 == 0:
                ids[:, 1] = ids[:, 0]                     # duplicates
            ids_dev = ids.to(dev)
            torch.cuda._sleep(2_000_000)                  # ~1 ms of GPU work before the copy
            st.seq += 1
            _post(rd, emb, ids_dev, nt, st.seq)
            rd.r.submit(torch.cuda.current_stream().cuda_stream, nt * heads, st.seq)
            st.expect.fill_(st.seq)
            out = pd._forward(emb, torch.empty(nt, device=dev))
            outs.append(out.clone())
            wants.append(data[ids.numpy().reshape(-1)].reshape(nt, heads * ROW))
            n_ids += ids.numel()
            n_uniq += np.unique(ids.numpy()).size
        torch.cuda.synchronize()
        for o, w in zip(outs, wants):
            np.testing.assert_array_equal(o.cpu().numpy(), w)
        s = rd.stats()
        assert s["jobs"] == 12 and s["errors"] == 0 and s["gpu_timeouts"] == 0 and s["gpu_calls"] == 12
        assert s["ids"] == n_ids and s["uniq"] == n_uniq, (s, n_ids, n_uniq)

        # CUDA graph: capture one 4-token step, replay it for new steps
        nt = 4
        x = torch.empty(nt, device=dev)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            with torch.cuda.graph(gr, stream=side):
                y = pd._forward(emb, x)
                yc = y.clone()
        torch.cuda.synchronize()
        for _ in range(5):
            ids = torch.randint(0, n_rows, (nt, heads), generator=g)
            torch.cuda._sleep(1_000_000)
            st.seq += 1
            _post(rd, emb, ids.to(dev), nt, st.seq)
            rd.r.submit(torch.cuda.current_stream().cuda_stream, nt * heads, st.seq)
            st.expect.fill_(st.seq)
            gr.replay()
            torch.cuda.synchronize()
            np.testing.assert_array_equal(yc.cpu().numpy(), data[ids.numpy().reshape(-1)].reshape(nt, heads * ROW))

        # timeout: expect a step that is never submitted -> the kernel gives up (stale rows), counted
        rd.timeout_ns = int(50e6)
        st.expect.fill_(st.seq + 1)
        pd._forward(emb, x)
        torch.cuda.synchronize()
        assert rd.stats()["gpu_timeouts"] == 1
        st.expect.fill_(st.seq)
        rd.r.stop()
    finally:
        os.unlink(path)
