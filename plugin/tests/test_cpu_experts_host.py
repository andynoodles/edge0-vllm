"""CPU only: the EDGE0_CPU_EXPERTS kernel (csrc/cpu_experts.cpp) on experts packed by a numpy port of vLLM's Marlin
repack (gptq_marlin_repack.cu 4-bit, marlin_permute_scales, nvfp4_marlin_process_scales), against an fp32 numpy
reference with the same roundings; and the pool's task split. Needs the built extension (x86-64 with AVX-512); no
GPU. The GPU test (test_cpu_experts.py) checks the same kernel against vLLM's real repack and Marlin output.
"""
import numpy as np
import pytest
import torch

LAT, FF = 1792, 640
_E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0., -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def _ext():
    try:
        from edge0_vllm import cpu_experts
        return cpu_experts.load()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"extension not available: {e}")


def e4m3_to_f32(b: np.ndarray) -> np.ndarray:
    b = b.astype(np.int32)
    e = (b >> 3) & 15
    m = b & 7
    v = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e - 7))
    return np.where((b & 0x7F) == 0x7F, np.nan, v).astype(np.float32) * np.where(b & 0x80, -1, 1)


def marlin_repack(codes: np.ndarray) -> np.ndarray:
    """codes uint8 [size_n, size_k] (0..15) -> Marlin int32 [size_k/16, size_n*2]."""
    size_n, size_k = codes.shape
    kt_n, nt_n = size_k // 16, size_n // 64
    out = np.zeros((kt_n, nt_n, 128), np.uint32)
    off = [0, 1, 8, 9]
    pack_idx = [0, 2, 4, 6, 1, 3, 5, 7]
    w = np.arange(128)
    th, warp = w // 4, w % 4
    n0 = warp * 16 + th // 4
    row = (th % 4) * 2
    for kt in range(kt_n):
        for nt in range(nt_n):
            vals = []
            for i in range(4):
                vals.append(codes[nt * 64 + n0, kt * 16 + row + off[i]])
            for i in range(4):
                vals.append(codes[nt * 64 + n0 + 8, kt * 16 + row + off[i]])
            word = np.zeros(128, np.uint32)
            for i in range(8):
                word |= vals[pack_idx[i]].astype(np.uint32) << np.uint32(4 * i)
            out[kt, nt] = word
    return out.reshape(kt_n, nt_n * 128).view(np.int32)


def marlin_scales(s: np.ndarray, factor: float) -> np.ndarray:
    """E4M3 scale values (fp32, >= 0) [size_n, size_k/16] -> Marlin bytes [size_k/16, size_n]."""
    t = s.T.copy()                                   # [K/16, N]
    rows, n = t.shape
    perm = np.array([i + 8 * j for i in range(8) for j in range(8)])
    t = t.reshape(-1, 64)[:, perm].reshape(rows, n)
    t = t.reshape(-1, 4)[:, [0, 2, 1, 3]].reshape(rows, n)
    h = (t.astype(np.float32) * factor).astype(np.float16) * np.float16(128)
    h[h < 2] = 0
    bits = (h.view(np.uint16).astype(np.uint32) << 1) & 0xFFFF
    return (bits >> 8).astype(np.uint8)


def bf16(x):
    return torch.from_numpy(np.asarray(x, np.float32)).to(torch.bfloat16).float().numpy()


def make_expert(rng):
    c13 = rng.integers(0, 16, (2 * FF, LAT), dtype=np.uint8)
    c2 = rng.integers(0, 16, (LAT, FF), dtype=np.uint8)
    b13 = rng.integers(0x18, 0x48, (2 * FF, LAT // 16)).astype(np.uint8)     # E4M3 scales ~0.0039 .. 0.5
    b2 = rng.integers(0x18, 0x48, (LAT, FF // 16)).astype(np.uint8)
    s13, s2 = e4m3_to_f32(b13), e4m3_to_f32(b2)
    f13 = 2.0 ** np.floor(np.log2(448 / s13.max()))
    f2 = 2.0 ** np.floor(np.log2(448 / s2.max()))
    g13, g2 = np.float32(rng.uniform(0.002, 0.02)), np.float32(rng.uniform(0.002, 0.02))
    rec = np.concatenate([marlin_repack(c13).view(np.uint8).ravel(), marlin_repack(c2).view(np.uint8).ravel(),
                          marlin_scales(s13, f13).ravel(), marlin_scales(s2, f2).ravel()])
    assert rec.nbytes == 1935360
    gm13 = np.float32(g13 * 2.0 ** 119 / f13)                  # nvfp4_marlin_process_global_scale / factor
    gm2 = np.float32(g2 * 2.0 ** 119 / f2)
    W13 = _E2M1[c13] * np.repeat(s13, 16, axis=1) * g13
    W2 = _E2M1[c2] * np.repeat(s2, 16, axis=1) * g2
    return rec, gm13, gm2, W13, W2


def reference(x, wr, W13, W2):
    """fp32 math with the Marlin / silu_and_mul roundings (bf16 at c1, silu, h; y = bf16(bf16(acc) * bf16(w g2)))."""
    hh = bf16(x.astype(np.float64) @ W13.T.astype(np.float64))
    gate, up = hh[:, :FF], hh[:, FF:]
    sil = bf16(gate / (1 + np.exp(-gate)))
    h = bf16(sil * up)
    return h, (h.astype(np.float64) @ W2.T.astype(np.float64)) * wr[:, None]


def test_kernel_vs_numpy_marlin():
    ext = _ext()
    rng = np.random.default_rng(0)
    rec, gm13, gm2, W13, W2 = make_expert(rng)
    buf = torch.from_numpy(rec.copy())
    p = buf.data_ptr()
    w13, w2 = p, p + 1146880
    s13, s2 = w2 + 573440, w2 + 573440 + 143360
    n = 4
    x = torch.from_numpy(rng.standard_normal((n, LAT)).astype(np.float32) * 0.5).to(torch.bfloat16)
    wr = torch.from_numpy(rng.uniform(0.05, 0.4, n).astype(np.float32))
    h = ext.expert_h(w13, s13, float(gm13), x).numpy()
    y = ext.expert_forward(w13, w2, s13, s2, float(gm13), float(gm2), x, wr).float().numpy()
    h_ref, y_ref = reference(x.float().numpy(), wr.numpy(), W13, W2)
    rel_h = np.linalg.norm(h - h_ref) / np.linalg.norm(h_ref)
    rel_y = np.linalg.norm(y - y_ref) / np.linalg.norm(y_ref)
    eq_h = float((h == h_ref.astype(np.float32)).mean())
    print(f"h rel {rel_h:.2e} equal {eq_h:.4f}; y rel {rel_y:.2e}")
    assert rel_h < 4e-3 and rel_y < 6e-3 and eq_h > 0.9


def test_pool_host_job_matches_single_thread():
    ext = _ext()
    from edge0_vllm import hostmem
    rng = np.random.default_rng(1)
    recs = [make_expert(rng) for _ in range(3)]
    pool_mem = np.concatenate([r[0] for r in recs])               # three slots, record per slot
    # carve like Tier.in_host: w13 | w2 | s13 | s2 arrays of 3 slots each
    n = 3
    sizes = (1146880, 573440, 143360, 71680)
    arrs = []
    for k, sz in enumerate(sizes):
        o = sum(sizes[:k])
        arrs.append(np.concatenate([r[0][o:o + sz] for r in recs]))
    tier = torch.from_numpy(np.concatenate(arrs).copy())
    base = tier.data_ptr()
    ptrs = [base]
    for sz in sizes[:-1]:
        ptrs.append(ptrs[-1] + n * sz)
    lay = dict(zip(("tasks", "route_ids", "route_w", "x", "y", "total", "header"), ext.layout(64)))
    mb = hostmem.alloc(lay["total"], "host_register", "cpu")
    pool = ext.Pool(mb.cpu.data_ptr(), 64, [0, 1, 2, 3], 50.0)
    try:
        pool.register_layer(5, *ptrs, n)
        mem = mb.cpu.numpy()
        tasks = np.zeros(n, dtype=[("slot", "<i4"), ("off", "<i4"), ("n", "<i4"), ("pad", "<i4"),
                                   ("g13", "<f4"), ("g2", "<f4"), ("p2", "<f4"), ("p3", "<f4")])
        rid, rw, want = [], [], {}
        r = 0
        for slot, (rec, gm13, gm2, _, _) in enumerate(recs):
            k = slot + 2
            tasks[slot] = (slot, len(rid), k, 0, gm13, gm2, 0, 0)
            x = torch.from_numpy(rng.standard_normal((k, LAT)).astype(np.float32) * 0.5).to(torch.bfloat16)
            w = torch.from_numpy(rng.uniform(0.05, 0.4, k).astype(np.float32))
            rb = torch.from_numpy(rec.copy())
            q = rb.data_ptr()
            y = ext.expert_forward(q, q + 1146880, q + 1720320, q + 1863680, float(gm13), float(gm2), x, w)
            for i in range(k):
                rid.append(r)
                rw.append(float(w[i]))
                mem[lay["x"] + r * 3584:lay["x"] + (r + 1) * 3584] = x[i].view(torch.int16).numpy().view(np.uint8)
                want[r] = y[i]
                r += 2
        mem[lay["tasks"]:lay["tasks"] + tasks.nbytes] = tasks.view(np.uint8)
        mem[lay["route_ids"]:lay["route_ids"] + 4 * len(rid)] = np.array(rid, np.int32).view(np.uint8)
        mem[lay["route_w"]:lay["route_w"] + 4 * len(rw)] = np.array(rw, np.float32).view(np.uint8)
        for k in range(20):
            pool.set_tiles(*((1, 2), (2, 4), (5, 7), (10, 28), (1, 1))[k % 5])
            mem[lay["y"]:lay["y"] + 64 * 3584] = 0
            pool.host_ring_and_wait(5, n, len(rid), 5.0)
            for rr, yw in want.items():
                got = torch.from_numpy(mem[lay["y"] + rr * 3584:lay["y"] + (rr + 1) * 3584].copy()).view(torch.bfloat16)
                assert torch.equal(got, yw), (k, rr)
        assert pool.stats()[0] == 20
    finally:
        pool.stop()


def test_kernel_speed_single_core():
    """Compute cost per expert and route, one core, weights in cache (DRAM is not involved)."""
    import time
    ext = _ext()
    rng = np.random.default_rng(2)
    rec, gm13, gm2, _, _ = make_expert(rng)
    buf = torch.from_numpy(rec.copy())
    p = buf.data_ptr()
    for n in (1, 4):
        x = torch.from_numpy(rng.standard_normal((n, LAT)).astype(np.float32)).to(torch.bfloat16)
        wr = torch.ones(n)
        ext.expert_forward(p, p + 1146880, p + 1720320, p + 1863680, float(gm13), float(gm2), x, wr)
        t0 = time.perf_counter()
        k = 20
        for _ in range(k):
            ext.expert_forward(p, p + 1146880, p + 1720320, p + 1863680, float(gm13), float(gm2), x, wr)
        dt = (time.perf_counter() - t0) / k
        print(f"{n} routes: {dt * 1e6:.0f} us per expert on one core ({dt * 1e6 / n:.0f} us per route)")


def test_settings_cpu_knobs():
    from edge0_vllm.config import Edge0Settings
    from edge0_vllm.cpu_experts import parse_cpus
    s = Edge0Settings.from_env({"EDGE0_CPU_EXPERTS": "1", "EDGE0_CPU_FRAC": "auto", "EDGE0_CPU_THREADS": "6"})
    assert s.cpu_experts and s.cpu_frac == "auto" and s.cpu_threads == 6 and s.cpu_max_routes == 8
    assert not Edge0Settings.from_env({}).cpu_experts
    for bad in ({"EDGE0_CPU_FRAC": "1.5"}, {"EDGE0_CPU_EXPERTS": "1", "EDGE0_PREFETCH": "8"},
                {"EDGE0_CPU_EXPERTS": "1", "EDGE0_MOE_IMPL": "torch"},
                {"EDGE0_CPU_EXPERTS": "1", "EDGE0_CPU_MIN_TOKENS": "9", "EDGE0_CPU_MAX_TOKENS": "4"}):
        with pytest.raises(ValueError):
            Edge0Settings.from_env(bad)
    assert parse_cpus("2-5,9", 5) == [2, 3, 4, 5, 9]
    with pytest.raises(ValueError):
        parse_cpus("2-3", 4)
