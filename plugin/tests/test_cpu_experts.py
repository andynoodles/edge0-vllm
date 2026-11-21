"""GPU + CPU: EDGE0_CPU_EXPERTS (docs/CPU-EXPERTS.md). The AVX-512 kernel over the Marlin record against the GPU's
Marlin path and the fp32 ModelOpt reference, the worker pool's task split, and the full op (dispatch -> gather +
doorbell -> Marlin -> wait -> combine) inside a captured CUDA graph against the GPU-only op. Real layer-0 experts of
EDGE0_CKPT. Skips without CUDA or EDGE0_CKPT.
"""
import types

import numpy as np
import pytest
import torch

from helpers import CKPT

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and CKPT), reason="needs CUDA and EDGE0_CKPT")
_C = {}


def _gx():
    import test_gpu_experts as g
    return g


def _ext():
    from edge0_vllm import cpu_experts
    return cpu_experts.load(verbose=False)


def _pool(n_hot):
    return _C.setdefault(("pool", n_hot), _gx()._pool(n_hot))


def _settings(**kw):
    d = dict(cpu_min_tokens=1, cpu_max_tokens=64, cpu_min_cold=1, cpu_max_routes=64, cpu_max_experts=512,
             cpu_timeout_ms=2000.0, cpu_cores="", cpu_threads=8, cpu_frac="0.5", cpu_frac_start=0.5, cpu_spin_ms=200.0)
    d.update(kw)
    return types.SimpleNamespace(**d)


class _Log:
    def info(self, *a):
        print(a[0] % a[1:] if len(a) > 1 else a[0])

    warning = info


def _bf16_ulps(a, b):
    ia = a.contiguous().view(torch.int16).to(torch.int32)
    ib = b.contiguous().view(torch.int16).to(torch.int32)
    return (ia - ib).abs()


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def _slot_ptrs(pool, slot):
    from edge0_vllm.config import S2_BYTES, S13_BYTES, W2_BYTES, W13_BYTES
    from edge0_vllm.cpu_experts import slot_pointers
    w13, w2, s13, s2, _ = slot_pointers(pool.cold)
    return w13 + slot * W13_BYTES, w2 + slot * W2_BYTES, s13 + slot * S13_BYTES, s2 + slot * S2_BYTES


@pytest.mark.parametrize("routes", [1, 5])
def test_expert_kernel_vs_marlin_and_reference(routes):
    from edge0_vllm import marlin_nvfp4 as mn
    from edge0_vllm.latent_moe import marlin_tier_direct
    ext = _ext()
    pool = _pool(0)
    ws = _C.setdefault("ws", _gx()._ws())
    t = _gx()._layer_tensors()
    g = torch.Generator().manual_seed(routes)
    worst = []
    for e in (0, 17, 300, 511):
        x = (torch.randn(routes, 1792, generator=g) * 0.5).to(torch.bfloat16)
        wr = torch.rand(routes, generator=g) * 0.3 + 0.01
        slot = int(pool.cold_map[e])
        p13, p2, ps13, ps2 = _slot_ptrs(pool, slot)
        g13, g2 = float(pool.cold.g13[slot]), float(pool.cold.g2[slot])
        h_cpu = ext.expert_h(p13, ps13, g13, x)
        y_cpu = ext.expert_forward(p13, p2, ps13, ps2, g13, g2, x, wr)
        xr = x.cuda()
        ids = torch.full((routes, 1), e, dtype=torch.int32, device="cuda")
        y_gpu = marlin_tier_direct(pool.cold, pool.cold_map, xr, ids, wr.cuda().view(-1, 1), ws, "cold", bsm=8).clone()
        h_gpu = ws.c2["cold"][:routes * 640].view(routes, 640).float().cpu()
        ref = mn.expert_forward_reference(xr, t["w13_weight"][e].cuda(), t["w13_weight_scale"][e].cuda(),
                                          float(t["w13_weight_scale_2"][e, 0]), t["w2_weight"][e].cuda(),
                                          t["w2_weight_scale"][e].cuda(), float(t["w2_weight_scale_2"][e]))
        ref = (ref * wr.cuda().view(-1, 1)).cpu()
        y_gpu = y_gpu.cpu()
        h_eq = float((h_cpu == h_gpu).float().mean())
        ul = _bf16_ulps(y_cpu, y_gpu)
        eq = float((ul == 0).float().mean())
        r_cpu, r_gpu = _rel(y_cpu, ref), _rel(y_gpu, ref)
        print(f"e={e} routes={routes}: h equal {h_eq:.4f} (rel {_rel(h_cpu, h_gpu):.2e}); y equal {eq:.4f}, "
              f"max ulp {int(ul.max())}, rel cpu-gpu {_rel(y_cpu, y_gpu):.2e}; vs fp32 ref: cpu {r_cpu:.3e} "
              f"gpu {r_gpu:.3e}")
        worst.append((h_eq, eq, r_cpu, r_gpu))
        assert _rel(h_cpu, h_gpu) < 1e-2, "G1 (gate/up/silu) disagrees with Marlin: layout or scale decode wrong"
        assert _rel(y_cpu, y_gpu) < 5e-3
        assert r_cpu < 1.2 * r_gpu + 1e-3
    assert min(w[1] for w in worst) > 0.9


def test_pool_job_matches_single_thread():
    """Host-rung job: 6 experts with 1..6 routes through the pool == expert_forward per expert (same code)."""
    from edge0_vllm.cpu_experts import CpuExperts
    ext = _ext()
    pool = _pool(0)
    cx = CpuExperts(_settings(cpu_threads=8), torch.device("cuda"), _Log(), ext=ext)
    try:
        cx.register(0, pool)
        lay = cx.lay
        mem = cx.buf.cpu.numpy()
        experts = [3, 99, 250, 400, 401, 511]
        g = torch.Generator().manual_seed(7)
        tasks = np.zeros(len(experts), dtype=[("slot", "<i4"), ("off", "<i4"), ("n", "<i4"), ("pad", "<i4"),
                                               ("g13", "<f4"), ("g2", "<f4"), ("p2", "<f4"), ("p3", "<f4")])
        rid, rw, want = [], [], {}
        r = 0
        for i, e in enumerate(experts):
            n = i + 1
            slot = int(pool.cold_map[e])
            tasks[i] = (slot, len(rid), n, 0, float(pool.cold.g13[slot]), float(pool.cold.g2[slot]), 0, 0)
            x = (torch.randn(n, 1792, generator=g) * 0.5).to(torch.bfloat16)
            w = torch.rand(n, generator=g) * 0.3 + 0.01
            y = ext.expert_forward(*_slot_ptrs(pool, slot), float(pool.cold.g13[slot]), float(pool.cold.g2[slot]),
                                   x, w)
            for k in range(n):
                rid.append(r)
                rw.append(float(w[k]))
                xs = x[k].view(torch.int16).numpy().view(np.uint8)
                mem[lay["x"] + r * 3584:lay["x"] + (r + 1) * 3584] = xs
                want[r] = y[k]
                r += 3                     # sparse route positions, like a real call
        mem[lay["tasks"]:lay["tasks"] + tasks.nbytes] = tasks.view(np.uint8)
        mem[lay["route_ids"]:lay["route_ids"] + 4 * len(rid)] = np.array(rid, np.int32).view(np.uint8)
        mem[lay["route_w"]:lay["route_w"] + 4 * len(rw)] = np.array(rw, np.float32).view(np.uint8)
        for rep in range(3):
            dt = cx.pool.host_ring_and_wait(0, len(experts), len(rid), 5.0)
            for rr, yw in want.items():
                got = torch.from_numpy(mem[lay["y"] + rr * 3584:lay["y"] + (rr + 1) * 3584].copy()).view(torch.bfloat16)
                assert torch.equal(got, yw), f"route {rr}"
            print(f"pool job {rep}: {len(experts)} experts {len(rid)} routes in {dt * 1e6:.0f} us")
        st = cx.stats()
        assert st["jobs"] == 3 and st["experts"] == 3 * len(experts)
    finally:
        cx.pool.stop()


@pytest.mark.parametrize("tokens", [4, 16, 32])
def test_op_in_cuda_graph_matches_gpu_only(tokens):
    from edge0_vllm.cpu_experts import CpuExperts
    from edge0_vllm.latent_moe import latent_experts_marlin
    gx = _gx()
    pool = _pool(200)
    ws = _C.setdefault("ws", gx._ws())
    grp = gx._layer_tensors()["expert_group"].to("cuda")
    cx = CpuExperts(_settings(cpu_threads=8, cpu_min_cold=1), torch.device("cuda"), _Log(), ext=_ext())
    try:
        cx.register(0, pool)
        xl, ids, w = gx._inputs(tokens, seed=11)
        sx, si, sw = xl.clone(), ids.clone(), w.clone()
        z_cpu = torch.empty_like(xl)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            latent_experts_marlin(pool, grp, sx, si, sw, ws, z_cpu, split=True, cpu=cx)   # warm-up
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            latent_experts_marlin(pool, grp, sx, si, sw, ws, z_cpu, split=True, cpu=cx)
        for seed in range(5):
            xl, ids, w = gx._inputs(tokens, seed=100 + seed)
            sx.copy_(xl)
            si.copy_(ids)
            sw.copy_(w)
            graph.replay()
            torch.cuda.synchronize()
            z_gpu = torch.empty_like(xl)
            latent_experts_marlin(pool, grp, xl, ids, w, ws, z_gpu, split=True)
            zr = gx._ref(xl, ids, w)
            r_split, r_ref, r_gpu = _rel(z_cpu, z_gpu), _rel(z_cpu, zr), _rel(z_gpu, zr)
            eq = float((z_cpu == z_gpu).float().mean())
            print(f"T={tokens} seed={seed}: split vs gpu-only rel {r_split:.2e} (equal {eq:.3f}), vs ref "
                  f"{r_ref:.3e} (gpu-only {r_gpu:.3e}), cpu sel {int(cx.cpu_sel.sum())}")
            assert r_split < 3e-3 and r_ref < 1.2 * r_gpu + 1e-3
        st = cx.stats()
        print(st)
        assert st["jobs"] >= 5 and st["timeouts"] == 0
    finally:
        cx.pool.stop()


@pytest.mark.parametrize("frac_arg", [None, -1.0])
def test_graph_mixes_calls_with_and_without_cpu_share(frac_arg):
    """Round 2 dispatch (exceed-2): a call without a CPU share returns early (no mapped writes, no doorbell, the gather
    leaves its CTA counter alone). Calls with and without a share alternate inside one captured graph, each output
    against the GPU-only op; frac_arg -1 = the auto path (frac read from the mailbox)."""
    from edge0_vllm.cpu_experts import CpuExperts
    from edge0_vllm.latent_moe import latent_experts_marlin
    gx = _gx()
    pool = _pool(200)
    ws = _C.setdefault("ws", gx._ws())
    grp = gx._layer_tensors()["expert_group"].to("cuda")
    cx = CpuExperts(_settings(cpu_threads=8, cpu_min_cold=1), torch.device("cuda"), _Log(), ext=_ext())
    if frac_arg is not None:
        cx.frac_arg = frac_arg
    plan = (10 ** 6, 1, 10 ** 6, 1, 1, 10 ** 6)          # min_cold per call: no share / share
    try:
        cx.register(0, pool)
        bufs = [tuple(t.clone() for t in gx._inputs(4, seed=7 + i)) for i in range(len(plan))]
        outs = [torch.empty_like(b[0]) for b in bufs]

        def body():
            for (sx, si, sw), z, mc in zip(bufs, outs, plan):
                cx.min_cold = mc
                latent_experts_marlin(pool, grp, sx, si, sw, ws, z, split=True, cpu=cx)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            body()
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        jobs0 = cx.stats()["jobs"]
        for rep in range(3):
            for i, (sx, si, sw) in enumerate(bufs):
                xl, ids, w = gx._inputs(4, seed=200 + 10 * rep + i)
                sx.copy_(xl)
                si.copy_(ids)
                sw.copy_(w)
            graph.replay()
            torch.cuda.synchronize()
            for (sx, si, sw), z, mc in zip(bufs, outs, plan):
                z_gpu = torch.empty_like(sx)
                latent_experts_marlin(pool, grp, sx, si, sw, ws, z_gpu, split=True)
                r = _rel(z, z_gpu)
                print(f"rep {rep} min_cold {mc}: split vs gpu-only rel {r:.2e}")
                assert r < 3e-3
        st = cx.stats()
        print(st)
        assert st["jobs"] - jobs0 == 3 * sum(1 for mc in plan if mc == 1) and st["timeouts"] == 0
    finally:
        cx.pool.stop()
