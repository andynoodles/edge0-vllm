"""GPU: ExpertPool loading (ModelOpt -> Marlin per expert, VRAM + pinned host tiers) and the op body
(gather -> hot / cold Marlin calls -> combine) against the fp32 reference, on real layer-0 experts of a converted
checkpoint (EDGE0_CKPT, e.g. /models/Edge0-145B-vllm-L4). Also: split vs sequential, swaps, CUDA-graph capture,
and a first zero-copy timing. Skips without CUDA or without EDGE0_CKPT. Not the full model: one layer only.
"""
import json
import os
import time

import pytest
import torch

from helpers import CKPT

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and CKPT), reason="needs CUDA and EDGE0_CKPT")
LAYER = 0
_CACHE = {}


def _layer_tensors():
    if "t" not in _CACHE:
        from safetensors import safe_open
        idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
        out = {}
        for name in ("w13_weight", "w13_weight_scale", "w13_weight_scale_2", "w2_weight", "w2_weight_scale",
                     "w2_weight_scale_2"):
            full = f"model.layers.{LAYER}.mlp.experts.{name}"
            with safe_open(os.path.join(CKPT, idx[full]), framework="pt") as f:
                out[name] = f.get_tensor(full)
        g = f"model.layers.{LAYER}.mlp.expert_group"
        with safe_open(os.path.join(CKPT, idx[g]), framework="pt") as f:
            out["expert_group"] = f.get_tensor(g)
        _CACHE["t"] = out
    return _CACHE["t"]


def _pool(n_hot: int, seed: int = 0, keep_reference: bool = False):
    from edge0_vllm.expert_pool import ExpertPool
    from edge0_vllm.placement import LayerPlacement
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(512, generator=g).tolist()
    pool = ExpertPool(LAYER, LayerPlacement(order=order, n_hot=n_hot), torch.device("cuda"), "host_register",
                      keep_reference=keep_reference)
    for name, t in _layer_tensors().items():
        if name != "expert_group":
            pool.load_tensor(name, t)
    pool.finalize()
    return pool


def _ws(tokens=64):
    from edge0_vllm.latent_moe import _MoeWorkspace
    return _MoeWorkspace(tokens * 10, torch.device("cuda"), torch.bfloat16)


def _inputs(t, seed=1):
    from edge0_vllm import kernels
    g = torch.Generator(device="cuda").manual_seed(seed)
    xl = (torch.randn(t, 8 * 1792, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    logits = torch.randn(t, 512, generator=g, device="cuda") * 2
    w, ids = kernels.route_topk_softmax(logits, 10)
    return xl, ids, w


def _run(pool, xl, ids, w, ws, split=True):
    from edge0_vllm.latent_moe import latent_experts_marlin
    grp = _layer_tensors()["expert_group"].to("cuda")
    z = torch.empty_like(xl)
    latent_experts_marlin(pool, grp, xl, ids, w, ws, z, split=split)
    return z


def _ref(xl, ids, w):
    from edge0_vllm.expert_pool import ModelOptCopy
    from edge0_vllm.latent_moe import latent_experts_torch
    t = _layer_tensors()
    ref = ModelOptCopy(w13=t["w13_weight"], w13_scale=t["w13_weight_scale"], w13_scale_2=t["w13_weight_scale_2"],
                       w2=t["w2_weight"], w2_scale=t["w2_weight_scale"], w2_scale_2=t["w2_weight_scale_2"])
    z = torch.empty(xl.shape, dtype=torch.float32, device="cuda")
    return latent_experts_torch(ref, t["expert_group"].to("cuda"), xl, ids, w, z)


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


@pytest.mark.parametrize("n_hot", [512, 0, 200])
@pytest.mark.parametrize("tokens", [1, 7, 64])
def test_marlin_pool_vs_reference(n_hot, tokens):
    pool = _CACHE.setdefault(("pool", n_hot), _pool(n_hot))
    ws = _CACHE.setdefault("ws", _ws())
    xl, ids, w = _inputs(tokens)
    z = _run(pool, xl, ids, w, ws, split=True)
    zr = _ref(xl, ids, w)
    assert torch.isfinite(z).all()
    rel = _rel(z, zr)
    print(f"n_hot={n_hot} T={tokens}: rel err vs fp32 reference {rel:.2e}")
    assert rel < 2e-2
    if n_hot == 200:
        z_seq = _run(pool, xl, ids, w, ws, split=False)
        assert torch.equal(z, z_seq)                       # two streams change nothing numerically
        z_all_hot = _run(_CACHE.setdefault(("pool", 512), _pool(512)), xl, ids, w, ws)
        assert _rel(z, z_all_hot) < 1e-3


def test_swap_keeps_results():
    pool = _pool(100, seed=5)
    ws = _ws(8)
    xl, ids, w = _inputs(8, seed=3)
    before = _run(pool, xl, ids, w, ws)
    e_hot, e_cold = pool.order[3], pool.order[100 + 7]
    pool.swap(3, 7)
    assert pool.order[3] == e_cold and pool.order[107] == e_hot
    assert int(pool.hot_map[e_cold]) == 3 and int(pool.hot_map[e_hot]) == -1
    assert int(pool.cold_map[e_hot]) == 7 and int(pool.cold_map[e_cold]) == -1
    after = _run(pool, xl, ids, w, ws)
    assert _rel(after, before) < 1e-3


def test_cuda_graph_capture_split():
    pool = _CACHE.setdefault(("pool", 200), _pool(200))
    ws = _ws(16)
    xl, ids, w = _inputs(4, seed=7)
    grp = _layer_tensors()["expert_group"].to("cuda")
    from edge0_vllm.latent_moe import latent_experts_marlin
    sx, si, sw = xl.clone(), ids.clone(), w.clone()
    sz = torch.empty_like(xl)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        latent_experts_marlin(pool, grp, sx, si, sw, ws, sz, split=True)          # warm-up (allocations)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        latent_experts_marlin(pool, grp, sx, si, sw, ws, sz, split=True)
    xl2, ids2, w2 = _inputs(4, seed=8)
    sx.copy_(xl2), si.copy_(ids2), sw.copy_(w2)
    graph.replay()
    torch.cuda.synchronize()
    eager = _run(pool, xl2, ids2, w2, ws)
    assert torch.equal(sz, eager)


def test_zero_copy_timing():
    """First numbers for DESIGN.md 4.1: one layer, all-hot vs all-cold, decode-like token counts."""
    ws = _CACHE.setdefault("ws", _ws())
    pools = {n: _CACHE.setdefault(("pool", n), _pool(n)) for n in (512, 0)}
    from edge0_vllm import kernels
    for tokens in (1, 8, 32, 64):
        xl, ids, w = _inputs(tokens, seed=11)
        distinct = int(torch.unique(ids).numel())
        res = {}
        for n_hot, pool in pools.items():
            for _ in range(3):
                _run(pool, xl, ids, w, ws)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            reps = 20
            for _ in range(reps):
                _run(pool, xl, ids, w, ws)
            torch.cuda.synchronize()
            res[n_hot] = (time.perf_counter() - t0) / reps
        cold_gbs = distinct * 1_935_360 / res[0] / 1e9
        print(f"T={tokens:3d} distinct experts {distinct:3d}: all-hot {res[512] * 1e3:.3f} ms, all-cold "
              f"{res[0] * 1e3:.3f} ms ({cold_gbs:.1f} GB/s effective zero-copy)")
    _ = kernels


def test_trace_bw_and_routes(tmp_path):
    """EDGE0_TRACE counters: graph-safe recording (also inside a captured graph), bw flush, routes ring dump."""
    import numpy as np
    from edge0_vllm.trace import ExpertTrace
    msgs = []

    class _Log:
        def info(self, fmt, *a):
            msgs.append(fmt % a)

    tr = ExpertTrace(frozenset({"bw", "routes"}), 2, 512, 10, torch.device("cuda"), 1e9, 64, str(tmp_path), _Log())
    hot_map = torch.full((512,), -1, dtype=torch.int32, device="cuda")
    hot_map[:256] = 0
    tr.register_pool(1, hot_map)
    ids = torch.tensor([[1, 2, 300, 301, 302, 3, 4, 5, 6, 7], [1, 2, 300, 400, 401, 3, 4, 5, 6, 8]],
                       dtype=torch.int32, device="cuda")
    tr.record(1, ids)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        tr.record(1, ids)                                   # warm-up outside capture
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        tr.record(1, ids)
    g.replay()
    torch.cuda.synchronize()
    # 3 executed calls (eager, warm-up, graph replay; capture itself runs nothing);
    # distinct experts per call: hot {1..8} = 8, cold {300, 301, 302, 400, 401} = 5
    assert int(tr.calls[1]) == 3 and float(tr.touch[1].sum()) == 3 * 13
    tr.maybe_flush(force=True)
    assert any("hot 8 cold 5" in m for m in msgs), msgs
    files = sorted(tmp_path.glob("routes-*.npz"))
    assert files
    d = np.load(files[0])
    assert d["layer1"].shape == (6, 10) and d["layer1"][0].tolist() == ids[0].tolist()   # 3 calls x 2 tokens


def test_swap_many_inside_captured_graph():
    """Batched in-place swaps between graph replays (what the adaptive cache does): maps and rows follow, the
    captured graph sees the new placement, results stay equal to the all-hot pool."""
    pool = _pool(100, seed=9)
    ws = _ws(16)
    grp = _layer_tensors()["expert_group"].to("cuda")
    from edge0_vllm.latent_moe import latent_experts_marlin
    xl, ids, w = _inputs(8, seed=21)
    sz = torch.empty_like(xl)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        latent_experts_marlin(pool, grp, xl, ids, w, ws, sz, split=True)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        latent_experts_marlin(pool, grp, xl, ids, w, ws, sz, split=True)
    graph.replay()
    torch.cuda.synchronize()
    before = sz.clone()
    # make every cold expert this batch touches hot
    touched = set(torch.unique(ids).tolist())
    cold_touched = [e for e in pool.order[100:] if e in touched]
    hot_untouched = [e for e in pool.order[:100] if e not in touched][:len(cold_touched)]
    slot = {e: i for i, e in enumerate(pool.order)}
    pool.swap_many([slot[e] for e in hot_untouched], [slot[e] - 100 for e in cold_touched])
    assert all(int(pool.hot_map[e]) >= 0 and int(pool.cold_map[e]) == -1 for e in cold_touched)
    assert all(int(pool.hot_map[e]) == -1 and int(pool.cold_map[e]) >= 0 for e in hot_untouched)
    assert sorted(pool.order) == list(range(512))
    graph.replay()
    torch.cuda.synchronize()
    eager = _run(pool, xl, ids, w, ws)
    assert torch.equal(sz, eager)
    assert _rel(sz, before) < 1e-3
    assert _rel(sz, _run(_CACHE.setdefault(("pool", 512), _pool(512)), xl, ids, w, ws)) < 1e-3


def test_adaptive_cache_swaps_hot_traffic_in():
    import types
    from edge0_vllm.adaptive import AdaptiveCache
    msgs = []

    class _Log:
        def info(self, fmt, *a):
            msgs.append(fmt % a)

        warning = info

    pool = _pool(50, seed=4)
    blk = types.SimpleNamespace(layer_idx=0, pool=pool)
    ac = AdaptiveCache([blk], torch.device("cuda"), _Log(),
                       env={"EDGE0_ADAPT_MIN_STEPS": "5", "EDGE0_ADAPT_EPOCH_S": "0", "EDGE0_ADAPT_SWAPS": "8"})
    ac.on_step()                                   # start: zero
    hotcold = pool.order[50:58]                    # 8 cold experts that get all the traffic
    ids = torch.tensor([hotcold + pool.order[:2]], dtype=torch.int32, device="cuda")
    for _ in range(40):
        ac.record(0, ids)
    ac.on_step()
    assert ac.total_swaps == 8, msgs
    assert all(int(pool.hot_map[e]) >= 0 for e in hotcold)
    assert sorted(pool.order) == list(range(512))
    assert any("swaps" in m for m in msgs)


def test_predict_trace(tmp_path):
    """EDGE0_TRACE=predict counters: a perfect predictor scores 100%, inside a captured graph too."""
    import types
    from edge0_vllm.trace import PredictTrace
    msgs = []

    class _Log:
        def info(self, fmt, *a):
            msgs.append(fmt % a)

    ids = torch.tensor([[1, 2, 300, 301, 302, 3, 4, 5, 6, 7]], dtype=torch.int32, device="cuda")
    logits = torch.full((1, 512), -10.0, device="cuda")
    logits[0, ids[0].long()] = torch.arange(10, 0, -1, device="cuda", dtype=torch.float32)
    b0 = types.SimpleNamespace(layer_idx=0, gate=lambda x: logits)
    b1 = types.SimpleNamespace(layer_idx=1, gate=lambda x: logits)
    pt = PredictTrace([b0, b1], 64, torch.device("cuda"), 1e9, _Log(), str(tmp_path))
    hot_map = torch.full((512,), -1, dtype=torch.int32, device="cuda")
    hot_map[:256] = 0
    x = torch.zeros(1, 2560, device="cuda")
    pt.stash_from(0, x)
    pt.score(1, ids, hot_map)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        pt.stash_from(0, x)
        pt.score(1, ids, hot_map)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        pt.stash_from(0, x)
        pt.score(1, ids, hot_map)
    g.replay()
    torch.cuda.synchronize()
    assert float(pt.route[1, 4, 0]) == 30 and float(pt.route[1, 4, 1]) == 9      # 3 calls x (10 routes, 3 cold)
    assert float(pt.route[1, 0, 0]) == 30 and float(pt.route[1, 0, 1]) == 9      # k'=10 recalls everything
    assert float(pt.step[1, 0, 2]) == 9 and float(pt.step[1, 0, 0]) == 9
    pt.maybe_flush(force=True)
    assert any("route recall 100% (cold 100%)" in m for m in msgs), msgs


def _pool_staged(n_hot: int, n_stage: int, seed: int = 0):
    from edge0_vllm.expert_pool import ExpertPool
    from edge0_vllm.placement import LayerPlacement
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(512, generator=g).tolist()
    pool = ExpertPool(LAYER, LayerPlacement(order=order, n_hot=n_hot), torch.device("cuda"), "host_register",
                      n_stage=n_stage)
    for name, t in _layer_tensors().items():
        if name != "expert_group":
            pool.load_tensor(name, t)
    pool.finalize()
    return pool


def test_prefetch_staging_eager_and_graph():
    """EDGE0_PREFETCH: staged experts move from the cold call to the hot call (staging slots) with unchanged
    results; the staging copy + join also work inside one captured graph with new inputs on replay."""
    import types
    from edge0_vllm.latent_moe import latent_experts_marlin
    from edge0_vllm.prefetch import Prefetcher

    class _Log:
        def info(self, *a):
            pass

    pool = _pool_staged(100, 8, seed=13)
    assert pool.hot.n == 108 and pool.hot.w13.shape[0] == 108
    ws = _ws(16)
    grp = _layer_tensors()["expert_group"].to("cuda")
    xl, ids, w = _inputs(4, seed=41)
    ref = _run(_CACHE.setdefault(("pool", 512), _pool(512)), xl, ids, w, ws)
    z0 = _run(pool, xl, ids, w, ws)                      # no staging yet
    assert _rel(z0, ref) < 1e-3
    # predictor that "knows" the cold experts of this batch
    cold_used = [e for e in torch.unique(ids).tolist() if int(pool.hot_map[e]) < 0][:8]
    logits = torch.full((4, 512), -20.0, device="cuda")
    logits[:, cold_used] = 5.0
    b0 = types.SimpleNamespace(layer_idx=0, pool=None, gate=None)
    b1 = types.SimpleNamespace(layer_idx=1, pool=pool, gate=lambda x: logits)
    pf = Prefetcher([b0, b1], 8, 0.02, 16, torch.device("cuda"), _Log())
    x = torch.zeros(4, 2560, device="cuda")
    pf.predict_and_copy(0, x)
    torch.cuda.synchronize()                       # maps are written on the prefetch stream
    staged = [e for e in cold_used if int(pool.stage_map[e]) >= 0]
    assert len(staged) == len(cold_used)
    assert all(int(pool.cold_eff[e]) == -1 and int(pool.hot_eff[e]) >= 100 for e in staged)
    z1 = torch.empty_like(xl)
    latent_experts_marlin(pool, grp, xl, ids, w, ws, z1, split=True)
    torch.cuda.synchronize()
    assert pool.pf_pending is None
    assert _rel(z1, ref) < 1e-3 and _rel(z1, z0) < 1e-3
    # staged rows really hold the experts' Marlin rows
    for e in staged:
        s = int(pool.stage_map[e])
        c = int(pool.cold_map[e])
        assert torch.equal(pool.hot.w13[s], pool.cold.w13[c]) and torch.equal(pool.hot.s2[s], pool.cold.s2[c])
        assert float(pool.hot.g13[s]) == float(pool.cold.g13[c])
    # graph: predict (layer 0) -> op (layer 1), replayed with other inputs and another prediction
    sx, si, sw, sz = xl.clone(), ids.clone(), w.clone(), torch.empty_like(xl)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        pf.predict_and_copy(0, x)
        latent_experts_marlin(pool, grp, sx, si, sw, ws, sz, split=True)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pf.predict_and_copy(0, x)
        latent_experts_marlin(pool, grp, sx, si, sw, ws, sz, split=True)
    xl2, ids2, w2 = _inputs(4, seed=42)
    sx.copy_(xl2), si.copy_(ids2), sw.copy_(w2)
    cold2 = [e for e in torch.unique(ids2).tolist() if int(pool.hot_map[e]) < 0][:5]
    logits.fill_(-20.0)
    logits[:, cold2] = 5.0
    graph.replay()
    torch.cuda.synchronize()
    assert all(int(pool.stage_map[e]) >= 0 for e in cold2)
    ref2 = _run(_CACHE[("pool", 512)], xl2, ids2, w2, ws)
    assert _rel(sz, ref2) < 1e-3
    # a swap resets the staging
    pool.swap_many([0], [0])
    assert int((pool.stage_map >= 0).sum()) == 0 and torch.equal(pool.hot_eff, pool.hot_map)


def test_tier_tuning_direct_call_matches_stock():
    """EDGE0_COLD_BSM / EDGE0_HOT_BSM / EDGE0_COLD_BLOCKS_PER_SM run the same kernels with another M block / CTA
    cap: results equal the stock fused_marlin_moe path within bf16 noise."""
    from edge0_vllm import latent_moe as lm
    pool = _CACHE.setdefault(("pool", 200), _pool(200))
    ws = _ws(64)
    xl, ids, w = _inputs(64, seed=31)
    base = _run(pool, xl, ids, w, ws)
    try:
        for hot, cold, bps in ((0, 64, -1), (16, 0, 1), (64, 64, 2)):
            lm.set_tier_tuning("hot", hot)
            lm.set_tier_tuning("cold", cold, bps)
            z = _run(pool, xl, ids, w, ws)
            assert _rel(z, base) < 1e-3, (hot, cold, bps)
    finally:
        lm.set_tier_tuning("hot", 0)
        lm.set_tier_tuning("cold", 0)
