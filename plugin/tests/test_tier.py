"""EDGE0_TIER (tier.py): the request-reactive expert tier. CPU: layout helpers and the decision rule. GPU (real
layer-0 experts of EDGE0_CKPT): copies + flips between replays of a captured op, results equal to the all-hot pool,
host mirrors equal to the device maps, evicted experts served correctly from their spare host slots, and the same
with EDGE0_CPU_EXPERTS (the CPU pool must follow the flipped cold map and the copied global scales)."""
import types

import numpy as np
import pytest
import torch

from helpers import CKPT

_GPU = torch.cuda.is_available() and bool(CKPT)


class _Log:
    def __init__(self):
        self.msgs = []

    def info(self, fmt, *a):
        self.msgs.append(fmt % a if a else fmt)

    warning = info


def _settings(**kw):
    d = dict(tier_swaps=48, tier_decay=0.915, tier_gain=1.5, tier_min_use=2.0, tier_max_tokens=64, tier_log_s=1e9,
             tier_copy="cudart", tier_free=4)
    d.update(kw)
    return types.SimpleNamespace(**d)


# ---------------------------------------------------------------------------------------------- CPU
def test_demote_and_storage_layout():
    from edge0_vllm.placement import LayerPlacement
    from edge0_vllm.tier import TierStorage, demote
    pl = {0: LayerPlacement(order=list(range(512)), n_hot=100), 1: LayerPlacement(order=list(range(512)), n_hot=12),
          2: LayerPlacement(order=list(range(512)), n_hot=512)}
    out, moved = demote(pl, 4)
    assert [out[l].n_hot for l in range(3)] == [96, 10, 512] and moved == 6       # keeps >= 10 hot, all-hot untouched
    st = TierStorage(out, 4, "cpu")
    assert st.n_free == {0: 4, 1: 4, 2: 0}
    assert st.n_h == {0: 100, 1: 14, 2: 512} and st.n_c == {0: 420, 1: 506, 2: 0}
    v = st.views(1)
    assert v.hot_g13.shape == (14,) and v.cold_g2.shape == (506,)
    assert v.hot_g13.data_ptr() == st.g[0, 100:].data_ptr()
    assert v.cold_g13.data_ptr() == st.g[0, st.Nh + 420:].data_ptr()
    assert v.hot_map.data_ptr() == st.maps[0, 1].data_ptr() and v.cold_map.data_ptr() == st.maps[1, 1].data_ptr()


# ---------------------------------------------------------------------------------------------- GPU
gpu = pytest.mark.skipif(not _GPU, reason="needs CUDA and EDGE0_CKPT")


def _gx():
    import test_gpu_experts as g
    return g


def _tier_pool(n_hot, seed=0, n_free=4):
    from edge0_vllm.expert_pool import ExpertPool
    from edge0_vllm.placement import LayerPlacement
    from edge0_vllm.tier import TierStorage, demote
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(512, generator=g).tolist()
    pl, _ = demote({0: LayerPlacement(order=order, n_hot=n_hot)}, n_free)
    st = TierStorage(pl, n_free, torch.device("cuda"))
    pool = ExpertPool(0, pl[0], torch.device("cuda"), "host_register", n_free=st.n_free[0], storage=st)
    for name, t in _gx()._layer_tensors().items():
        if name != "expert_group":
            pool.load_tensor(name, t)
    pool.finalize()
    return pool, st


def _make_tier(pool, st, **kw):
    from edge0_vllm.tier import ReactiveTier
    log = _Log()
    blk = types.SimpleNamespace(layer_idx=0, pool=pool)
    return ReactiveTier([blk], _settings(**kw), torch.device("cuda"), log, 64, storage=st), log


def _capture(pool, tier, ids_shape, cpu=None, seed=31):
    """Graph of [tier.record -> op] on static inputs (what one decode step of the model does for this layer)."""
    from edge0_vllm.latent_moe import latent_experts_marlin
    gx = _gx()
    ws = gx._ws(16)
    grp = gx._layer_tensors()["expert_group"].to("cuda")
    xl, ids, w = gx._inputs(ids_shape, seed=seed)
    sx, si, sw = xl.clone(), ids.clone(), w.clone()
    sz = torch.empty_like(xl)

    def body():
        tier.record(0, si)
        latent_experts_marlin(pool, grp, sx, si, sw, ws, sz, split=True, cpu=cpu)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        body()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    return graph, (sx, si, sw, sz), ws, grp


def _all_hot(xl, ids, w):
    gx = _gx()
    from edge0_vllm.latent_moe import latent_experts_marlin
    pool = gx._CACHE.setdefault(("pool", 512), gx._pool(512))
    ws = gx._CACHE.setdefault("ws", gx._ws())
    z = torch.empty_like(xl)
    latent_experts_marlin(pool, gx._layer_tensors()["expert_group"].to("cuda"), xl, ids, w, ws, z, split=True)
    return z


def _steer(si, experts):
    """Route every token to `experts` (10 distinct ids per token)."""
    t = si.shape[0]
    rows = [experts[(i * 3 + j) % len(experts)] for i in range(t) for j in range(10)]
    ids = torch.tensor(rows, dtype=torch.int32, device="cuda").view(t, 10)
    # make each row distinct
    for i in range(t):
        r = []
        k = 0
        while len(r) < 10:
            e = experts[(i * 3 + k) % len(experts)]
            if e not in r:
                r.append(e)
            k += 1
        ids[i] = torch.tensor(r, dtype=torch.int32)
    si.copy_(ids)


@gpu
@pytest.mark.parametrize("use_cpu", [False, True])
def test_tier_swaps_in_traffic_between_graph_replays(use_cpu):
    from edge0_vllm.tier import check_consistency
    pool, st = _tier_pool(100, seed=3)
    assert pool.n_hot == 96 and pool.n_free == 4 and pool.hot.n == 100 and pool.cold.n == 420
    cx = None
    if use_cpu:
        from edge0_vllm.cpu_experts import CpuExperts
        import test_cpu_experts as tc
        cx = CpuExperts(tc._settings(cpu_threads=8, cpu_min_cold=1), torch.device("cuda"), _Log(), ext=tc._ext())
        cx.register(0, pool)
    try:
        tier, log = _make_tier(pool, st, tier_swaps=6)
        graph, (sx, si, sw, sz), ws, grp = _capture(pool, tier, 4, cpu=cx)
        cold_set = [e for e in pool.order[96:] if e not in pool.order[:96]][:12]
        _steer(si, cold_set)
        tier.on_step()                       # start: drops warm-up counts
        rels = []
        for step in range(30):
            graph.replay()
            tier.on_step()
            torch.cuda.synchronize()
            ref = _all_hot(sx, si, sw)
            rels.append(_gx()._rel(sz, ref))
        torch.cuda.synchronize()
        assert check_consistency(tier) == []
        hot_now = [e for e in cold_set if tier.is_hot[0, e]]
        assert len(hot_now) == 12, (hot_now, tier.usage[0, cold_set])
        assert all(int(pool.hot_map[e]) >= 0 and int(pool.cold_map[e]) == -1 for e in cold_set)
        assert max(rels) < 3e-3 if use_cpu else max(rels) < 1e-3, rels
        assert tier.total_swaps == 12
        # the evicted experts now live in spare host slots: route traffic to them and compare again
        victims = [e for e in range(512) if not tier.is_hot[0, e] and pool.order.index(e) < 96]
        assert len(victims) == 12
        _steer(si, victims[:12])
        graph.replay()
        torch.cuda.synchronize()
        assert _gx()._rel(sz, _all_hot(sx, si, sw)) < (3e-3 if use_cpu else 1e-3)
        print(f"use_cpu={use_cpu}: {tier.total_swaps} swaps, max rel {max(rels):.2e}")
    finally:
        if cx is not None:
            cx.pool.stop()


@gpu
def test_tier_counts_and_stats():
    pool, st = _tier_pool(200, seed=4)
    tier, log = _make_tier(pool, st, tier_swaps=0, tier_log_s=0.0)
    ids = torch.tensor([pool.order[:5] + pool.order[300:305]], dtype=torch.int32, device="cuda")
    tier.on_step()
    tier.record(0, ids)
    tier.record(0, ids)
    tier.on_step()                    # snapshot
    torch.cuda.synchronize()
    tier.on_step()                    # consume + log
    u = tier.usage[0]
    assert u[pool.order[0]] == pytest.approx(2.0) and u[pool.order[300]] == pytest.approx(2.0)
    assert any("cold 5.00 hot 5.00" in m and "routes from VRAM 50.0%" in m for m in log.msgs), log.msgs
    assert tier.total_swaps == 0


@gpu
def test_tier_copy_modes_agree():
    """cudart and torch copy paths move the same bytes."""
    for mode in ("cudart", "torch"):
        pool, st = _tier_pool(60, seed=9)
        tier, _ = _make_tier(pool, st, tier_copy=mode, tier_swaps=4)
        x = [e for e in pool.order[56:]][:4]
        tier.on_step()
        for _ in range(3):
            tier.record(0, torch.tensor([x * 2 + pool.order[:2]], dtype=torch.int32, device="cuda"))
            tier.on_step()
            torch.cuda.synchronize()
        tier.on_step()
        torch.cuda.synchronize()
        assert all(tier.is_hot[0, e] for e in x), mode
        ref, _ = _tier_pool(512, seed=9)
        for e in x:
            hs = int(pool.hot_map[e])
            rs = int(ref.hot_map[e])
            for a, b in ((pool.hot.w13, ref.hot.w13), (pool.hot.w2, ref.hot.w2),
                         (pool.hot.s13.view(torch.uint8), ref.hot.s13.view(torch.uint8))):
                assert torch.equal(a[hs], b[rs]), (mode, e)
            assert float(pool.hot.g13[hs]) == float(ref.hot.g13[rs]) and float(pool.hot.g2[hs]) == float(ref.hot.g2[rs])
        assert np.isfinite(tier.usage).all()


@gpu
@pytest.mark.parametrize("copy", ["batch", "cudart"])
@pytest.mark.parametrize("use_cpu", [False, True])
def test_tier_early_resident_next_step(copy, use_cpu):
    """EDGE0_TIER_EARLY: on_step only snapshots; pre_sync (next step's staging) decides from the step that just ran and
    flips on the copy stream; post_sync makes the next replay wait for it. So the experts a step touched are hot in
    the very next replay (late mode: one replay later), results stay equal to the all-hot pool, host mirrors equal
    the device maps, and evicted experts are served from their spare host slots (also by the CPU path)."""
    from edge0_vllm.tier import check_consistency
    pool, st = _tier_pool(100, seed=5)
    cx = None
    if use_cpu:
        from edge0_vllm.cpu_experts import CpuExperts
        import test_cpu_experts as tc
        cx = CpuExperts(tc._settings(cpu_threads=8, cpu_min_cold=1), torch.device("cuda"), _Log(), ext=tc._ext())
        cx.register(0, pool)
    try:
        tier, log = _make_tier(pool, st, tier_swaps=6, tier_early=True, tier_copy=copy, tier_log_s=0.0,
                               tier_min_use=1.0, tier_gain=0.5)
        assert tier.early and tier.copier.mode == copy
        graph, (sx, si, sw, sz), ws, grp = _capture(pool, tier, 4, cpu=cx)
        cold_set = [e for e in pool.order[96:]][:4]
        _steer(si, cold_set + pool.order[:6])
        tier.on_step()                       # start: drops warm-up counts
        rels, hot_after = [], []
        for step in range(6):
            graph.replay()
            tier.on_step()                   # snapshot only
            tier.pre_sync()                  # waits for the replay, decides, copies, flips on the copy stream
            tier.post_sync()                 # the next replay waits for the flip
            torch.cuda.synchronize()
            rels.append(_gx()._rel(sz, _all_hot(sx, si, sw)))
            hot_after.append(sum(int(pool.hot_map[e]) >= 0 for e in cold_set))
        assert hot_after[0] == 4, hot_after          # resident right after the step that touched them
        assert check_consistency(tier) == []
        assert tier.total_swaps == 4
        assert max(rels) < (3e-3 if use_cpu else 1e-3), rels
        victims = [e for e in range(512) if not tier.is_hot[0, e] and pool.order.index(e) < 96]
        assert len(victims) == 4
        _steer(si, victims + pool.order[:6])
        graph.replay()
        torch.cuda.synchronize()
        assert _gx()._rel(sz, _all_hot(sx, si, sw)) < (3e-3 if use_cpu else 1e-3)
        # the next rounds bring the victims back (min usage 1, gain 0.5): still consistent
        for _ in range(4):
            graph.replay()
            tier.on_step()
            tier.pre_sync()
            tier.post_sync()
        torch.cuda.synchronize()
        assert check_consistency(tier) == []
        assert all(tier.is_hot[0, e] for e in victims)
        assert _gx()._rel(sz, _all_hot(sx, si, sw)) < (3e-3 if use_cpu else 1e-3)
        tier.n_calls = max(tier.n_calls, 1)
        tier._log(1.0)
        assert any("EARLY: host waited" in m for m in log.msgs), log.msgs
        print(f"copy={copy} use_cpu={use_cpu}: swaps {tier.total_swaps}, max rel {max(rels):.2e}, hot after step "
              f"{hot_after}")
    finally:
        if cx is not None:
            cx.pool.stop()


@gpu
@pytest.mark.parametrize("use_cpu", [False, True])
def test_tier_shadow_eviction_without_writeback(use_cpu):
    """EDGE0_TIER_SHADOW: an expert swapped in keeps its host slot; when it is evicted again it goes back there with
    no D2H, and is served correctly from it (GPU cold path and CPU path)."""
    from edge0_vllm.tier import TOP_K, check_consistency
    pool, st = _tier_pool(TOP_K + 8, seed=7)          # after demote: TOP_K pinned + 4 evictable hot experts
    assert pool.n_hot == TOP_K + 4
    cx = None
    if use_cpu:
        from edge0_vllm.cpu_experts import CpuExperts
        import test_cpu_experts as tc
        cx = CpuExperts(tc._settings(cpu_threads=8, cpu_min_cold=1), torch.device("cuda"), _Log(), ext=tc._ext())
        cx.register(0, pool)
    try:
        tier, log = _make_tier(pool, st, tier_swaps=8, tier_early=True, tier_shadow=True, tier_copy="batch",
                               tier_min_use=1.0, tier_gain=0.5)
        assert tier.shadow_on
        graph, (sx, si, sw, sz), ws, grp = _capture(pool, tier, 4, cpu=cx)
        cold = [e for e in pool.order[pool.n_hot:]]
        A, B = cold[:4], cold[4:8]
        pin = pool.order[:6]

        def run(steps):
            for _ in range(steps):
                graph.replay()
                tier.on_step()
                tier.pre_sync()
                tier.post_sync()
            torch.cuda.synchronize()

        _steer(si, A + pin)
        tier.on_step()
        run(3)
        assert all(tier.is_hot[0, e] for e in A) and tier.n_d2h == 4      # first evictions: plan-hot -> write-back
        assert all(tier.shadow[0, e] >= 0 for e in A)
        d2h0, sw0 = tier.n_d2h, tier.total_swaps
        _steer(si, B + pin)
        for _ in range(12):
            run(1)
            if all(tier.is_hot[0, e] for e in B):
                break
        assert all(tier.is_hot[0, e] for e in B) and not any(tier.is_hot[0, e] for e in A)
        assert tier.total_swaps - sw0 == 4 and tier.n_d2h == d2h0, (tier.n_d2h, d2h0)   # A evicted to its shadows
        assert check_consistency(tier) == []
        _steer(si, A + pin)                    # A is cold again, read from its shadow slots
        graph.replay()
        torch.cuda.synchronize()
        assert _gx()._rel(sz, _all_hot(sx, si, sw)) < (3e-3 if use_cpu else 1e-3)
        run(6)                                 # back and forth: B's eviction goes to its shadows as well
        assert check_consistency(tier) == []
        assert _gx()._rel(sz, _all_hot(sx, si, sw)) < (3e-3 if use_cpu else 1e-3)
        print(f"use_cpu={use_cpu}: swaps {tier.total_swaps}, write-backs {tier.n_d2h}, stolen {tier.stolen}")
    finally:
        if cx is not None:
            cx.pool.stop()


@gpu
def test_tier_shadow_steal():
    """Spare host slots exhausted and a victim without a shadow: it takes the shadow slot of the most used hot expert
    (made in an earlier round), whose own later eviction then writes back. Every expert stays served correctly."""
    from edge0_vllm.tier import TOP_K, check_consistency
    pool, st = _tier_pool(TOP_K + 12, seed=11)         # TOP_K pinned + 8 evictable plan-hot experts, 4 spares
    tier, log = _make_tier(pool, st, tier_swaps=8, tier_early=True, tier_shadow=True, tier_copy="batch",
                           tier_min_use=1.0, tier_gain=0.5)
    graph, (sx, si, sw, sz), ws, grp = _capture(pool, tier, 4)
    cold = [e for e in pool.order[pool.n_hot:]]
    plan_hot = [e for e in pool.order[TOP_K:pool.n_hot]]
    A, B, C = cold[:4], cold[4:8], cold[8:12]
    pin = pool.order[:6]

    def run(steps):
        for _ in range(steps):
            graph.replay()
            tier.on_step()
            tier.pre_sync()
            tier.post_sync()
        torch.cuda.synchronize()

    def check(experts):
        _steer(si, experts + pin)
        graph.replay()
        torch.cuda.synchronize()
        assert _gx()._rel(sz, _all_hot(sx, si, sw)) < 1e-3, experts
        assert check_consistency(tier) == []

    _steer(si, A + pin)
    tier.on_step()
    run(3)
    assert all(tier.is_hot[0, e] for e in A) and tier.n_d2h == 4 and not tier.spare_h[0]
    _steer(si, B + pin)
    run(3)
    assert all(tier.is_hot[0, e] for e in B)
    assert tier.stolen == 4 and tier.n_d2h == 8, (tier.stolen, tier.n_d2h)       # the other 4 plan-hot experts
    assert all(tier.shadow[0, e] < 0 for e in A) and all(tier.shadow[0, e] >= 0 for e in B)
    evicted = [e for e in plan_hot if not tier.is_hot[0, e]]
    assert len(evicted) == 8
    check(evicted[:4] + evicted[4:6] + A[:4])          # written-back experts, incl. the ones in stolen slots
    check(evicted[6:8] + B + A[:4])
    _steer(si, C + pin)                                # evicts A or B (lowest usage): A writes back, B does not
    run(8)
    check(C + A[:2] + B[:2] + evicted[:2])
    check(A[2:] + B[2:] + evicted[2:8])
    print(f"swaps {tier.total_swaps}, write-backs {tier.n_d2h}, stolen {tier.stolen}")


@gpu
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tier_gpu_decision_matches_numpy(seed):
    """EDGE0_TIER_GPU_DECIDE: the side-stream decision picks the same pairs and gains as ``_decide`` (distinct usage
    values, so no ties), including the gain rule, min usage, pinned experts and the swap cap."""
    pool, st = _tier_pool(100, seed=seed)
    tier, _ = _make_tier(pool, st, tier_swaps=3, tier_early=True, tier_gpu_decide=True)
    assert tier.gpu_decide
    rng = np.random.default_rng(seed)
    u = rng.random((tier.L, tier.E)) * 6.0
    tier.usage[:] = u
    tier.u_dev.copy_(torch.from_numpy(u).float())
    ref = tier._decide()
    ev = torch.cuda.Event()
    ev.record()
    tier._decide_gpu(ev)
    got = tier._read_gpu_decision()
    assert [(l, x, v) for _, l, x, v in got] == [(l, x, v) for _, l, x, v in ref], (got, ref)
    assert np.allclose([g for g, *_ in got], [g for g, *_ in ref], atol=1e-5)
    assert 0 < len(ref) <= 3


@gpu
def test_tier_batch_copy_moves_same_bytes():
    """EDGE0_TIER_COPY=batch (one cudaMemcpyBatchAsync per direction) in early mode: the incoming experts' slots hold
    exactly the reference pool's bytes and scales, the evicted ones' spare host slots their VRAM bytes."""
    from edge0_vllm.tier import check_consistency
    pool, st = _tier_pool(60, seed=9)
    tier, _ = _make_tier(pool, st, tier_copy="batch", tier_swaps=4, tier_early=True)
    x = [e for e in pool.order[56:]][:4]
    tier.on_step()
    for _ in range(3):
        tier.record(0, torch.tensor([x * 2 + pool.order[:2]], dtype=torch.int32, device="cuda"))
        tier.on_step()
        tier.pre_sync()
        tier.post_sync()
    torch.cuda.synchronize()
    assert tier.copier.mode == "batch"
    assert all(tier.is_hot[0, e] for e in x)
    assert check_consistency(tier) == []
    ref, _ = _tier_pool(512, seed=9)
    for e in x:
        hs, rs = int(pool.hot_map[e]), int(ref.hot_map[e])
        for a, b in ((pool.hot.w13, ref.hot.w13), (pool.hot.w2, ref.hot.w2),
                     (pool.hot.s13.view(torch.uint8), ref.hot.s13.view(torch.uint8)),
                     (pool.hot.s2.view(torch.uint8), ref.hot.s2.view(torch.uint8))):
            assert torch.equal(a[hs], b[rs]), e
        assert float(pool.hot.g13[hs]) == float(ref.hot.g13[rs]) and float(pool.hot.g2[hs]) == float(ref.hot.g2[rs])
    moved = 0
    for e in range(512):
        cs = int(pool.cold_map[e])
        if cs >= pool.n_cold:              # a spare host slot: written by this round's D2H
            rs = int(ref.hot_map[e])
            for i, t in enumerate((ref.hot.w13, ref.hot.w2, ref.hot.s13, ref.hot.s2)):
                assert torch.equal(pool.cold.cpu[i][cs].view(torch.uint8), t[rs].cpu().view(torch.uint8)), (e, i)
            moved += 1
    assert moved == 4
