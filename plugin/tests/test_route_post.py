"""GPU: ``EDGE0_ROUTE_FUSE`` (kernels.route_post, round 2 agent exceed-2) against the unfused routing tail of
``Edge0LatentMoeBlock.route`` (skip_pad's two ``torch.where`` + ``zeros_like``, the tier's ``full_like`` + ``where`` +
``tier_count``): same ids, weights and counter row, with and without padding rows; plus the time of both inside a
captured CUDA graph of 48 calls (printed, not asserted). Skips without CUDA."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

E, K = 512, 10


def _unfused(ids, w, pad, pad_ids, row, hot_map):
    from edge0_vllm import kernels
    if pad is not None and pad_ids is not None:
        pm = pad.view(-1, 1)
        ids = torch.where(pm, pad_ids, ids)
        w = torch.where(pm, torch.zeros_like(w), w)
    if row is not None:
        cid = ids if pad is None else torch.where(pad.view(-1, 1), torch.full_like(ids, -1), ids)
        kernels.tier_count(cid, row, hot_map)
    return w, ids


def _inputs(t, seed, n_pad):
    from edge0_vllm import kernels
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(t, E, generator=g, device="cuda") * 2
    w, ids = kernels.route_topk_softmax(logits, K)
    pad = torch.zeros(t, dtype=torch.bool, device="cuda")
    if n_pad:
        pad[t - n_pad:] = True
    hot_map = torch.where(torch.rand(E, generator=g, device="cuda") < 0.4,
                          torch.arange(E, device="cuda", dtype=torch.int32), torch.full((E,), -1, device="cuda",
                                                                                          dtype=torch.int32))
    pad_ids = torch.arange(K, dtype=torch.int32, device="cuda").view(1, K) * 3
    return ids.contiguous(), w.contiguous(), pad, pad_ids, hot_map


@pytest.mark.parametrize("t,n_pad", [(1, 0), (4, 0), (4, 1), (12, 3), (16, 0), (16, 5), (64, 7), (100, 0)])
@pytest.mark.parametrize("mode", ["both", "pad_only", "count_only", "no_mask"])
def test_route_post_matches_unfused(t, n_pad, mode):
    from edge0_vllm import kernels
    ids, w, pad, pad_ids, hot_map = _inputs(t, 7 + t + n_pad, n_pad)
    use_pad = None if mode == "no_mask" else pad
    use_padids = None if mode == "count_only" else pad_ids
    r0 = torch.rand(E + kernels.TIER_STATS, device="cuda") if mode != "pad_only" else None
    r1 = r0.clone() if r0 is not None else None
    w0, i0 = _unfused(ids, w, use_pad, use_padids, r0, hot_map)
    w1, i1 = kernels.route_post(ids, w, use_pad, use_padids, r1, hot_map)
    torch.cuda.synchronize()
    assert torch.equal(i0, i1)
    assert torch.equal(w0, w1)
    if r0 is not None:
        assert torch.equal(r0, r1)


def test_route_post_graph_time():
    from edge0_vllm import kernels
    t, L = 4, 48
    ins = [_inputs(t, 100 + i, 0) for i in range(L)]
    rows_a = torch.zeros(L, E + kernels.TIER_STATS, device="cuda")
    rows_b = rows_a.clone()
    res = {}
    for name, fn, rows in (("unfused", _unfused, rows_a), ("fused", kernels.route_post, rows_b)):
        def body():
            out = []
            for i, (ids, w, pad, pad_ids, hot_map) in enumerate(ins):
                out.append(fn(ids, w, pad, pad_ids, rows[i], hot_map))
            return out
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            body()
        torch.cuda.current_stream().wait_stream(s)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            outs = body()
        for _ in range(5):
            gr.replay()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(200):
            gr.replay()
        e1.record()
        torch.cuda.synchronize()
        res[name] = e0.elapsed_time(e1) * 1e3 / (200 * L)
        res[name + "_out"] = outs
    print(f"routing tail per MoE call in a graph of 48 (T=4, padding mask + tier count): unfused {res['unfused']:.2f} us, "
          f"fused {res['fused']:.2f} us")
    for (wa, ia), (wb, ib) in zip(res["unfused_out"], res["fused_out"]):
        assert torch.equal(ia, ib) and torch.equal(wa, wb)
    torch.testing.assert_close(rows_a, rows_b, rtol=0, atol=0)
