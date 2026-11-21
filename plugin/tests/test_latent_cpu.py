"""The latent MoE algebra of the op (route -> route-as-token gather -> experts -> deterministic combine -> U_cat),
on CPU with the torch fallbacks, against loops and against Strata's per-group oracle on real Edge0 data."""
import numpy as np
import pytest
import torch

from edge0_vllm import kernels
from edge0_vllm import marlin_nvfp4 as mn

from helpers import SRC_MODEL, expert_record_to_modelopt, have_src_model, strata_formats, to_torch_fp8

sf = strata_formats()
G, L, E, K = 8, 1792, 512, 10


def test_gather_routes_matches_loop():
    torch.manual_seed(0)
    t = 5
    xl = torch.randn(t, G * L)
    grp = torch.randint(0, G, (E,), dtype=torch.int32)
    ids = torch.randint(0, E, (t, K), dtype=torch.int32)
    out = kernels.gather_routes(xl, ids, grp, L)
    for tt in range(t):
        for j in range(K):
            g = int(grp[ids[tt, j]])
            assert torch.equal(out[tt * K + j], xl[tt, g * L:(g + 1) * L])


@pytest.mark.parametrize("tiers", ["both", "hot", "cold"])
def test_combine_routes_tiers_and_order(tiers):
    torch.manual_seed(1)
    t = 4
    grp = torch.randint(0, G, (E,), dtype=torch.int32)
    ids = torch.stack([torch.randperm(E)[:K] for _ in range(t)]).to(torch.int32)
    hot_map = torch.full((E,), -1, dtype=torch.int32)
    hot = torch.randperm(E)[:200]
    hot_map[hot] = torch.arange(200, dtype=torch.int32)
    y_true = torch.randn(t * K, L)
    garbage = torch.full((t * K, L), float("nan"))                   # skipped rows of a tier are never read
    is_hot = hot_map[ids.long().reshape(-1)] >= 0
    y_hot = torch.where(is_hot[:, None], y_true, garbage)
    y_cold = torch.where(is_hot[:, None], garbage, y_true)
    if tiers == "hot":
        hot_map[:] = 0
        y_hot, y_cold = y_true, None
    elif tiers == "cold":
        hot_map[:] = -1
        y_hot, y_cold = None, y_true
    z = kernels.combine_routes(y_hot, y_cold, ids, grp, hot_map, G, L, torch.float32)
    ref = torch.zeros(t, G, L)
    for tt in range(t):
        for j in range(K):
            ref[tt, int(grp[ids[tt, j]])] += y_true[tt * K + j]
    assert torch.isfinite(z).all()
    assert torch.allclose(z.view(t, G, L), ref, atol=1e-5)


def test_touch_count_and_routing():
    cnt = torch.zeros(E)
    ids = torch.tensor([[3, 3, 5], [5, 7, -1]], dtype=torch.int32)
    kernels.touch_count(ids, cnt, touch=True)
    assert cnt[3] == 1 and cnt[5] == 1 and cnt[7] == 1 and cnt.sum() == 3
    kernels.touch_count(ids, cnt, touch=False)
    assert cnt[3] == 3 and cnt[5] == 3
    logits = torch.randn(6, E) * 3
    w, i = kernels.route_topk_softmax(logits, K)
    if sf is not None:
        for r in range(6):
            ids_ref, w_ref = sf.route(logits[r].numpy())
            assert i[r].tolist() == ids_ref.tolist()
            assert np.allclose(w[r].numpy(), w_ref, rtol=1e-5)


@pytest.mark.skipif(sf is None or not have_src_model(), reason="needs the source model folder")
def test_latent_moe_vs_strata_oracle_layer0():
    """Real layer-0 weights (router, D/U, the routed experts): the plugin's formulation, computed in fp32 with the
    plugin's own routing / gather / combine / ModelOpt-layout expert code, equals Strata's per-group computation."""
    m = sf.EdgeModel(SRC_MODEL)
    layer = 0
    rng = np.random.default_rng(3)
    x = rng.standard_normal(2560).astype(np.float32) * 0.5
    ref = sf.latent_moe_reference(m, layer, x, with_shared=False)

    router = m.pack().tensor(f"blk.{layer}.ffn_gate_inp.weight")             # (512, 2560)
    logits = torch.from_numpy(router @ x).view(1, E)
    w, ids = kernels.route_topk_softmax(logits, K)
    assert ids[0].tolist() == ref["ids"].tolist()
    labels = torch.from_numpy(m.labels()[layer].astype(np.int32))
    xt = torch.from_numpy(x)
    xl = torch.zeros(1, G * L)
    du = {}
    for g in sorted(set(labels[ids[0].long()].tolist())):
        D, U = m.latent_du(layer, g)
        du[g] = U
        xl[0, g * L:(g + 1) * L] = torch.from_numpy(D) @ xt
    x_routes = kernels.gather_routes(xl, ids, labels, L)
    y = torch.zeros(K, L)
    for j, e in enumerate(ids[0].tolist()):
        t = expert_record_to_modelopt(m.expert_raw(layer, e), sf)
        y[j] = mn.expert_forward_reference(
            x_routes[j:j + 1], torch.from_numpy(t["w13"]), to_torch_fp8(t["s13"]), float(t["g13"][0]),
            torch.from_numpy(t["w2"]), to_torch_fp8(t["s2"]), float(t["g2"]))[0] * w[0, j]
    z = kernels.combine_routes(y, None, ids, labels, torch.zeros(E, dtype=torch.int32), G, L, torch.float32)
    out = torch.zeros(2560)
    for g, U in du.items():                                                  # U_cat @ z restricted to used groups
        out += torch.from_numpy(U) @ z[0, g * L:(g + 1) * L]
    rel = float((out - torch.from_numpy(ref["routed"])).norm() / torch.from_numpy(ref["routed"]).norm())
    assert rel < 1e-5, rel
