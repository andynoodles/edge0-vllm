"""edge0_vllm.q8gemv: Triton Q8_0 GEMV over b12x's packed layout, with latent D/U group skipping.

CPU: ``TRITON_INTERPRET=1 python -m pytest tests/test_q8gemv.py -k interp`` (Triton interpreter, small shapes).
GPU: the same checks at the real shapes, plus equality-level agreement with b12x's own Q8_0 kernel.
"""
import os

import pytest
import torch

GPU = torch.cuda.is_available()
INTERP = os.environ.get("TRITON_INTERPRET") == "1"


def _rand_blocks(n, k, dev, gen):
    kb = k // 32
    d = (torch.rand((n, kb), generator=gen) * 2e-3 + 1e-4).to(torch.float16)
    q = torch.randint(-127, 128, (n, kb, 32), generator=gen, dtype=torch.int8)
    blocks = torch.empty((n, kb, 34), dtype=torch.uint8)
    blocks[..., :2] = d.view(torch.uint8).view(n, kb, 2)
    blocks[..., 2:] = q.view(torch.uint8)
    return blocks.to(dev)


def _pack_like_b12x(blocks):
    """b12x ``pack_iq2_xs_weight`` for q8_0 (N % 128 == 0): payload tiles [N/128, K/32, 128, 32], bases [K/32, N, 2]."""
    n, kb, _ = blocks.shape
    values = blocks[..., 2:].reshape(n // 128, 128, kb, 32).permute(0, 2, 1, 3).contiguous().view(n, kb * 32)
    meta = blocks[..., :2].permute(1, 0, 2).contiguous()
    return values, meta


def _ref(x, blocks):
    n, kb, _ = blocks.shape
    d = blocks[..., :2].contiguous().view(torch.float16).float()
    q = blocks[..., 2:].contiguous().view(torch.int8).float()
    return x.float() @ (q * d).reshape(n, kb * 32).t()


def _check(dev, n, k, m, use_dot, mask_mode=0, ngroups=8):
    from edge0_vllm import q8gemv
    gen = torch.Generator().manual_seed(n + k + m)
    blocks = _rand_blocks(n, k, dev, gen)
    values, meta = _pack_like_b12x(blocks)
    x = torch.randn((m, k), generator=gen).to(torch.bfloat16).to(dev)
    ids = eg = None
    if mask_mode:
        eg = (torch.arange(512, dtype=torch.int32) % 8).to(dev)
        ids = (torch.randint(0, 64, (m, 10), generator=gen, dtype=torch.int32) * 8 +
               torch.randint(0, ngroups, (m, 10), generator=gen, dtype=torch.int32)).to(dev)
        ids[0, :ngroups] = torch.arange(ngroups, dtype=torch.int32, device=dev)
        if mask_mode == 2:
            x.view(m, 8, -1)[:, ngroups:] = 0     # z is zero on unused groups
    y = q8gemv.q8_gemv(x, values, meta, n, k, ids=ids, expert_group=eg, mask_mode=mask_mode, use_dot=use_dot)
    ref = _ref(x, blocks)
    if mask_mode == 1:
        g = q8gemv.LATENT_DIM
        y, ref = y[:, :ngroups * g], ref[:, :ngroups * g]
    err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
    assert err < 8e-3, err       # bf16 output rounding
    return y


@pytest.mark.skipif(not INTERP, reason="needs TRITON_INTERPRET=1")
def test_interp_plain():
    # (the interpreter's bf16 tl.dot is not faithful: only the CUDA-core variant is checked on CPU)
    _check("cpu", 256, 128, 3, False)


@pytest.mark.skipif(not INTERP, reason="needs TRITON_INTERPRET=1")
@pytest.mark.parametrize("mode", [1, 2])
def test_interp_mask(mode, monkeypatch):
    from edge0_vllm import q8gemv
    # shrink the latent geometry so the interpreter stays fast: 8 groups of 64 rows (D) / 2 kb (U)
    monkeypatch.setattr(q8gemv, "LATENT_DIM", 64)
    if mode == 1:
        _check("cpu", 512, 64, 2, False, mask_mode=1, ngroups=3)
    else:
        _check("cpu", 128, 512, 2, False, mask_mode=2, ngroups=3)


def _check_shared(dev, m, n_ff, h, use_dot):
    """routed + sigmoid(x.wg) * down(silu(g) * u) through the two epilogues vs an fp32 reference."""
    from edge0_vllm import q8gemv
    gen = torch.Generator().manual_seed(7 + m)
    bgu = _rand_blocks(2 * n_ff, h, dev, gen)
    bdn = _rand_blocks(h, n_ff, dev, gen)
    vgu, mgu = _pack_like_b12x(bgu)
    vdn, mdn = _pack_like_b12x(bdn)
    x = torch.randn((m, h), generator=gen).to(torch.bfloat16).to(dev)
    routed = torch.randn((m, h), generator=gen).to(torch.bfloat16).to(dev)
    wg = (torch.randn(h, generator=gen) * 0.05).to(torch.bfloat16).to(dev)
    hh = q8gemv.q8_gemv(x, vgu, mgu, 2 * n_ff, h, epi=q8gemv.EPI_SILU_MUL, use_dot=use_dot)
    y = q8gemv.q8_gemv(hh, vdn, mdn, h, n_ff, epi=q8gemv.EPI_GATED_ADD, add=routed, xg=x, wg=wg, use_dot=use_dot)
    gu = _ref(x, bgu)
    hr = torch.nn.functional.silu(gu[:, :n_ff]) * gu[:, n_ff:]
    ref = routed.float() + torch.sigmoid(x.float() @ wg.float())[:, None] * _ref(hr.to(torch.bfloat16), bdn)
    err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
    assert err < 1e-2, err


def test_lmhead_requant_recovers_q8():
    """EDGE0_LMHEAD=q8: re-quantizing the BF16 head (= bf16(q*d) of GGUF Q8_0 blocks) gives back the blocks
    (ggml blocks always hold a |q| = 127 element, so amax recovers d up to bf16 rounding)."""
    from edge0_vllm.lmhead_q8 import quantize_q8_0_blocks
    gen = torch.Generator().manual_seed(3)
    n, k = 512, 256
    blocks = _rand_blocks(n, k, "cpu", gen)
    q = blocks[..., 2:].contiguous().view(torch.int8)
    q[..., 0] = 127                                     # ggml: d = amax / 127 -> one element at +-127
    blocks[..., 2:] = q.view(torch.uint8)
    w = (q.float() * blocks[..., :2].contiguous().view(torch.float16).float()).reshape(n, k).to(torch.bfloat16)
    rq = quantize_q8_0_blocks(w)
    q2 = rq[..., 2:].contiguous().view(torch.int8)
    same_q = (q2 == q).float().mean().item()
    off = (q2.int() - q.int()).abs().max().item()
    err = (_ref(torch.eye(k), rq) - _ref(torch.eye(k), blocks)).abs().max().item()
    assert same_q > 0.98 and off <= 1, (same_q, off)
    assert err < 2e-3 * 1.05, err


@pytest.mark.skipif(not INTERP, reason="needs TRITON_INTERPRET=1")
def test_interp_shared():
    _check_shared("cpu", 2, 128, 128, False)


@pytest.mark.skipif(not GPU, reason="needs a GPU")
@pytest.mark.parametrize("m", [1, 4])
def test_gpu_shared(m):
    _check_shared("cuda", m, 640, 2560, True)


@pytest.mark.skipif(not GPU, reason="needs a GPU")
@pytest.mark.parametrize("nk", [(14336, 2560), (2560, 14336), (1280, 2560), (2560, 640)])
@pytest.mark.parametrize("m", [1, 4, 8])
def test_gpu_shapes(nk, m):
    _check("cuda", nk[0], nk[1], m, True)


@pytest.mark.skipif(not GPU, reason="needs a GPU")
@pytest.mark.parametrize("ng", [1, 4, 8])
def test_gpu_mask(ng):
    _check("cuda", 14336, 2560, 4, True, mask_mode=1, ngroups=ng)
    _check("cuda", 2560, 14336, 4, True, mask_mode=2, ngroups=ng)
