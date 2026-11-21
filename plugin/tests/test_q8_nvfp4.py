"""Q8_0 dequant-on-load and the NVFP4 checkpoint layout, against the numpy oracle (strata_formats.py)."""
import numpy as np
import pytest
import torch

from edge0_vllm import marlin_nvfp4 as mn
from edge0_vllm import q8

from helpers import (SRC_MODEL, expert_record_to_modelopt, ggml_nvfp4_rows_to_modelopt, have_src_model,
                     strata_formats, synthetic_ggml_nvfp4_rows, to_torch_fp8)

sf = strata_formats()


def test_q8_roundtrip_and_oracle():
    torch.manual_seed(0)
    w = torch.randn(96, 256) * 0.05
    blocks = q8.quant_q8_0(w)
    assert q8.is_q8_blocks(blocks) and blocks.shape == (96, 8, 34)
    got = q8.dequant_q8_0(blocks, torch.float32)
    d = w.reshape(96, 8, 32).abs().amax(-1, keepdim=True) / 127              # per-block step
    assert ((got - w).reshape(96, 8, 32).abs() <= d * 0.5 + 1e-5).all()
    if sf is not None:
        ref = sf.deq_q8_0(blocks.numpy()).reshape(96, 256)
        assert np.array_equal(got.numpy(), ref)                      # bit-exact fp32 vs the oracle
        bf = q8.dequant_q8_0(blocks, torch.bfloat16)
        assert torch.equal(bf, torch.from_numpy(ref).to(torch.bfloat16))   # RNE, like the converter


def test_q8_meta_and_loader_wrapper():
    meta = q8.dequant_q8_0(torch.empty(14336, 80, 34, dtype=torch.uint8, device="meta"))
    assert meta.device.type == "meta" and meta.shape == (14336, 2560) and meta.dtype == torch.bfloat16
    calls = []

    def loader(param, loaded, *args, **kw):
        calls.append((loaded.dtype, tuple(loaded.shape), getattr(loaded, "shard_id", None), args))
        param.data.copy_(loaded)

    wrapped = q8.wrap_weight_loader(loader)
    assert q8.wrap_weight_loader(wrapped) is wrapped
    p = torch.nn.Parameter(torch.zeros(64, 64, dtype=torch.bfloat16), requires_grad=False)
    b = q8.quant_q8_0(torch.randn(64, 64))
    b.shard_id = (0, 1, 2)
    wrapped(p, b, 7)
    assert calls[-1] == (torch.bfloat16, (64, 64), (0, 1, 2), (7,))
    assert torch.equal(p.data, q8.dequant_q8_0(b))
    plain = torch.ones(64, 64, dtype=torch.bfloat16)
    wrapped(p, plain)                                                # BF16 tensors pass through untouched
    assert calls[-1][0] == torch.bfloat16 and torch.equal(p.data, plain)


@pytest.mark.skipif(sf is None or not have_src_model(), reason="needs the source model folder")
def test_q8_real_latent_rows():
    """Four D rows of layer 3 from latent_dense_q8.bin (2720 B per row): dequant == oracle."""
    du = sf.LatentDense(SRC_MODEL / "latent_dense_q8.bin", 8)
    off = 3 * du.layer_bytes + 100 * du.d_row
    raw = np.fromfile(du.path, dtype=np.uint8, count=4 * du.d_row, offset=off)
    got = q8.dequant_q8_0(torch.from_numpy(raw.reshape(4, 80, 34)), torch.float32).numpy()
    ref = sf.deq_q8_0(raw).reshape(4, 2560)
    assert np.array_equal(got, ref)


def test_unpack_codes_order():
    # byte m: element 2m in the low nibble, 2m+1 in the high nibble (ModelOpt / vLLM break_fp4_bytes)
    b = torch.tensor([[0x21, 0x43]], dtype=torch.uint8)
    assert mn.unpack_codes(b).tolist() == [[1, 2, 3, 4]]
    assert mn.e2m1_table()[torch.tensor([0, 1, 7, 9, 15])].tolist() == [0.0, 0.5, 6.0, -0.5, -6.0]


@pytest.mark.skipif(sf is None, reason="needs edge0/convert/strata_formats.py")
def test_contract_repack_synthetic():
    """Random ggml NVFP4 rows -> contract repack -> plugin dequant == oracle deq_nvfp4 (exact, fp32)."""
    rng = np.random.default_rng(1)
    for k in (1792, 640):
        raw = synthetic_ggml_nvfp4_rows(rng, 8, k)
        codes, scales = ggml_nvfp4_rows_to_modelopt(raw, k)
        got = mn.dequant_modelopt_nvfp4(torch.from_numpy(codes), to_torch_fp8(scales), 0.75).numpy()
        ref = np.stack([sf.deq_nvfp4(r) for r in raw]) * np.float32(0.75)
        assert np.array_equal(got, ref), k


@pytest.mark.skipif(sf is None or not have_src_model(), reason="needs the source model folder")
@pytest.mark.parametrize("layer,expert", [(0, 7), (47, 511)])
def test_contract_repack_real_expert(layer, expert):
    """A real experts.bin record through the contract repack decodes exactly like Strata's, and one expert's
    SwiGLU in the plugin's reference matches the oracle."""
    m = sf.EdgeModel(SRC_MODEL)
    blob = m.expert_raw(layer, expert)
    t = expert_record_to_modelopt(blob, sf)
    assert t["g13"][0] == t["g13"][1]                                  # w13 shares one global scale
    for s in (t["s13"], t["s2"]):                                     # contract asserts (DESIGN.md 1.5)
        assert not (s & 0x80).any() and not (s == 0x7F).any()
    ref = m.expert(layer, expert)
    w13 = mn.dequant_modelopt_nvfp4(torch.from_numpy(t["w13"]), to_torch_fp8(t["s13"]), float(t["g13"][0]))
    w2 = mn.dequant_modelopt_nvfp4(torch.from_numpy(t["w2"]), to_torch_fp8(t["s2"]), float(t["g2"]))
    assert np.array_equal(w13[:640].numpy(), ref["gate"]) and np.array_equal(w13[640:].numpy(), ref["up"])
    assert np.array_equal(w2.numpy(), ref["down"])
    x = torch.from_numpy(np.random.default_rng(2).standard_normal((3, 1792)).astype(np.float32) * 0.1)
    y = mn.expert_forward_reference(x, torch.from_numpy(t["w13"]), to_torch_fp8(t["s13"]), float(t["g13"][0]),
                                    torch.from_numpy(t["w2"]), to_torch_fp8(t["s2"]), float(t["g2"])).numpy()
    xn = x.numpy()
    y_ref = (sf.silu(xn @ ref["gate"].T) * (xn @ ref["up"].T)) @ ref["down"].T
    assert np.allclose(y, y_ref, rtol=1e-4, atol=1e-6)
