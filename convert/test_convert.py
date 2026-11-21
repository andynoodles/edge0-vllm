#!/usr/bin/env python3
"""edge0/convert/test_convert.py - unit tests for the converter's codecs, repack and writer (no model files needed).

    python3 test_convert.py            # plain asserts; also runs under pytest
Optional cross-checks run when the libraries exist (vLLM image): ml_dtypes (bf16 / e4m3 casts), safetensors + torch
(safe_open of a written shard).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ckpt_contract as C  # noqa: E402
import strata_formats as SF  # noqa: E402
from convert_checkpoint import quantize_nvfp4, repack_ggml_nvfp4  # noqa: E402
from st_io import (E2M1_VALUES, E4M3FN_STRICT, ShardReader, ShardWriter, bf16_bits, bf16_to_f32,  # noqa: E402
                   e2m1_encode, e4m3_encode)


def _random_blocks(rng, shape_lead, nb):
    b = rng.integers(0, 256, size=shape_lead + (nb, 36), dtype=np.uint8)
    b[..., :4] &= 0x7F                                   # valid UE4M3 scales (bit 7 clear) ...
    b[..., :4][b[..., :4] == 0x7F] = 0x7E                # ... and no NaN pattern
    return b


def test_repack_matches_contract_pseudocode():
    rng = np.random.default_rng(0)
    blocks = _random_blocks(rng, (3, 5), 4)              # 3 x 5 rows, K = 4 * 64 = 256
    packed, scales = repack_ggml_nvfp4(blocks)
    assert packed.shape == (3, 5, 128) and scales.shape == (3, 5, 16)
    for i in range(3):
        for r in range(5):
            row = blocks[i, r]
            code = np.zeros(256, np.uint8)
            sc = np.zeros(16, np.uint8)
            for b in range(4):                           # DESIGN 1.5 pseudo-code, literally
                d, qs = row[b, :4], row[b, 4:]
                for s in range(4):
                    for j in range(16):
                        code[64 * b + 16 * s + j] = (qs[8 * s + (j % 8)] >> (4 * (j // 8))) & 0xF
                    sc[4 * b + s] = d[s]
            out = (code[0::2] | (code[1::2] << 4)).astype(np.uint8)
            assert np.array_equal(out, packed[i, r]) and np.array_equal(sc, scales[i, r])


def test_repack_decode_equals_strata_decode():
    rng = np.random.default_rng(1)
    blocks = _random_blocks(rng, (7,), 28)               # 7 rows of K = 1792
    want = SF.deq_nvfp4(blocks.reshape(-1)).reshape(7, 1792)
    packed, scales = repack_ggml_nvfp4(blocks)
    codes = np.stack([packed & 0x0F, packed >> 4], axis=-1).reshape(7, 1792)
    got = (SF.E2M1[codes].reshape(7, 112, 16) * E4M3FN_STRICT[scales][..., None]).reshape(7, 1792)
    assert np.array_equal(got, want)


def test_bf16_rne():
    x = np.array([0.0, -0.0, 1.0, 1.00390625, 1.005859375, 1.0078125, 3.14159265, -2.5e-8, 1e30, -65504.0],
                 np.float32)
    rng = np.random.default_rng(2)
    x = np.concatenate([x, rng.standard_normal(100000).astype(np.float32) * 10 ** rng.uniform(-6, 6, 100000)
                        .astype(np.float32)])
    b = bf16_bits(x)
    back = bf16_to_f32(b)
    # nearest: no other bf16 neighbour is closer
    up = bf16_to_f32((b.astype(np.int32) + 1).astype(np.uint16))
    dn = bf16_to_f32((b.astype(np.int32) - 1).astype(np.uint16))
    err = np.abs(back.astype(np.float64) - x)
    d_up = np.nan_to_num(np.abs(up.astype(np.float64) - x), nan=np.inf)   # wrap-around neighbours are NaN
    d_dn = np.nan_to_num(np.abs(dn.astype(np.float64) - x), nan=np.inf)
    assert (err <= d_up).all() and (err <= d_dn).all()
    assert bf16_bits(np.array([1.00390625], np.float32))[0] == 0x3F80      # tie -> even (down)
    assert bf16_bits(np.array([1.01171875], np.float32))[0] == 0x3F82      # tie -> even (up)
    try:
        import ml_dtypes
        assert np.array_equal(b, x.astype(ml_dtypes.bfloat16).view(np.uint16))
    except ImportError:
        pass


def test_e4m3_encode():
    codes = np.arange(0x7F, dtype=np.uint8)
    assert np.array_equal(e4m3_encode(E4M3FN_STRICT[codes]), codes)
    assert e4m3_encode(np.array([1000.0]))[0] == 0x7E                     # saturate at 448
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 460, 50000)
    c = e4m3_encode(x)
    v = E4M3FN_STRICT[c].astype(np.float64)
    pos = E4M3FN_STRICT[:0x7F].astype(np.float64)
    best = np.abs(pos[None, :] - np.minimum(x, 448)[:, None]).min(1)
    assert np.allclose(np.abs(v - np.minimum(x, 448)), best)
    try:
        import ml_dtypes
        y = rng.uniform(0, 448, 50000).astype(np.float32)
        assert np.array_equal(e4m3_encode(y), y.astype(ml_dtypes.float8_e4m3fn).view(np.uint8))
    except ImportError:
        pass


def test_e2m1_encode():
    grid = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
    assert np.array_equal(e2m1_encode(grid), np.arange(8))
    assert np.array_equal(e2m1_encode(-grid[1:]), np.arange(1, 8) | 8)
    assert e2m1_encode(np.array([0.25], np.float32))[0] == 0               # tie -> even code 0
    assert e2m1_encode(np.array([0.75], np.float32))[0] == 2               # tie 0.5/1.0 -> code 2 (1.0)
    assert e2m1_encode(np.array([2.5], np.float32))[0] == 4                # tie 2/3 -> code 4 (2.0)
    assert e2m1_encode(np.array([5.0], np.float32))[0] == 6                # tie 4/6 -> code 6 (4.0)
    assert e2m1_encode(np.array([100.0], np.float32))[0] == 7
    assert np.array_equal(E2M1_VALUES[e2m1_encode(np.array([-3.2, 1.3], np.float32))], [-3.0, 1.5])


def test_quantize_q2_like_values():
    rng = np.random.default_rng(4)
    d = rng.uniform(1e-4, 1e-2, size=(64, 40)).astype(np.float32)          # one Q2_0 scale per 64 values
    codes = rng.integers(0, 4, size=(64, 40, 64))
    w = ((codes - 1) * d[..., None]).reshape(64, 2560).astype(np.float32)
    g = np.float32(np.abs(w).max()) / np.float32(6 * 448)
    packed, s8, deq = quantize_nvfp4(w, g)
    assert packed.shape == (64, 1280) and s8.shape == (64, 160)
    assert ((s8 & 0x80) == 0).all() and (s8 != 0x7F).all()
    cd = np.stack([packed & 0x0F, packed >> 4], axis=-1).reshape(64, 2560)
    redo = (E2M1_VALUES[cd].reshape(64, 160, 16) * (E4M3FN_STRICT[s8] * g)[..., None]).reshape(64, 2560)
    assert np.array_equal(redo, deq)
    rel = np.linalg.norm(deq - w) / np.linalg.norm(w)
    assert rel < 0.06, rel


def test_writer_roundtrip():
    rng = np.random.default_rng(5)
    specs = C.shard_layout([C.Spec("a.bf16", "BF16", (3, 5)), C.Spec("odd.f32", "F32", (1,)),
                            C.Spec("s.f32", "F32", ()), C.Spec("u8", "U8", (4, 2, 34)),
                            C.Spec("e4", "F8_E4M3", (2, 8)), C.Spec("i32", "I32", (6,)), C.Spec("f16", "F16", (2, 1, 4))])
    assert [s.name for s in specs][-2:] == ["odd.f32", "s.f32"]
    data = {"a.bf16": rng.integers(0, 65535, (3, 5), dtype=np.uint16), "odd.f32": np.array([0.5], np.float32),
            "s.f32": np.array(2.0, np.float32), "u8": rng.integers(0, 255, (4, 2, 34), dtype=np.uint8),
            "e4": rng.integers(0, 0x7E, (2, 8), dtype=np.uint8), "i32": np.arange(6, dtype=np.int32),
            "f16": rng.standard_normal((2, 1, 4)).astype(np.float16)}
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "x.safetensors")
        w = ShardWriter(p, specs, sync_every=64)
        for s in specs[::-1]:                                              # any tensor order, in pieces
            a = data[s.name].reshape(-1).view(np.uint8)
            w.write(s.name, a[: len(a) // 2])
            w.write(s.name, a[len(a) // 2:])
        w.finish()
        assert not os.path.exists(p + ".partial")
        r = ShardReader(p)
        for k, v in data.items():
            assert np.array_equal(r.tensor(k), v), k
        hdr = json.loads(open(p, "rb").read()[8:8 + int.from_bytes(open(p, "rb").read(8), "little")])
        assert all(hdr[s.name]["data_offsets"][0] % 8 == 0 for s in specs if s.nbytes % 8 == 0)
        try:
            import torch
            from safetensors import safe_open
            with safe_open(p, framework="pt") as f:
                assert f.get_tensor("e4").dtype == torch.float8_e4m3fn and f.get_tensor("s.f32").shape == ()
                assert f.get_tensor("a.bf16").dtype == torch.bfloat16
                assert torch.equal(f.get_tensor("i32"), torch.arange(6, dtype=torch.int32))
        except ImportError:
            pass


def test_contract_counts():
    cfg = C.build_config(48, True)
    ql = cfg["quantization_config"]["quantized_layers"]
    assert len(ql) == 371 and len(C.q8_module_names(48, True)) == 370
    assert "quant_algo" not in cfg["quantization_config"] and cfg["eos_token_id"] == 248044
    assert sum(s.nbytes for s in C.expert_specs(0)) == 990910464
    assert len(cfg["layer_types"]) == 48 and cfg["layer_types"][3] == "indexed_attention"
    assert C.GDN_V_PERM[:6] == [0, 16, 32, 1, 17, 33] and sorted(C.GDN_V_PERM) == list(range(48))
    plan = C.shard_plan(4, False)
    assert [f for f, _ in plan] == ["model-%05d-of-00005.safetensors" % i for i in range(1, 6)]
    names = [s.name for _, specs in C.shard_plan(48, True) for s in specs]
    assert len(names) == len(set(names))
    for f, specs in C.shard_plan(48, True):
        for s in specs:
            assert C.shard_name(C.expected_shard_of(s.name, 48, True), 50) == f


if __name__ == "__main__":
    n = 0
    for k, fn in sorted(globals().items()):
        if k.startswith("test_") and callable(fn):
            fn()
            n += 1
            print("ok", k)
    print("%d tests passed" % n)
