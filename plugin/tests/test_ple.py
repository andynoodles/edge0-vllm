"""PLE: vLLM's n-gram ids vs Strata's (MODEL-SPEC 3.1), the GGUF table-layout patch, and the threaded reader."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from edge0_vllm import ple

from helpers import SRC_MODEL, strata_formats

sf = strata_formats()
EOS = 248044
MULT = [23703573157769, 20109073645365, 8052911324071]
VOCAB = [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
         20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]


def _vllm_hash_module():
    """What Qwen4ExpNGramEmbedding derives from the Edge0 config (seed 1234, vocab 248320, base 20M, 3-gram,
    8 heads per n-gram, PLE dense layer 0), as the CPU-id namespace vLLM itself uses (copy_module_for_cpu_ids)."""
    from vllm.models.qwen4_exp.nvidia.ngram_embedding import Qwen4ExpNGramEmbedding as N
    mult = N._make_layer_multipliers(ngram_size=3, unigram_vocab_size=248320, seed=1234, ple_dense_layer_id=0)
    sizes, offsets, total = N._make_vocab_layout(ngram_vocab_size_base=20_000_000, ngram_heads=16,
                                                 ple_dense_layer_id=0)
    ns = SimpleNamespace(layer_multipliers=torch.tensor(mult), ngram_heads_vocab_sizes=torch.tensor(sizes),
                         ngram_heads_offsets=torch.tensor(offsets), eos_token_id=EOS, ngram_size=3,
                         heads_per_ngram=8, _shift_precompute=N._shift_precompute, _shift_apply=N._shift_apply)
    return N, ns, mult, sizes, offsets, total


def test_hash_constants_match_gguf():
    _, _, mult, sizes, offsets, total = _vllm_hash_module()
    assert mult == MULT and sizes == VOCAB
    assert offsets[-1] == 300001275 and total == 320001446
    assert -(-total // 128) * 128 == 320001536                      # padded rows = ple_rows of the contract


def _strata_rows(seq):
    c = sf.PleConsts(MULT, VOCAB, list(np.cumsum([0] + VOCAB[:-1])), EOS)
    return sf.ngram_rows_for_sequence(seq, c).astype(np.int64)


@pytest.mark.skipif(sf is None, reason="needs edge0/convert/strata_formats.py")
def test_ngram_ids_prefill_batch_vs_strata():
    N, ns, *_ = _vllm_hash_module()
    rng = np.random.default_rng(0)
    seqs = [rng.integers(0, 248320, 11).tolist(), [5, 7, EOS, 9, 11, EOS, EOS, 3], [0, 0, 0], [248319]]
    seqs[0][4] = EOS
    ids = torch.tensor([t for s in seqs for t in s], dtype=torch.int64)
    qsl = torch.tensor(np.cumsum([0] + [len(s) for s in seqs]), dtype=torch.int64)
    ctx = torch.full((len(seqs), 2), EOS, dtype=torch.int64)       # fresh sequences: context = EOS
    got = N.compute_ngram_ids(ns, ids, qsl, ctx).numpy()
    ref = np.concatenate([_strata_rows(s) for s in seqs])
    assert got.shape == ref.shape == (len(ids), 16)
    assert np.array_equal(got, ref)


@pytest.mark.skipif(sf is None, reason="needs edge0/convert/strata_formats.py")
def test_ngram_ids_decode_continuation_vs_strata():
    """Decode step: one new token per request, the previous two tokens come in ngram_context (as model_state
    builds it); must equal the rows Strata computes for that position of the whole sequence."""
    N, ns, *_ = _vllm_hash_module()
    full = [[11, 22, 33, 44], [EOS, 5, 6], [7, EOS, 8], [9]]
    last = torch.tensor([s[-1] for s in full], dtype=torch.int64)
    qsl = torch.arange(len(full) + 1, dtype=torch.int64)
    ctx = torch.tensor([[s[-3] if len(s) >= 3 else EOS, s[-2] if len(s) >= 2 else EOS] for s in full],
                       dtype=torch.int64)
    got = N.compute_ngram_ids(ns, last, qsl, ctx).numpy()
    ref = np.stack([_strata_rows(s)[-1] for s in full])
    assert np.array_equal(got, ref)


def _fake_ckpt(tmp_path, rows=1000, row_bytes=160, offset=352, real_target=None):
    ck = tmp_path / "ckpt"
    ck.mkdir()
    edge0 = {"format": "edge0-vllm-v1", "latent_dim": 1792, "latent_groups": 8, "expert_intermediate_size": 640,
             "expert_format": "modelopt_nvfp4_stacked", "ple_file": "ple-fp8.gguf", "ple_data_offset": offset,
             "ple_rows": rows, "ple_row_bytes": row_bytes, "ple_scale": 0.0002, "profile_file": "profile.bin"}
    (ck / "config.json").write_text(json.dumps({"model_type": "qwen4_exp_text", "edge0": edge0}))
    if real_target is not None:
        os.symlink(real_target, ck / "ple-fp8.gguf")
    else:
        data = np.random.default_rng(0).integers(0, 256, offset + rows * row_bytes, dtype=np.uint8)
        data[:4] = np.frombuffer(b"GGUF", np.uint8)
        (ck / "ple-fp8.gguf").write_bytes(data.tobytes())
    (ck / "model-00001-of-00002.safetensors").write_bytes(b"")
    return ck


def test_table_layout_patch(tmp_path):
    from vllm.models.qwen4_exp.nvidia.ple_pageable import ShardLocation, TableLayout
    ck = _fake_ckpt(tmp_path)
    calls = []

    def orig(*a, **kw):
        calls.append((a, kw))
        return "orig"

    f = ple.make_discover_table_layout(orig, TableLayout, ShardLocation)
    files = [str(ck / "model-00001-of-00002.safetensors")]
    lay = f(files, 1, 900, 160, torch.float8_e4m3fn, 512, suffix="weight")
    assert lay.row_bytes == 160 and lay.rows_per_shard == 1000 and lay.num_rows == 900
    assert lay.shards == (ShardLocation(str((ck / "ple-fp8.gguf").resolve()), 352, 1000),)
    assert f(files, 1, 900, 160, torch.float8_e4m3fn, 512, suffix="weight_scale") == "orig"
    with pytest.raises(ValueError):
        f(files, 1, 2000, 160, torch.float8_e4m3fn, 512)                # more rows than the table
    with pytest.raises(ValueError):
        f(files, 1, 900, 80, torch.float8_e4m3fn, 512)                  # wrong row width
    other = tmp_path / "other"
    other.mkdir()
    assert f([str(other / "x.safetensors")], 1, 900, 160, torch.float8_e4m3fn, 512) == "orig"


@pytest.mark.skipif(not (SRC_MODEL / "ple-fp8.gguf").is_file(), reason="source model folder not mounted")
def test_table_layout_real_file(tmp_path):
    """The contract's offset/rows/row_bytes match the real 51.2 GB file (stat + 4-byte magic only)."""
    ck = _fake_ckpt(tmp_path, rows=320001536, row_bytes=160, offset=352, real_target=SRC_MODEL / "ple-fp8.gguf")
    path, spec = ple.ple_table_location(ck)
    assert spec.ple_rows == 320001536 and os.path.getsize(path) == 352 + 320001536 * 160


def test_threaded_reader_equals_serial(tmp_path):
    from vllm.models.qwen4_exp.nvidia import ngram_embedding as ne
    from vllm.models.qwen4_exp.nvidia.ple_pageable import ShardLocation, TableLayout
    ck = _fake_ckpt(tmp_path, rows=5000)
    path = str(ck / "ple-fp8.gguf")
    layout = TableLayout(num_rows=5000, row_bytes=160, rows_per_shard=5000, shards=(ShardLocation(path, 352, 5000),))
    plane = ne._FilePlane(layout, {})
    ids = np.unique(np.random.default_rng(1).integers(0, 5000, 3000))
    serial = np.zeros((len(ids), 160), np.uint8)
    plane.read(ids, serial)
    raw = np.fromfile(path, np.uint8)[352:].reshape(5000, 160)
    assert np.array_equal(serial, raw[ids])
    fake_ne = SimpleNamespace(_FilePlane=type("FP", (ne._FilePlane,), {}))
    ple._install_threaded_reader(fake_ne, 8)
    threaded = np.zeros_like(serial)
    fake_ne._FilePlane.read(plane, ids, threaded)
    assert np.array_equal(threaded, serial)
