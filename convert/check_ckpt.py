#!/usr/bin/env python3
"""edge0/convert/check_ckpt.py - validate a converted Edge0 checkpoint against the contract (DESIGN.md 1.7).

    # contract only (names, dtypes, shapes, shard rule, index, config, small files); numpy + stdlib, seconds
    python3 check_ckpt.py --ckpt /models/Edge0-145B-vllm
    # + lossless / numeric checks against the Strata folder (items 2-4); a few minutes, < 1.5 GB RSS
    python3 check_ckpt.py --ckpt /models/Edge0-145B-vllm --src /models/Edge0-145B-NVFP4 --level deep
    # + tokenizer (item 5) and safe_open(framework="pt") of every shard (item 6); run inside the vLLM image
    python3 check_ckpt.py --ckpt ... --src ... --level full --hf-ref /tokref [--corpus sharegpt.json] [--strata-ref f]

--write-results merges the results into <ckpt>/edge0_convert.json["checks"].  Exit status 0 = all checks passed.
Every check derives the expected bytes from the contract text independently of convert_checkpoint.py (row maps are
re-derived here, decoding uses strata_formats.py's readers as the oracle).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ckpt_contract as C  # noqa: E402
import strata_formats as SF  # noqa: E402
from st_io import DTYPE_BYTES, E4M3FN_STRICT, ShardReader, bf16_bits, bf16_to_f32, read_header  # noqa: E402


class Report:
    def __init__(self, name):
        self.name, self.ok, self.problems, self.info = name, True, [], {}

    def fail(self, msg):
        self.ok = False
        if len(self.problems) < 50:
            self.problems.append(msg)

    def expect(self, cond, msg):
        if not cond:
            self.fail(msg)
        return bool(cond)

    def as_dict(self):
        return {"ok": self.ok, "problems": self.problems, **self.info}


def _diff_dict(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a:
                out.append("%s%s missing" % (path, k))
            elif k not in b:
                out.append("%s%s unexpected" % (path, k))
            else:
                out += _diff_dict(a[k], b[k], "%s%s." % (path, k))
    elif a != b or type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        out.append("%s: %r != expected %r" % (path.rstrip("."), a if len(repr(a)) < 120 else repr(a)[:120],
                                                 b if len(repr(b)) < 120 else repr(b)[:120]))
    return out


# ============================================================================== 1.7 item 1: contract
def check_contract(ckpt: Path, src: Path | None = None) -> tuple[dict, dict]:
    R = Report("contract")
    cfg = json.loads((ckpt / "config.json").read_text())
    nl, mtp = C.config_layout(cfg)
    R.info.update(num_layers=nl, mtp=mtp)
    exp = C.build_config(nl, mtp, cfg.get("edge0", {}).get("source_dir", C.SOURCE_DIR_DEFAULT))
    for d in _diff_dict(cfg, exp):
        R.fail("config.json " + d)
    R.expect(isinstance(cfg.get("eos_token_id"), int) and cfg["eos_token_id"] == 248044, "eos_token_id must be int 248044")
    R.expect("quant_algo" not in cfg.get("quantization_config", {}), "quantization_config has a top-level quant_algo")
    gen = json.loads((ckpt / "generation_config.json").read_text())
    for d in _diff_dict(gen, C.GENERATION_CONFIG):
        R.fail("generation_config.json " + d)
    for f in C.TOKENIZER_FILES:
        R.expect((ckpt / f).is_file(), "missing %s" % f)
    if (ckpt / "tokenizer_config.json").is_file():
        tc = json.loads((ckpt / "tokenizer_config.json").read_text())
        for k, v in (("eos_token", "<|im_end|>"), ("pad_token", "<|endoftext|>"), ("add_bos_token", False),
                     ("tokenizer_class", "Qwen2Tokenizer")):
            R.expect(tc.get(k) == v, "tokenizer_config.json %s=%r" % (k, tc.get(k)))
        R.expect(bool(tc.get("chat_template")), "tokenizer_config.json has no chat_template")
    for f in C.COPIED_FILES + ("edge0_convert.json",):
        if not (ckpt / f).is_file():
            R.info.setdefault("warnings", []).append("missing %s" % f)

    # shards and index
    plan = C.shard_plan(nl, mtp)
    expected = {s.name: (fname, s) for fname, specs in plan for s in specs}
    idx = json.loads((ckpt / "model.safetensors.index.json").read_text())
    wm = idx.get("weight_map", {})
    R.expect(set(wm) == set(expected), "index weight_map: %d unexpected, %d missing (e.g. %s / %s)" % (
        len(set(wm) - set(expected)), len(set(expected) - set(wm)), sorted(set(wm) - set(expected))[:3],
        sorted(set(expected) - set(wm))[:3]))
    total = sum(s.nbytes for _, s in expected.values())
    R.expect(idx.get("metadata", {}).get("total_size") == total,
             "index total_size %s != %d" % (idx.get("metadata", {}).get("total_size"), total))
    nshards = C.num_shards(nl, mtp)
    for name, f in wm.items():
        if name in expected:
            want = C.shard_name(C.expected_shard_of(name, nl, mtp), nshards)
            R.expect(f == want == expected[name][0], "%s in %s, shard rule says %s" % (name, f, want))
    files = sorted({f for f, _ in plan})
    present = sorted(p.name for p in ckpt.glob("model-*.safetensors"))
    R.expect(present == files, "shard files %s != expected %d files" % (present[:3] + ["..."], len(files)))
    n_checked = 0
    for fname, specs in plan:
        p = ckpt / fname
        if not p.is_file():
            R.fail("missing shard %s" % fname)
            continue
        hdr, base = read_header(p)
        want = {s.name: s for s in specs}
        R.expect(set(hdr) == set(want), "%s: tensor set differs (%d vs %d)" % (fname, len(hdr), len(want)))
        spans = []
        for name, h in hdr.items():
            s = want.get(name)
            if s is None:
                continue
            b, e = h["data_offsets"]
            R.expect(h["dtype"] == s.dtype and list(h["shape"]) == list(s.shape),
                     "%s: %s %s, contract %s %s" % (name, h["dtype"], h["shape"], s.dtype, list(s.shape)))
            R.expect(e - b == s.nbytes, "%s: %d bytes, contract %d" % (name, e - b, s.nbytes))
            spans.append((b, e))
            n_checked += 1
        spans.sort()
        pos = 0
        for b, e in spans:
            R.expect(b == pos, "%s: gap/overlap at %d" % (fname, b))
            pos = e
        R.expect(p.stat().st_size == base + pos, "%s: file size %d != %d" % (fname, p.stat().st_size, base + pos))
    R.info["tensors_checked"] = n_checked
    R.info["total_size"] = total

    # PLE hardlink, profile, draft vocab
    ple = ckpt / cfg.get("edge0", {}).get("ple_file", "ple-fp8.gguf")
    if R.expect(ple.is_file(), "missing %s" % ple):
        st = ple.stat()
        R.expect(st.st_size == C.PLE_DATA_OFFSET + C.PLE_ROWS * C.PLE_ROW_BYTES, "ple size %d" % st.st_size)
        R.expect(st.st_nlink >= 2, "ple-fp8.gguf has st_nlink %d: not a hardlink" % st.st_nlink)
        if src is not None:
            R.expect(os.path.samefile(ple, src / "ple-fp8.gguf"), "ple-fp8.gguf is not the same inode as the source")
        pt = SF.PleTable(ple)
        R.expect((pt.t.offset, pt.n_rows, pt.scale) == (C.PLE_DATA_OFFSET, C.PLE_ROWS, C.PLE_SCALE),
                 "ple layout %s" % ((pt.t.offset, pt.n_rows, pt.scale),))
        R.info["ple"] = {"inode": st.st_ino, "nlink": st.st_nlink}
    prof = ckpt / "profile.bin"
    if R.expect(prof.is_file(), "missing profile.bin"):
        p = SF.read_profile(prof)
        R.expect(prof.stat().st_size == 196632 and p["n_ranked"] == 24576, "profile.bin header/size")
        if src is not None:
            R.expect(prof.read_bytes() == (src / "profile.bin").read_bytes(), "profile.bin differs from source")
    dvp = ckpt / "draft_vocab.json"
    if R.expect(dvp.is_file(), "missing draft_vocab.json"):
        dv = json.loads(dvp.read_text())
        R.expect(len(dv) == C.DRAFT_VOCAB_SIZE and dv == sorted(set(dv)) and dv[0] >= 0 and dv[-1] < C.VOCAB,
                 "draft_vocab.json is not 106,299 sorted unique ids")
        if src is not None:
            R.expect(dv == sorted(SF.read_draft_vocab(src / "mtp" / "draft_vocab.bin").tolist()),
                     "draft_vocab.json differs from mtp/draft_vocab.bin")
    return R.as_dict(), cfg


# ============================================================================== helpers for the deep checks
class Ckpt:
    def __init__(self, ckpt: Path):
        self.dir = ckpt
        idx = json.loads((ckpt / "model.safetensors.index.json").read_text())
        self.wm = idx["weight_map"]
        self._r = {}

    def r(self, name) -> ShardReader:
        f = self.wm[name]
        if f not in self._r:
            self._r[f] = ShardReader(self.dir / f)
        return self._r[f]

    def info(self, name):
        return self.r(name).info(name)

    def rows(self, name, r0, n=1):
        return self.r(name).rows(name, r0, n)

    def tensor(self, name):
        return self.r(name).tensor(name)

    def close(self):
        for r in self._r.values():
            r.close()


def _bf16_ref(x: np.ndarray) -> np.ndarray:
    """bf16 RNE bit patterns from an independent implementation when available (ml_dtypes), else st_io's."""
    try:
        import ml_dtypes
        return np.asarray(x, np.float32).astype(ml_dtypes.bfloat16).view(np.uint16)
    except ImportError:
        return bf16_bits(x)


def _vsrc(i):          # independent re-derivation of DESIGN 1.5: HF head i = 3k + r <- GGUF head 16r + k
    return 16 * (i % 3) + i // 3


def _gg_rows(m: SF.EdgeModel, name, rows):
    return np.stack([m.skeleton().raw(name, int(r), 1) for r in rows])


def _du_rows(m: SF.EdgeModel, layer, which, rows):
    du = m.latent()
    base = layer * du.layer_bytes + (0 if which == "D" else du.d_bytes)
    rb = du.d_row if which == "D" else du.u_row
    return np.stack([np.fromfile(du.path, np.uint8, rb, offset=base + int(r) * rb) for r in rows])


def _sample_rows(n, rng, k=6):
    return sorted(set([0, n - 1] + rng.integers(0, n, k).tolist()))


def expected_rows(m: SF.EdgeModel, name: str, rows, mtp_index=None):
    """For a 2-D-or-more tensor stored as raw source bytes (Q8 blocks / BF16), the expected bytes of `rows`
    (dim 0), shape (len(rows), row_bytes).  Returns None if the tensor is not a raw-row kind."""
    rows = list(rows)
    if name.startswith("mtp."):
        for ck, sname, dt, shape in C.mtp_dense_table():
            if ck == name and dt in ("U8", "BF16"):
                r = mtp_index[sname]
                rb = r["bytes"] // r["rows"]
                return np.stack([np.fromfile(m.dir / "mtp" / "dense.bin", np.uint8, rb, offset=r["off"] + q * rb)
                                 for q in rows])
        return None
    if name == "model.hyper_connection_mixer.input_mix_weight_down.weight":
        return _gg_rows(m, "output_hc_down.weight", rows)
    if name == "model.hyper_connection_mixer.input_mix_weight_up.weight":
        return _gg_rows(m, "output_hc_up.weight", rows)
    p = name.split(".")
    if p[0] != "model" or p[1] != "layers":
        return None
    l, rest = int(p[2]), ".".join(p[3:])
    b = "blk.%d." % l
    hc = {"attn_hyper_connection": "hc_attn_", "mlp_hyper_connection": "hc_ffn_"}
    for k, g in hc.items():
        for suf, gs in (("input_mix_weight_down.weight", "down.weight"), ("input_mix_weight_up.weight", "up.weight"),
                        ("block_inject_weight.weight", "inject.weight")):
            if rest == k + "." + suf:
                return _gg_rows(m, b + g + gs, rows)
    if rest == "linear_attn.in_proj_qkv.weight":
        src = [r if r < 4096 else 4096 + 128 * _vsrc((r - 4096) // 128) + (r - 4096) % 128 for r in rows]
        return _gg_rows(m, b + "attn_qkv.weight", src)
    if rest == "linear_attn.in_proj_z.weight":
        return _gg_rows(m, b + "attn_gate.weight", [128 * _vsrc(r // 128) + r % 128 for r in rows])
    if rest in ("linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight"):
        return _gg_rows(m, b + ("ssm_beta.weight" if "_b." in rest else "ssm_alpha.weight"), [_vsrc(r) for r in rows])
    if rest == "linear_attn.out_proj.weight":
        src = _gg_rows(m, b + "ssm_out.weight", rows).reshape(len(rows), 192, 34)
        out = np.empty_like(src)
        for i in range(48):
            out[:, 4 * i:4 * i + 4] = src[:, 4 * _vsrc(i):4 * _vsrc(i) + 4]
        return out.reshape(len(rows), -1)
    if rest == "self_attn.o_proj.weight":
        return _gg_rows(m, b + "attn_output.weight", rows)
    if rest == "self_attn.indexer.index_qk_proj.weight":
        return np.stack([m.skeleton().raw(b + "indexer.q_proj.weight", r, 1) if r < 512 else
                         m.skeleton().raw(b + "indexer.k_proj.weight", r - 512, 1) for r in rows])
    if rest == "mlp.gate.weight":
        return _gg_rows(m, b + "ffn_gate_inp.weight", rows)
    if rest == "mlp.shared_expert_gate.weight":
        return _gg_rows(m, b + "ffn_gate_inp_shexp.weight", rows)
    for proj in ("gate", "up", "down"):
        if rest == "mlp.shared_expert.%s_proj.weight" % proj:
            return _gg_rows(m, b + "ffn_%s_shexp.weight" % proj, rows)
    if rest == "mlp.latent_down.weight":
        return _du_rows(m, l, "D", rows)
    if rest == "mlp.latent_up.weight":
        return np.stack([np.concatenate(list(_du_rows(m, l, "U", [2560 * g + r for g in range(8)]))) for r in rows])
    if rest == "ple.key_proj.weight":
        return _gg_rows(m, b + "ple_key.weight", rows)
    if rest == "ple.value_proj.weight":
        return _gg_rows(m, b + "ple_value.weight", rows)
    return None


Q8_TO_BF16 = {"model.embed_tokens.weight": "token_embd.weight", "lm_head.weight": "output.weight"}


def expected_small(m: SF.EdgeModel, name: str, mtp_index=None):
    """Whole expected value (numpy, final dtype) of the small transformed tensors (F32/F16/I32), else None."""
    sk = m.skeleton()
    m1 = lambda g: sk.tensor(g) - np.float32(1.0)                     # noqa: E731
    if name.startswith("mtp."):
        for ck, sname, dt, shape in C.mtp_dense_table():
            if ck == name and dt == "F32":
                return SF.read_mtp_tensor(m.dir / "mtp", sname, mtp_index).reshape(shape) - np.float32(1.0)
        return None
    if name == "model.hyper_connection_mixer.hc_norm.weight":
        return m1("output_hc_norm.weight")
    p = name.split(".")
    if p[0] != "model" or p[1] != "layers":
        return None
    l, rest = int(p[2]), ".".join(p[3:])
    b = "blk.%d." % l
    perm = [_vsrc(i) for i in range(48)]
    table = {"attn_hyper_connection.hc_norm.weight": lambda: m1(b + "hc_attn_norm.weight"),
             "mlp_hyper_connection.hc_norm.weight": lambda: m1(b + "hc_ffn_norm.weight"),
             "linear_attn.A_log": lambda: np.log(-sk.tensor(b + "ssm_a").astype(np.float64)).astype(np.float32)[perm],
             "linear_attn.dt_bias": lambda: sk.tensor(b + "ssm_dt.bias")[perm],
             "linear_attn.norm.weight": lambda: sk.tensor(b + "ssm_norm.weight"),
             "self_attn.q_norm.weight": lambda: m1(b + "attn_q_norm.weight"),
             "self_attn.k_norm.weight": lambda: m1(b + "attn_k_norm.weight"),
             "self_attn.indexer.q_layernorm.weight": lambda: m1(b + "indexer.q_norm.weight"),
             "self_attn.indexer.k_layernorm.weight": lambda: m1(b + "indexer.k_norm.weight"),
             "mlp.expert_group": lambda: m.labels()[l].astype(np.int32),
             "ple.norm_key.weight": lambda: m1(b + "ple_norm_key.weight"),
             "ple.norm_query.weight": lambda: m1(b + "ple_norm_query.weight"),
             "ple.norm_conv.weight": lambda: m1(b + "ple_norm_conv.weight"),
             "ple.ple_embedding.ngram_embedding.weight_scale": lambda: np.array([C.PLE_SCALE], np.float32)}
    if rest in table:
        return table[rest]()
    if rest == "linear_attn.conv1d.weight":
        w = sk.tensor(b + "ssm_conv1d.weight")                       # (10240, 4): row = channel, 4 taps
        rows = [c if c < 4096 else 4096 + 128 * _vsrc((c - 4096) // 128) + (c - 4096) % 128 for c in range(10240)]
        return w[rows].reshape(10240, 1, 4)
    if rest == "ple.conv1d.weight":
        return sk.tensor(b + "ple_conv1d.weight").astype(np.float16).reshape(10240, 1, 4)
    return None


def decode_converted_expert(ck: Ckpt, layer: int, e: int) -> dict:
    """ModelOpt NVFP4 (byte m = value 2m low nibble, 2m+1 high nibble) -> float32, the same multiply order as
    strata_formats.decode_expert_blob: (E2M1 * e4m3(scale)) * scale_2."""
    p = "model.layers.%d.mlp.experts." % layer

    def deq(w, s):
        codes = np.stack([w & 0x0F, w >> 4], axis=-1).reshape(w.shape[0], -1)          # (N, K)
        v = SF.E2M1[codes].reshape(w.shape[0], s.shape[1], 16) * E4M3FN_STRICT[s][..., None]
        return v.reshape(w.shape[0], -1)

    w13 = deq(ck.rows(p + "w13_weight", e)[0], ck.rows(p + "w13_weight_scale", e)[0])
    w2 = deq(ck.rows(p + "w2_weight", e)[0], ck.rows(p + "w2_weight_scale", e)[0])
    g13 = ck.rows(p + "w13_weight_scale_2", e)[0]
    g2 = ck.rows(p + "w2_weight_scale_2", e)[0]
    return {"gate": w13[:640] * g13[0], "up": w13[640:] * g13[1], "down": w2 * g2, "scales": (g13[0], g13[1], g2)}


def deq_q8_tensor(ck: Ckpt, name) -> np.ndarray:
    t = ck.tensor(name)
    return SF.deq_q8_0(t.reshape(-1)).reshape(t.shape[0], t.shape[1] * 32)


# ============================================================================== 1.7 item 2: lossless
def check_lossless(ck: Ckpt, m: SF.EdgeModel, nl: int, mtp: bool, expert_layers, n_random: int, seed=0) -> dict:
    R = Report("lossless")
    rng = np.random.default_rng(seed)
    mtp_index = SF.read_mtp_index(m.dir / "mtp") if mtp else None
    counts = {"raw_rows": 0, "bf16_rows": 0, "small": 0, "experts": 0, "mtp_experts": 0}
    for name in sorted(ck.wm):
        if ".mlp.experts." in name:
            continue
        h = ck.info(name)
        if name in Q8_TO_BF16 or (name.startswith("model.layers.") and any(
                name.endswith("self_attn.%s.weight" % x) for x in ("q_proj", "k_proj", "v_proj"))):
            if name in Q8_TO_BF16:
                g = Q8_TO_BF16[name]
            else:
                l = int(name.split(".")[2])
                g = "blk.%d.attn_%s.weight" % (l, name.split(".")[-2][0])
            rows = _sample_rows(h["shape"][0], rng, 6)
            if name == "model.embed_tokens.weight":
                rows = sorted(set(rows + [248044, 248045, 248046]))
            got = np.stack([ck.rows(name, r)[0] for r in rows])
            want = _bf16_ref(np.stack([m.skeleton().tensor(g, r, 1)[0] for r in rows]))
            R.expect(np.array_equal(got, want), "%s: bf16 rows %s differ from bf16(deq_q8_0(%s))" % (name, rows, g))
            counts["bf16_rows"] += len(rows)
            continue
        small = expected_small(m, name, mtp_index)
        if small is not None:
            got = ck.tensor(name)
            R.expect(got.dtype == small.dtype and got.shape == small.shape and np.array_equal(got, small),
                     "%s: value differs from the transformed source (%s %s vs %s %s)" % (
                         name, got.dtype, got.shape, small.dtype, small.shape))
            counts["small"] += 1
            continue
        rows = _sample_rows(h["shape"][0], rng, 6)
        want = expected_rows(m, name, rows, mtp_index)
        if want is None:
            R.fail("%s: no source rule in the checker" % name)
            continue
        got = np.stack([ck.rows(name, r)[0].view(np.uint8).reshape(-1) for r in rows])
        R.expect(got.shape == want.shape and np.array_equal(got, want), "%s: rows %s differ from the source" % (name, rows))
        counts["raw_rows"] += len(rows)
    # experts: all of the listed layers + random pairs, float32 equality with EdgeModel.expert
    pairs = [(l, e) for l in expert_layers for e in range(C.N_EXPERT)]
    pairs += [(int(rng.integers(0, nl)), int(rng.integers(0, C.N_EXPERT))) for _ in range(n_random)]
    bad = 0
    t0 = time.time()
    for l, e in pairs:
        a = decode_converted_expert(ck, l, e)
        b = m.expert(l, e)
        same = all(np.array_equal(a[k], b[k]) for k in ("gate", "up", "down"))
        if not same:
            bad += 1
            if bad <= 5:
                R.fail("expert (%d, %d): decode differs (max abs %.3g)" % (l, e, max(
                    float(np.nanmax(np.abs(a[k] - b[k]))) for k in ("gate", "up", "down"))))
        counts["experts"] += 1
    R.info["expert_seconds"] = round(time.time() - t0, 1)
    R.info["expert_mismatches"] = bad
    R.info["expert_layers_full"] = list(expert_layers)
    if mtp:
        errs = []
        for e in sorted(set([0, 511] + rng.integers(0, 512, 6).tolist())):
            ref = SF.read_mtp_expert(m.dir / "mtp", e)
            p = "mtp.layers.0.mlp.experts.%d." % e
            for proj, key in (("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")):
                w = ck.tensor(p + proj + ".weight")
                s = ck.tensor(p + proj + ".weight_scale")
                g = ck.tensor(p + proj + ".weight_scale_2")
                R.expect(ck.tensor(p + proj + ".input_scale") == np.float32(1.0), "%s input_scale" % p)
                codes = np.stack([w & 0x0F, w >> 4], axis=-1).reshape(w.shape[0], -1)
                v = (SF.E2M1[codes].reshape(w.shape[0], s.shape[1], 16) * E4M3FN_STRICT[s][..., None] * g).reshape(
                    w.shape[0], -1)
                R.expect(np.isfinite(v).all(), "%s%s: non-finite" % (p, proj))
                errs.append(float(np.linalg.norm(v - ref[key]) / np.linalg.norm(ref[key])))
            R.expect(ck.tensor(p + "gate_proj.weight_scale_2") == ck.tensor(p + "up_proj.weight_scale_2"),
                     "%s: gate/up weight_scale_2 differ" % p)
            counts["mtp_experts"] += 1
        R.info["mtp_requant_rel_err_sample_max"] = max(errs)
        R.info["mtp_requant_rel_err_sample_mean"] = float(np.mean(errs))
        R.expect(max(errs) < 0.1, "MTP re-quantization error %.3g looks wrong" % max(errs))
    R.info["counts"] = counts
    return R.as_dict()


# ============================================================================== 1.7 item 3: GDN reorder
def _silu(x):
    return x / (1.0 + np.exp(-x))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def gdn_steps(W, xs, pairing):
    """MODEL-SPEC 2.3 GDN over a few tokens from zero state.  W: dict of float arrays; pairing(h) -> k head."""
    conv_state = np.zeros((3, 10240), np.float64)
    S = np.zeros((48, 128, 128), np.float64)
    outs = []
    for x in xs:
        xf = np.asarray(x, np.float32)
        qkv = (W["qkv"] @ xf).astype(np.float64)                             # big matrices stay float32
        z = (W["z"] @ xf).astype(np.float64)
        inp = np.concatenate([conv_state, qkv[None]], 0)                       # (4, 10240), row 0 oldest
        conv = (W["conv"] * inp.T).sum(1)                                     # W["conv"] (10240, 4), tap 0 oldest
        conv_state = inp[1:]
        h = _silu(conv)
        q = h[:2048].reshape(16, 128)
        k = h[2048:4096].reshape(16, 128)
        q = q / np.sqrt((q ** 2).sum(1, keepdims=True) + 1e-6) / np.sqrt(128)
        k = k / np.sqrt((k ** 2).sum(1, keepdims=True) + 1e-6)
        v = h[4096:].reshape(48, 128)
        beta = _sigmoid(W["b"] @ x)
        g = _softplus(W["a"] @ x + W["dt"]) * W["A"]
        o = np.empty((48, 128))
        for hh in range(48):
            kh = pairing(hh)
            S[hh] *= np.exp(g[hh])
            d = (v[hh] - S[hh].T @ k[kh]) * beta[hh]
            S[hh] += np.outer(k[kh], d)
            o[hh] = S[hh].T @ q[kh]
        y = o / np.sqrt((o ** 2).mean(1, keepdims=True) + 1e-6) * W["norm"] * _sigmoid(z.reshape(48, 128))
        outs.append((W["out"] @ y.reshape(-1).astype(np.float32)).astype(np.float64))
    return np.stack(outs)


def check_gdn(ck: Ckpt, m: SF.EdgeModel, layer: int, seed=1) -> dict:
    R = Report("gdn_reorder")
    sk = m.skeleton()
    b = "blk.%d." % layer
    a = "model.layers.%d.linear_attn." % layer
    f64 = lambda x: np.asarray(x, np.float64)                                   # noqa: E731
    Wg = {"qkv": sk.tensor(b + "attn_qkv.weight"), "z": sk.tensor(b + "attn_gate.weight"),
          "b": f64(sk.tensor(b + "ssm_beta.weight")), "a": f64(sk.tensor(b + "ssm_alpha.weight")),
          "conv": f64(sk.tensor(b + "ssm_conv1d.weight")), "A": f64(sk.tensor(b + "ssm_a")),
          "dt": f64(sk.tensor(b + "ssm_dt.bias")), "norm": f64(sk.tensor(b + "ssm_norm.weight")),
          "out": sk.tensor(b + "ssm_out.weight")}
    Wh = {"qkv": deq_q8_tensor(ck, a + "in_proj_qkv.weight"), "z": deq_q8_tensor(ck, a + "in_proj_z.weight"),
          "b": f64(bf16_to_f32(ck.tensor(a + "in_proj_b.weight"))), "a": f64(bf16_to_f32(ck.tensor(a + "in_proj_a.weight"))),
          "conv": f64(ck.tensor(a + "conv1d.weight")).reshape(10240, 4), "A": -np.exp(f64(ck.tensor(a + "A_log"))),
          "dt": f64(ck.tensor(a + "dt_bias")), "norm": f64(ck.tensor(a + "norm.weight")),
          "out": deq_q8_tensor(ck, a + "out_proj.weight")}
    rng = np.random.default_rng(seed)
    xs = rng.standard_normal((3, C.HIDDEN))
    yg = gdn_steps(Wg, xs, lambda h: h % 16)          # GGUF: tiled pairing
    yh = gdn_steps(Wh, xs, lambda h: h // 3)          # HF/vLLM: grouped pairing
    rel = float(np.linalg.norm(yh - yg) / np.linalg.norm(yg))
    # control: the converted tensors with the WRONG (tiled) pairing must disagree, or the check proves nothing
    ctrl = float(np.linalg.norm(gdn_steps(Wh, xs, lambda h: h % 16) - yg) / np.linalg.norm(yg))
    R.info.update(layer=layer, tokens=3, rel_err=rel, control_rel_err_wrong_pairing=ctrl)
    R.expect(rel < 1e-5, "GDN layer %d: converted (grouped) vs GGUF (tiled) rel err %.3g" % (layer, rel))
    R.expect(ctrl > 1e-2, "GDN control did not detect a wrong pairing (rel %.3g)" % ctrl)
    return R.as_dict()


# ============================================================================== 1.7 item 4: latent MoE (D5)
def latent_moe_converted(ck: Ckpt, layer: int, x: np.ndarray, cache: dict) -> np.ndarray:
    p = "model.layers.%d.mlp." % layer
    if cache.get("layer") != layer:
        cache.clear()
        cache.update(layer=layer, router=bf16_to_f32(ck.tensor(p + "gate.weight")),
                     D=deq_q8_tensor(ck, p + "latent_down.weight"), U=deq_q8_tensor(ck, p + "latent_up.weight"),
                     grp=ck.tensor(p + "expert_group"),
                     sg=bf16_to_f32(ck.tensor(p + "shared_expert_gate.weight")).reshape(-1),
                     shg=deq_q8_tensor(ck, p + "shared_expert.gate_proj.weight"),
                     shu=deq_q8_tensor(ck, p + "shared_expert.up_proj.weight"),
                     shd=deq_q8_tensor(ck, p + "shared_expert.down_proj.weight"))
    c = cache
    ids, w = SF.route(c["router"] @ x)
    xl = (c["D"] @ x).reshape(C.LATENT_GROUPS, C.LATENT)                 # xl = D_cat x, all 8 groups
    z = np.zeros((C.LATENT_GROUPS, C.LATENT), np.float32)
    grp = c["grp"][ids]
    for j in range(len(ids)):                                              # route-as-token, combine per group
        ex = decode_converted_expert(ck, layer, int(ids[j]))
        xr = xl[grp[j]]
        hdn = SF.silu(ex["gate"] @ xr) * (ex["up"] @ xr)
        z[grp[j]] += w[j] * (ex["down"] @ hdn)
    routed = c["U"] @ z.reshape(-1)                                         # one U_cat GEMM
    shared = SF.sigmoid(float(c["sg"] @ x)) * (c["shd"] @ (SF.silu(c["shg"] @ x) * (c["shu"] @ x)))
    return routed + shared


def check_latent(ck: Ckpt, m: SF.EdgeModel, layers, n_x=4, seed=2) -> dict:
    R = Report("latent_moe")
    rng = np.random.default_rng(seed)
    errs = []
    for l in layers:
        cache = {}
        for _ in range(n_x):
            x = rng.standard_normal(C.HIDDEN).astype(np.float32)
            ref = SF.latent_moe_reference(m, l, x)["out"]
            got = latent_moe_converted(ck, l, x, cache)
            rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
            errs.append(rel)
            R.expect(rel < 1e-5, "layer %d: D5 on converted tensors vs latent_moe_reference rel err %.3g" % (l, rel))
        cache.clear()
    R.info.update(layers=list(layers), samples=len(errs), rel_err_max=max(errs), rel_err_mean=float(np.mean(errs)))
    return R.as_dict()


# ============================================================================== 1.7 item 6: safe_open
def check_safe_open(ckpt: Path) -> dict:
    R = Report("safe_open")
    import torch
    from safetensors import safe_open
    tdt = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I32": torch.int32, "U8": torch.uint8,
           "F8_E4M3": torch.float8_e4m3fn}
    n = 0
    for p in sorted(ckpt.glob("model-*.safetensors")):
        hdr, _ = read_header(p)
        with safe_open(str(p), framework="pt") as f:
            keys = list(f.keys())
            R.expect(sorted(keys) == sorted(hdr), "%s: keys differ from the raw header" % p.name)
            for k in keys:
                h = hdr[k]
                sl = f.get_slice(k)
                R.expect(list(sl.get_shape()) == list(h["shape"]), "%s shape" % k)
                if np.prod(h["shape"], dtype=np.int64) * DTYPE_BYTES[h["dtype"]] <= 1 << 16 or not h["shape"]:
                    t = f.get_tensor(k)
                else:
                    t = sl[0:1]
                R.expect(t.dtype == tdt[h["dtype"]], "%s: torch dtype %s for %s" % (k, t.dtype, h["dtype"]))
                n += 1
    R.info["tensors_opened"] = n
    R.info["torch"] = torch.__version__
    return R.as_dict()


# ============================================================================== driver
def run_checks(ckpt, src=None, level="contract", hf_ref=None, corpus=None, strata_ref=None, expert_layers=None,
               n_random=64) -> dict:
    ckpt = Path(ckpt)
    src = Path(src) if src else None
    t0 = time.time()
    out = {"level": level, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    def guarded(key, fn):
        t = time.time()
        try:
            out[key] = fn()
        except Exception as e:                                   # a crash is a failed check, with the traceback
            out[key] = {"ok": False, "problems": ["%s: %s" % (type(e).__name__, e)],
                        "traceback": traceback.format_exc()[-2000:]}
        out[key]["seconds"] = round(time.time() - t, 1)
        print("[check] %-12s %s (%.0f s)%s" % (key, "PASS" if out[key]["ok"] else "FAIL", out[key]["seconds"],
                                               "" if out[key]["ok"] else "  " + "; ".join(out[key]["problems"][:3])),
              flush=True)

    holder = {}

    def contract():
        res, holder["cfg"] = check_contract(ckpt, src)
        return res

    guarded("contract", contract)
    if level in ("deep", "full"):
        if src is None:
            raise ValueError("--level %s needs --src" % level)
        cfg = holder.get("cfg") or json.loads((ckpt / "config.json").read_text())
        nl, mtp = C.config_layout(cfg)
        m = SF.EdgeModel(src)
        ck = Ckpt(ckpt)
        if expert_layers is None:
            expert_layers = sorted({0, 23, 47} & set(range(nl))) or [0]
        guarded("lossless", lambda: check_lossless(ck, m, nl, mtp, expert_layers, n_random))
        gdn_layer = 0
        guarded("gdn_reorder", lambda: check_gdn(ck, m, gdn_layer))
        lat_layers = sorted({0, nl // 2, nl - 1})
        guarded("latent_moe", lambda: check_latent(ck, m, lat_layers))
        ck.close()
    if level == "full":
        def tok():
            import hf_tokenizer
            r = hf_tokenizer.check(ckpt, (src or Path(C.SOURCE_DIR_DEFAULT)) / "tokenizer", hf_ref, corpus,
                                   strata_ref=strata_ref)
            r.pop("think_off_prompt_ids", None)
            return r
        guarded("tokenizer", tok)
        guarded("safe_open", lambda: check_safe_open(ckpt))
    out["ok"] = all(v.get("ok", True) for v in out.values() if isinstance(v, dict))
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--src", help="Strata model folder (needed for --level deep/full)")
    ap.add_argument("--level", default="contract", choices=("contract", "deep", "full"))
    ap.add_argument("--hf-ref", default="/home/andychen/work/edge0-bench/tokenizer-qwen38")
    ap.add_argument("--corpus")
    ap.add_argument("--strata-ref")
    ap.add_argument("--expert-layers", help="comma list of layers whose 512 experts are all decoded (default 0,23,47)")
    ap.add_argument("--expert-random", type=int, default=64)
    ap.add_argument("--write-results", action="store_true", help="store the results in edge0_convert.json['checks']")
    ap.add_argument("--json", help="also write the results to this file")
    a = ap.parse_args()
    el = [int(x) for x in a.expert_layers.split(",")] if a.expert_layers else None
    res = run_checks(a.ckpt, a.src, a.level, a.hf_ref, a.corpus, a.strata_ref, el, a.expert_random)
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "traceback"} if isinstance(v, dict) else v)
                      for k, v in res.items()}, indent=1, ensure_ascii=False, default=str)[:20000])
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    if a.write_results:
        p = Path(a.ckpt) / "edge0_convert.json"
        rec = json.loads(p.read_text()) if p.is_file() else {}
        rec["checks"] = res
        tmp = Path(str(p) + ".tmp")
        tmp.write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str))
        os.replace(tmp, p)
    print("ALL CHECKS PASSED" if res["ok"] else "SOME CHECKS FAILED")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
