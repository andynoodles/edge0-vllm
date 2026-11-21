#!/usr/bin/env python3
"""edge0/convert/convert_checkpoint.py - Strata model folder (Edge0-145B-NVFP4) -> vLLM checkpoint (DESIGN.md 1).

    python3 convert_checkpoint.py --src /models/Edge0-145B-NVFP4 --out /models/Edge0-145B-vllm-L4 --num-layers 4 --no-mtp
    python3 convert_checkpoint.py --src /models/Edge0-145B-NVFP4 --out /models/Edge0-145B-vllm --jobs 6

Streaming and low memory: every source is mmapped read-only and consumed in slices (one GGUF tensor, one D/U layer,
or `--chunk-experts` expert records at a time); each slice is copied out and its pages dropped (madvise +
posix_fadvise DONTNEED); every shard is written through st_io.ShardWriter (pwrite into a `.partial`, periodic
fdatasync + DONTNEED, fsync + rename at the end).  Peak RSS is about 0.4 GB per job.

Resumable: a shard whose final file exists with exactly the expected header and size is skipped (`--force` rewrites
everything).  The small files (config, tokenizer, hardlink, profile, draft vocab, index) are rewritten on every run.
At the end the acceptance checks of DESIGN 1.7 run (check_ckpt.py) and their results go into edge0_convert.json; a
failed check makes the exit status 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ckpt_contract as C  # noqa: E402
import strata_formats as SF  # noqa: E402
from st_io import (E2M1_VALUES, E4M3FN_STRICT, ShardWriter, SrcFile, Throttle, bf16_bits, e2m1_encode,  # noqa: E402
                   e4m3_encode, fadvise_dontneed, read_header)

REC = 1935376                     # one latent expert record (MODEL-SPEC 5.1)
GU_BYTES = 2 * 640 * 1008         # gate + up = 1,290,240
TAIL_OFF = GU_BYTES + 1792 * 360  # 1,935,360
LAYER_STRIDE = 512 * REC          # 990,912,512


# ============================================================================== NVFP4 repack (DESIGN 1.5)
def repack_ggml_nvfp4(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ggml block_nvfp4 rows -> ModelOpt NVFP4 (lossless).  blocks: uint8 [..., nb, 36] (d[4] then qs[32]).
    Returns (packed uint8 [..., nb*32], scales uint8 [..., nb*4]).
    ggml: sub-block s, byte j<8 holds value 16s+j (low nibble) and 16s+j+8 (high nibble).
    ModelOpt: byte m of the row holds value 2m (low nibble) and 2m+1 (high nibble).
    Inside a 16-value sub-block with source bytes q[0..7]: out[m] = lo(q[2m]) | lo(q[2m+1])<<4 for m<4 and
    out[m] = hi(q[2m-8]) | hi(q[2m-7])<<4 for m>=4."""
    lead, nb = blocks.shape[:-2], blocks.shape[-2]
    qs = blocks[..., 4:].reshape(*lead, nb, 4, 8)
    a, b = qs[..., 0::2], qs[..., 1::2]
    out = np.empty(qs.shape, np.uint8)
    out[..., :4] = (a & 0x0F) | ((b & 0x0F) << 4)
    out[..., 4:] = (a >> 4) | (b & 0xF0)
    return out.reshape(*lead, nb * 32), np.ascontiguousarray(blocks[..., :4]).reshape(*lead, nb * 4)


def fix_scales(s: np.ndarray) -> tuple[int, int]:
    """In place: E4M3 scale bytes with bit 7 set or equal to 0x7F -> 0x00 (DESIGN 1.5).  Returns the two counts."""
    b7 = (s & 0x80) != 0
    nan = s == 0x7F
    n7, n_nan = int(b7.sum()), int(nan.sum())
    if n7 or n_nan:
        s[b7 | nan] = 0
    return n7, n_nan


def scale_stats(s: np.ndarray) -> dict:
    return {"n": int(s.size), "lt_2m6": int(((s & 0x78) == 0).sum()), "zero": int((s == 0).sum())}


# ============================================================================== source access
GG = {"F32": 0, "F16": 1, "Q8_0": 8, "BF16": 30}


class Source:
    """The Strata model folder; every big read goes through SrcFile (copy + drop)."""

    def __init__(self, src_dir, throttle: Throttle):
        self.dir = Path(src_dir)
        self.throttle = throttle
        self.skel = SF.Gguf(self.dir / "model-skeleton.gguf")
        self._files = {}
        self.labels = SF.read_labels(self.dir / "labels.bin")
        self.layers = SF.read_native_experts(self.dir / "native_experts.txt")

    def f(self, rel) -> SrcFile:
        if rel not in self._files:
            self._files[rel] = SrcFile(self.dir / rel, self.throttle)
        return self._files[rel]

    def close(self):
        for f in self._files.values():
            f.close()
        self._files = {}

    def gg(self, name: str, type_name: str, shape: tuple, row0: int = 0, nrows: int | None = None) -> np.ndarray:
        """Raw bytes of rows [row0, row0+nrows) of a skeleton tensor, after asserting its GGUF type and shape
        (shape in GGUF order, ne0 first)."""
        t = self.skel.tensors[name]
        if t.type_id != GG[type_name] or tuple(t.shape) != tuple(shape):
            raise ValueError("%s: GGUF %s %s, contract expects %s %s" % (name, t.type_name, t.shape, type_name, shape))
        rb = SF.row_bytes(t.type_id, t.shape[0])
        total = t.nbytes // rb
        nrows = total - row0 if nrows is None else nrows
        assert 0 <= row0 and row0 + nrows <= total
        return self.f("model-skeleton.gguf").read(t.offset + row0 * rb, nrows * rb)

    def gg_rows(self, name: str) -> int:
        t = self.skel.tensors[name]
        return t.nbytes // SF.row_bytes(t.type_id, t.shape[0])


# ============================================================================== dense shard producers
def _m1(raw: np.ndarray) -> np.ndarray:
    return raw.view(np.float32) - np.float32(1.0)


def _q8_bf16(src: Source, name: str, shape: tuple, chunk_rows: int = 4096):
    n = shape[1]
    for r0 in range(0, n, chunk_rows):
        k = min(chunk_rows, n - r0)
        yield bf16_bits(SF.deq_q8_0(src.gg(name, "Q8_0", shape, r0, k)))


def _vperm_rows(raw: np.ndarray, row_bytes: int, head_rows: int) -> np.ndarray:
    """Reorder 48 v-heads of `head_rows` rows each: HF head i <- GGUF head GDN_V_PERM[i]."""
    v = raw.reshape(C.GDN_V_HEADS, head_rows, row_bytes)
    return v[C.GDN_V_PERM].reshape(-1)


def produce_dense(src: Source, name: str):
    """Yield the bytes of checkpoint tensor `name` (shard 1) in order."""
    H, HC, LR = C.HIDDEN, C.HC_DIM, C.HC_LR
    if name == "model.embed_tokens.weight":
        yield from _q8_bf16(src, "token_embd.weight", (H, C.VOCAB))
        return
    if name == "lm_head.weight":
        yield from _q8_bf16(src, "output.weight", (H, C.VOCAB))
        return
    m = re.fullmatch(r"model\.hyper_connection_mixer\.(\w+)\.weight", name)
    if m:
        k = m.group(1)
        if k == "hc_norm":
            yield _m1(src.gg("output_hc_norm.weight", "F32", (HC,)))
        elif k == "input_mix_weight_down":
            yield src.gg("output_hc_down.weight", "BF16", (HC, LR))
        elif k == "input_mix_weight_up":
            yield src.gg("output_hc_up.weight", "BF16", (LR, HC))
        else:
            raise KeyError(name)
        return
    m = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", name)
    if not m:
        raise KeyError(name)
    l, rest = int(m.group(1)), m.group(2)
    b = "blk.%d." % l
    m = re.fullmatch(r"(attn|mlp)_hyper_connection\.(\w+)\.weight", rest)
    if m:
        g = b + "hc_%s_" % ("attn" if m.group(1) == "attn" else "ffn")
        k = m.group(2)
        if k == "hc_norm":
            yield _m1(src.gg(g + "norm.weight", "F32", (HC,)))
        elif k == "input_mix_weight_down":
            yield src.gg(g + "down.weight", "BF16", (HC, LR))
        elif k == "input_mix_weight_up":
            yield src.gg(g + "up.weight", "BF16", (LR, HC))
        elif k == "block_inject_weight":
            yield src.gg(g + "inject.weight", "BF16", (HC, 4))
        else:
            raise KeyError(name)
        return
    # ---- GDN
    if rest == "linear_attn.in_proj_qkv.weight":
        rb = 80 * 34
        raw = src.gg(b + "attn_qkv.weight", "Q8_0", (H, 10240))
        yield raw[:C.GDN_QK * rb]
        yield _vperm_rows(raw[C.GDN_QK * rb:], rb, C.GDN_HEAD)
        return
    if rest == "linear_attn.in_proj_z.weight":
        yield _vperm_rows(src.gg(b + "attn_gate.weight", "Q8_0", (H, 6144)), 80 * 34, C.GDN_HEAD)
        return
    if rest in ("linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight"):
        g = "ssm_beta.weight" if rest.endswith("_b.weight") else "ssm_alpha.weight"
        yield _vperm_rows(src.gg(b + g, "BF16", (H, C.GDN_V_HEADS)), H * 2, 1)
        return
    if rest == "linear_attn.conv1d.weight":
        w = src.gg(b + "ssm_conv1d.weight", "F32", (4, 10240)).view(np.float32).reshape(10240, 4)
        yield w[:C.GDN_QK]
        yield w[C.GDN_QK:].reshape(C.GDN_V_HEADS, C.GDN_HEAD, 4)[C.GDN_V_PERM]
        return
    if rest == "linear_attn.A_log":
        a = src.gg(b + "ssm_a", "F32", (C.GDN_V_HEADS,)).view(np.float32)
        if not (a < 0).all():
            raise ValueError("%sssm_a has non-negative entries" % b)
        yield np.log(-a.astype(np.float64)).astype(np.float32)[C.GDN_V_PERM]
        return
    if rest == "linear_attn.dt_bias":
        yield src.gg(b + "ssm_dt.bias", "F32", (C.GDN_V_HEADS,)).view(np.float32)[C.GDN_V_PERM]
        return
    if rest == "linear_attn.norm.weight":
        yield src.gg(b + "ssm_norm.weight", "F32", (C.GDN_HEAD,))
        return
    if rest == "linear_attn.out_proj.weight":
        raw = src.gg(b + "ssm_out.weight", "Q8_0", (6144, H)).reshape(H, C.GDN_V_HEADS, 4 * 34)
        yield raw[:, C.GDN_V_PERM]
        return
    # ---- QSA
    if rest == "self_attn.q_proj.weight":
        yield from _q8_bf16(src, b + "attn_q.weight", (H, 12288))
        return
    if rest in ("self_attn.k_proj.weight", "self_attn.v_proj.weight"):
        yield from _q8_bf16(src, b + ("attn_k.weight" if "k_proj" in rest else "attn_v.weight"), (H, 512))
        return
    if rest == "self_attn.o_proj.weight":
        yield src.gg(b + "attn_output.weight", "Q8_0", (6144, H))
        return
    if rest in ("self_attn.q_norm.weight", "self_attn.k_norm.weight"):
        yield _m1(src.gg(b + ("attn_q_norm.weight" if "q_norm" in rest else "attn_k_norm.weight"), "F32", (256,)))
        return
    if rest == "self_attn.indexer.index_qk_proj.weight":
        yield src.gg(b + "indexer.q_proj.weight", "BF16", (H, 512))
        yield src.gg(b + "indexer.k_proj.weight", "BF16", (H, 128))
        return
    if rest in ("self_attn.indexer.q_layernorm.weight", "self_attn.indexer.k_layernorm.weight"):
        yield _m1(src.gg(b + ("indexer.q_norm.weight" if ".q_" in rest else "indexer.k_norm.weight"), "F32", (128,)))
        return
    # ---- MoE (non-expert part)
    if rest == "mlp.gate.weight":
        yield src.gg(b + "ffn_gate_inp.weight", "BF16", (H, C.N_EXPERT))
        return
    if rest == "mlp.shared_expert_gate.weight":
        yield src.gg(b + "ffn_gate_inp_shexp.weight", "BF16", (H, 1))
        return
    m = re.fullmatch(r"mlp\.shared_expert\.(gate|up|down)_proj\.weight", rest)
    if m:
        p = m.group(1)
        yield src.gg(b + "ffn_%s_shexp.weight" % p, "Q8_0", (C.EXPERT_FF, H) if p == "down" else (H, C.EXPERT_FF))
        return
    if rest in ("mlp.latent_down.weight", "mlp.latent_up.weight"):
        G, ELL = C.LATENT_GROUPS, C.LATENT
        d_row, u_row = H // 32 * 34, ELL // 32 * 34              # 2720, 1904
        d_bytes, u_bytes = G * ELL * d_row, G * H * u_row
        layer_bytes = d_bytes + u_bytes
        du = src.f("latent_dense_q8.bin")
        if du.size != C.N_LAYERS_FULL * layer_bytes:
            raise ValueError("latent_dense_q8.bin: %d B, expected %d" % (du.size, C.N_LAYERS_FULL * layer_bytes))
        if rest == "mlp.latent_down.weight":
            yield du.read(l * layer_bytes, d_bytes)                # D_cat rows 0..14335, group g = rows 1792g..
        else:
            u = du.read(l * layer_bytes + d_bytes, u_bytes).reshape(G, H, u_row)
            yield u.transpose(1, 0, 2)                             # U_cat row r = [U_0 row r | ... | U_7 row r]
        return
    if rest == "mlp.expert_group":
        lab = src.labels[l]
        if lab.min() < 0 or lab.max() >= C.LATENT_GROUPS:
            raise ValueError("labels.bin layer %d out of range" % l)
        yield lab.astype("<i4")
        return
    # ---- PLE (layer 1)
    if rest == "ple.key_proj.weight":
        yield src.gg(b + "ple_key.weight", "BF16", (H, HC))
        return
    if rest == "ple.value_proj.weight":
        yield src.gg(b + "ple_value.weight", "BF16", (H, H))
        return
    m = re.fullmatch(r"ple\.norm_(key|query|conv)\.weight", rest)
    if m:
        yield _m1(src.gg(b + "ple_norm_%s.weight" % m.group(1), "F32", (HC,)))
        return
    if rest == "ple.conv1d.weight":
        yield src.gg(b + "ple_conv1d.weight", "F16", (4, HC))
        return
    if rest == "ple.ple_embedding.ngram_embedding.weight_scale":
        pt = SF.PleTable(src.dir / "ple-fp8.gguf")
        if pt.scale != C.PLE_SCALE:
            raise ValueError("PLE scale %r != contract %r" % (pt.scale, C.PLE_SCALE))
        yield np.array([pt.scale], np.float32)
        return
    raise KeyError(name)


def write_dense_shard(src: Source, path: Path, specs, throttle) -> dict:
    t0 = time.time()
    w = ShardWriter(path, specs, throttle)
    try:
        for s in specs:
            for part in produce_dense(src, s.name):
                w.write(s.name, part)
            if w.cursor[s.name] != s.nbytes:
                raise ValueError("%s: produced %d B, contract %d B" % (s.name, w.cursor[s.name], s.nbytes))
        w.finish()
    except BaseException:
        w.abort()
        raise
    return {"seconds": round(time.time() - t0, 1), "bytes": w.size}


# ============================================================================== expert shards
def write_expert_shard(src_dir: str, path: str, layer: int, chunk: int, throttle_mbps: float) -> dict:
    """One layer's routed experts -> the six stacked ModelOpt NVFP4 tensors.  Runs in a worker process."""
    t0 = time.time()
    throttle = Throttle(throttle_mbps)
    L = SF.read_native_experts(Path(src_dir) / "native_experts.txt")[layer]
    if (L.gu_type, L.d_type, L.blob_bytes, L.offset) != (40, 40, REC, layer * LAYER_STRIDE):
        raise ValueError("native_experts.txt layer %d: %s" % (layer, L))
    f = SrcFile(Path(src_dir) / "experts.bin", throttle)
    specs = C.shard_layout(C.expert_specs(layer))
    p = "model.layers.%d.mlp.experts." % layer
    w = ShardWriter(path, specs, throttle)
    st = {"scale_bit7": 0, "scale_7f": 0, "tail3_nonzero": 0, "w13": {"n": 0, "lt_2m6": 0, "zero": 0},
          "w2": {"n": 0, "lt_2m6": 0, "zero": 0}, "s_gate_ne_s_up": 0}
    try:
        for e0 in range(0, C.N_EXPERT, chunk):
            n = min(chunk, C.N_EXPERT - e0)
            blob = f.read(L.offset + e0 * REC, n * REC).reshape(n, REC)
            tail = np.ascontiguousarray(blob[:, TAIL_OFF:]).view("<f4").reshape(n, 4)
            ne = tail[:, 0] != tail[:, 1]
            if ne.any():
                st["s_gate_ne_s_up"] += int(ne.sum())
                raise ValueError("layer %d experts %s: s_gate != s_up" % (layer, (np.nonzero(ne)[0] + e0).tolist()[:8]))
            st["tail3_nonzero"] += int((tail[:, 3] != 0).sum())
            w13, s13 = repack_ggml_nvfp4(blob[:, :GU_BYTES].reshape(n, 2 * C.EXPERT_FF, C.LATENT // 64, 36))
            w2, s2 = repack_ggml_nvfp4(blob[:, GU_BYTES:TAIL_OFF].reshape(n, C.LATENT, C.EXPERT_FF // 64, 36))
            del blob
            for s, key in ((s13, "w13"), (s2, "w2")):
                a, b = fix_scales(s)
                st["scale_bit7"] += a
                st["scale_7f"] += b
                for k, v in scale_stats(s).items():
                    st[key][k] += v
            w.write(p + "w13_weight", w13)
            w.write(p + "w13_weight_scale", s13)
            w.write(p + "w13_weight_scale_2", tail[:, 0:2])
            w.write(p + "w2_weight", w2)
            w.write(p + "w2_weight_scale", s2)
            w.write(p + "w2_weight_scale_2", tail[:, 2])
        w.finish()
    except BaseException:
        w.abort()
        raise
    finally:
        f.close()
    for key in ("w13", "w2"):
        st[key]["frac_lt_2m6"] = st[key]["lt_2m6"] / st[key]["n"]
        st[key]["frac_zero"] = st[key]["zero"] / st[key]["n"]
    return {"seconds": round(time.time() - t0, 1), "bytes": w.size, "stats": st}


def _expert_job(args):
    return args[2], write_expert_shard(*args)


# ============================================================================== MTP shard
def quantize_nvfp4(w: np.ndarray, gscale: np.float32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standard ModelOpt NVFP4 of a (N, K) float32 matrix with a given global scale: per-16 block scale
    e4m3(amax16 / (6 * g)), codes = nearest E2M1 of value / (scale * g).  Returns (packed [N, K/2],
    scales [N, K/16], dequantized float32 [N, K])."""
    N, K = w.shape
    blk = w.reshape(N, K // 16, 16)
    amax = np.abs(blk).max(axis=-1)
    s8 = e4m3_encode(amax / (np.float32(6.0) * gscale))
    sdec = E4M3FN_STRICT[s8] * gscale                                       # float32
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(sdec[..., None] > 0, blk / sdec[..., None], np.float32(0))
    codes = e2m1_encode(q)
    deq = (E2M1_VALUES[codes] * sdec[..., None]).reshape(N, K)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << np.uint8(4))).reshape(N, K // 2)
    return packed, s8, deq


def _rel(a, b) -> float:
    nb = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b)) / nb if nb > 0 else 0.0


def write_mtp_shard(src: Source, path: Path, throttle) -> dict:
    t0 = time.time()
    mdir = src.dir / "mtp"
    idx = SF.read_mtp_index(mdir)
    table = C.mtp_dense_table()
    want = {sname for _, sname, _, _ in table}
    if set(idx) != want:
        raise ValueError("mtp/dense.txt names differ from the contract: extra %s missing %s"
                         % (sorted(set(idx) - want), sorted(want - set(idx))))
    specs = C.shard_layout(C.mtp_specs())
    w = ShardWriter(path, specs, throttle)
    dense = src.f("mtp/dense.bin")
    ex = src.f("mtp/experts.bin")
    if ex.size != C.N_EXPERT * SF.MTP_BLOB:
        raise ValueError("mtp/experts.bin size %d" % ex.size)
    errs = {"gate_proj": [], "up_proj": [], "down_proj": []}
    scalars = {}
    try:
        for ck, sname, dt, shape in table:
            r = idx[sname]
            raw = dense.read(r["off"], r["bytes"])
            if dt == "U8":
                ok = r["kind"] == "q8_0" and (r["rows"], r["cols"] // 32, 34) == shape
                data = raw
            elif dt == "BF16":
                ok = r["kind"] == "bf16" and (r["rows"], r["cols"]) == shape
                data = raw
            else:
                ok = r["kind"] == "f32" and r["rows"] == 1 and (r["cols"],) == shape
                data = _m1(raw)
            if not ok or raw.size != int(np.prod(shape)) * (4 if dt == "F32" else 2 if dt == "BF16" else 1):
                raise ValueError("mtp %s: dense.txt %s vs contract %s %s" % (sname, r, dt, shape))
            w.write(ck, data)
        for e in range(C.N_EXPERT):
            d = SF.read_mtp_expert(mdir, e)
            fadvise_dontneed(ex.fd, e * SF.MTP_BLOB, SF.MTP_BLOB)
            throttle.account(SF.MTP_BLOB)
            g_gu = np.float32(max(np.abs(d["gate"]).max(), np.abs(d["up"]).max())) / np.float32(6 * 448)
            g_d = np.float32(np.abs(d["down"]).max()) / np.float32(6 * 448)
            p = "mtp.layers.0.mlp.experts.%d." % e
            for proj, key, g in (("gate_proj", "gate", g_gu), ("up_proj", "up", g_gu), ("down_proj", "down", g_d)):
                packed, s8, deq = quantize_nvfp4(d[key], g)
                errs[proj].append(_rel(deq, d[key]))
                w.write(p + proj + ".weight", packed)
                w.write(p + proj + ".weight_scale", s8)
                scalars[p + proj + ".weight_scale_2"] = g
                scalars[p + proj + ".input_scale"] = np.float32(1.0)
        for k, v in scalars.items():
            w.write(k, np.array(v, np.float32))
        w.finish()
    except BaseException:
        w.abort()
        raise
    st = {k: {"rel_err_max": float(np.max(v)), "rel_err_mean": float(np.mean(v)),
              "worst_expert": int(np.argmax(v))} for k, v in errs.items()}
    return {"seconds": round(time.time() - t0, 1), "bytes": w.size, "stats": {"requant_rel_frobenius_error": st}}


# ============================================================================== small files
def write_json(path: Path, obj, indent=2) -> None:
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def shard_done(path: Path, specs) -> bool:
    if not path.is_file():
        return False
    try:
        hdr, base = read_header(path)
    except Exception:
        return False
    off = 0
    for s in specs:
        h = hdr.get(s.name)
        if h is None or h["dtype"] != s.dtype or list(h["shape"]) != list(s.shape) or h["data_offsets"] != [off, off + s.nbytes]:
            return False
        off += s.nbytes
    return len(hdr) == len(specs) and path.stat().st_size == base + off


def link_ple(src_dir: Path, out: Path) -> dict:
    srcp, dst = src_dir / "ple-fp8.gguf", out / "ple-fp8.gguf"
    pt = SF.PleTable(srcp)
    size = srcp.stat().st_size
    info = {"offset": pt.t.offset, "rows": pt.n_rows, "row_bytes": pt.t.shape[0], "scale": pt.scale, "size": size}
    if (pt.t.offset, pt.n_rows, pt.t.shape[0], pt.scale) != (C.PLE_DATA_OFFSET, C.PLE_ROWS, C.PLE_ROW_BYTES, C.PLE_SCALE):
        raise ValueError("ple-fp8.gguf layout %s differs from the contract" % info)
    if size != C.PLE_DATA_OFFSET + C.PLE_ROWS * C.PLE_ROW_BYTES:
        raise ValueError("ple-fp8.gguf size %d" % size)
    if dst.exists():
        if not os.path.samefile(srcp, dst):
            raise ValueError("%s exists and is not a hardlink of %s (never copy the PLE table)" % (dst, srcp))
    else:
        os.link(srcp, dst)            # raises EXDEV across filesystems: the 51 GB table is never copied
    info["inode"] = os.stat(dst).st_ino
    info["nlink"] = os.stat(dst).st_nlink
    return info


def write_small_files(a, src_dir: Path, out: Path, mtp: bool) -> dict:
    import hf_tokenizer
    rep = {}
    write_json(out / "config.json", C.build_config(a.num_layers, mtp, a.source_dir_record))
    write_json(out / "generation_config.json", C.GENERATION_CONFIG)
    rep["tokenizer"] = hf_tokenizer.build(src_dir / "tokenizer", a.hf_tokenizer, out)
    rep["ple"] = link_ple(src_dir, out)
    prof = SF.read_profile(src_dir / "profile.bin")
    if (prof["n_layers"], prof["n_expert"], prof["n_ranked"]) != (48, 512, 24576):
        raise ValueError("profile.bin header %s" % {k: prof[k] for k in ("n_layers", "n_expert", "n_ranked")})
    shutil.copyfile(src_dir / "profile.bin", out / "profile.bin")
    dv = SF.read_draft_vocab(src_dir / "mtp" / "draft_vocab.bin")
    uniq = sorted(set(int(x) for x in dv))
    if len(uniq) != C.DRAFT_VOCAB_SIZE or len(dv) != len(uniq) or uniq[0] < 0 or uniq[-1] >= C.VOCAB:
        raise ValueError("draft_vocab.bin: %d ids, %d unique, range %d..%d" % (len(dv), len(uniq), uniq[0], uniq[-1]))
    with open(out / "draft_vocab.json", "w") as f:
        json.dump(uniq, f, separators=(",", ":"))
    rep["draft_vocab"] = {"ids": len(uniq), "min": uniq[0], "max": uniq[-1]}
    for name in C.COPIED_FILES:
        if (src_dir / name).is_file():
            shutil.copyfile(src_dir / name, out / name)
    return rep


def write_index(out: Path, plan) -> dict:
    wm, total = {}, 0
    for fname, specs in plan:
        for s in specs:
            wm[s.name] = fname
            total += s.nbytes
    idx = {"metadata": {"total_size": total}, "weight_map": dict(sorted(wm.items()))}
    write_json(out / "model.safetensors.index.json", idx)
    return {"tensors": len(wm), "total_size": total}


def _sha256_files(paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()


# ============================================================================== main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, help="Strata model folder (Edge0-145B-NVFP4)")
    ap.add_argument("--out", required=True, help="output checkpoint directory (same filesystem as --src)")
    ap.add_argument("--num-layers", type=int, default=48)
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--hf-tokenizer", default="/home/andychen/work/edge0-bench/tokenizer-qwen38",
                    help="Qwen3.8 HF tokenizer dir used as the structural template and for the identity check")
    ap.add_argument("--source-dir-record", default=C.SOURCE_DIR_DEFAULT,
                    help="value written to config.json edge0.source_dir (the host path, also when run in docker)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel expert-shard workers")
    ap.add_argument("--chunk-experts", type=int, default=32)
    ap.add_argument("--throttle-mbps", type=float, default=0.0, help="cap read+write MB/s per process (0 = none)")
    ap.add_argument("--force", action="store_true", help="rewrite shards that are already complete")
    ap.add_argument("--checks", default="full", choices=("none", "contract", "deep", "full"),
                    help="acceptance checks at the end (DESIGN 1.7); 'full' adds tokenizer + safe_open (needs the image)")
    ap.add_argument("--corpus", help="ShareGPT json for the tokenizer encode comparison (optional)")
    ap.add_argument("--strata-ref", help="strata-ref.json for the prompt-token check (optional)")
    a = ap.parse_args(argv)

    if not 2 <= a.num_layers <= C.N_LAYERS_FULL:
        ap.error("--num-layers must be 2..48 (layer 1 carries the PLE)")
    mtp = not a.no_mtp
    src_dir, out = Path(a.src).resolve(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.partial"):
        p.unlink()
    throttle = Throttle(a.throttle_mbps)
    rec_path = out / "edge0_convert.json"
    old = json.loads(rec_path.read_text()) if rec_path.is_file() else {}
    rec = {"format": "edge0-vllm-v1", "converter": "edge0/convert/convert_checkpoint.py",
           "converter_sha256": _sha256_files([HERE / f for f in ("convert_checkpoint.py", "ckpt_contract.py",
                                                                  "st_io.py", "hf_tokenizer.py", "strata_formats.py")]),
           "argv": sys.argv if argv is None else argv, "host": platform.node(), "python": sys.version.split()[0],
           "numpy": np.__version__, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "src": {"dir": str(src_dir), "record": a.source_dir_record,
                   "files": {n: os.path.getsize(src_dir / n) for n in
                             ("model-skeleton.gguf", "experts.bin", "latent_dense_q8.bin", "labels.bin", "profile.bin",
                              "ple-fp8.gguf", "mtp/dense.bin", "mtp/experts.bin")}},
           "num_layers": a.num_layers, "mtp": mtp, "shards": dict(old.get("shards", {}))}
    t_all = time.time()
    print("[convert] %s -> %s (layers %d, mtp %s)" % (src_dir, out, a.num_layers, mtp), flush=True)
    rec.update(write_small_files(a, src_dir, out, mtp))
    print("[convert] small files done; tokenizer: %s" % rec["tokenizer"]["branch"], flush=True)

    plan = C.shard_plan(a.num_layers, mtp)
    stale = {f.name for f in out.glob("model-*.safetensors")} - {fname for fname, _ in plan}
    if stale:
        raise SystemExit("stale shards from another layout in %s: %s (remove them)" % (out, sorted(stale)[:4]))
    todo_main, todo_exp = [], []
    for i, (fname, specs) in enumerate(plan):
        path = out / fname
        if not a.force and shard_done(path, specs):
            print("[convert] %s complete, skipped" % fname, flush=True)
            rec["shards"].setdefault(fname, {})["skipped_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        elif i == 0 or (mtp and i == len(plan) - 1):
            todo_main.append((i, fname, path, specs))
        else:
            todo_exp.append((str(src_dir), str(path), i - 1, a.chunk_experts, a.throttle_mbps))

    def done_expert(layer, r):
        fname = plan[layer + 1][0]
        rec["shards"][fname] = r
        write_json(rec_path, rec)
        print("[convert] %s (layer %d) %.2f GB in %.0f s" % (fname, layer, r["bytes"] / 1e9, r["seconds"]), flush=True)

    # with --jobs > 1 the expert shards run in a process pool WHILE this process writes the dense and MTP shards
    pool = mp.get_context("spawn").Pool(a.jobs) if todo_exp and a.jobs > 1 else None
    results = pool.imap_unordered(_expert_job, todo_exp) if pool else None
    try:
        src = Source(src_dir, throttle)
        for i, fname, path, specs in todo_main:
            r = write_dense_shard(src, path, specs, throttle) if i == 0 else write_mtp_shard(src, path, throttle)
            rec["shards"][fname] = r
            write_json(rec_path, rec)
            print("[convert] %s %.2f GB in %.0f s" % (fname, r["bytes"] / 1e9, r["seconds"]), flush=True)
        src.close()
        if pool:
            for layer, r in results:
                done_expert(layer, r)
        else:
            for args in todo_exp:
                done_expert(*_expert_job(args))
    finally:
        if pool:
            pool.terminate()
            pool.join()
    rec["index"] = write_index(out, plan)
    ex = [v.get("stats") for k, v in rec["shards"].items() if v.get("stats") and "w13" in v.get("stats", {})]
    if ex:
        rec["expert_totals"] = {
            "layers_with_stats": len(ex),
            "scale_bit7": sum(s["scale_bit7"] for s in ex), "scale_7f": sum(s["scale_7f"] for s in ex),
            "s_gate_ne_s_up": sum(s["s_gate_ne_s_up"] for s in ex), "tail3_nonzero": sum(s["tail3_nonzero"] for s in ex),
            "w13_frac_lt_2m6": sum(s["w13"]["lt_2m6"] for s in ex) / sum(s["w13"]["n"] for s in ex),
            "w2_frac_lt_2m6": sum(s["w2"]["lt_2m6"] for s in ex) / sum(s["w2"]["n"] for s in ex),
            "w13_frac_lt_2m6_max_layer": max(s["w13"]["frac_lt_2m6"] for s in ex),
            "w2_frac_lt_2m6_max_layer": max(s["w2"]["frac_lt_2m6"] for s in ex)}
    rec["convert_seconds"] = round(time.time() - t_all, 1)
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(rec_path, rec)
    print("[convert] shards and index done in %.0f s" % rec["convert_seconds"], flush=True)

    ok = True
    if a.checks != "none":
        import check_ckpt
        res = check_ckpt.run_checks(out, src_dir if a.checks != "contract" else None, level=a.checks,
                                    hf_ref=a.hf_tokenizer, corpus=a.corpus, strata_ref=a.strata_ref)
        rec["checks"] = res
        ok = res["ok"]
        write_json(rec_path, rec)
        print("[convert] checks (%s): %s" % (a.checks, "PASS" if ok else "FAIL"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
