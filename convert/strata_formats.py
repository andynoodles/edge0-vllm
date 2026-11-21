#!/usr/bin/env python3
"""edge0/convert/strata_formats.py - numpy reference readers for the Edge0-145B-NVFP4 model folder.

Every format here is transcribed from the edge0-strata source (MIT) and is documented in
edge0/docs/MODEL-SPEC.md, section "File formats".  Only numpy is needed; nothing is executed from the
model files, and every reader touches small slices (memmap + explicit offsets), so it is safe to run next to a
serving engine.

    python3 strata_formats.py check     --model /home/andychen/models/Edge0-145B-NVFP4
    python3 strata_formats.py skeleton  --model /home/andychen/models/Edge0-145B-NVFP4 [--out skeleton-tensors.txt]

Library use (all arrays are float32, matrices are (out_features, in_features) = torch nn.Linear layout):

    m = EdgeModel("/home/andychen/models/Edge0-145B-NVFP4")
    e = m.expert(layer=0, expert=7)            # {"gate": (640,1792), "up": (640,1792), "down": (1792,640), "scales": (sg,su,sd)}
    D, U = m.latent_du(layer=0, group=3)       # D: (1792, 2560)   U: (2560, 1792)
    g = m.labels()[0, 7]                       # routing group (0..7) of expert 7 in layer 0
    rows = m.ple().rows([0, 12345])            # (2, 160) PLE table rows
    ids = ngram_rows([tok], [[-1, -1]], m.ple_consts())   # (1, 16) PLE row ids
    W = m.skeleton().tensor("blk.0.attn_qkv.weight")     # (10240, 2560) from Q8_0
    p = read_profile(".../profile.bin")
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ============================================================================== constants (model geometry)
N_LAYERS, N_EMBD, N_EXPERT, TOP_K = 48, 2560, 512, 10
ELL, N_FF, N_GROUPS = 1792, 640, 8              # latent width, expert FFN width, D/U routing groups
HC, HC_LR = 4, 320                              # hyper-connection streams, low rank
N_VOCAB = 248320
PLE_HEAD_DIM, PLE_N_HEADS, NGRAM_SIZE, HEADS_PER_NGRAM = 160, 16, 3, 8
PLE_EOS = 248044                                # qwen4exp.ple.eos_token_id (<|endoftext|>)
TOKEN_NULL = -1

# ============================================================================== GGML block types
# type id -> (name, elements per block, bytes per block)
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 8: ("Q8_0", 32, 34), 24: ("I8", 1, 1),
    30: ("BF16", 1, 2), 39: ("MXFP4", 32, 17), 40: ("NVFP4", 64, 36), 42: ("Q2_0", 64, 18),
}


def row_bytes(type_id: int, n: int) -> int:
    name, blk, size = GGML_TYPES[type_id]
    assert n % blk == 0, (name, n, blk)
    return n // blk * size


# ----------------------------------------------------------------------------- scalar codecs
def _e4m3fn_table() -> np.ndarray:
    """FP8 E4M3 ("fn": no inf; 0x7F/0xFF are NaN, which Strata reads as 0).  src/kernels/ngram.cpp Fp8Table."""
    t = np.zeros(256, np.float32)
    for x in range(256):
        s, e, m = x >> 7, (x >> 3) & 15, x & 7
        if e == 15 and m == 7:
            f = 0.0
        elif e == 0:
            f = (m / 8.0) * 2.0 ** -6
        else:
            f = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        t[x] = -f if s else f
    return t


def _ue4m3_table() -> np.ndarray:
    """NVFP4 block scale: unsigned E4M3, bit 7 ignored, 0x00/0x7F -> 0.  ggml_ue4m3_to_fp32 WITHOUT its *0.5
    (ggml halves the scale because kvalues_mxfp4 holds doubled E2M1 values; here E2M1 is not doubled)."""
    t = np.zeros(256, np.float32)
    for x in range(256):
        if x in (0, 0x7F):
            continue
        e, m = (x >> 3) & 15, x & 7
        t[x] = m * 2.0 ** -9 if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return t


E4M3FN = _e4m3fn_table()
UE4M3 = _ue4m3_table()
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def deq_f32(raw: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(raw).view(np.float32).copy()


def deq_f16(raw: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(raw).view(np.float16).astype(np.float32)


def deq_bf16(raw: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(raw).view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def deq_q8_0(raw: np.ndarray) -> np.ndarray:
    """block_q8_0: fp16 d, 32 x int8.  value = d * q."""
    b = np.ascontiguousarray(raw).reshape(-1, 34)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    return (b[:, 2:].view(np.int8).astype(np.float32) * d).reshape(-1)


def deq_q4_0(raw: np.ndarray) -> np.ndarray:
    """block_q4_0: fp16 d, 16 bytes; element j in qs[j] low nibble, j+16 in qs[j] high nibble; value = (code-8)*d."""
    b = np.ascontiguousarray(raw).reshape(-1, 18)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    qs = b[:, 2:]
    q = np.concatenate([qs & 15, qs >> 4], axis=1).astype(np.float32)
    return ((q - 8.0) * d).reshape(-1)


def deq_nvfp4(raw: np.ndarray) -> np.ndarray:
    """ggml block_nvfp4 (type 40), 64 values in 36 bytes: d[4] (UE4M3, one per 16 values) then qs[32].
    Sub-block s (values 16s..16s+15) keeps its codes in qs[8s..8s+7]: byte j holds value j in the LOW nibble and
    value j+8 in the HIGH nibble.  value = E2M1[code] * ue4m3(d[s]).  (ggml-quants.c dequantize_row_nvfp4,
    strata iq_kernels.cu dq_nvfp4.)  The per-tensor (per-expert) global scale is applied by the caller."""
    b = np.ascontiguousarray(raw).reshape(-1, 36)
    sc = UE4M3[b[:, :4]]                                   # (nb, 4)
    qs = b[:, 4:].reshape(-1, 4, 8)                        # (nb, sub, byte)
    codes = np.concatenate([qs & 15, qs >> 4], axis=2)     # (nb, sub, 16): j<8 low nibbles, j>=8 high nibbles
    return (E2M1[codes] * sc[:, :, None]).reshape(-1)


def deq_q2_0(raw: np.ndarray) -> np.ndarray:
    """Strata/ggml-fork Q2_0 (type 42): fp16 d + 16 bytes, 64 values; value 4i+k at bits 2k of byte i;
    value = (code - 1) * d (grid {-1,0,1,2})."""
    b = np.ascontiguousarray(raw).reshape(-1, 18)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    qs = b[:, 2:]
    codes = np.stack([(qs >> s) & 3 for s in (0, 2, 4, 6)], axis=2).reshape(-1, 64).astype(np.float32)
    return ((codes - 1.0) * d).reshape(-1)


DEQ = {0: deq_f32, 1: deq_f16, 2: deq_q4_0, 8: deq_q8_0, 30: deq_bf16, 40: deq_nvfp4, 42: deq_q2_0,
       24: lambda r: np.ascontiguousarray(r).view(np.int8).astype(np.float32)}


# ============================================================================== GGUF (own minimal parser)
@dataclass
class GgufTensor:
    name: str
    shape: tuple          # GGUF order: shape[0] = ne0 (contiguous / input features)
    type_id: int
    offset: int           # absolute file offset of the data
    nbytes: int

    @property
    def type_name(self) -> str:
        return GGML_TYPES.get(self.type_id, ("type%d" % self.type_id, 0, 0))[0]


class Gguf:
    """GGUF v3 header + tensor directory.  String/number arrays longer than `keep_array` are skipped (their
    length is recorded) so parsing the 248K-token tokenizer arrays stays cheap."""

    _SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

    def __init__(self, path, keep_array: int = 4096):
        self.path = str(path)
        self.kv: dict = {}
        self.tensors: dict[str, GgufTensor] = {}
        with open(self.path, "rb") as f:
            magic, ver = f.read(4), struct.unpack("<I", f.read(4))[0]
            if magic != b"GGUF" or ver not in (2, 3):
                raise ValueError("%s: not a GGUF v2/v3 file" % path)
            n_t, n_kv = struct.unpack("<QQ", f.read(16))
            for _ in range(n_kv):
                key = self._str(f)
                vt = struct.unpack("<I", f.read(4))[0]
                self.kv[key] = self._value(f, vt, keep_array)
            infos = []
            for _ in range(n_t):
                name = self._str(f)
                nd = struct.unpack("<I", f.read(4))[0]
                shape = struct.unpack("<%dQ" % nd, f.read(8 * nd))
                ty, off = struct.unpack("<IQ", f.read(12))
                infos.append((name, shape, ty, off))
            align = int(self.kv.get("general.alignment", 32))
            self.data_start = (f.tell() + align - 1) // align * align
        size = os.path.getsize(self.path)
        for name, shape, ty, off in infos:
            n = int(np.prod(shape))
            nb = row_bytes(ty, n) if ty in GGML_TYPES else -1
            t = GgufTensor(name, tuple(int(s) for s in shape), ty, self.data_start + off, nb)
            if nb >= 0 and t.offset + nb > size:
                raise ValueError("%s: tensor %s runs past the end of the file" % (path, name))
            self.tensors[name] = t

    @staticmethod
    def _str(f) -> str:
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")

    def _value(self, f, vt, keep_array):
        if vt in self._SCALAR:
            fmt = self._SCALAR[vt]
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
        if vt == 8:
            return self._str(f)
        if vt == 9:
            et, n = struct.unpack("<IQ", f.read(12))
            if et == 8:
                if n > keep_array:
                    for _ in range(n):
                        f.seek(struct.unpack("<Q", f.read(8))[0], 1)
                    return ("<string array>", n)
                return [self._str(f) for _ in range(n)]
            fmt = self._SCALAR[et]
            sz = struct.calcsize(fmt)
            if n > keep_array:
                f.seek(sz * n, 1)
                return ("<array>", n)
            return list(struct.unpack("<%d%s" % (n, fmt[1]), f.read(sz * n)))
        raise ValueError("unknown GGUF value type %d" % vt)

    def raw(self, name: str, row0: int = 0, nrows: int | None = None) -> np.ndarray:
        """Raw bytes of rows [row0, row0+nrows) of a 2-D (or 1-D) tensor; a row is shape[0] elements."""
        t = self.tensors[name]
        rb = row_bytes(t.type_id, t.shape[0])
        total = t.nbytes // rb
        nrows = total - row0 if nrows is None else nrows
        assert 0 <= row0 and row0 + nrows <= total, (name, row0, nrows, total)
        return np.fromfile(self.path, dtype=np.uint8, count=nrows * rb, offset=t.offset + row0 * rb)

    def tensor(self, name: str, row0: int = 0, nrows: int | None = None) -> np.ndarray:
        """float32 rows: shape (nrows, shape[0]) for a 2-D tensor ((n,) for 1-D).  Row r = output feature r."""
        t = self.tensors[name]
        out = DEQ[t.type_id](self.raw(name, row0, nrows))
        return out if len(t.shape) == 1 else out.reshape(-1, t.shape[0])


# ============================================================================== pack index (index.txt, dense.bin)
INDEX_FIELDS = ("file", "kind", "src_off", "src_bytes", "dst_off", "dst_bytes", "ne0", "ne1", "code_bits",
                "code_bias", "group_elems", "codebook", "has_offset", "codes_bytes", "scales_bytes",
                "offset_bytes", "scales_fp16", "act_kind")
# kind: 0 = quantized, served natively from model-skeleton.gguf (row carries the shape only),
#       2 = F32, 4 = BF16 raw, 5 = F16 raw  (src/core/weights.cpp WeightTable::load)
KIND_DEQ = {2: deq_f32, 4: deq_bf16, 5: deq_f16}


def read_index(path) -> tuple[dict, dict]:
    header, rows = {}, {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            p = line.split()
            if len(p) >= 7 and p[1] == "align":
                header = {"align": int(p[2]), "pool": int(p[4]), "tensors": int(p[6])}
            continue
        p = line.split()
        if len(p) != 19:
            raise ValueError("index.txt row with %d fields: %r" % (len(p), line))
        rows[p[0]] = dict(zip(INDEX_FIELDS, map(int, p[1:])))
    return header, rows


class Pack:
    """index.txt + dense.bin (file 0).  Only the float tensors (kinds 2/4/5) have bytes here."""

    def __init__(self, model_dir):
        self.dir = Path(model_dir)
        self.header, self.rows = read_index(self.dir / "index.txt")

    def tensor(self, name: str) -> np.ndarray:
        r = self.rows[name]
        if r["kind"] not in KIND_DEQ:
            raise KeyError("%s is quantized (kind %d): read it from model-skeleton.gguf" % (name, r["kind"]))
        assert r["file"] == 0, name
        raw = np.fromfile(self.dir / "dense.bin", dtype=np.uint8, count=r["src_bytes"], offset=r["src_off"])
        v = KIND_DEQ[r["kind"]](raw)
        return v.reshape(r["ne1"], r["ne0"]) if r["ne1"] > 0 else v


# ============================================================================== routed experts (experts.bin)
@dataclass
class ExpertLayer:
    layer: int
    gu_type: int
    d_type: int
    offset: int
    blob_bytes: int


def read_native_experts(path) -> list[ExpertLayer]:
    """native_experts.txt: '# strata native experts v6: latent A-only (expert_embd 1792, expert_ff 640)' then one
    line per layer: layer gu_type d_type offset blob_bytes."""
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = list(map(int, line.split()[:5]))
        out.append(ExpertLayer(*p))
    return out


@dataclass
class BlobLayout:
    """src/kernels/cuda/iq_kernels.cu native_expert_layout(gu_type, d_type, n_embd=ELL, n_ff)."""
    gu_type: int = 40
    d_type: int = 40
    n_in: int = ELL
    n_ff: int = N_FF
    gu_row: int = field(init=False)
    d_row: int = field(init=False)
    up_off: int = field(init=False)
    down_off: int = field(init=False)
    tail_off: int = field(init=False)
    bytes: int = field(init=False)

    def __post_init__(self):
        self.gu_row = row_bytes(self.gu_type, self.n_in)          # 1792/64*36 = 1008
        self.d_row = row_bytes(self.d_type, self.n_ff)            # 640/64*36 = 360
        self.up_off = self.n_ff * self.gu_row                     # 645,120
        self.down_off = 2 * self.up_off                           # 1,290,240
        self.tail_off = self.down_off + self.n_in * self.d_row    # 1,935,360
        self.bytes = self.tail_off + (16 if 40 in (self.gu_type, self.d_type) else 0)   # 1,935,376


def decode_expert_blob(blob: np.ndarray, L: BlobLayout, apply_scale: bool = True) -> dict:
    """One expert record -> gate (n_ff, n_in), up (n_ff, n_in), down (n_in, n_ff), float32.
    NVFP4 tail: 4 x f32 {s_gate, s_up, s_down, 0} (ModelOpt weight_scale_2 per projection)."""
    assert blob.size == L.bytes, (blob.size, L.bytes)
    gate = DEQ[L.gu_type](blob[:L.up_off]).reshape(L.n_ff, L.n_in)
    up = DEQ[L.gu_type](blob[L.up_off:L.down_off]).reshape(L.n_ff, L.n_in)
    down = DEQ[L.d_type](blob[L.down_off:L.tail_off]).reshape(L.n_in, L.n_ff)
    tail = np.frombuffer(blob[L.tail_off:L.tail_off + 16].tobytes(), np.float32) if L.bytes > L.tail_off \
        else np.array([1, 1, 1, 0], np.float32)
    if apply_scale:
        gate, up, down = gate * tail[0], up * tail[1], down * tail[2]
    return {"gate": gate, "up": up, "down": down, "scales": tuple(float(x) for x in tail[:3]), "tail": tail}


# ============================================================================== latent D/U (latent_dense_q8.bin / _q4.bin)
class LatentDense:
    """src/core/latent_dense.cpp.  Per layer l: D [G*ELL rows, H in] then U [G*H rows, ELL in], each row a run of
    Q8_0 (fmt 8, 34 B/32) or Q4_0 (fmt 4, 18 B/32) blocks; layers back to back, no header, no padding.
    Group g's D is rows [g*ELL, (g+1)*ELL); its U is rows [g*H, (g+1)*H)."""

    def __init__(self, path, fmt: int, n_layers=N_LAYERS, groups=N_GROUPS, n_embd=N_EMBD, ell=ELL):
        self.path, self.fmt = str(path), fmt
        self.L, self.G, self.H, self.ELL = n_layers, groups, n_embd, ell
        self.ty = 8 if fmt == 8 else 2
        self.d_row, self.u_row = row_bytes(self.ty, n_embd), row_bytes(self.ty, ell)
        self.d_bytes, self.u_bytes = groups * ell * self.d_row, groups * n_embd * self.u_row
        self.layer_bytes = self.d_bytes + self.u_bytes
        size = os.path.getsize(self.path)
        if size != n_layers * self.layer_bytes:
            raise ValueError("%s: %d B, expected %d for fmt %d" % (path, size, n_layers * self.layer_bytes, fmt))

    def _rows(self, off, nrows, rb):
        return np.fromfile(self.path, dtype=np.uint8, count=nrows * rb, offset=off)

    def d(self, layer: int, group: int) -> np.ndarray:
        off = layer * self.layer_bytes + group * self.ELL * self.d_row
        return DEQ[self.ty](self._rows(off, self.ELL, self.d_row)).reshape(self.ELL, self.H)

    def u(self, layer: int, group: int) -> np.ndarray:
        off = layer * self.layer_bytes + self.d_bytes + group * self.H * self.u_row
        return DEQ[self.ty](self._rows(off, self.H, self.u_row)).reshape(self.H, self.ELL)

    def layer(self, layer: int):
        """Whole layer: D_cat (G*ELL, H) = [D_0; ...; D_7] and U_cat (H, G*ELL) = [U_0 | ... | U_7], so that
        xl_all = D_cat @ x and out = U_cat @ concat_g(z_g) (MODEL-SPEC.md section 2.6).  ~147 MB each in float32."""
        d = DEQ[self.ty](self._rows(layer * self.layer_bytes, self.G * self.ELL, self.d_row)).reshape(self.G * self.ELL, self.H)
        u = DEQ[self.ty](self._rows(layer * self.layer_bytes + self.d_bytes, self.G * self.H, self.u_row))
        u = u.reshape(self.G, self.H, self.ELL).transpose(1, 0, 2).reshape(self.H, self.G * self.ELL)
        return d, u


def read_labels(path, n_layers=N_LAYERS, n_expert=N_EXPERT) -> np.ndarray:
    """labels.bin: int32 [n_layers, n_expert], the D/U group (0..7) of every routed expert."""
    v = np.fromfile(path, dtype="<i4")
    assert v.size == n_layers * n_expert, v.size
    return v.reshape(n_layers, n_expert)


# ============================================================================== PLE (ple-fp8.gguf) and the n-gram hash
@dataclass
class PleConsts:
    mult: list           # 3 x uint64  (qwen4exp.ple.layer_multipliers)
    vocab: list          # 16 x uint64 (qwen4exp.ple.head_vocab_sizes)
    offset: list         # 16 x uint64 (qwen4exp.ple.head_offsets)
    eos: int = PLE_EOS


def ple_consts_from_gguf(g: Gguf) -> PleConsts:
    kv = g.kv
    return PleConsts(list(kv["qwen4exp.ple.layer_multipliers"]), list(kv["qwen4exp.ple.head_vocab_sizes"]),
                     list(kv["qwen4exp.ple.head_offsets"]), int(kv.get("qwen4exp.ple.eos_token_id", PLE_EOS)))


def ngram_rows(tokens, prev, c: PleConsts) -> np.ndarray:
    """src/kernels/ngram.cpp ngram_rows.  tokens: (n,), prev: (n, 2) predecessors OLDEST FIRST, -1 = none.
    ctx[0] = token; walking back s = 1, 2: a missing (-1 / <0) or EOS predecessor cuts the history, and every
    position at or beyond the cut reads EOS.  mixed_n = XOR_{j<n} (ctx[j] * mult[j]) mod 2^64 for n = 2, 3;
    row[(n-2)*8 + g] = mixed_n % vocab[h] + offset[h].  Returns uint32 (n, 16)."""
    M = (1 << 64) - 1
    out = np.zeros((len(tokens), PLE_N_HEADS), np.uint32)
    for i, tok in enumerate(tokens):
        ctx = [int(tok)]
        cut = False
        for s in (1, 2):
            t = TOKEN_NULL if cut else int(prev[i][2 - s])
            cut = cut or t < 0 or t == c.eos
            ctx.append(c.eos if cut else t)
        for n in (2, 3):
            mixed = (ctx[0] * c.mult[0]) & M
            for j in range(1, n):
                mixed ^= (ctx[j] * c.mult[j]) & M
            for gidx in range(HEADS_PER_NGRAM):
                h = (n - 2) * HEADS_PER_NGRAM + gidx
                out[i, h] = mixed % c.vocab[h] + c.offset[h]
    return out


def ngram_rows_for_sequence(seq, c: PleConsts) -> np.ndarray:
    """Row ids for every position of a token sequence that starts a fresh context (no predecessors)."""
    prev = [[seq[i - 2] if i >= 2 else TOKEN_NULL, seq[i - 1] if i >= 1 else TOKEN_NULL] for i in range(len(seq))]
    return ngram_rows(seq, prev, c)


class PleTable:
    """ple-fp8.gguf: one tensor per_layer_token_embd.weight, GGUF type I8 (raw FP8 E4M3 bytes), shape
    [160, 320001536]: a row is 160 contiguous bytes.  value = e4m3fn(byte) * strata.ple.scale."""

    def __init__(self, path):
        self.g = Gguf(path)
        self.t = self.g.tensors["per_layer_token_embd.weight"]
        fmt = self.g.kv.get("strata.ple.format")
        if fmt != "f8_e4m3" or self.t.type_id != 24 or self.t.shape[0] != PLE_HEAD_DIM:
            raise ValueError("unexpected PLE table: format %r type %s shape %s" % (fmt, self.t.type_name, self.t.shape))
        self.scale = float(self.g.kv["strata.ple.scale"])
        self.n_rows = self.t.shape[1]

    def rows(self, ids) -> np.ndarray:
        out = np.empty((len(ids), PLE_HEAD_DIM), np.float32)
        with open(self.g.path, "rb") as f:
            for i, r in enumerate(ids):
                r = int(r)
                assert 0 <= r < self.n_rows, r
                f.seek(self.t.offset + r * PLE_HEAD_DIM)
                out[i] = E4M3FN[np.frombuffer(f.read(PLE_HEAD_DIM), np.uint8)] * self.scale
        return out

    def embedding(self, rows16) -> np.ndarray:
        """16 row ids -> the 2560-wide PLE input, flattened HEAD-SLOWEST (row h at [160h, 160h+160))."""
        return self.rows(rows16).reshape(-1)


# ============================================================================== profile.bin
def read_profile(path) -> dict:
    """'STRP', <5I version n_layers n_expert slots n_ranked>, n_ranked x <HH layer expert> (best first), then
    int32 table [n_layers][n_expert] = rank slot or -1.  tools/make_profile.py, src/core/expert_cache.cpp."""
    blob = Path(path).read_bytes()
    if blob[:4] != b"STRP":
        raise ValueError("not a Strata profile")
    ver, nl, ne, slots, n = struct.unpack_from("<5I", blob, 4)
    ranked = np.frombuffer(blob, "<u2", count=2 * n, offset=24).reshape(n, 2)
    table = np.frombuffer(blob, "<i4", count=nl * ne, offset=24 + 4 * n).reshape(nl, ne)
    return {"version": ver, "n_layers": nl, "n_expert": ne, "slots": slots, "n_ranked": n,
            "ranked": ranked, "table": table}


# ============================================================================== MTP draft layer (mtp/)
def read_mtp_index(mtp_dir) -> dict:
    """mtp/dense.txt: 'name kind rows cols offset bytes', kind q8_0 | bf16 | f32 (f32 norms already 1+w).
    rows = output features, cols = input features (Q8_0 rows of `cols`)."""
    out = {}
    for line in (Path(mtp_dir) / "dense.txt").read_text().split("\n"):
        p = line.split()
        if len(p) == 6:
            out[p[0]] = {"kind": p[1], "rows": int(p[2]), "cols": int(p[3]), "off": int(p[4]), "bytes": int(p[5])}
    return out


def read_mtp_tensor(mtp_dir, name: str, index: dict | None = None) -> np.ndarray:
    r = (index or read_mtp_index(mtp_dir))[name]
    raw = np.fromfile(Path(mtp_dir) / "dense.bin", dtype=np.uint8, count=r["bytes"], offset=r["off"])
    v = {"q8_0": deq_q8_0, "bf16": deq_bf16, "f32": deq_f32}[r["kind"]](raw)
    return v.reshape(r["rows"], r["cols"]) if r["rows"] > 1 else v


MTP_BLOB = 3 * (N_EMBD * N_FF * 18 // 64)     # 1,382,400


def read_mtp_expert(mtp_dir, e: int) -> dict:
    """mtp/experts.bin (tools/mtp_rt.py blob_of): FULL-WIDTH (2560) Q2_0 experts, 512 x 1,382,400 B, each:
    [gate/up code planes interleaved by row (2r = gate r, 2r+1 = up r), 1280 rows x 40 blocks x 16 B]
    [down codes 2560 x 10 x 16] [gate/up fp16 scales 1280 x 40 x 2] [down scales 2560 x 10 x 2]."""
    blob = np.fromfile(Path(mtp_dir) / "experts.bin", dtype=np.uint8, count=MTP_BLOB, offset=e * MTP_BLOB)
    gu_c, dn_c = 2 * N_FF * (N_EMBD // 64) * 16, N_EMBD * (N_FF // 64) * 16
    gu_s, dn_s = 2 * N_FF * (N_EMBD // 64) * 2, N_EMBD * (N_FF // 64) * 2
    o = np.cumsum([0, gu_c, dn_c, gu_s, dn_s])
    def blocks(codes, scales, nblk):
        b = np.empty((nblk, 18), np.uint8)
        b[:, :2] = scales.reshape(nblk, 2)
        b[:, 2:] = codes.reshape(nblk, 16)
        return deq_q2_0(b.reshape(-1))
    gu = blocks(blob[o[0]:o[1]], blob[o[2]:o[3]], 2 * N_FF * N_EMBD // 64).reshape(2 * N_FF, N_EMBD)
    dn = blocks(blob[o[1]:o[2]], blob[o[3]:o[4]], N_EMBD * N_FF // 64).reshape(N_EMBD, N_FF)
    return {"gate": gu[0::2], "up": gu[1::2], "down": dn}


def read_draft_vocab(path) -> np.ndarray:
    """mtp/draft_vocab.bin: int32 token ids the draft head may propose (its logits are computed over these rows of
    output.weight only)."""
    return np.fromfile(path, dtype="<i4")


# ============================================================================== whole-folder convenience
class EdgeModel:
    def __init__(self, model_dir, du_fmt: int = 8):
        self.dir = Path(model_dir)
        self.du_fmt = du_fmt
        self._skel = self._ple = self._pack = self._du = self._labels = None
        self.layers = read_native_experts(self.dir / "native_experts.txt")

    def skeleton(self) -> Gguf:
        self._skel = self._skel or Gguf(self.dir / "model-skeleton.gguf")
        return self._skel

    def pack(self) -> Pack:
        self._pack = self._pack or Pack(self.dir)
        return self._pack

    def ple(self) -> PleTable:
        self._ple = self._ple or PleTable(self.dir / "ple-fp8.gguf")
        return self._ple

    def ple_consts(self) -> PleConsts:
        return ple_consts_from_gguf(self.skeleton())

    def labels(self) -> np.ndarray:
        if self._labels is None:
            self._labels = read_labels(self.dir / "labels.bin")
        return self._labels

    def latent(self) -> LatentDense:
        if self._du is None:
            name = "latent_dense_q8.bin" if self.du_fmt == 8 else "latent_dense_q4.bin"
            self._du = LatentDense(self.dir / name, self.du_fmt)
        return self._du

    def latent_du(self, layer: int, group: int):
        return self.latent().d(layer, group), self.latent().u(layer, group)

    def expert_raw(self, layer: int, expert: int) -> np.ndarray:
        L = self.layers[layer]
        return np.fromfile(self.dir / "experts.bin", dtype=np.uint8, count=L.blob_bytes,
                           offset=L.offset + expert * L.blob_bytes)

    def expert(self, layer: int, expert: int, apply_scale: bool = True) -> dict:
        L = self.layers[layer]
        lay = BlobLayout(L.gu_type, L.d_type)
        assert lay.bytes == L.blob_bytes, (lay.bytes, L.blob_bytes)
        return decode_expert_blob(self.expert_raw(layer, expert), lay, apply_scale)


# ============================================================================== reference math for one token
def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def route(logits: np.ndarray, k: int = TOP_K):
    """softmax over all experts, stable descending sort (ties -> lower index), top-k, renormalise with the sum
    clamped below at 2^-14; routed scaling factor 1, no bias.  (router_top10.hpp / native_router.hpp)"""
    p = np.exp(logits - logits.max())
    p /= p.sum()
    ids = np.argsort(-p, kind="stable")[:k]
    w = p[ids] / max(p[ids].sum(), 2.0 ** -14)
    return ids, w.astype(np.float32)


def latent_moe_reference(m: EdgeModel, layer: int, x: np.ndarray, with_shared: bool = True) -> dict:
    """The MoE half of one layer for ONE token, x = the 2560-wide output of the FFN hyper-connection read.
    out = sum_j w_j * U_{g_j} . down_j( silu(gate_j . D_{g_j} x) * (up_j . D_{g_j} x) )
          + sigmoid(gate_inp_shexp . x) * down_sh( silu(gate_sh x) * up_sh x )     (g_j = labels[layer, e_j])."""
    pk, sk = m.pack(), m.skeleton()
    logits = pk.tensor("blk.%d.ffn_gate_inp.weight" % layer) @ x                 # (512,)
    ids, w = route(logits)
    lab = m.labels()[layer]
    out = np.zeros(N_EMBD, np.float32)
    xl_cache = {}
    for e, wj in zip(ids, w):
        g = int(lab[e])
        if g not in xl_cache:
            D, U = m.latent_du(layer, g)
            xl_cache[g] = (D @ x, U)
        xl, U = xl_cache[g]
        ex = m.expert(layer, int(e))
        h = silu(ex["gate"] @ xl) * (ex["up"] @ xl)
        out += wj * (U @ (ex["down"] @ h))
    res = {"ids": ids, "weights": w, "groups": lab[ids], "routed": out.copy()}
    if with_shared:
        gsh = sk.tensor("blk.%d.ffn_gate_shexp.weight" % layer)
        ush = sk.tensor("blk.%d.ffn_up_shexp.weight" % layer)
        dsh = sk.tensor("blk.%d.ffn_down_shexp.weight" % layer)
        gate = sigmoid(float(pk.tensor("blk.%d.ffn_gate_inp_shexp.weight" % layer).reshape(-1) @ x))
        res["shared"] = gate * (dsh @ (silu(gsh @ x) * (ush @ x)))
        res["shared_gate"] = gate
        out = out + res["shared"]
    res["out"] = out
    return res


# ============================================================================== CLI
def _stats(name, a):
    a = np.asarray(a, np.float32)
    print("  %-44s shape=%-16s min=% .4g max=% .4g mean=% .4g rms=%.4g nan=%d inf=%d"
          % (name, tuple(a.shape), a.min(), a.max(), a.mean(), float(np.sqrt((a.astype(np.float64) ** 2).mean())),
             int(np.isnan(a).sum()), int(np.isinf(a).sum())))
    return bool(np.isfinite(a).all())


def cmd_skeleton(a):
    g = Gguf(Path(a.model) / "model-skeleton.gguf")
    lines = ["# model-skeleton.gguf tensor directory: name type shape(GGUF order, ne0 first) bytes abs_offset",
             "# %d tensors, data starts at %d" % (len(g.tensors), g.data_start)]
    for t in g.tensors.values():
        lines.append("%s %s %s %d %d" % (t.name, t.type_name, list(t.shape), t.nbytes, t.offset))
    lines.append("# metadata")
    for k, v in g.kv.items():
        s = repr(v)
        lines.append("KV %s = %s" % (k, s if len(s) < 400 else s[:400] + "...(%d chars)" % len(s)))
    text = "\n".join(lines) + "\n"
    if a.out:
        Path(a.out).write_text(text)
        print("wrote %s (%d tensors)" % (a.out, len(g.tensors)))
    else:
        sys.stdout.write(text)


def cmd_check(a):
    ok = True
    m = EdgeModel(a.model, du_fmt=a.du_fmt)
    d = Path(a.model)
    print("== native_experts.txt / experts.bin")
    lay = BlobLayout()
    print("  blob layout: gu_row %d d_row %d up_off %d down_off %d tail_off %d bytes %d"
          % (lay.gu_row, lay.d_row, lay.up_off, lay.down_off, lay.tail_off, lay.bytes))
    for L in m.layers:
        ok &= L.blob_bytes == lay.bytes and L.offset == L.layer * N_EXPERT * lay.bytes and (L.gu_type, L.d_type) == (40, 40)
    ok &= os.path.getsize(d / "experts.bin") == N_LAYERS * N_EXPERT * lay.bytes
    print("  48 layers x 512 x %d B consistent: %s" % (lay.bytes, ok))
    pk = m.pack()
    for layer, e in ((0, 0), (0, 511), (47, 300)):
        ex = m.expert(layer, e)
        print(" layer %d expert %d scales(gate,up,down)=%s tail[3]=%g" % (layer, e, ex["scales"], ex["tail"][3]))
        for k in ("gate", "up", "down"):
            ok &= _stats("%s" % k, ex[k])
        # ModelOpt NVFP4 picks each 16-value block scale as amax/6, so with the right sub-block grouping nearly every
        # sub-block holds a code of magnitude 6 (E2M1 code 7 or 15)
        raw = m.expert_raw(layer, e)
        b = raw[:lay.tail_off].reshape(-1, 36)
        qs = b[:, 4:].reshape(-1, 4, 8)
        codes = np.concatenate([qs & 15, qs >> 4], axis=2)
        frac = float(((codes & 7) == 7).any(axis=2).mean())
        print("    sub-blocks holding |E2M1| = 6: %.4f" % frac)
        # dense.bin also carries blk.N.ffn_{gate,up,down}_exps.{scale,input_scale}; the engine never reads them (the
        # blob tail is what every NVFP4 kernel multiplies by) and they do not match the tails: report only
        sp = [float(pk.tensor("blk.%d.ffn_%s_exps.scale" % (layer, p))[e]) for p in ("gate", "up", "down")]
        si = [float(pk.tensor("blk.%d.ffn_%s_exps.input_scale" % (layer, p))[e]) for p in ("gate", "up", "down")]
        print("    (unused) dense.bin .scale gate/up/down = %.4g/%.4g/%.4g  .input_scale = %.4g/%.4g/%.4g"
              % tuple(sp + si))
    print("== labels.bin")
    lab = m.labels()
    counts = np.stack([np.bincount(lab[l], minlength=N_GROUPS) for l in range(N_LAYERS)])
    ok &= lab.min() >= 0 and lab.max() < N_GROUPS
    print("  range %d..%d; experts per group, layer 0: %s; layer 47: %s; min/max over all layers %d/%d"
          % (lab.min(), lab.max(), counts[0].tolist(), counts[47].tolist(), counts.min(), counts.max()))
    print("== latent_dense (fmt %d)" % a.du_fmt)
    for layer, g in ((0, 0), (47, 7)):
        D, U = m.latent_du(layer, g)
        ok &= _stats("D layer %d group %d" % (layer, g), D) & _stats("U layer %d group %d" % (layer, g), U)
    Dc, Uc = m.latent().layer(5)
    same = np.array_equal(Dc[3 * ELL:4 * ELL], m.latent().d(5, 3)) and np.array_equal(Uc[:, 3 * ELL:4 * ELL], m.latent().u(5, 3))
    ok &= same
    print("  layer 5 D_cat %s U_cat %s slice of group 3 == per-group read: %s" % (Dc.shape, Uc.shape, same))
    del Dc, Uc
    print("== profile.bin")
    p = read_profile(d / "profile.bin")
    ok &= p["n_ranked"] == N_LAYERS * N_EXPERT and (p["table"] >= 0).all()
    top = p["ranked"][:8].tolist()
    print("  version %d %dx%d slots %d ranked %d; top-8 (layer,expert): %s" % (p["version"], p["n_layers"],
                                                                             p["n_expert"], p["slots"], p["n_ranked"], top))
    ok &= all(p["table"][l, e] == i for i, (l, e) in enumerate(p["ranked"][:1000].tolist()))
    per_layer_top4k = np.bincount(p["ranked"][:4096, 0], minlength=N_LAYERS)
    print("  layers of the first 4096 ranked pairs: min %d max %d per layer" % (per_layer_top4k.min(), per_layer_top4k.max()))
    print("== model-skeleton.gguf")
    sk = m.skeleton()
    print("  %d tensors, arch %s, data_start %d" % (len(sk.tensors), sk.kv.get("general.architecture"), sk.data_start))
    for name in ("blk.0.attn_qkv.weight", "blk.3.attn_q.weight", "blk.0.ffn_gate_shexp.weight"):
        ok &= _stats(name + " rows 0..7", sk.tensor(name, 0, 8))
    for name in ("blk.3.attn_q_norm.weight", "blk.3.attn_k_norm.weight", "blk.3.indexer.q_norm.weight",
                 "blk.0.hc_attn_norm.weight", "output_hc_norm.weight", "blk.0.ssm_norm.weight", "blk.0.ssm_a",
                 "blk.0.ssm_dt.bias", "blk.1.ple_norm_key.weight", "blk.1.ple_norm_query.weight",
                 "blk.1.ple_norm_conv.weight", "blk.0.ssm_conv1d.weight", "blk.1.ple_conv1d.weight"):
        ok &= _stats(name, sk.tensor(name))
    ok &= _stats("token_embd rows 248044..248047", sk.tensor("token_embd.weight", 248044, 4))
    a1 = sk.tensor("blk.0.ffn_gate_inp.weight")
    a2 = pk.tensor("blk.0.ffn_gate_inp.weight")
    same = a1.shape == a2.shape and np.array_equal(a1, a2)
    ok &= same
    print("  blk.0.ffn_gate_inp.weight skeleton vs dense.bin: %s %s" % (a1.shape, "identical" if same else "DIFFER"))
    print("== ple-fp8.gguf")
    pt = m.ple()
    c = m.ple_consts()
    print("  rows %d scale %r; last head ends at %d" % (pt.n_rows, pt.scale, c.offset[-1] + c.vocab[-1]))
    ok &= c.offset[-1] + c.vocab[-1] <= pt.n_rows
    ok &= all(c.offset[i + 1] == c.offset[i] + c.vocab[i] for i in range(PLE_N_HEADS - 1))
    seq = [248045, 872, 198, 9707, 11, 1879]           # <|im_start|> user \n ... (token ids only for the hash test)
    rows = ngram_rows_for_sequence(seq, c)
    print("  hash rows for a 6-token sequence, position 0:", rows[0].tolist())
    print("  position 5:", rows[5].tolist())
    emb = pt.embedding(rows[5])
    ok &= _stats("ple embedding (2560) at position 5", emb)
    ok &= _stats("ple rows 0..3", pt.rows([0, 1, 2, 3]))
    print("== mtp/")
    mi = read_mtp_index(d / "mtp")
    print("  %d dense tensors; experts.bin %d B = 512 x %d" % (len(mi), os.path.getsize(d / "mtp/experts.bin"), MTP_BLOB))
    ok &= os.path.getsize(d / "mtp/experts.bin") == N_EXPERT * MTP_BLOB
    for name in ("pre_fc_norm_embedding.weight", "hyper_connection_mixer.hc_norm.weight", "mlp.gate.weight",
                 "fc_embedding.weight"):
        ok &= _stats("mtp " + name, read_mtp_tensor(d / "mtp", name, mi))
    me = read_mtp_expert(d / "mtp", 5)
    for k in ("gate", "up", "down"):
        ok &= _stats("mtp expert 5 " + k, me[k])
    dv = read_draft_vocab(d / "mtp/draft_vocab.bin")
    print("  draft_vocab: %d ids, range %d..%d, sorted %s" % (dv.size, dv.min(), dv.max(), bool((np.diff(dv) > 0).all())))
    if a.moe:
        print("== latent MoE reference, layer 0, one random unit-RMS token")
        rng = np.random.default_rng(0)
        x = rng.standard_normal(N_EMBD).astype(np.float32)
        r = latent_moe_reference(m, 0, x)
        print("  experts %s\n  weights %s\n  groups  %s\n  shared gate %.4f" % (r["ids"].tolist(),
              np.round(r["weights"], 4).tolist(), r["groups"].tolist(), r["shared_gate"]))
        ok &= _stats("routed out", r["routed"]) & _stats("shared out", r["shared"]) & _stats("moe out", r["out"])
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--model", required=True)
    c.add_argument("--du-fmt", type=int, default=8, choices=(4, 8))
    c.add_argument("--moe", action="store_true", help="also run the one-token latent MoE reference for layer 0")
    s = sub.add_parser("skeleton")
    s.add_argument("--model", required=True)
    s.add_argument("--out")
    a = ap.parse_args()
    return cmd_check(a) if a.cmd == "check" else cmd_skeleton(a)


if __name__ == "__main__":
    sys.exit(main())
