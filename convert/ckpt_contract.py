#!/usr/bin/env python3
"""edge0/convert/ckpt_contract.py - the converted-checkpoint contract (edge0/docs/DESIGN.md section 1) as code.

One place that knows every tensor name, dtype, shape and shard, plus the exact config.json /
generation_config.json contents.  convert_checkpoint.py writes what this module describes and check_ckpt.py
validates a directory against it, so the converter and the validator cannot drift apart silently.  The values are
transcribed from DESIGN.md 1.2 / 1.3 / 1.5; if the design changes, change this file (and only this file) first.

Pure Python + the standard library (no numpy), so the model agent can import it anywhere.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# ----------------------------------------------------------------------------- geometry (DESIGN 1.2)
N_LAYERS_FULL = 48
HIDDEN = 2560
VOCAB = 248320
N_EXPERT = 512
LATENT = 1792
LATENT_GROUPS = 8
EXPERT_FF = 640
HC_DIM = 4 * HIDDEN            # 10240
HC_LR = 320
GDN_QK = 4096                  # q 2048 + k 2048 channels of in_proj_qkv / conv1d
GDN_V_HEADS, GDN_K_HEADS, GDN_HEAD = 48, 16, 128
PLE_LAYER = 1                  # HF ple_layer_ids=[2] is 1-based; vLLM attaches the PLE to layer_idx 1
PLE_SCALE = 0.00019931793212890625
PLE_DATA_OFFSET = 352
PLE_ROWS = 320001536
PLE_ROW_BYTES = 160
DRAFT_VOCAB_SIZE = 106299
SOURCE_DIR_DEFAULT = "/home/andychen/models/Edge0-145B-NVFP4"

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "U8": 1, "F8_E4M3": 1}

Q8_RECIPE = {"quant_algo": "Q8_0", "group_size": 32, "block_payload_bytes": 34, "packing": "ggml"}
MTP_EXPERT_RECIPE = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
MTP_EXPERT_KEY = "mtp.layers.0.mlp.experts"


def is_qsa(layer: int) -> bool:
    return layer % 4 == 3


def gdn_v_src(i: int) -> int:
    """HF (grouped) v-head i = 3k + r  <-  GGUF (tiled) v-head 16r + k   (k < 16, r < 3).  DESIGN 1.5 / MODEL-SPEC 7."""
    k, r = divmod(i, GDN_V_HEADS // GDN_K_HEADS)
    return GDN_K_HEADS * r + k


GDN_V_PERM = [gdn_v_src(i) for i in range(GDN_V_HEADS)]      # HF head i takes GGUF head GDN_V_PERM[i]


@dataclass(frozen=True)
class Spec:
    name: str
    dtype: str
    shape: tuple

    @property
    def numel(self) -> int:
        return math.prod(self.shape)          # () -> 1 (0-dim scalar)

    @property
    def nbytes(self) -> int:
        return self.numel * DTYPE_BYTES[self.dtype]


def _q8(name, n, k):
    assert k % 32 == 0
    return Spec(name, "U8", (n, k // 32, 34))


# ----------------------------------------------------------------------------- tensor catalogue (DESIGN 1.5)
def dense_specs(num_layers: int) -> list[Spec]:
    """Shard 1: every tensor that is not an expert and not MTP, in catalogue order."""
    S = []
    S += [Spec("model.embed_tokens.weight", "BF16", (VOCAB, HIDDEN)),
          Spec("lm_head.weight", "BF16", (VOCAB, HIDDEN)),
          Spec("model.hyper_connection_mixer.hc_norm.weight", "F32", (HC_DIM,)),
          Spec("model.hyper_connection_mixer.input_mix_weight_down.weight", "BF16", (HC_LR, HC_DIM)),
          Spec("model.hyper_connection_mixer.input_mix_weight_up.weight", "BF16", (HC_DIM, HC_LR))]
    for l in range(num_layers):
        p = "model.layers.%d." % l
        for hc in ("attn_hyper_connection", "mlp_hyper_connection"):
            S += [Spec(p + hc + ".hc_norm.weight", "F32", (HC_DIM,)),
                  Spec(p + hc + ".input_mix_weight_down.weight", "BF16", (HC_LR, HC_DIM)),
                  Spec(p + hc + ".input_mix_weight_up.weight", "BF16", (HC_DIM, HC_LR)),
                  Spec(p + hc + ".block_inject_weight.weight", "BF16", (4, HC_DIM))]
        if is_qsa(l):
            a = p + "self_attn."
            S += [Spec(a + "q_proj.weight", "BF16", (24 * 512, HIDDEN)),
                  Spec(a + "k_proj.weight", "BF16", (512, HIDDEN)),
                  Spec(a + "v_proj.weight", "BF16", (512, HIDDEN)),
                  _q8(a + "o_proj.weight", HIDDEN, 6144),
                  Spec(a + "q_norm.weight", "F32", (256,)),
                  Spec(a + "k_norm.weight", "F32", (256,)),
                  Spec(a + "indexer.index_qk_proj.weight", "BF16", (640, HIDDEN)),
                  Spec(a + "indexer.q_layernorm.weight", "F32", (128,)),
                  Spec(a + "indexer.k_layernorm.weight", "F32", (128,))]
        else:
            a = p + "linear_attn."
            S += [_q8(a + "in_proj_qkv.weight", 10240, HIDDEN),
                  _q8(a + "in_proj_z.weight", 6144, HIDDEN),
                  Spec(a + "in_proj_b.weight", "BF16", (GDN_V_HEADS, HIDDEN)),
                  Spec(a + "in_proj_a.weight", "BF16", (GDN_V_HEADS, HIDDEN)),
                  Spec(a + "conv1d.weight", "F32", (10240, 1, 4)),
                  Spec(a + "A_log", "F32", (GDN_V_HEADS,)),
                  Spec(a + "dt_bias", "F32", (GDN_V_HEADS,)),
                  Spec(a + "norm.weight", "F32", (GDN_HEAD,)),
                  _q8(a + "out_proj.weight", HIDDEN, 6144)]
        m = p + "mlp."
        S += [Spec(m + "gate.weight", "BF16", (N_EXPERT, HIDDEN)),
              Spec(m + "shared_expert_gate.weight", "BF16", (1, HIDDEN)),
              _q8(m + "shared_expert.gate_proj.weight", EXPERT_FF, HIDDEN),
              _q8(m + "shared_expert.up_proj.weight", EXPERT_FF, HIDDEN),
              _q8(m + "shared_expert.down_proj.weight", HIDDEN, EXPERT_FF),
              _q8(m + "latent_down.weight", LATENT_GROUPS * LATENT, HIDDEN),
              _q8(m + "latent_up.weight", HIDDEN, LATENT_GROUPS * LATENT),
              Spec(m + "expert_group", "I32", (N_EXPERT,))]
        if l == PLE_LAYER:
            e = p + "ple."
            S += [Spec(e + "key_proj.weight", "BF16", (HC_DIM, HIDDEN)),
                  Spec(e + "value_proj.weight", "BF16", (HIDDEN, HIDDEN)),
                  Spec(e + "norm_key.weight", "F32", (HC_DIM,)),
                  Spec(e + "norm_query.weight", "F32", (HC_DIM,)),
                  Spec(e + "norm_conv.weight", "F32", (HC_DIM,)),
                  Spec(e + "conv1d.weight", "F16", (HC_DIM, 1, 4)),
                  Spec(e + "ple_embedding.ngram_embedding.weight_scale", "F32", (1,))]
    return S


EXPERT_SUFFIXES = ("w13_weight", "w13_weight_scale", "w13_weight_scale_2", "w2_weight", "w2_weight_scale",
                   "w2_weight_scale_2")


def expert_specs(layer: int) -> list[Spec]:
    """Shard layer+2: the six stacked ModelOpt NVFP4 tensors of one layer (logical expert order)."""
    p = "model.layers.%d.mlp.experts." % layer
    E, N13, N2 = N_EXPERT, 2 * EXPERT_FF, LATENT
    return [Spec(p + "w13_weight", "U8", (E, N13, LATENT // 2)),
            Spec(p + "w13_weight_scale", "F8_E4M3", (E, N13, LATENT // 16)),
            Spec(p + "w13_weight_scale_2", "F32", (E, 2)),
            Spec(p + "w2_weight", "U8", (E, N2, EXPERT_FF // 2)),
            Spec(p + "w2_weight_scale", "F8_E4M3", (E, N2, EXPERT_FF // 16)),
            Spec(p + "w2_weight_scale_2", "F32", (E,))]


# MTP dense tensors: (checkpoint name, mtp/dense.txt name, dtype, shape).  f32 entries are norms stored 1+w -> -1.
def mtp_dense_table() -> list[tuple[str, str, str, tuple]]:
    T = []

    def add(ck, src, dt, shape):
        T.append((ck, src, dt, shape))

    def q8(ck, src, n, k):
        add(ck, src, "U8", (n, k // 32, 34))

    q8("mtp.fc_embedding.weight", "fc_embedding.weight", HIDDEN, HIDDEN)
    q8("mtp.fc_hidden.weight", "fc_hidden.weight", HIDDEN, HIDDEN)
    add("mtp.pre_fc_norm_embedding.weight", "pre_fc_norm_embedding.weight", "F32", (HIDDEN,))
    add("mtp.pre_fc_norm_hidden.weight", "pre_fc_norm_hidden.weight", "F32", (HC_DIM,))
    add("mtp.hyper_connection_mixer.hc_norm.weight", "hyper_connection_mixer.hc_norm.weight", "F32", (HC_DIM,))
    add("mtp.hyper_connection_mixer.input_mix_weight_down.weight",
        "hyper_connection_mixer.input_mix_weight_down.weight", "BF16", (HC_LR, HC_DIM))
    add("mtp.hyper_connection_mixer.input_mix_weight_up.weight",
        "hyper_connection_mixer.input_mix_weight_up.weight", "BF16", (HC_DIM, HC_LR))
    for hc in ("attn_hyper_connection", "mlp_hyper_connection"):
        add("mtp.layers.0.%s.hc_norm.weight" % hc, "%s.hc_norm.weight" % hc, "F32", (HC_DIM,))
        add("mtp.layers.0.%s.input_mix_weight_down.weight" % hc, "%s.input_mix_weight_down.weight" % hc, "BF16",
            (HC_LR, HC_DIM))
        add("mtp.layers.0.%s.input_mix_weight_up.weight" % hc, "%s.input_mix_weight_up.weight" % hc, "BF16",
            (HC_DIM, HC_LR))
        add("mtp.layers.0.%s.block_inject_weight.weight" % hc, "%s.block_inject_weight.weight" % hc, "BF16",
            (4, HC_DIM))
    a = "mtp.layers.0.self_attn."
    q8(a + "q_proj.weight", "self_attn.q_proj.weight", 24 * 512, HIDDEN)
    q8(a + "k_proj.weight", "self_attn.k_proj.weight", 512, HIDDEN)
    q8(a + "v_proj.weight", "self_attn.v_proj.weight", 512, HIDDEN)
    q8(a + "o_proj.weight", "self_attn.o_proj.weight", HIDDEN, 6144)
    q8(a + "indexer.index_qk_proj.weight", "self_attn.indexer.index_qk_proj.weight", 640, HIDDEN)
    add(a + "q_norm.weight", "self_attn.q_norm.weight", "F32", (256,))
    add(a + "k_norm.weight", "self_attn.k_norm.weight", "F32", (256,))
    add(a + "indexer.q_layernorm.weight", "self_attn.indexer.q_layernorm.weight", "F32", (128,))
    add(a + "indexer.k_layernorm.weight", "self_attn.indexer.k_layernorm.weight", "F32", (128,))
    m = "mtp.layers.0.mlp."
    add(m + "gate.weight", "mlp.gate.weight", "BF16", (N_EXPERT, HIDDEN))
    add(m + "shared_expert_gate.weight", "mlp.shared_expert_gate.weight", "BF16", (1, HIDDEN))
    q8(m + "shared_expert.gate_proj.weight", "mlp.shared_expert.gate_proj.weight", EXPERT_FF, HIDDEN)
    q8(m + "shared_expert.up_proj.weight", "mlp.shared_expert.up_proj.weight", EXPERT_FF, HIDDEN)
    q8(m + "shared_expert.down_proj.weight", "mlp.shared_expert.down_proj.weight", HIDDEN, EXPERT_FF)
    return T


MTP_PROJ_SHAPES = {"gate_proj": (EXPERT_FF, HIDDEN), "up_proj": (EXPERT_FF, HIDDEN), "down_proj": (HIDDEN, EXPERT_FF)}


def mtp_expert_specs(e: int) -> list[Spec]:
    p = "mtp.layers.0.mlp.experts.%d." % e
    S = []
    for proj, (n, k) in MTP_PROJ_SHAPES.items():
        S += [Spec(p + proj + ".weight", "U8", (n, k // 2)),
              Spec(p + proj + ".weight_scale", "F8_E4M3", (n, k // 16)),
              Spec(p + proj + ".weight_scale_2", "F32", ()),
              Spec(p + proj + ".input_scale", "F32", ())]
    return S


def mtp_specs() -> list[Spec]:
    S = [Spec(ck, dt, shape) for ck, _, dt, shape in mtp_dense_table()]
    for e in range(N_EXPERT):
        S += mtp_expert_specs(e)
    return S


def shard_layout(specs: list[Spec]) -> list[Spec]:
    """Byte order inside a shard: catalogue order, except that tensors whose size is not a multiple of 8 go last, so
    every other tensor starts 8-byte aligned (safetensors forbids padding between tensors; its own writer sorts by
    alignment for the same reason)."""
    return [s for s in specs if s.nbytes % 8 == 0] + [s for s in specs if s.nbytes % 8 != 0]


def num_shards(num_layers: int, mtp: bool) -> int:
    return 1 + num_layers + (1 if mtp else 0)


def shard_name(i: int, total: int) -> str:
    return "model-%05d-of-%05d.safetensors" % (i, total)


def shard_plan(num_layers: int, mtp: bool) -> list[tuple[str, list[Spec]]]:
    """[(file name, specs in file byte order)], shard 1 dense, shard l+2 experts of layer l, last shard MTP."""
    total = num_shards(num_layers, mtp)
    plan = [(shard_name(1, total), shard_layout(dense_specs(num_layers)))]
    for l in range(num_layers):
        plan.append((shard_name(l + 2, total), shard_layout(expert_specs(l))))
    if mtp:
        plan.append((shard_name(total, total), shard_layout(mtp_specs())))
    return plan


def expected_shard_of(name: str, num_layers: int, mtp: bool) -> int:
    """The shard rule of DESIGN 1.1, applied to a tensor name (1-based shard index)."""
    total = num_shards(num_layers, mtp)
    if name.startswith("mtp."):
        return total
    if ".mlp.experts." in name:
        return int(name.split(".")[2]) + 2
    return 1


# ----------------------------------------------------------------------------- config.json (DESIGN 1.2)
def layer_types(num_layers: int) -> list[str]:
    return ["indexed_attention" if is_qsa(l) else "linear_attention" for l in range(num_layers)]


def q8_module_names(num_layers: int, mtp: bool) -> list[str]:
    """The quantized_layers keys with the Q8_0 recipe (370 for the full model with MTP)."""
    out = []
    for l in range(num_layers):
        p = "model.layers.%d." % l
        if is_qsa(l):
            out.append(p + "self_attn.o_proj")
        else:
            out += [p + "linear_attn.in_proj_qkv", p + "linear_attn.in_proj_z", p + "linear_attn.out_proj"]
        out += [p + "mlp.shared_expert.gate_proj", p + "mlp.shared_expert.up_proj", p + "mlp.shared_expert.down_proj",
                p + "mlp.latent_down", p + "mlp.latent_up"]
    if mtp:
        out += ["mtp.fc_embedding", "mtp.fc_hidden"]
        out += ["mtp.layers.0.self_attn.%s" % x for x in ("q_proj", "k_proj", "v_proj", "o_proj")]
        out += ["mtp.layers.0.self_attn.indexer.index_qk_proj"]
        out += ["mtp.layers.0.mlp.shared_expert.%s" % x for x in ("gate_proj", "up_proj", "down_proj")]
    return out


def build_config(num_layers: int, mtp: bool, source_dir: str = SOURCE_DIR_DEFAULT) -> dict:
    ql = {name: dict(Q8_RECIPE) for name in q8_module_names(num_layers, mtp)}
    if mtp:
        ql[MTP_EXPERT_KEY] = dict(MTP_EXPERT_RECIPE)
    return {
        "architectures": ["Edge0ForCausalLM"],
        "model_type": "qwen4_exp_text",
        "dtype": "bfloat16",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "num_hidden_layers": num_layers,
        "layer_types": layer_types(num_layers),
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "hidden_act": "silu",
        "attention_bias": False,
        "rms_norm_eps": 1e-06,
        "max_position_embeddings": 262144,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25,
                            "mrope_section": [11, 11, 10], "mrope_interleaved": True},
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "output_gate_type": "sigmoid",
        "num_experts": N_EXPERT,
        "num_experts_per_tok": 10,
        "norm_topk_prob": True,
        "moe_intermediate_size": EXPERT_FF,
        "shared_expert_intermediate_size": EXPERT_FF,
        "hc_count": 4,
        "hc_lowrank": HC_LR,
        "ple_layer_ids": [2],
        "ple_embed_dim": HIDDEN,
        "ple_conv_kernel_size": 4,
        "ngram_size": 3,
        "heads_per_ngram": 8,
        "ngram_vocab_size_base": 20000000,
        "make_ngram_vocab_size_divisible_by": 128,
        "seed": 1234,
        "split_ngram_parts": 512,
        "ple_embedding_dtype": "float8_e4m3fn",
        "indexer_n_heads": 4,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 128,
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
        "mtp_num_hidden_layers": 1,
        "tie_word_embeddings": False,
        "bos_token_id": 248044,
        "pad_token_id": 248044,
        "eos_token_id": 248044,
        "quantization_config": {
            "quant_method": "edge0",
            "dense_format": "ggml_q8_0",
            "quantized_layers": ql,
        },
        "edge0": {
            "format": "edge0-vllm-v1",
            "latent_dim": LATENT,
            "latent_groups": LATENT_GROUPS,
            "expert_intermediate_size": EXPERT_FF,
            "expert_format": "modelopt_nvfp4_stacked",
            "ple_file": "ple-fp8.gguf",
            "ple_data_offset": PLE_DATA_OFFSET,
            "ple_rows": PLE_ROWS,
            "ple_row_bytes": PLE_ROW_BYTES,
            "ple_scale": PLE_SCALE,
            "profile_file": "profile.bin",
            "draft_vocab_file": "draft_vocab.json",
            "source_dir": source_dir,
        },
    }


GENERATION_CONFIG = {"eos_token_id": [248046, 248044], "pad_token_id": 248044, "temperature": 1.0, "top_k": 20,
                     "top_p": 0.95}

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
                   "chat_template.jinja")
COPIED_FILES = ("README.md", "LICENSE", "NOTICE")
THINK_OFF_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def config_layout(cfg: dict) -> tuple[int, bool]:
    """(num_layers, has_mtp) of a config.json written by the converter."""
    return int(cfg["num_hidden_layers"]), MTP_EXPERT_KEY in cfg.get("quantization_config", {}).get(
        "quantized_layers", {})


if __name__ == "__main__":
    for nl, mtp in ((48, True), (4, False)):
        plan = shard_plan(nl, mtp)
        tot = sum(s.nbytes for _, specs in plan for s in specs)
        print("num_layers=%d mtp=%s: %d shards, %d tensors, %d bytes (%.2f GB); shard 1 %d B, expert shard %d B%s"
              % (nl, mtp, len(plan), sum(len(s) for _, s in plan), tot, tot / 1e9,
                 sum(s.nbytes for s in plan[0][1]), sum(s.nbytes for s in plan[1][1]),
                 ", mtp shard %d B" % sum(s.nbytes for s in plan[-1][1]) if mtp else ""))
        print("  quantized_layers: %d (Q8_0 %d)" % (len(build_config(nl, mtp)["quantization_config"]["quantized_layers"]),
                                                    len(q8_module_names(nl, mtp))))
