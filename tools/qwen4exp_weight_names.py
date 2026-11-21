"""List the HF checkpoint tensor names/shapes a Qwen4Exp (Qwen3.8-Flash-Next) model has, on the meta device.

Shapes follow the Edge0-145B skeleton GGUF (hidden 2560, 24 q heads / 2 kv heads x 256, GDN 16 k / 48 v heads x 128,
HC 4 streams rank 320, QSA indexer 4 x 128, shared expert 640, PLE on layer 1: 16 n-gram heads x 160 = 2560).
Routed experts are reduced to 4 (names only) and the PLE vocab base to 1000 (meta tensors anyway).

docker run --rm --memory 3g --cpus 2 -v /home/andychen/work/edge0-vllm:/workspace/edge0 --entrypoint bash \
    nvcr.io/nvidia/vllm:26.11-py3 -c 'nice -n 19 python -I /workspace/edge0/tools/qwen4exp_weight_names.py'
"""
import re

import torch
from transformers import Qwen4ExpTextConfig
from transformers.models.qwen4_exp import modeling_qwen4_exp as m

cfg = Qwen4ExpTextConfig(
    vocab_size=248320, hidden_size=2560, num_hidden_layers=4, num_attention_heads=24, num_key_value_heads=2,
    head_dim=256, linear_num_key_heads=16, linear_num_value_heads=48, linear_key_head_dim=128,
    linear_value_head_dim=128, linear_conv_kernel_dim=4, moe_intermediate_size=640,
    shared_expert_intermediate_size=640, num_experts=4, num_experts_per_tok=2, hc_count=4, hc_lowrank=320,
    ple_layer_ids=[1], ple_embed_dim=2560, ngram_size=3, heads_per_ngram=8, ngram_vocab_size_base=1000,
    indexer_n_heads=4, indexer_kv_heads=1, indexer_head_dim=128, indexer_budget=2048, indexer_compress_ratio=4,
    eos_token_id=248044, max_position_embeddings=262144,
    rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                     "mrope_section": [11, 11, 10], "mrope_interleaved": True},
)
print("layer_types", cfg.layer_types)
cls = getattr(m, "Qwen4ExpForCausalLM", None) or getattr(m, "Qwen4ExpTextForCausalLM", None)
print("class", cls.__name__, [n for n in dir(m) if n.startswith("Qwen4Exp") and "For" in n])
with torch.device("meta"):
    model = cls(cfg)
for k, v in model.state_dict().items():
    mm = re.search(r"layers\.(\d+)\.", k)
    if mm and int(mm.group(1)) not in (0, 3):
        continue
    if re.search(r"experts\.(\d+)\.", k) and not re.search(r"experts\.0\.", k):
        continue
    print(f"{k:80s} {str(v.dtype):16s} {list(v.shape)}")
for name, buf in model.named_buffers():
    if "layers.0." in name or "layers.3." in name:
        print("buffer", name, list(buf.shape))
