"""GPU: one real ``Edge0LatentMoeBlock`` (layer 0 of EDGE0_CKPT) built from vLLM's layers, loaded through vLLM's
weight mapper + AutoWeightsLoader (Q8_0 -> BF16 dequant loader, fused shared-expert gate_up shards, expert pools),
run through the registered ``vllm::edge0_latent_moe`` op, against Strata's float32 oracle
(``strata_formats.latent_moe_reference`` on the *source* model files, shared expert included).
A block-level unit test: the decoder model itself is not started."""
import json
import os

import numpy as np
import pytest
import torch

from helpers import CKPT, SRC_MODEL, have_src_model, init_single_process_dist, make_vllm_config, strata_formats

sf = strata_formats()
pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and CKPT and sf is not None and have_src_model()),
                                reason="needs CUDA, EDGE0_CKPT and the source model folder")
PREFIX = "model.layers.0.mlp"


def _block_weights(ckpt):
    from safetensors import safe_open
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    for name, fn in sorted(idx.items()):
        if name.startswith(PREFIX + "."):
            with safe_open(os.path.join(ckpt, fn), framework="pt") as f:
                yield name, f.get_tensor(name)


@pytest.mark.parametrize("n_hot_gib", ["0.05", "0"])
def test_block_vs_strata_oracle(monkeypatch, n_hot_gib):
    for k, v in {"EDGE0_EXPERT_VRAM_GIB": n_hot_gib, "EDGE0_PLACEMENT": "profile", "EDGE0_EMBED_HOST": "0"}.items():
        monkeypatch.setenv(k, v)
    from edge0_vllm import config as C, runtime
    C.reset_settings()
    vc = make_vllm_config(CKPT)
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.models.utils import AutoWeightsLoader
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpModel
    from vllm.utils.torch_utils import set_default_torch_dtype
    with set_current_vllm_config(vc):
        init_single_process_dist(vc)
        runtime.setup(vc)
        from edge0_vllm.latent_moe import Edge0LatentMoeBlock
        with set_default_torch_dtype(torch.bfloat16), torch.device("cuda"):
            block = Edge0LatentMoeBlock(vc, prefix=PREFIX)
        n_hot = block.pool.n_hot
        experts, rest = {}, []
        for name, t in Qwen4ExpModel.hf_to_vllm_mapper.apply(_block_weights(CKPT)):
            short = name[len(PREFIX) + 1:]
            if short.startswith("experts."):
                block.load_expert_tensor(short[len("experts."):], t)
            else:
                rest.append((short, t))
        loaded = AutoWeightsLoader(block).load_weights(rest)
        assert {"gate.weight", "latent_down.weight", "latent_up.weight", "shared_expert.gate_up_proj.weight",
                "shared_expert.down_proj.weight", "shared_expert_gate.weight", "expert_group"} <= set(loaded), loaded
        block.finalize()
        # dequantized D equals the oracle's Q8 rows
        m = sf.EdgeModel(SRC_MODEL)
        D1, _ = m.latent_du(0, 1)
        assert torch.equal(block.latent_down.weight[1792:2 * 1792].float().cpu(),
                           torch.from_numpy(D1).to(torch.bfloat16).float())
        rng = np.random.default_rng(0)
        xs = rng.standard_normal((5, 2560)).astype(np.float32) * 0.6
        with torch.no_grad():
            out = block(torch.from_numpy(xs).to("cuda", torch.bfloat16)).float().cpu().numpy()
        same_ids = 0
        for i in range(xs.shape[0]):
            ref = sf.latent_moe_reference(m, 0, xs[i], with_shared=True)
            rel = np.linalg.norm(out[i] - ref["out"]) / np.linalg.norm(ref["out"])
            logits = block.gate(torch.from_numpy(xs[i:i + 1]).to("cuda", torch.bfloat16))[0]
            from edge0_vllm.kernels import route_topk_softmax
            ids = route_topk_softmax(logits.float(), 10)[1][0].cpu().tolist()
            same_ids += sorted(ids) == sorted(ref["ids"].tolist())
            print(f"n_hot={n_hot} token {i}: rel err vs Strata oracle {rel:.2e}, same top-10 {sorted(ids) == sorted(ref['ids'].tolist())}")
            assert rel < 3e-2
        assert same_ids >= 4
