"""Whole-model checks without weights in memory: vLLM resolves the checkpoint to Edge0ForCausalLM through the plugin,
builds the model on the ``meta`` device (real CUDA platform, so vLLM's own backend selection runs), and the
checkpoint's tensors - shapes and dtypes from the safetensors headers, no data read - load through
``Edge0ForCausalLM.load_weights`` with every parameter accounted for.

Needs CUDA (vLLM's platform detection) and EDGE0_CKPT (e.g. /models/Edge0-145B-vllm-L4 or the full checkpoint).
"""
import json
import os

import pytest
import torch

from helpers import CKPT, init_single_process_dist

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and CKPT), reason="needs CUDA and EDGE0_CKPT")

_ST_DTYPES = {"U8": torch.uint8, "I8": torch.int8, "I32": torch.int32, "I64": torch.int64, "F16": torch.float16,
              "BF16": torch.bfloat16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn}


def _meta_weights(ckpt):
    """(name, meta tensor) for every checkpoint tensor, in file order like vLLM's safetensors iterator."""
    from safetensors import safe_open
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    for fn in sorted(set(idx.values())):
        with safe_open(os.path.join(ckpt, fn), framework="pt") as f:
            for name in f.keys():
                sl = f.get_slice(name)
                yield name, torch.empty(sl.get_shape(), dtype=_ST_DTYPES[sl.get_dtype()], device="meta")


def test_meta_build_and_load_names(monkeypatch):
    for k, v in {"EDGE0_EXPERT_VRAM_GIB": "0.5", "EDGE0_PLACEMENT": "profile", "EDGE0_EMBED_HOST": "0",
                 "EDGE0_DENSE": "bf16"}.items():
        monkeypatch.setenv(k, v)
    from edge0_vllm import config as C
    C.reset_settings()
    import edge0_vllm
    edge0_vllm.register()
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    ea = EngineArgs(model=CKPT, quantization="edge0", dtype="bfloat16", max_model_len=4096, max_num_seqs=4,
                    max_num_batched_tokens=512, enforce_eager=True, skip_tokenizer_init=True,
                    gpu_memory_utilization=0.5, engram_config={"cpu_offload": False, "host_file_gather": True})
    vc = ea.create_engine_config()
    assert vc.model_config.architecture == "Edge0ForCausalLM"
    assert vc.model_config.quantization == "edge0" and vc.quant_config.get_name() == "edge0"
    assert vc.model_config.is_hybrid
    with set_current_vllm_config(vc):
        init_single_process_dist(vc)
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
        from vllm.model_executor.model_loader.utils import initialize_model
        from vllm.utils.torch_utils import set_default_torch_dtype
        with set_default_torch_dtype(vc.model_config.dtype), torch.device("meta"):   # as BaseModelLoader does
            model = initialize_model(vllm_config=vc)
        from edge0_vllm.latent_moe import Edge0LatentMoeBlock
        n_layers = vc.model_config.hf_text_config.num_hidden_layers
        assert type(model).__name__ == "Edge0ForCausalLM"
        assert all(isinstance(layer.mlp, Edge0LatentMoeBlock) for layer in model.model.layers)
        assert model.num_moe_layers == 0                       # no stock FusedMoE in the target model
        assert len(model.edge0.blocks) == n_layers
        assert sum(b.pool.n_hot for b in model.edge0.blocks) == model.edge0.hot_slots
        mlp0 = model.model.layers[0].mlp
        assert isinstance(mlp0.latent_down.quant_method, UnquantizedLinearMethod)   # Q8 dequantized at load
        assert mlp0.latent_down.weight.dtype == torch.bfloat16 and mlp0.latent_down.weight.shape == (14336, 2560)
        assert model.model.layers[0].linear_attn.in_proj_qkvz.weight.dtype == torch.bfloat16
        kinds = {type(m.quant_method).__name__ for m in model.modules() if hasattr(m, "quant_method")
                 and m.quant_method is not None}
        print("quant methods in use:", sorted(kinds))

        # Meta tensors carry no data: make vLLM's copy helper shape-check only (some small params, e.g. the PLE
        # file-gather state, are created on a real device even under the meta default device).
        import importlib
        real_dev = []

        def copy_weight_meta(dst, src):
            assert dst.numel() == src.numel() or src.numel() == 1, (tuple(dst.shape), tuple(src.shape))
            if dst.device.type != "meta":
                real_dev.append(tuple(dst.shape))
            return dst
        for mod in ("vllm.model_executor.utils", "vllm.model_executor.parameter", "vllm.model_executor.layers.linear",
                    "vllm.model_executor.layers.vocab_parallel_embedding", "vllm.model_executor.model_loader.weight_utils"):
            m = importlib.import_module(mod)
            if hasattr(m, "copy_weight_"):
                monkeypatch.setattr(m, "copy_weight_", copy_weight_meta)
        loaded = model.load_weights(_meta_weights(CKPT))
        print("params on a real device during the meta load:", real_dev)
        params = {n for n, _ in model.named_parameters()}
        missing = sorted(params - loaded)
        print(f"{len(params)} parameters, {len(loaded)} names loaded; not loaded: {missing}")
        # The PLE table is served from ple-fp8.gguf by the file-gather backend, never loaded as a tensor.
        assert set(missing) <= {"model.layers.1.ple.ple_embedding.ngram_embedding.weight"}, missing
        for b in model.edge0.blocks:
            assert b.pool.missing() == [], (b.prefix, b.pool.missing())
        bufs = {n for n, _ in model.named_buffers()}
        assert all(f"model.layers.{l}.mlp.expert_group" in bufs and f"model.layers.{l}.mlp.expert_group" in loaded
                   for l in range(n_layers))
        model.process_weights_after_loading()                  # meta: finalize bookkeeping only
