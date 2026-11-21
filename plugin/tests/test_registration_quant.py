"""Plugin registration (entry point, model registry, arch tables) and the ``edge0`` quantization config."""
import torch

ARCH = "Edge0ForCausalLM"


def _contract_quant_config(num_layers=4, mtp=False):
    rec = {"quant_algo": "Q8_0", "group_size": 32, "block_payload_bytes": 34, "packing": "ggml"}
    layers = {}
    for l in range(num_layers):
        if l % 4 != 3:
            for n in ("in_proj_qkv", "in_proj_z", "out_proj"):
                layers[f"model.layers.{l}.linear_attn.{n}"] = dict(rec)
        else:
            layers[f"model.layers.{l}.self_attn.o_proj"] = dict(rec)
        for n in ("gate_proj", "up_proj", "down_proj"):
            layers[f"model.layers.{l}.mlp.shared_expert.{n}"] = dict(rec)
        for n in ("latent_down", "latent_up"):
            layers[f"model.layers.{l}.mlp.{n}"] = dict(rec)
    if mtp:
        layers["mtp.layers.0.mlp.experts"] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    return {"quant_method": "edge0", "dense_format": "ggml_q8_0", "quantized_layers": layers}


def test_entry_point_and_register():
    from importlib.metadata import entry_points
    eps = [e for e in entry_points(group="vllm.general_plugins") if e.name == "edge0"]
    assert eps and eps[0].value == "edge0_vllm:register"
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    from vllm.model_executor.models import ModelRegistry
    assert ARCH in ModelRegistry.get_supported_archs()
    import vllm.config.engram as engram
    import vllm.config.vllm as vcfg
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP
    assert MODELS_CONFIG_MAP[ARCH] is MODELS_CONFIG_MAP["Qwen4ExpForCausalLM"]
    assert ARCH in vcfg.DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES
    assert engram._NGRAM_LAYER_FIELDS[ARCH] == "ple_layer_ids"
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS, get_quantization_config
    from edge0_vllm.quant import Edge0QuantConfig
    assert "edge0" in QUANTIZATION_METHODS and get_quantization_config("edge0") is Edge0QuantConfig


def test_quant_config_dispatch(monkeypatch):
    from edge0_vllm import config as C
    from edge0_vllm.quant import Edge0QuantConfig
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    cfg_dict = _contract_quant_config(num_layers=8, mtp=True)
    assert Edge0QuantConfig.override_quantization_method(cfg_dict, None) == "edge0"
    assert Edge0QuantConfig.override_quantization_method({"quant_method": "modelopt"}, None) is None
    # ModelOpt must not hijack it either (no top-level quant_algo)
    from vllm.model_executor.layers.quantization.modelopt import ModelOptMixedPrecisionConfig
    assert ModelOptMixedPrecisionConfig.override_quantization_method(cfg_dict, None) is None
    try:
        Edge0QuantConfig.from_config({**cfg_dict, "quant_algo": "MIXED_PRECISION"})
        raise AssertionError("top-level quant_algo must be rejected")
    except ValueError:
        pass
    qc = Edge0QuantConfig.from_config(cfg_dict)
    assert qc.get_name() == "edge0"
    from edge0_vllm.model import Edge0ForCausalLM
    qc.packed_modules_mapping = Edge0ForCausalLM.packed_modules_mapping
    r = qc._resolve_quant_algo
    assert r("model.layers.0.linear_attn.in_proj_qkvz") == "Q8_0"       # fused [in_proj_qkv, in_proj_z]
    assert r("model.layers.0.linear_attn.out_proj") == "Q8_0"
    assert r("model.layers.0.linear_attn.in_proj_ba") is None           # BF16
    assert r("model.layers.3.self_attn.o_proj") == "Q8_0"
    assert r("model.layers.3.self_attn.qkv_proj") is None               # BF16 q/k/v + indexer
    assert r("model.layers.5.mlp.shared_expert.gate_up_proj") == "Q8_0"
    assert r("model.layers.5.mlp.shared_expert.down_proj") == "Q8_0"
    assert r("model.layers.5.mlp.latent_down") == "Q8_0" and r("model.layers.5.mlp.latent_up") == "Q8_0"
    assert r("model.layers.1.ple.kv_proj") is None
    assert r("model.layers.5.attn_hyper_connection.input_mix_weight_down_block_inject") is None
    assert r("mtp.layers.0.mlp.experts") == "W4A16_NVFP4"

    monkeypatch.setenv("EDGE0_DENSE", "bf16")
    C.reset_settings()
    from edge0_vllm.quant import make_q8_dequant_linear_method
    m = make_q8_dequant_linear_method()
    assert type(m) is UnquantizedLinearMethod                           # keeps vLLM's sm_120 skinny-GEMM plans

    import vllm.model_executor.parameter as vparam
    monkeypatch.setattr(vparam, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(vparam, "get_tensor_model_parallel_world_size", lambda: 1)
    layer = torch.nn.Module()
    seen = {}

    def loader(param, w, *a):
        seen["w"] = w
        param.data.copy_(w)

    m.create_weights(layer, 64, [32, 32], 64, 64, torch.bfloat16, weight_loader=loader)
    from edge0_vllm.q8 import quant_q8_0, dequant_q8_0
    b = quant_q8_0(torch.randn(64, 64))
    layer.weight.weight_loader(layer.weight, b)
    assert seen["w"].dtype == torch.bfloat16 and torch.equal(layer.weight.data, dequant_q8_0(b))
    C.reset_settings()
