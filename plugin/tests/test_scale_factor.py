"""``marlin_nvfp4.scale_factor`` (chunked, round 2 agent exceed-2) == vLLM's ``_nvfp4_compute_scale_factor`` on the bf16
cast of the whole stacked tensor: random E4M3 block scales of a layer's shape at several magnitudes, plus zeros,
negative entries and the E4M3 maximum. Runs on CUDA when available, else on the CPU."""
import pytest
import torch

vllm_fp4 = pytest.importorskip("vllm.model_executor.layers.quantization.utils.marlin_utils_fp4")


def _dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("scale,shape", [(1.0, (512, 1280, 112)), (0.02, (512, 1792, 40)), (30.0, (64, 128, 16)),
                                         (1e-3, (64, 128, 16))])
def test_scale_factor_matches_vllm(scale, shape):
    from edge0_vllm import marlin_nvfp4 as mn
    g = torch.Generator().manual_seed(int(scale * 1000) + shape[0])
    x = (torch.rand(shape, generator=g) * scale).to(torch.float8_e4m3fn)
    want = float(vllm_fp4._nvfp4_compute_scale_factor(x.to(_dev()).to(mn.PARAM_DTYPE), mn.PARAM_DTYPE))
    got = mn.scale_factor(x, _dev())
    assert got == want, (got, want)


def test_scale_factor_edge_cases():
    from edge0_vllm import marlin_nvfp4 as mn
    z = torch.zeros(8, 16, 4).to(torch.float8_e4m3fn)
    assert mn.scale_factor(z, _dev()) == float(vllm_fp4._nvfp4_compute_scale_factor(z.to(_dev()).to(mn.PARAM_DTYPE),
                                                                                     mn.PARAM_DTYPE)) == 1.0
    n = (torch.rand(8, 16, 4) * 2 - 1).to(torch.float8_e4m3fn)          # negative entries are masked out by vLLM
    assert mn.scale_factor(n, _dev(), chunk=3) == float(
        vllm_fp4._nvfp4_compute_scale_factor(n.to(_dev()).to(mn.PARAM_DTYPE), mn.PARAM_DTYPE))
    big = torch.full((4, 4, 4), 448.0).to(torch.float8_e4m3fn)
    assert mn.scale_factor(big, _dev()) == float(
        vllm_fp4._nvfp4_compute_scale_factor(big.to(_dev()).to(mn.PARAM_DTYPE), mn.PARAM_DTYPE))
