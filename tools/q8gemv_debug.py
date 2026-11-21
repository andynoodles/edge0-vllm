"""Tiny interpreter debug of edge0_vllm.q8gemv (TRITON_INTERPRET=1, CPU)."""
import sys

import torch

sys.path.insert(0, "/workspace/edge0/plugin")
sys.path.insert(0, "/workspace/edge0/plugin/tests")
from test_q8gemv import _pack_like_b12x, _rand_blocks, _ref  # noqa: E402

from edge0_vllm import q8gemv  # noqa: E402

for n, k, m, dot in [(128, 32, 1, False), (128, 32, 1, True), (128, 64, 2, True), (256, 128, 3, True)]:
    gen = torch.Generator().manual_seed(0)
    b = _rand_blocks(n, k, "cpu", gen)
    v, meta = _pack_like_b12x(b)
    x = torch.randn((m, k), generator=gen).to(torch.bfloat16)
    y = q8gemv.q8_gemv(x, v, meta, n, k, use_dot=dot)
    r = _ref(x, b)
    print(n, k, m, dot, "y", y[0, :4].tolist(), "ref", r[0, :4].tolist(),
          "err", ((y.float() - r).abs().max() / r.abs().max()).item())
