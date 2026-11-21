"""Post-GPU_FREE probe of what the 26.11 image + RTX 5090 give for host-resident experts. Run ONLY after
/home/andychen/work/edge0-vllm-GPU_FREE exists:

  docker run --rm --gpus all --ipc=host --ulimit memlock=-1 -v /home/andychen/work/edge0-vllm:/workspace/edge0 \
      --entrypoint bash nvcr.io/nvidia/vllm:26.11-py3 -c 'python -I /workspace/edge0/tools/gpu_host_probe.py'

Checks, in order:
  1. CUDA works in the container (driver 595.99 = CUDA 13.2 vs image CUDA 13.5: minor-version compatibility).
  2. vLLM custom ops needed on sm_120 load (cutlass FP4 GEMM / grouped GEMM, Marlin MoE).
  3. Device attributes that decide how cold experts can live in host memory.
  4. TidalMoE-style VMM mapping: one VA range, first half device pages, second half host pages (cuMemCreate HOST_NUMA).
  5. Bandwidths (GB/s): HBM read, H2D copy-engine copy from pinned memory, SM zero-copy read of pinned host memory
     (UVA), SM zero-copy read of VMM host pages, and an expert-sized gather (1,935,376-byte rows) from host to VRAM.
Uses <= 2 GiB pinned host memory and <= 3 GiB VRAM.
"""
import os
import time

import torch

GIB = 1 << 30
REC = 1_935_376          # one Edge0 NVFP4 latent expert record (experts.bin)


def line(k, v):
    print(f"{k:56s} {v}", flush=True)


def timeit(fn, reps=5):
    fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps / 1e3


print("== 1. CUDA")
line("torch", torch.__version__)
line("torch.version.cuda", torch.version.cuda)
line("cuda available", torch.cuda.is_available())
dev = torch.device("cuda:0")
p = torch.cuda.get_device_properties(dev)
line("device", f"{p.name} sm_{p.major}{p.minor} {p.total_memory / GIB:.1f} GiB, {p.multi_processor_count} SMs")
try:
    from cuda.bindings import driver as cu
    cu.cuInit(0)
    line("driver CUDA version", cu.cuDriverGetVersion()[1])
except Exception as e:  # noqa: BLE001
    line("cuda.bindings", f"ERR {e}")
x = torch.randn(1024, 1024, device=dev)
line("matmul ok", bool(torch.isfinite(x @ x).all()))
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _add(a, b, n, B: tl.constexpr):
        o = tl.program_id(0) * B + tl.arange(0, B)
        m = o < n
        tl.store(b + o, tl.load(a + o, mask=m) + 1, mask=m)
    a = torch.zeros(4096, device=dev)
    b = torch.empty_like(a)
    _add[(4,)](a, b, 4096, B=1024)
    line("triton kernel ok", bool((b == 1).all()))
except Exception as e:  # noqa: BLE001
    line("triton", f"ERR {type(e).__name__}: {e}")

print("== 2. vLLM ops")
try:
    import vllm._custom_ops as ops  # noqa: F401
    for ns in ("_C", "_moe_C"):
        o = getattr(torch.ops, ns)
        have = [n for n in ("cutlass_scaled_fp4_mm", "cutlass_fp4_group_mm", "scaled_fp4_quant",
                            "scaled_fp4_experts_quant", "cutlass_group_gemm_supported", "moe_wna16_marlin_gemm",
                            "topk_softmax", "moe_align_block_size", "shuffle_rows") if hasattr(o, n)]
        line(f"torch.ops.{ns}", have)
    if hasattr(torch.ops._C, "cutlass_group_gemm_supported"):
        line("cutlass_group_gemm_supported(120)", torch.ops._C.cutlass_group_gemm_supported(120))
except Exception as e:  # noqa: BLE001
    line("vllm ops", f"ERR {type(e).__name__}: {e}")
try:
    from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe, has_flashinfer_b12x_moe
    line("flashinfer cutlass fused moe / b12x", (has_flashinfer_cutlass_fused_moe(), has_flashinfer_b12x_moe()))
except Exception as e:  # noqa: BLE001
    line("flashinfer", f"ERR {e}")

print("== 3. device attributes")
A = cu.CUdevice_attribute
for n in ("CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED",
          "CU_DEVICE_ATTRIBUTE_HOST_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED",
          "CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED",
          "CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID", "CU_DEVICE_ATTRIBUTE_NUMA_CONFIG",
          "CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS",
          "CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES",
          "CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM",
          "CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED", "CU_DEVICE_ATTRIBUTE_INTEGRATED",
          "CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT", "CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED"):
    try:
        line(n.replace("CU_DEVICE_ATTRIBUTE_", ""), cu.cuDeviceGetAttribute(getattr(A, n), 0)[1])
    except Exception as e:  # noqa: BLE001
        line(n, f"ERR {e}")


def ck(ret):
    err, *vals = ret if isinstance(ret, tuple) else (ret,)
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{err}: {cu.cuGetErrorString(err)[1]}")
    return vals[0] if len(vals) == 1 else vals


class _CAI:
    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 3,
                                         "strides": None}


def vmm_split(nbytes_dev, nbytes_host, host_loc):
    """One VA range: [0, dev) device pages, [dev, dev+host) host pages (the TidalMoE _alloc_segments layout)."""
    def prop(t, i):
        q = cu.CUmemAllocationProp()
        q.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        q.location.type = t
        q.location.id = i
        return q
    dp = prop(cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE, 0)
    hp = prop(host_loc, 0)
    rec = cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
    g = max(int(ck(cu.cuMemGetAllocationGranularity(dp, rec))), int(ck(cu.cuMemGetAllocationGranularity(hp, rec))))
    nd, nh = -(-nbytes_dev // g) * g, -(-nbytes_host // g) * g
    base = int(ck(cu.cuMemAddressReserve(nd + nh, g, 0, 0)))
    hd = ck(cu.cuMemCreate(nd, dp, 0))
    hh = ck(cu.cuMemCreate(nh, hp, 0))
    ck(cu.cuMemMap(cu.CUdeviceptr(base), nd, 0, hd, 0))
    ck(cu.cuMemMap(cu.CUdeviceptr(base + nd), nh, 0, hh, 0))
    acc = cu.CUmemAccessDesc()
    acc.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    acc.location.id = 0
    acc.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    ck(cu.cuMemSetAccess(cu.CUdeviceptr(base), nd + nh, [acc], 1))
    cai = _CAI(base, nd + nh)
    t = torch.as_tensor(cai, device=dev)
    return t, nd, nh, g, (cai, hd, hh)


print("== 4. VMM device+host on one VA range")
vmm = None
for loc in ("CU_MEM_LOCATION_TYPE_HOST_NUMA", "CU_MEM_LOCATION_TYPE_HOST"):
    try:
        t, nd, nh, g, keep = vmm_split(256 << 20, 1 << 30, getattr(cu.CUmemLocationType, loc))
        t[: nd].fill_(1)
        t[nd:].fill_(2)
        torch.cuda.synchronize()
        ok = int(t[:nd].sum()) == nd and int(t[nd:].to(torch.int64).sum()) == 2 * nh
        line(f"vmm split with {loc}", f"OK={ok} granularity={g} dev={nd >> 20}MiB host={nh >> 20}MiB")
        vmm = (t, nd, nh, keep)
        break
    except Exception as e:  # noqa: BLE001
        line(f"vmm split with {loc}", f"FAILED {e}")

print("== 5. bandwidth (GB/s)")
n = 1 << 30
d = torch.empty(n, dtype=torch.uint8, device=dev).fill_(1)
line("HBM read (sum over 1 GiB)", f"{n / timeit(lambda: d.view(torch.int32).sum()) / 1e9:.0f}")
h = torch.empty(n, dtype=torch.uint8, pin_memory=True).fill_(1)
line("H2D copy engine, pinned 1 GiB", f"{n / timeit(lambda: d.copy_(h, non_blocking=True)) / 1e9:.1f}")
line("D2H copy engine, pinned 1 GiB", f"{n / timeit(lambda: h.copy_(d, non_blocking=True)) / 1e9:.1f}")
try:
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    hv = get_accelerator_view_from_cpu_tensor(h)
    line("SM zero-copy read, pinned (UVA view, sum)", f"{n / timeit(lambda: hv.view(torch.int32).sum()) / 1e9:.1f}")
    rows = n // REC
    hv2 = hv[: rows * REC].view(rows, REC)
    for k in (8, 64, 256):
        idx = torch.randperm(rows, device=dev)[:k]
        out = torch.empty(k, REC, dtype=torch.uint8, device=dev)
        sec = timeit(lambda: torch.index_select(hv2, 0, idx, out=out))
        line(f"SM gather of {k} expert records host->VRAM", f"{k * REC / sec / 1e9:.1f} ({sec * 1e3:.2f} ms)")
    for k in (8, 64, 256):
        idx = torch.randperm(rows)[:k].tolist()
        out = torch.empty(k, REC, dtype=torch.uint8, device=dev)
        hrows = h[: rows * REC].view(rows, REC)

        def ce():
            for j, r in enumerate(idx):
                out[j].copy_(hrows[r], non_blocking=True)
        sec = timeit(ce)
        line(f"copy-engine copies of {k} expert records", f"{k * REC / sec / 1e9:.1f} ({sec * 1e3:.2f} ms)")
except Exception as e:  # noqa: BLE001
    line("UVA", f"ERR {type(e).__name__}: {e}")
if vmm is not None:
    t, nd, nh, _ = vmm
    hp = t[nd:]
    line("SM zero-copy read, VMM host pages (sum)", f"{nh / timeit(lambda: hp.view(torch.int32).sum()) / 1e9:.1f}")
print("== done", time.strftime("%F %T"), "pid", os.getpid())
