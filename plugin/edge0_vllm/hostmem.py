"""Pinned, device-mapped host memory for cold experts (DESIGN.md 2.6).

Each allocation returns a ``HostBuffer`` with a CPU ``uint8`` view (host writes) and a CUDA ``uint8`` view of the
same bytes (UVA: kernels read it over PCIe, zero-copy). Backends (``EDGE0_COLD_ALLOC``):

* ``host_register`` (default): anonymous ``mmap`` (+ ``MADV_HUGEPAGE``), then ``cudaHostRegister(Portable|Mapped)``.
  Exact size: unlike ``torch.empty(pin_memory=True)``, whose caching host allocator rounds up to a power of two.
* ``host_alloc``: ``cudaHostAlloc(Portable|Mapped)`` through cuda-python.
* ``torch_pinned``: vLLM's own UVA-offload mechanism (``UVAOffloader``): ``torch.empty(pin_memory=True)`` plus
  ``get_accelerator_view_from_cpu_tensor``. The simple fallback if the other two misbehave; costs up to ~2x RAM for
  large blocks because of the power-of-two rounding, so it is meant for small/debug pools.

On a CPU-only process (tests) ``device`` may be ``cpu`` or ``meta``; then the CPU view doubles as the "device"
view (``cpu``) or a meta tensor is returned without allocating (``meta``).
"""
from __future__ import annotations

import ctypes
import mmap
from dataclasses import dataclass

import numpy as np
import torch

_KEEP: list = []        # owners that must outlive every view (process lifetime)
cudaHostRegisterPortable = 1
cudaHostRegisterMapped = 2


@dataclass
class HostBuffer:
    cpu: torch.Tensor          # uint8 [nbytes] host view
    dev: torch.Tensor          # uint8 [nbytes] device view (UVA) of the same bytes
    backend: str

    @property
    def nbytes(self) -> int:
        return self.cpu.numel()

    def carve(self, offset: int, nbytes: int, dtype: torch.dtype, shape: tuple[int, ...]):
        """(cpu, dev) typed views of bytes [offset, offset + nbytes)."""
        def view(t):
            return t[offset:offset + nbytes].view(dtype).view(shape)
        return view(self.cpu), view(self.dev)


def _cuda_view(cpu: torch.Tensor) -> torch.Tensor:
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    return get_accelerator_view_from_cpu_tensor(cpu)


def _check_cuda(err, what: str) -> None:
    code = int(err[0] if isinstance(err, tuple) else err)
    if code != 0:
        raise RuntimeError(f"{what} failed with CUDA error {code}")


def _alloc_host_register(nbytes: int) -> HostBuffer:
    buf = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE)
    if hasattr(mmap, "MADV_HUGEPAGE"):
        try:
            buf.madvise(mmap.MADV_HUGEPAGE)
        except OSError:
            pass
    cpu = torch.from_numpy(np.frombuffer(buf, dtype=np.uint8))
    _check_cuda(torch.cuda.cudart().cudaHostRegister(cpu.data_ptr(), nbytes,
                                                     cudaHostRegisterPortable | cudaHostRegisterMapped),
                f"cudaHostRegister({nbytes / 2**30:.2f} GiB)")
    _KEEP.append(buf)
    return HostBuffer(cpu=cpu, dev=_cuda_view(cpu), backend="host_register")


def _alloc_host_alloc(nbytes: int) -> HostBuffer:
    from cuda.bindings import runtime as cudart
    err, ptr = cudart.cudaHostAlloc(nbytes, cudaHostRegisterPortable | cudaHostRegisterMapped)
    _check_cuda(err, f"cudaHostAlloc({nbytes / 2**30:.2f} GiB)")
    raw = (ctypes.c_uint8 * nbytes).from_address(int(ptr))
    cpu = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8))
    _KEEP.append(raw)
    return HostBuffer(cpu=cpu, dev=_cuda_view(cpu), backend="host_alloc")


def _alloc_torch_pinned(nbytes: int) -> HostBuffer:
    cpu = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    return HostBuffer(cpu=cpu, dev=_cuda_view(cpu), backend="torch_pinned")


def alloc(nbytes: int, backend: str, device: torch.device | str = "cuda") -> HostBuffer:
    """``nbytes`` of pinned host memory with a device view on ``device``."""
    device = torch.device(device)
    nbytes = int(nbytes)
    if device.type == "meta":
        t = torch.empty(nbytes, dtype=torch.uint8, device="meta")
        return HostBuffer(cpu=t, dev=t, backend="meta")
    if device.type == "cpu":
        t = torch.zeros(nbytes, dtype=torch.uint8)
        return HostBuffer(cpu=t, dev=t, backend="cpu")
    if nbytes == 0:
        return HostBuffer(cpu=torch.empty(0, dtype=torch.uint8), dev=torch.empty(0, dtype=torch.uint8, device=device),
                          backend=backend)
    if backend == "host_register":
        return _alloc_host_register(nbytes)
    if backend == "host_alloc":
        return _alloc_host_alloc(nbytes)
    if backend == "torch_pinned":
        return _alloc_torch_pinned(nbytes)
    raise ValueError(f"unknown host allocator {backend!r}")


def is_host_memory(t: torch.Tensor) -> bool | None:
    """True when a CUDA tensor's memory is host memory (UVA view), None if it cannot be determined."""
    if not t.is_cuda:
        return None
    try:
        from cuda.bindings import driver as cu
        err, mt = cu.cuPointerGetAttribute(cu.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMORY_TYPE, t.data_ptr())
        if err != cu.CUresult.CUDA_SUCCESS:
            return None
        return int(mt) == int(cu.CUmemorytype.CU_MEMORYTYPE_HOST)
    except Exception:  # noqa: BLE001
        return None
