// EDGE0_PLE_DEFER=1: the in-graph half of the deferred PLE staging (ple_reader.cpp has the host half).
//
// ple_wait_copy: every CTA's thread 0 spins (with __nanosleep) until the mapped flag reaches *expect, the step's
// sequence number that prepare_inputs wrote into device memory with a fill kernel before the graph launch; then the
// CTA copies its share of the rows from pinned, device-mapped host memory (ld.global.cv: fetched again on every
// replay) into the device staging buffer the PLE forward reads. expect == 0 (no deferred step yet, dummy runs)
// never waits. A wait longer than timeout_ns gives up (stale rows; counted in stats[2]) rather than hang the GPU.
// stats (mapped host memory, written by CTA 0 only): [0] calls, [1] ns spent waiting, [2] timeouts, [3] calls that
// waited at all.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>

namespace e0ple {

__device__ __forceinline__ uint64_t globaltimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void __launch_bounds__(256) ple_wait_copy_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst,
                                                            int64_t n16, const volatile uint64_t* flag,
                                                            const int64_t* expect, uint64_t timeout_ns,
                                                            volatile uint64_t* stats) {
    if (threadIdx.x == 0) {
        const int64_t want = *expect;
        if (want > 0 && (int64_t) *flag < want) {
            const uint64_t t0 = globaltimer();
            bool timed_out = false;
            while ((int64_t) *flag < want) {
                __nanosleep(100);
                if (globaltimer() - t0 > timeout_ns) {
                    timed_out = true;
                    break;
                }
            }
            if (blockIdx.x == 0) {
                stats[1] += globaltimer() - t0;
                stats[3] += 1;
                if (timed_out) stats[2] += 1;
            }
        }
        if (blockIdx.x == 0) stats[0] += 1;
        __threadfence_system();
    }
    __syncthreads();
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (int64_t) gridDim.x * blockDim.x)
        dst[i] = __ldcv(src + i);
}

void cuda_wait_copy(torch::Tensor src, torch::Tensor dst, int64_t nbytes, int64_t flag_dev, torch::Tensor expect,
                    int64_t timeout_ns, int64_t stats_dev) {
    TORCH_CHECK(nbytes % 16 == 0, "ple wait_copy: nbytes must be a multiple of 16");
    TORCH_CHECK(src.numel() * src.element_size() >= nbytes && dst.numel() * dst.element_size() >= nbytes);
    TORCH_CHECK(((uintptr_t) src.data_ptr() % 16) == 0 && ((uintptr_t) dst.data_ptr() % 16) == 0);
    TORCH_CHECK(expect.scalar_type() == torch::kInt64 && expect.is_cuda());
    const c10::cuda::CUDAGuard guard(dst.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t n16 = nbytes / 16;
    int blocks = (int) std::min<int64_t>((n16 + 255) / 256, 64);
    if (blocks < 1) blocks = 1;
    ple_wait_copy_kernel<<<blocks, 256, 0, stream>>>((const uint4*) src.data_ptr(), (uint4*) dst.data_ptr(), n16,
                                                     (const volatile uint64_t*) flag_dev, expect.data_ptr<int64_t>(),
                                                     (uint64_t) timeout_ns, (volatile uint64_t*) stats_dev);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// post_ids (doorbell wake-up of the reader): write the step's ids into pinned, mapped host memory, fence, then the
// step's sequence number into the mapped ids flag the reader thread spins on. One CTA; launched outside the graphs.
__global__ void __launch_bounds__(256) post_ids_kernel(const int64_t* __restrict__ ids, int64_t n, int64_t* dst,
                                                       volatile uint64_t* flag, uint64_t seq) {
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) dst[i] = ids[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *flag = seq;
        __threadfence_system();
    }
}

void cuda_post_ids(torch::Tensor ids, int64_t dst_dev, int64_t flag_dev, int64_t seq) {
    TORCH_CHECK(ids.scalar_type() == torch::kInt64 && ids.is_cuda() && ids.is_contiguous());
    const c10::cuda::CUDAGuard guard(ids.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    post_ids_kernel<<<1, 256, 0, stream>>>(ids.data_ptr<int64_t>(), ids.numel(), (int64_t*) dst_dev,
                                           (volatile uint64_t*) flag_dev, (uint64_t) seq);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace e0ple
