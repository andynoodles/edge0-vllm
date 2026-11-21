// Edge0 CPU cold experts, GPU side (docs/CPU-EXPERTS.md 3.2): the three graph-capturable kernels around one MoE call.
//
//   dispatch  (1 CTA x 512 threads, one per logical expert): route histogram, deterministic choice of the CPU's cold
//             experts, cold_map_eff / cpu_sel in VRAM, the job (tasks, route lists) in mapped memory. `frac` >= 0 is
//             the fixed split (a kernel argument), < 0 = auto (read from the mailbox).
//   gather    (1 CTA per route): the Marlin input row (VRAM) and, for CPU routes, the same row in mapped memory; the
//             last CTA bumps the sequence number in VRAM and rings the doorbell in mapped memory (Strata's
//             doorbell_ring_kernel: the value comes from memory, never a captured literal).
//   wait      (1 thread): returns at once without a CPU share, else spins on mapped `done == seq` with a timeout.
//
// `state` (int64 [16], VRAM): u32[0] seq, u32[1] CTA counter of gather, u32[2] n_cpu of the current call;
// u64[2] doorbell time (globaltimer), u64[3..6] running totals mirrored into the mailbox (calls, gpu_ns, wait_ns,
// timeouts).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/block/block_scan.cuh>

#include "mailbox.h"

namespace e0 {
namespace {

__device__ __forceinline__ uint64_t globaltimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

constexpr int kE = 512;

__global__ void __launch_bounds__(kE) dispatch_kernel(
    const int32_t* __restrict__ ids, const float* __restrict__ w, int R, int T, const int32_t* __restrict__ cold_map,
    const float* __restrict__ g13, const float* __restrict__ g2, int layer, int min_cold, int max_routes_e,
    int max_experts, float frac_arg, uint8_t* mbox, MboxLayout lay, int32_t* __restrict__ cold_map_eff,
    int32_t* __restrict__ cpu_sel, uint32_t* state) {
    // Round 2 (agent exceed-2): the c1 critical path runs this once per layer. Only the work a CPU share needs touches
    // mapped memory: a fixed split passes `frac_arg` (no PCIe read of the mailbox; < 0 = auto, read MboxHeader.frac,
    // and only when the call has enough cold experts), the eligible / cold counts share one block scan, a call
    // without a CPU share returns before the route scan and writes nothing to mapped memory, and only threads that
    // wrote mapped memory fence.
    __shared__ int cnt[kE];
    __shared__ int cur[kE];
    __shared__ int ssel[kE];
    __shared__ float sfrac;
    typedef cub::BlockScan<int, kE> Scan;
    __shared__ typename Scan::TempStorage tmp;
    MboxHeader* h = (MboxHeader*) mbox;
    const int e = threadIdx.x;
    const int cm = cold_map[e];
    cnt[e] = 0;
    __syncthreads();
    for (int r = e; r < R; r += kE) {
        const int x = ids[r];
        if (x >= 0 && x < kE) atomicAdd(&cnt[x], 1);
    }
    __syncthreads();
    const int c = cnt[e];
    const int is_cold = (cm >= 0 && c > 0) ? 1 : 0;
    const int elig = (is_cold && c <= max_routes_e) ? 1 : 0;
    int packed, total;                       // (cold << 16) | eligible: both counts are <= 512
    Scan(tmp).ExclusiveSum((is_cold << 16) | elig, packed, total);
    const int rank = packed & 0xffff;
    const int n_elig = total & 0xffff;
    const int n_cold = total >> 16;
    int ncpu = 0;
    if (n_cold >= min_cold) {                // block-uniform
        if (frac_arg < 0.f) {
            if (e == 0) sfrac = h->frac;     // volatile: the auto split updates it
            __syncthreads();
        }
        const float frac = frac_arg < 0.f ? sfrac : frac_arg;
        if (frac > 0.f) {
            ncpu = (int) floorf(frac * (float) n_cold + 0.5f);
            ncpu = min(min(ncpu, max_experts), n_elig);
        }
    }
    const int sel = (elig && rank < ncpu) ? 1 : 0;
    cold_map_eff[e] = sel ? -1 : cm;
    cpu_sel[e] = sel;
    if (ncpu == 0) {                         // block-uniform: nothing for the CPU, gather / wait see n_cpu 0
        if (e == 0) state[2] = 0u;
        return;
    }
    __syncthreads();                         // tmp is reused
    int off, n_routes;
    Scan(tmp).ExclusiveSum(sel ? c : 0, off, n_routes);
    ssel[e] = sel;
    cur[e] = off;
    bool wrote = false;
    if (sel) {
        Task t;
        t.slot = cm;
        t.off = off;
        t.n = c;
        t._pad = 0;
        t.g13 = g13[cm];
        t.g2 = g2[cm];
        t._pad2[0] = t._pad2[1] = 0.f;
        ((Task*) (mbox + lay.tasks))[rank] = t;
        wrote = true;
    }
    __syncthreads();
    int32_t* rid = (int32_t*) (mbox + lay.route_ids);
    float* rw = (float*) (mbox + lay.route_w);
    for (int r = e; r < R; r += kE) {
        const int x = ids[r];
        if (x >= 0 && x < kE && ssel[x]) {
            const int p = atomicAdd(&cur[x], 1);
            rid[p] = r;
            rw[p] = w[r];
            wrote = true;
        }
    }
    if (e == 0) {
        h->layer = (uint32_t) layer;
        h->n_experts = (uint32_t) ncpu;
        h->n_routes = (uint32_t) n_routes;
        h->n_tokens = (uint32_t) T;
        state[2] = (uint32_t) ncpu;
        wrote = true;
    }
    if (wrote) __threadfence_system();
}

__global__ void __launch_bounds__(128) gather_kernel(
    const uint4* __restrict__ xl, int64_t xl_stride4, const int32_t* __restrict__ ids, const int32_t* __restrict__ group,
    int K, uint4* __restrict__ out, const int32_t* __restrict__ cpu_sel, uint8_t* mbox, MboxLayout lay,
    uint32_t* state) {
    constexpr int ROW4 = LAT * 2 / 16;     // 224 uint4 per bf16 row
    const int r = blockIdx.x;
    const int t = r / K;
    int e = ids[r];
    const bool ok = e >= 0 && e < kE;
    e = ok ? e : 0;
    const int g = group[e];
    const bool cpu = ok && cpu_sel[e];
    const uint4* src = xl + (int64_t) t * xl_stride4 + (int64_t) g * ROW4;
    uint4* dst = out + (int64_t) r * ROW4;
    uint4* mx = (uint4*) (mbox + lay.x) + (int64_t) r * ROW4;
    for (int i = threadIdx.x; i < ROW4; i += blockDim.x) {
        const uint4 v = src[i];
        dst[i] = v;
        if (cpu) mx[i] = v;
    }
    if (state[2] == 0) return;              // no CPU share in this call (dispatch): no doorbell, nothing mapped
    if (cpu) __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned prev = atomicAdd(&state[1], 1u);
        if (prev == gridDim.x - 1) {           // last CTA: every row is written and fenced
            state[1] = 0;
            if (state[2] > 0) {
                const uint32_t s = state[0] + 1;
                state[0] = s;
                ((uint64_t*) state)[2] = globaltimer();
                __threadfence_system();
                ((MboxHeader*) mbox)->seq = s;  // volatile store
                __threadfence_system();
            }
        }
    }
}

__global__ void wait_kernel(uint32_t* state, uint8_t* mbox, uint64_t timeout_ns) {
    if (threadIdx.x != 0 || state[2] == 0) return;
    MboxHeader* h = (MboxHeader*) mbox;
    uint64_t* st = (uint64_t*) state;
    const uint32_t want = state[0];
    const uint64_t t0 = globaltimer();
    bool timed_out = false;
    while (h->done != want) {
        __nanosleep(64);
        if (globaltimer() - t0 > timeout_ns) {
            timed_out = true;
            break;
        }
    }
    const uint64_t t1 = globaltimer();
    __threadfence_system();
    st[3] += 1;
    st[4] += t0 - st[2];
    st[5] += t1 - t0;
    if (timed_out) st[6] += 1;
    h->calls = st[3];           // posted writes: the CPU pool's auto split reads them
    h->gpu_ns = st[4];
    h->wait_ns = st[5];
    h->timeouts = st[6];
}

}  // namespace

void cuda_dispatch(torch::Tensor ids, torch::Tensor w, torch::Tensor cold_map, torch::Tensor g13, torch::Tensor g2,
                   int64_t layer, int64_t min_cold, int64_t max_routes_e, int64_t max_experts, int64_t mbox_dev,
                   int64_t max_routes, torch::Tensor cold_map_eff, torch::Tensor cpu_sel, torch::Tensor state,
                   double frac) {
    TORCH_CHECK(ids.is_cuda() && ids.scalar_type() == torch::kInt32 && ids.is_contiguous());
    TORCH_CHECK(w.scalar_type() == torch::kFloat32 && w.is_contiguous() && w.numel() == ids.numel());
    TORCH_CHECK(cold_map.numel() == kE && cold_map.scalar_type() == torch::kInt32);
    TORCH_CHECK(cold_map_eff.numel() == kE && cpu_sel.numel() == kE);
    const int R = (int) ids.numel();
    TORCH_CHECK(R <= max_routes, "edge0 cpu experts: ", R, " routes > mailbox capacity ", max_routes);
    const int T = ids.dim() == 2 ? (int) ids.size(0) : R;
    const c10::cuda::CUDAGuard guard(ids.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    dispatch_kernel<<<1, kE, 0, stream>>>(
        ids.data_ptr<int32_t>(), w.data_ptr<float>(), R, T, cold_map.data_ptr<int32_t>(), g13.data_ptr<float>(),
        g2.data_ptr<float>(), (int) layer, (int) min_cold, (int) max_routes_e, (int) max_experts, (float) frac,
        (uint8_t*) mbox_dev,
        mbox_layout((int) max_routes), cold_map_eff.data_ptr<int32_t>(), cpu_sel.data_ptr<int32_t>(),
        (uint32_t*) state.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void cuda_gather(torch::Tensor xl, torch::Tensor ids, torch::Tensor group, torch::Tensor out, torch::Tensor cpu_sel,
                 int64_t mbox_dev, int64_t max_routes, torch::Tensor state) {
    TORCH_CHECK(xl.is_cuda() && xl.element_size() == 2 && xl.stride(1) == 1 && xl.size(1) % LAT == 0);
    TORCH_CHECK(out.element_size() == 2 && out.is_contiguous() && out.size(1) == LAT);
    TORCH_CHECK(ids.scalar_type() == torch::kInt32 && ids.is_contiguous() && ids.dim() == 2);
    TORCH_CHECK(((uintptr_t) xl.data_ptr() % 16) == 0 && (xl.stride(0) * 2) % 16 == 0);
    const int R = (int) ids.numel();
    const int K = (int) ids.size(1);
    TORCH_CHECK(R <= max_routes && out.size(0) >= R);
    if (R == 0) return;
    const c10::cuda::CUDAGuard guard(xl.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    gather_kernel<<<R, 128, 0, stream>>>(
        (const uint4*) xl.data_ptr(), xl.stride(0) * 2 / 16, ids.data_ptr<int32_t>(), group.data_ptr<int32_t>(), K,
        (uint4*) out.data_ptr(), cpu_sel.data_ptr<int32_t>(), (uint8_t*) mbox_dev, mbox_layout((int) max_routes),
        (uint32_t*) state.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void cuda_wait(torch::Tensor state, int64_t mbox_dev, int64_t timeout_ns) {
    const c10::cuda::CUDAGuard guard(state.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    wait_kernel<<<1, 32, 0, stream>>>((uint32_t*) state.data_ptr<int64_t>(), (uint8_t*) mbox_dev,
                                      (uint64_t) timeout_ns);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace e0
