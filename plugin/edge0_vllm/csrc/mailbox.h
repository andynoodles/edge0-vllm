// Edge0 CPU cold experts (docs/CPU-EXPERTS.md): the mailbox shared by the GPU kernels (cpu_experts_cuda.cu) and the
// CPU pool (cpu_experts.cpp). It lives in pinned, mapped host memory: the GPU writes the job and rings, the CPU
// computes and raises `done`. One job at a time (MoE layers run one after the other on one stream).
#pragma once
#include <cstddef>
#include <cstdint>

namespace e0 {

constexpr int LAT = 1792;               // latent width (w13 input, w2 output)
constexpr int FF = 640;                 // expert intermediate size
constexpr int N13 = 2 * FF;             // w13 output columns (gate 0..639, up 640..1279)
constexpr int KT13 = LAT / 16;          // 112 k-tiles of w13
constexpr int NT13 = N13 / 64;          // 20 n-tiles of w13
constexpr int KT2 = FF / 16;            // 40 k-tiles of w2
constexpr int NT2 = LAT / 64;           // 28 n-tiles of w2
constexpr int MAX_EXPERTS = 512;
constexpr size_t W13_BYTES = (size_t) KT13 * NT13 * 128 * 4;   // 1,146,880
constexpr size_t W2_BYTES = (size_t) KT2 * NT2 * 128 * 4;      // 573,440
constexpr size_t S13_BYTES = (size_t) KT13 * N13;              // 143,360
constexpr size_t S2_BYTES = (size_t) KT2 * LAT;                // 71,680

struct alignas(64) MboxHeader {
    volatile uint32_t seq;              // line 0: GPU -> CPU doorbell (incremented once per job)
    uint32_t _p0[15];
    volatile uint32_t done;             // line 1: CPU -> GPU, = seq of the last completed job
    uint32_t _p1[15];
    uint32_t layer, n_experts, n_routes, n_tokens;   // line 2: the job, written by the dispatch kernel
    uint32_t _p2[12];
    volatile float frac;                // line 3: share of cold experts the CPU takes (host-written, GPU-read)
    uint32_t _p3[15];
    // line 4: GPU statistics (written by the wait kernel, posted writes only; running totals)
    volatile uint64_t calls;            // calls with a CPU share
    volatile uint64_t gpu_ns;           // sum over those calls: doorbell -> wait-kernel start (GPU side of the call)
    volatile uint64_t wait_ns;          // sum: time the wait kernel spun on `done`
    volatile uint64_t timeouts;         // waits that gave up (should stay 0)
    uint64_t _p4[4];
};
static_assert(sizeof(MboxHeader) == 320, "mailbox header layout");

struct Task {                           // one CPU expert of the job
    int32_t slot;                       // cold-tier slot of the layer
    int32_t off;                        // first entry in route_ids / route_w
    int32_t n;                          // routes of this expert
    int32_t _pad;
    float g13, g2;                      // Marlin-processed global scales of the slot
    float _pad2[2];
};
static_assert(sizeof(Task) == 32, "task layout");

struct MboxLayout {
    size_t tasks, route_ids, route_w, x, y, total;
};

inline size_t mbox_align(size_t v) { return (v + 4095) & ~(size_t) 4095; }

// x / y rows are indexed by the route's position r in the MoE call (r < max_routes); route_ids / route_w by the
// position in the job (experts' route lists back to back, `Task::off`).
inline MboxLayout mbox_layout(int max_routes) {
    MboxLayout m{};
    size_t o = 4096;
    m.tasks = o;     o = mbox_align(o + (size_t) MAX_EXPERTS * sizeof(Task));
    m.route_ids = o; o = mbox_align(o + (size_t) max_routes * 4);
    m.route_w = o;   o = mbox_align(o + (size_t) max_routes * 4);
    m.x = o;         o = mbox_align(o + (size_t) max_routes * LAT * 2);
    m.y = o;         o = mbox_align(o + (size_t) max_routes * LAT * 2);
    m.total = o;
    return m;
}

}  // namespace e0
