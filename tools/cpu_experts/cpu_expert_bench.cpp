// CPU cold-expert throughput on this box with Strata's own NVFP4 AVX-512 kernel (docs/PROFILE.md section 5).
//
// Build (on the 5090 host; reads Strata's source and headers, writes nothing there):
//   S=~/work/edge0/edge0-strata
//   g++ -O3 -march=native -fopenmp -std=c++17 -I$S/include -I$S/third_party/ggml \
//       tools/cpu_experts/cpu_expert_bench.cpp $S/src/kernels/cpu/nvfp4_avx512.cpp -o /tmp/cpu_expert_bench
//
// Modes:
//   read  <GiB> <threads> <seconds>                       DRAM read bandwidth (AVX-512 loads), GB/s every 0.5 s
//   expert <GiB> <threads> <experts/call> <tokens> <seconds>
//          one Edge0 latent expert = 1,935,360 B: gate+up 1280 rows x 1792 (28 NVFP4 blocks/row) and down
//          1792 rows x 640 (10 blocks/row), the record the vLLM pool keeps per slot. Each "call" is one layer's CPU
//          share: <experts/call> random experts from a pool of <GiB> (>> L3), each serving <tokens> routes: Q8_0
//          activations -> gate/up rows (SiLU*up) -> Q8_0 ff -> down rows. Rows split over threads (OpenMP).
//          Prints us per call, us per expert and weight GB/s, every 0.5 s.
#include "strata/kernels/cpu/nvfp4_avx512.hpp"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <immintrin.h>
#include <omp.h>
#include <sys/mman.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::kernels::cpu;
using clk = std::chrono::steady_clock;

static constexpr int LAT = 1792, FF = 640;
static constexpr size_t GU_ROW = (LAT / QK_NVFP4) * sizeof(block_nvfp4);   // 28 x 36 = 1008 B
static constexpr size_t DN_ROW = (FF / QK_NVFP4) * sizeof(block_nvfp4);    // 10 x 36 = 360 B
static constexpr size_t GU_BYTES = 2 * FF * GU_ROW;                         // 1,290,240
static constexpr size_t EXPERT_BYTES = GU_BYTES + LAT * DN_ROW;             // 1,935,360

static double now() { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); }  // wall clock: aligned with the GPU loop's time.time()

static uint8_t* big_alloc(size_t n) {
    void* p = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); std::exit(1); }
    madvise(p, n, MADV_HUGEPAGE);
    return (uint8_t*) p;
}

static void fill_random(uint8_t* p, size_t n, int threads) {
#pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t i = 0; i < n / 8; ++i) {
        uint64_t x = i * 0x9E3779B97F4A7C15ull;
        x ^= x >> 29;
        // keep UE4M3 scale bytes sane-ish: any byte works for timing (values are never checked)
        ((uint64_t*) p)[i] = x & 0x3F3F3F3F3F3F3F3Full;
    }
}

static void quantize_q8_0(const float* x, int n, block_q8_0* out) {
    for (int b = 0; b < n / 32; ++b) {
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) amax = std::fmax(amax, std::fabs(x[b * 32 + j]));
        const float d = amax / 127.f, id = d ? 1.f / d : 0.f;
        out[b].d = _cvtss_sh(d, 0);
        for (int j = 0; j < 32; ++j) out[b].qs[j] = (int8_t) std::lrintf(x[b * 32 + j] * id);
    }
}

static int mode_read(double gib, int threads, double seconds) {
    const size_t n = (size_t) (gib * (1ull << 30)) & ~(size_t) 4095;
    uint8_t* buf = big_alloc(n);
    fill_random(buf, n, threads);
    struct alignas(64) Ctr { volatile double b = 0; };
    std::vector<Ctr> bytes(threads);
    volatile long long sink = 0;
    const double t_end = now() + seconds;
    const size_t CH = 4u << 20;                                   // 4 MiB chunks
    double last = now(), last_b = 0, tot_all = 0, t_first = last;
#pragma omp parallel num_threads(threads)
    {
        const int t = omp_get_thread_num();
        const size_t lo = (n / threads * t) & ~(size_t) 255, hi = (n / threads * (t + 1)) & ~(size_t) 255;
        __m512i acc = _mm512_setzero_si512();
        bool done = false;
        while (!done) {
            for (size_t c = lo; c < hi && !done; c += CH) {
                const size_t ce = c + CH < hi ? c + CH : hi;
                for (size_t o = c; o < ce; o += 256) {
                    acc = _mm512_add_epi64(acc, _mm512_load_si512((const void*) (buf + o)));
                    acc = _mm512_add_epi64(acc, _mm512_load_si512((const void*) (buf + o + 64)));
                    acc = _mm512_add_epi64(acc, _mm512_load_si512((const void*) (buf + o + 128)));
                    acc = _mm512_add_epi64(acc, _mm512_load_si512((const void*) (buf + o + 192)));
                }
                bytes[t].b = bytes[t].b + (double) (ce - c);
                const double tn = now();
                if (tn >= t_end) done = true;
                if (t == 0 && tn - last >= 0.5) {
                    double tot = 0;
                    for (auto& b : bytes) tot += b.b;
                    std::printf("%.3f read %.1f GB/s\n", tn, (tot - last_b) / (tn - last) / 1e9);
                    std::fflush(stdout);
                    last = tn; last_b = tot;
                }
            }
        }
        sink = sink + _mm512_reduce_add_epi64(acc);
    }
    for (auto& b : bytes) tot_all += b.b;
    std::printf("mean read %.1f GB/s over %.1f s (%d threads)\n", tot_all / (now() - t_first) / 1e9,
                now() - t_first, threads);
    return 0;
}

static int mode_expert(double gib, int threads, int per_call, int nt, double seconds) {
    const size_t n_exp = (size_t) (gib * (1ull << 30)) / EXPERT_BYTES;
    uint8_t* pool = big_alloc(n_exp * EXPERT_BYTES);
    fill_random(pool, n_exp * EXPERT_BYTES, threads);
    std::mt19937 rng(1);
    std::uniform_int_distribution<size_t> pick(0, n_exp - 1);
    std::normal_distribution<float> nd(0.f, 1.f);
    // per expert-slot of a call: activations (nt tokens), ff (nt x 640), out (nt x 1792)
    std::vector<std::vector<float>> x(per_call * nt, std::vector<float>(LAT));
    for (auto& v : x) for (auto& f : v) f = nd(rng);
    std::vector<std::vector<block_q8_0>> xq(per_call * nt, std::vector<block_q8_0>(LAT / 32));
    std::vector<std::vector<float>> ff(per_call * nt, std::vector<float>(FF));
    std::vector<std::vector<block_q8_0>> fq(per_call * nt, std::vector<block_q8_0>(FF / 32));
    std::vector<std::vector<float>> out(per_call * nt, std::vector<float>(LAT));
    std::vector<const uint8_t*> e(per_call);
    const int GU_CH = 64, DN_CH = 128;                       // rows per task
    const int gu_tasks = FF / GU_CH, dn_tasks = LAT / DN_CH;  // 10, 14
    std::printf("pool %zu experts (%.2f GiB), %d threads, %d experts/call, %d tokens/expert\n", n_exp,
                n_exp * EXPERT_BYTES / 1073741824.0, threads, per_call, nt);
    const double t_end = now() + seconds;
    double last = now();
    long calls = 0;
    double busy = 0;
    while (now() < t_end) {
        for (int i = 0; i < per_call; ++i) e[i] = pool + pick(rng) * EXPERT_BYTES;
        const double t0 = now();
#pragma omp parallel num_threads(threads)
        {
#pragma omp for schedule(static)
            for (int i = 0; i < per_call * nt; ++i) quantize_q8_0(x[i].data(), LAT, xq[i].data());
#pragma omp for schedule(dynamic, 1)
            for (int task = 0; task < per_call * gu_tasks; ++task) {
                const int i = task / gu_tasks, r0 = (task % gu_tasks) * GU_CH;
                const void* act[8];
                float* f[8];
                for (int t = 0; t < nt; ++t) { act[t] = xq[i * nt + t].data(); f[t] = ff[i * nt + t].data(); }
                nvfp4_512_gu_rows(e[i], GU_ROW, FF * GU_ROW, LAT, act, nt, f, r0, r0 + GU_CH, 1.f, 1.f);
            }
#pragma omp for schedule(static)
            for (int i = 0; i < per_call * nt; ++i) quantize_q8_0(ff[i].data(), FF, fq[i].data());
#pragma omp for schedule(dynamic, 1)
            for (int task = 0; task < per_call * dn_tasks; ++task) {
                const int i = task / dn_tasks, r0 = (task % dn_tasks) * DN_CH;
                const void* act[8];
                float* o[8];
                for (int t = 0; t < nt; ++t) { act[t] = fq[i * nt + t].data(); o[t] = out[i * nt + t].data(); }
                nvfp4_512_rows(e[i] + GU_BYTES, DN_ROW, FF, act, nt, o, r0, r0 + DN_CH, 1.f);
            }
        }
        busy += now() - t0;
        ++calls;
        const double tn = now();
        if (tn - last >= 0.5) {
            const double per = busy / calls;
            std::printf("%.3f expert %.1f us/call %.2f us/expert %.1f GB/s\n", tn, per * 1e6, per * 1e6 / per_call,
                        per_call * EXPERT_BYTES / per / 1e9);
            std::fflush(stdout);
            last = tn; busy = 0; calls = 0;
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 5 && !std::strcmp(argv[1], "read")) return mode_read(atof(argv[2]), atoi(argv[3]), atof(argv[4]));
    if (argc >= 7 && !std::strcmp(argv[1], "expert"))
        return mode_expert(atof(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atof(argv[6]));
    std::fprintf(stderr, "usage: read <GiB> <threads> <s> | expert <GiB> <threads> <experts/call> <tokens> <s>\n");
    return 2;
}
