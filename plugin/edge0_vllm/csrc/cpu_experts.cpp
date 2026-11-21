// Edge0 CPU cold experts: the worker pool and the AVX-512 kernels over the Marlin NVFP4 expert record
// (docs/CPU-EXPERTS.md). The design follows Strata's CPU expert pool (src/kernels/cpu/pool.cpp: pinned busy-polling
// workers, doorbell / flag hand-off with the GPU through mapped memory; MIT, see NOTICE); the kernels are new because
// the plugin's cold pool keeps experts in vLLM's Marlin layout, not Strata's.
//
// Marlin tile (vLLM csrc/quantization/marlin/gptq_marlin_repack.cu, 4-bit, bf16 activations): weights int32
// [K/16, size_n * 2]; tile (kt, nt) = 16 k x 64 n = 128 words at (kt * n_tiles + nt) * 128. Word w = th * 4 + warp
// holds column n0 = warp * 16 + th / 4 and n0 + 8, rows (th % 4) * 2 + {0, 1, 8, 9}; nibble i holds
// vals[{0,2,4,6,1,3,5,7}[i]], vals[0..3] = column n0 at rows +{0,1,8,9}, vals[4..7] = column n0 + 8. A zmm of 16
// consecutive words (v = w / 16) has lane l at columns (l % 4) * 16 + v (+8) and rows (l / 4) * 2 + {0,1,8,9}.
//
// Scales (marlin_permute_scales + nvfp4_marlin_process_scales): per (k-tile, column) one byte, the high byte of
// fp16(s * factor * 2^7) << 1 ("S0E5M3"); within each 64-column chunk, column n = c * 16 + v + 8 h sits at byte
// 8 v + 4 (c >> 1) + (c & 1) + 2 h. Decoded value S = s * factor * 2^7; together with Marlin's power-of-two dequant
// offset, the GPU's fp32 accumulator equals ours times 2^-126, which is folded into the global scale.
//
// Rounding follows the Marlin MoE epilogue (marlin_moe_wna16/marlin_template.h) and vLLM's silu_and_mul:
//   GEMM1  c1 = bf16(acc * g13)                      (mul_topk_weights = false: fp32 multiply, one rounding)
//   act    h  = bf16(float(bf16(silu(gate))) * up)
//   GEMM2  y  = bf16(bf16(acc) * bf16(w_route * g2))  (mul_topk_weights = true: __hmul2 after the bf16 cast)
#include <torch/extension.h>

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "mailbox.h"

namespace e0 {
namespace {

inline uint64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}

// ------------------------------------------------------------------------------------------------ vector helpers
alignas(64) const float kE2M1[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                     -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

struct Consts {
    alignas(64) int32_t xidx[4][16];      // lane l -> row (l/4)*2 + {0, 8, 1, 9}[o] of the k-tile
    alignas(64) int32_t sidx[2][2][16];   // [v & 1][h] lane l -> lane of the decoded scale vector F[v >> 1]
    Consts() {
        const int off[4] = {0, 8, 1, 9};
        for (int o = 0; o < 4; ++o)
            for (int l = 0; l < 16; ++l) xidx[o][l] = (l / 4) * 2 + off[o];
        for (int vo = 0; vo < 2; ++vo)
            for (int h = 0; h < 2; ++h)
                for (int l = 0; l < 16; ++l) {
                    const int c = l % 4;
                    sidx[vo][h][l] = 8 * vo + 4 * (c >> 1) + (c & 1) + 2 * h;
                }
    }
};
const Consts kC;

inline __m512 bf16_round(__m512 x) {      // round to nearest even at bf16 precision, kept as fp32
    __m512i b = _mm512_castps_si512(x);
    const __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(b, 16), _mm512_set1_epi32(1));
    b = _mm512_add_epi32(b, _mm512_add_epi32(lsb, _mm512_set1_epi32(0x7FFF)));
    return _mm512_castsi512_ps(_mm512_and_si512(b, _mm512_set1_epi32((int) 0xFFFF0000u)));
}

inline __m256i to_bf16_bits(__m512 x_rounded) {   // already bf16-exact fp32 -> 16 x uint16
    return _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(x_rounded), 16));
}

inline __m512 bf16_load16(const uint16_t* p) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*) p)), 16));
}

inline __m512 exp512(__m512 x) {          // Cephes expf, ~1 ulp
    x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-87.3f)), _mm512_set1_ps(88.3f));
    const __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504088896341f)),
                                          _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693359375f), x);
    r = _mm512_fnmadd_ps(n, _mm512_set1_ps(-2.12194440e-4f), r);
    __m512 p = _mm512_set1_ps(1.9875691500E-4f);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.3981999507E-3f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(8.3334519073E-3f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(4.1665795894E-2f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.6666665459E-1f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(5.0000001201E-1f));
    p = _mm512_fmadd_ps(p, _mm512_mul_ps(r, r), _mm512_add_ps(r, _mm512_set1_ps(1.f)));
    return _mm512_scalef_ps(p, n);
}

// ------------------------------------------------------------------------------------------------ the kernel
// One n-tile column (64 outputs) of a Marlin NVFP4 matrix times one activation vector x [KT * 16] (fp32).
// w: first word of tile (0, nt); wstride: words between k-tiles (= n_tiles * 128). s: scale byte 0 of chunk nt at
// k-tile 0; sstride: bytes between k-tiles (= size_n). out[n] (n = 0..63 within the tile) = sum_k W(k, n) S x[k]
// without the global scale.
constexpr int kPrefetch = 3;   // k-tiles ahead

template <int KT>
inline void tile_gemv(const uint32_t* __restrict w, size_t wstride, const uint8_t* __restrict s, size_t sstride,
                      const float* __restrict x, float* __restrict out) {
    const __m512 tab = _mm512_load_ps(kE2M1);
    __m512 acc[16];
    for (int i = 0; i < 16; ++i) acc[i] = _mm512_setzero_ps();
    for (int kt = 0; kt < KT; ++kt) {
        const uint32_t* wt = w + (size_t) kt * wstride;
        const uint8_t* st = s + (size_t) kt * sstride;
        if (kt + kPrefetch < KT) {
            const char* pw = (const char*) (wt + kPrefetch * wstride);
            for (int c = 0; c < 512; c += 64) _mm_prefetch(pw + c, _MM_HINT_T0);
            _mm_prefetch((const char*) (st + kPrefetch * sstride), _MM_HINT_T0);
        }
        const __m512 xk = _mm512_loadu_ps(x + kt * 16);
        const __m512 X0 = _mm512_permutexvar_ps(_mm512_load_si512(kC.xidx[0]), xk);
        const __m512 X8 = _mm512_permutexvar_ps(_mm512_load_si512(kC.xidx[1]), xk);
        const __m512 X1 = _mm512_permutexvar_ps(_mm512_load_si512(kC.xidx[2]), xk);
        const __m512 X9 = _mm512_permutexvar_ps(_mm512_load_si512(kC.xidx[3]), xk);
        const __m512i sb = _mm512_loadu_si512((const void*) st);
        __m512 F[4];
        for (int a = 0; a < 4; ++a) {
            const __m512i b = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(sb, a));
            const __m512i bits = _mm512_add_epi32(_mm512_slli_epi32(b, 20), _mm512_set1_epi32(112 << 23));
            F[a] = _mm512_maskz_mov_ps(_mm512_test_epi32_mask(b, b), _mm512_castsi512_ps(bits));
        }
#pragma GCC unroll 8
        for (int v = 0; v < 8; ++v) {
            const __m512i q = _mm512_loadu_si512((const void*) (wt + v * 16));
            __m512 t0 = _mm512_mul_ps(_mm512_permutexvar_ps(q, tab), X0);                              // nib 0: k+0
            __m512 t1 = _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(q, 8), tab), X0);       // nib 2: k+0, n+8
            t0 = _mm512_fmadd_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(q, 4), tab), X8, t0);        // nib 1: k+8
            t1 = _mm512_fmadd_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(q, 12), tab), X8, t1);       // nib 3
            t0 = _mm512_fmadd_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(q, 16), tab), X1, t0);       // nib 4: k+1
            t1 = _mm512_fmadd_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(q, 24), tab), X1, t1);       // nib 6
            t0 = _mm512_fmadd_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(q, 20), tab), X9, t0);       // nib 5: k+9
            t1 = _mm512_fmadd_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(q, 28), tab), X9, t1);       // nib 7
            const __m512 S0 = _mm512_permutexvar_ps(_mm512_load_si512(kC.sidx[v & 1][0]), F[v >> 1]);
            const __m512 S1 = _mm512_permutexvar_ps(_mm512_load_si512(kC.sidx[v & 1][1]), F[v >> 1]);
            acc[2 * v] = _mm512_fmadd_ps(t0, S0, acc[2 * v]);
            acc[2 * v + 1] = _mm512_fmadd_ps(t1, S1, acc[2 * v + 1]);
        }
    }
    for (int v = 0; v < 8; ++v)
        for (int h = 0; h < 2; ++h) {
            const __m512 a = acc[2 * v + h];
            const __m256 s8 = _mm256_add_ps(_mm512_castps512_ps256(a),
                                            _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(a), 1)));
            const __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(s8), _mm256_extractf128_ps(s8, 1));
            alignas(16) float t[4];
            _mm_store_ps(t, s4);
            for (int c = 0; c < 4; ++c) out[c * 16 + v + 8 * h] = t[c];
        }
}

struct LayerPtrs {
    const uint8_t* w13 = nullptr;     // slot s at w13 + s * W13_BYTES (and likewise for the others)
    const uint8_t* w2 = nullptr;
    const uint8_t* s13 = nullptr;
    const uint8_t* s2 = nullptr;
    int n = 0;
};

// G1 for one route: 64 intermediate features j*64 .. j*64+63 (gate tile j, up tile j + 10) -> h (bf16-exact fp32)
inline void g1_route(const LayerPtrs& L, int slot, int j, float g13, const float* xf, float* h64) {
    const uint32_t* w = (const uint32_t*) (L.w13 + (size_t) slot * W13_BYTES);
    const uint8_t* s = L.s13 + (size_t) slot * S13_BYTES;
    alignas(64) float gate[64], up[64];
    tile_gemv<KT13>(w + (size_t) j * 128, (size_t) NT13 * 128, s + j * 64, N13, xf, gate);
    tile_gemv<KT13>(w + (size_t) (j + FF / 64) * 128, (size_t) NT13 * 128, s + (j + FF / 64) * 64, N13, xf, up);
    const __m512 g = _mm512_set1_ps(std::ldexp(g13, -126));
    for (int c = 0; c < 64; c += 16) {
        const __m512 gb = bf16_round(_mm512_mul_ps(_mm512_load_ps(gate + c), g));
        const __m512 ub = bf16_round(_mm512_mul_ps(_mm512_load_ps(up + c), g));
        const __m512 sil = bf16_round(_mm512_div_ps(gb, _mm512_add_ps(_mm512_set1_ps(1.f),
                                                                      exp512(_mm512_sub_ps(_mm512_setzero_ps(), gb)))));
        _mm512_storeu_ps(h64 + c, bf16_round(_mm512_mul_ps(sil, ub)));
    }
}

// G2 for one route: output columns nt*64 .. nt*64+63 -> y (bf16 bits)
inline void g2_route(const LayerPtrs& L, int slot, int nt, float g2, float wroute, const float* h, uint16_t* y64) {
    const uint32_t* w = (const uint32_t*) (L.w2 + (size_t) slot * W2_BYTES);
    const uint8_t* s = L.s2 + (size_t) slot * S2_BYTES;
    alignas(64) float acc[64];
    tile_gemv<KT2>(w + (size_t) nt * 128, (size_t) NT2 * 128, s + nt * 64, LAT, h, acc);
    // tw = bf16(fp32(w * g2)); y = bf16(bf16(acc) * tw * 2^-126)  (the 2^-126 is exact: powers of two, normal range)
    alignas(64) float twv[16];
    const float tw = std::ldexp(_mm512_cvtss_f32(bf16_round(_mm512_set1_ps(wroute * g2))), -126);
    for (int i = 0; i < 16; ++i) twv[i] = tw;
    const __m512 t = _mm512_load_ps(twv);
    for (int c = 0; c < 64; c += 16) {
        const __m512 a = bf16_round(_mm512_load_ps(acc + c));
        _mm256_storeu_si256((__m256i*) (y64 + c), to_bf16_bits(bf16_round(_mm512_mul_ps(a, t))));
    }
}

inline void bf16_row_to_f32(const uint16_t* src, float* dst, int n) {
    for (int i = 0; i < n; i += 16) _mm512_store_ps(dst + i, bf16_load16(src + i));
}

void pin_thread(int cpu) {
    if (cpu < 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

constexpr int G1_PAIRS = FF / 64;          // 10 (gate n-tile j, up n-tile j + 10) pairs per expert

}  // namespace

// ------------------------------------------------------------------------------------------------ the pool
class Pool {
   public:
    Pool(int64_t mbox_ptr, int64_t max_routes, std::vector<int64_t> cpus, double spin_ms)
        : mbox_((uint8_t*) mbox_ptr), max_routes_((int) max_routes), lay_(mbox_layout((int) max_routes)),
          spin_ns_((uint64_t) (spin_ms * 1e6)) {
        if (!__builtin_cpu_supports("avx512f") || !__builtin_cpu_supports("avx512bw") ||
            !__builtin_cpu_supports("avx512vl") || !__builtin_cpu_supports("avx512dq"))
            throw std::runtime_error("edge0 CPU experts need AVX-512 F/BW/VL/DQ");
        hdr_ = (MboxHeader*) mbox_;
        tasks_ = (const Task*) (mbox_ + lay_.tasks);
        rid_ = (const int32_t*) (mbox_ + lay_.route_ids);
        rw_ = (const float*) (mbox_ + lay_.route_w);
        x_ = (const uint16_t*) (mbox_ + lay_.x);
        y_ = (uint16_t*) (mbox_ + lay_.y);
        H_.reset((float*) std::aligned_alloc(64, mbox_align((size_t) max_routes_ * FF * 4)));
        layers_.resize(256);
        for (auto& a : g1done_) a.store(0);
        const uint32_t s0 = __atomic_load_n(&hdr_->seq, __ATOMIC_ACQUIRE);
        claimed_.store(s0);
        ticket_.store((uint64_t) s0 << 32 | 0xFFFFFFFFull);
        start_seq_ = s0;
        for (size_t i = 0; i < cpus.size(); ++i) threads_.emplace_back([this, i, c = (int) cpus[i]] { worker(c); });
    }
    ~Pool() { stop(); }

    void stop() {
        stop_.store(true);
        for (auto& t : threads_)
            if (t.joinable()) t.join();
        threads_.clear();
    }

    void register_layer(int64_t layer, int64_t w13, int64_t w2, int64_t s13, int64_t s2, int64_t n) {
        if (layer < 0 || layer >= (int64_t) layers_.size()) throw std::out_of_range("layer");
        layers_[layer] = LayerPtrs{(const uint8_t*) w13, (const uint8_t*) w2, (const uint8_t*) s13,
                                   (const uint8_t*) s2, (int) n};
    }

    void set_frac(double f) { hdr_->frac = (float) f; }
    // task size: (gate, up) n-tile pairs per G1 task (1, 2, 5, 10) and w2 n-tiles per G2 task (1, 2, 4, 7, 14, 28);
    // takes effect at the next job
    void set_tiles(int64_t g1_pairs, int64_t g2_tiles) {
        if (g1_pairs < 1 || G1_PAIRS % g1_pairs || g2_tiles < 1 || NT2 % g2_tiles)
            throw std::invalid_argument("set_tiles: g1_pairs must divide 10 and g2_tiles 28");
        g1_pairs_ = (int) g1_pairs;
        g2_tiles_ = (int) g2_tiles;
    }
    double frac() const { return hdr_->frac; }
    void set_auto(bool on, double lo, double hi) {
        auto_ = on;
        auto_lo_ = lo;
        auto_hi_ = hi;
    }

    std::vector<double> stats() const {
        return {(double) jobs_.load(), (double) experts_.load(), (double) routes_.load(), (double) cpu_ns_.load() * 1e-9,
                (double) hdr_->calls, (double) hdr_->gpu_ns * 1e-9, (double) hdr_->wait_ns * 1e-9,
                (double) hdr_->timeouts, (double) hdr_->frac, (double) g2_waits_.load(), (double) auto_updates_.load()};
    }

    // Tests / microbenchmarks: run one job written by the host (tasks, routes, x already in the mailbox), from the
    // calling thread: ring, then wait for done.
    double host_ring_and_wait(int64_t layer, int64_t n_experts, int64_t n_routes, double timeout_s) {
        hdr_->layer = (uint32_t) layer;
        hdr_->n_experts = (uint32_t) n_experts;
        hdr_->n_routes = (uint32_t) n_routes;
        const uint32_t s = hdr_->seq + 1;
        const uint64_t t0 = now_ns();
        __atomic_store_n(&hdr_->seq, s, __ATOMIC_RELEASE);
        while (__atomic_load_n(&hdr_->done, __ATOMIC_ACQUIRE) != s) {
            _mm_pause();
            if (now_ns() - t0 > (uint64_t) (timeout_s * 1e9)) throw std::runtime_error("host job timed out");
        }
        return (now_ns() - t0) * 1e-9;
    }

   private:
    struct JobView {
        uint32_t s;
        int n_experts, n_g1, n_tasks, layer;
        int g1_pairs, g1_per, g2_tiles, g2_per;   // task size of this job
    };

    void worker(int cpu) {
        pin_thread(cpu);
        alignas(64) float xf[LAT];
        uint32_t last = start_seq_;
        uint64_t idle_since = now_ns();
        uint32_t spins = 0;
        while (!stop_.load(std::memory_order_relaxed)) {
            const uint32_t s = __atomic_load_n(&hdr_->seq, __ATOMIC_ACQUIRE);
            if (s == last) {
                _mm_pause();
                if ((++spins & 1023u) == 0 && now_ns() - idle_since > spin_ns_) {
                    timespec ts{0, 50000};
                    nanosleep(&ts, nullptr);
                }
                continue;
            }
            last = s;
            run_job(s, xf);
            idle_since = now_ns();
        }
    }

    void run_job(uint32_t s, float* xf) {
        JobView j{};
        uint32_t c = claimed_.load(std::memory_order_acquire);
        bool leader = false;
        while ((int32_t) (s - c) > 0) {
            if (claimed_.compare_exchange_weak(c, s, std::memory_order_acq_rel)) {
                leader = true;
                break;
            }
        }
        if (leader) {
            job_t0_ = now_ns();
            const int ne = (int) hdr_->n_experts;
            jver_.fetch_add(1, std::memory_order_acq_rel);          // seqlock: odd while j_ is being written
            const int g1p = g1_pairs_, g2t = g2_tiles_, g1n = G1_PAIRS / g1p, g2n = NT2 / g2t;
            j_ = JobView{s, ne, ne * g1n, ne * (g1n + g2n), (int) hdr_->layer, g1p, g1n, g2t, g2n};
            jver_.fetch_add(1, std::memory_order_acq_rel);
            for (int e = 0; e < ne && e < MAX_EXPERTS; ++e) g1done_[e].store(0, std::memory_order_relaxed);
            finished_.store(0, std::memory_order_relaxed);
            if (j_.n_tasks == 0) __atomic_store_n(&hdr_->done, s, __ATOMIC_RELEASE);
            ticket_.store((uint64_t) s << 32, std::memory_order_release);
        } else {
            for (;;) {   // wait until the leader of s published the job (or a newer job started: we are late)
                const uint32_t ts = (uint32_t) (ticket_.load(std::memory_order_acquire) >> 32);
                if (ts == s) break;
                if ((int32_t) (ts - s) > 0 || (int32_t) (c - s) > 0) return;
                _mm_pause();
                c = claimed_.load(std::memory_order_acquire);
            }
        }
        {   // published before the ticket (release / acquire); a worker late by a whole job may race the next
            // leader's write, so read it under the seqlock and drop out if it is not job s
            const uint32_t v1 = jver_.load(std::memory_order_acquire);
            j = j_;
            std::atomic_thread_fence(std::memory_order_acquire);
            if ((v1 & 1u) || jver_.load(std::memory_order_relaxed) != v1 || j.s != s) return;
        }
        const LayerPtrs& L = layers_[j.layer];
        for (;;) {
            uint64_t v = ticket_.load(std::memory_order_acquire);
            if ((uint32_t) (v >> 32) != s) return;
            const uint32_t idx = (uint32_t) v;
            if ((int) idx >= j.n_tasks) return;
            if (!ticket_.compare_exchange_weak(v, v + 1, std::memory_order_acq_rel)) continue;
            exec(j, L, (int) idx, xf);
            if (finished_.fetch_add(1, std::memory_order_acq_rel) + 1 == j.n_tasks) finish(j);
        }
    }

    void exec(const JobView& j, const LayerPtrs& L, int idx, float* xf) {
        if (idx < j.n_g1) {
            const int e = idx / j.g1_per, jj = idx % j.g1_per;
            const Task t = tasks_[e];
            for (int q = 0; q < j.g1_pairs; ++q) {   // pair-outer: a pair's 112 KB stay in L2 for every route
                const int pair = jj * j.g1_pairs + q;
                for (int i = t.off; i < t.off + t.n; ++i) {
                    bf16_row_to_f32(x_ + (size_t) rid_[i] * LAT, xf, LAT);
                    g1_route(L, t.slot, pair, t.g13, xf, H_.get() + (size_t) i * FF + pair * 64);
                }
            }
            g1done_[e].fetch_add(1, std::memory_order_release);
        } else {
            const int k = idx - j.n_g1;
            const int e = k / j.g2_per, jj = k % j.g2_per;
            const Task t = tasks_[e];
            if (g1done_[e].load(std::memory_order_acquire) != j.g1_per) {
                g2_waits_.fetch_add(1, std::memory_order_relaxed);
                while (g1done_[e].load(std::memory_order_acquire) != j.g1_per) _mm_pause();
            }
            for (int q = 0; q < j.g2_tiles; ++q) {
                const int nt = jj * j.g2_tiles + q;
                for (int i = t.off; i < t.off + t.n; ++i)
                    g2_route(L, t.slot, nt, t.g2, rw_[i], H_.get() + (size_t) i * FF,
                             y_ + (size_t) rid_[i] * LAT + nt * 64);
            }
        }
    }

    void finish(const JobView& j) {
        __atomic_store_n(&hdr_->done, j.s, __ATOMIC_RELEASE);
        const uint64_t dt = now_ns() - job_t0_;
        jobs_.fetch_add(1, std::memory_order_relaxed);
        experts_.fetch_add(j.n_experts, std::memory_order_relaxed);
        routes_.fetch_add(hdr_->n_routes, std::memory_order_relaxed);
        cpu_ns_.fetch_add(dt, std::memory_order_relaxed);
        if (auto_) control();
    }

    // EDGE0_CPU_FRAC=auto: move the CPU share towards the point where both sides of a call take the same time
    // (Strata's refined PCIe share). Runs on the worker that completed a job, at most every 0.5 s.
    void control() {
        const uint64_t t = now_ns();
        if (t - ctl_t_ < 500000000ull) return;
        const uint64_t calls = hdr_->calls, gns = hdr_->gpu_ns, jobs = jobs_.load(), cns = cpu_ns_.load();
        const uint64_t dc = calls - ctl_calls_, dj = jobs - ctl_jobs_;
        if (ctl_t_ != 0 && dc >= 200 && dj >= 200) {
            const double tg = (double) (gns - ctl_gns_) / dc, tc = (double) (cns - ctl_cns_) / dj;
            const double f = hdr_->frac;
            if (tg > 0 && tc > 0 && f > 0 && f < 1) {
                const double fs = (f / tc) / (f / tc + (1 - f) / tg);
                double nf = f + 0.5 * (fs - f);
                nf = nf < auto_lo_ ? auto_lo_ : (nf > auto_hi_ ? auto_hi_ : nf);
                hdr_->frac = (float) nf;
                auto_updates_.fetch_add(1);
            }
        }
        if (ctl_t_ == 0 || (dc >= 200 && dj >= 200)) {
            ctl_t_ = t;
            ctl_calls_ = calls;
            ctl_gns_ = gns;
            ctl_jobs_ = jobs;
            ctl_cns_ = cns;
        }
    }

    uint8_t* mbox_;
    int max_routes_;
    MboxLayout lay_;
    uint64_t spin_ns_;
    MboxHeader* hdr_ = nullptr;
    const Task* tasks_ = nullptr;
    const int32_t* rid_ = nullptr;
    const float* rw_ = nullptr;
    const uint16_t* x_ = nullptr;
    uint16_t* y_ = nullptr;
    struct Free {
        void operator()(float* p) const { std::free(p); }
    };
    std::unique_ptr<float, Free> H_;
    std::vector<LayerPtrs> layers_;
    uint32_t start_seq_ = 0;
    alignas(64) std::atomic<uint32_t> claimed_{0};
    alignas(64) std::atomic<uint64_t> ticket_{0};
    alignas(64) std::atomic<int> finished_{0};
    alignas(64) std::atomic<uint32_t> jver_{0};
    JobView j_{};
    uint64_t job_t0_ = 0;
    alignas(64) std::atomic<int> g1done_[MAX_EXPERTS];
    std::atomic<bool> stop_{false};
    std::vector<std::thread> threads_;
    std::atomic<uint64_t> jobs_{0}, experts_{0}, routes_{0}, cpu_ns_{0}, g2_waits_{0}, auto_updates_{0};
    int g1_pairs_ = 1, g2_tiles_ = 2;
    bool auto_ = false;
    double auto_lo_ = 0.05, auto_hi_ = 0.9;
    uint64_t ctl_t_ = 0, ctl_calls_ = 0, ctl_gns_ = 0, ctl_jobs_ = 0, ctl_cns_ = 0;
};

// ------------------------------------------------------------------------------------------------ test entry points
// One expert on the calling thread: x [n, LAT] bf16 (host), route weights [n] -> y [n, LAT] bf16. The four pointers
// are the expert's slot (Marlin record: w13, w2, s13, s2), g13 / g2 its Marlin-processed global scales.
torch::Tensor expert_forward(int64_t w13, int64_t w2, int64_t s13, int64_t s2, double g13, double g2,
                             torch::Tensor x, torch::Tensor wr) {
    TORCH_CHECK(x.device().is_cpu() && x.scalar_type() == torch::kBFloat16 && x.dim() == 2 && x.size(1) == LAT);
    TORCH_CHECK(wr.device().is_cpu() && wr.scalar_type() == torch::kFloat32 && wr.numel() == x.size(0));
    x = x.contiguous();
    const int n = (int) x.size(0);
    auto y = torch::empty({n, LAT}, x.options());
    LayerPtrs L{(const uint8_t*) w13, (const uint8_t*) w2, (const uint8_t*) s13, (const uint8_t*) s2, 1};
    alignas(64) float xf[LAT];
    alignas(64) float h[FF];
    const uint16_t* xs = (const uint16_t*) x.data_ptr();
    uint16_t* ys = (uint16_t*) y.data_ptr();
    const float* w = wr.data_ptr<float>();
    for (int i = 0; i < n; ++i) {
        bf16_row_to_f32(xs + (size_t) i * LAT, xf, LAT);
        for (int j = 0; j < G1_PAIRS; ++j) g1_route(L, 0, j, (float) g13, xf, h + j * 64);
        for (int nt = 0; nt < NT2; ++nt) g2_route(L, 0, nt, (float) g2, w[i], h, ys + (size_t) i * LAT + nt * 64);
    }
    return y;
}

// Debug: the G1 output (h = silu(gate) * up, bf16-exact) of one expert for x [n, LAT] -> [n, FF] fp32.
torch::Tensor expert_h(int64_t w13, int64_t s13, double g13, torch::Tensor x) {
    TORCH_CHECK(x.device().is_cpu() && x.scalar_type() == torch::kBFloat16 && x.dim() == 2 && x.size(1) == LAT);
    x = x.contiguous();
    const int n = (int) x.size(0);
    auto h = torch::empty({n, FF}, x.options().dtype(torch::kFloat32));
    LayerPtrs L{(const uint8_t*) w13, nullptr, (const uint8_t*) s13, nullptr, 1};
    alignas(64) float xf[LAT];
    for (int i = 0; i < n; ++i) {
        bf16_row_to_f32((const uint16_t*) x.data_ptr() + (size_t) i * LAT, xf, LAT);
        for (int j = 0; j < G1_PAIRS; ++j) g1_route(L, 0, j, (float) g13, xf, h.data_ptr<float>() + i * FF + j * 64);
    }
    return h;
}

std::vector<int64_t> layout(int64_t max_routes) {
    const MboxLayout m = mbox_layout((int) max_routes);
    return {(int64_t) m.tasks, (int64_t) m.route_ids, (int64_t) m.route_w, (int64_t) m.x, (int64_t) m.y,
            (int64_t) m.total, (int64_t) sizeof(MboxHeader)};
}

// CUDA side (cpu_experts_cuda.cu)
void cuda_dispatch(torch::Tensor ids, torch::Tensor w, torch::Tensor cold_map, torch::Tensor g13, torch::Tensor g2,
                   int64_t layer, int64_t min_cold, int64_t max_routes_e, int64_t max_experts, int64_t mbox_dev,
                   int64_t max_routes, torch::Tensor cold_map_eff, torch::Tensor cpu_sel, torch::Tensor state,
                   double frac);
void cuda_gather(torch::Tensor xl, torch::Tensor ids, torch::Tensor group, torch::Tensor out, torch::Tensor cpu_sel,
                 int64_t mbox_dev, int64_t max_routes, torch::Tensor state);
void cuda_wait(torch::Tensor state, int64_t mbox_dev, int64_t timeout_ns);

}  // namespace e0

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using e0::Pool;
    pybind11::class_<Pool>(m, "Pool")
        .def(pybind11::init<int64_t, int64_t, std::vector<int64_t>, double>())
        .def("register_layer", &Pool::register_layer)
        .def("set_frac", &Pool::set_frac)
        .def("set_tiles", &Pool::set_tiles)
        .def("frac", &Pool::frac)
        .def("set_auto", &Pool::set_auto)
        .def("stats", &Pool::stats)
        .def("host_ring_and_wait", &Pool::host_ring_and_wait, pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("stop", &Pool::stop, pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("expert_forward", &e0::expert_forward);
    m.def("expert_h", &e0::expert_h);
    m.def("layout", &e0::layout);
    m.def("dispatch", &e0::cuda_dispatch);
    m.def("gather", &e0::cuda_gather);
    m.def("wait", &e0::cuda_wait);
}
