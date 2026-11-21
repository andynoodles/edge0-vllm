// EDGE0_PLE_DEFER=1 (plugin/edge0_vllm/ple_defer.py): PLE rows read off the critical path.
//
// vLLM's host_file_gather PLE stages a step's rows in prepare_inputs behind a full stream sync: the GPU idles while
// the host reads ~64 random 160-byte rows from NVMe (1.3-1.6 ms per c1 verify step). Here prepare_inputs only
// enqueues the n-gram ids' D2H copy and an event, and returns; the step's CUDA graph is launched at once. A reader
// thread (this file) waits for the event, reads the rows (posix_fadvise WILLNEED for all rows, then preadv each,
// as vLLM's _FilePlane.read does), scatters them into pinned, device-mapped host memory and publishes the step's
// sequence number in a mapped flag. Inside the graph, the PLE embedding's forward (layer 1) runs ple_wait_copy
// (ple_wait.cu): spin until flag >= the step's sequence number, then copy the rows into the device staging buffer.
// Layer 0 and the launch overlap the read.
//
// Ordering: step s+1's ids D2H is enqueued after step s's graph on the same stream, so the reader writes the shared
// host buffers for s+1 only after step s's wait-copy kernel has consumed them.
#include <torch/extension.h>

#include <cuda_runtime.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace e0ple {

void cuda_wait_copy(torch::Tensor src, torch::Tensor dst, int64_t nbytes, int64_t flag_dev, torch::Tensor expect,
                    int64_t timeout_ns, int64_t stats_dev);
void cuda_post_ids(torch::Tensor ids, int64_t dst_dev, int64_t flag_dev, int64_t seq);

static double now_s() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

class Reader {
  public:
    // fds / bases: one per shard (row r lives in shard r / rows_per_shard at bases[shard] + (r % rows_per_shard) *
    // row_bytes). host_ids: pinned int64 [max_ids] (the D2H target). rows: pinned mapped uint8 [max_ids, row_bytes].
    // flag: pinned mapped uint64 (the published sequence number).
    Reader(std::vector<int64_t> fds, std::vector<int64_t> bases, int64_t rows_per_shard, int64_t row_bytes,
           int64_t n_rows, int64_t host_ids, int64_t rows, int64_t flag, int64_t max_ids, int64_t device,
           int64_t cpu, int64_t ids_flag)
        : fds_(std::move(fds)), bases_(std::move(bases)), rows_per_shard_(rows_per_shard), row_bytes_(row_bytes),
          n_rows_(n_rows), host_ids_((const int64_t*) host_ids), rows_((uint8_t*) rows),
          flag_((volatile uint64_t*) flag), max_ids_(max_ids), device_((int) device), cpu_((int) cpu),
          ids_flag_((volatile uint64_t*) ids_flag) {
        if (fds_.empty() || fds_.size() != bases_.size()) throw std::runtime_error("ple reader: bad shard list");
        tmp_.resize((size_t) max_ids_ * row_bytes_);
        order_.reserve(max_ids_);
        for (int i = 0; i < kEvents; ++i) {
            if (cudaSetDevice(device_) != cudaSuccess ||
                cudaEventCreateWithFlags(&events_[i], cudaEventDisableTiming) != cudaSuccess)
                throw std::runtime_error("ple reader: cudaEventCreate failed");
        }
        worker_ = std::thread([this] { run(); });
    }

    ~Reader() { stop(); }

    // Main thread, inside prepare_inputs, after the ids' D2H copy was enqueued on `stream`.
    void submit(int64_t stream, int64_t num_ids, int64_t seq) {
        if (num_ids > max_ids_) throw std::runtime_error("ple reader: step has more ids than the staging buffer");
        std::unique_lock<std::mutex> lk(mu_);
        cv_space_.wait(lk, [this] { return (int) q_.size() < kEvents - 2 || stop_; });
        const int slot = (int) (seq % kEvents);
        if (ids_flag_ == nullptr && cudaEventRecord(events_[slot], (cudaStream_t) stream) != cudaSuccess)
            throw std::runtime_error("ple reader: cudaEventRecord failed");
        q_.push_back(Job{slot, num_ids, (uint64_t) seq, now_s()});
        cv_.notify_one();
    }

    // jobs, ids, unique rows, read seconds, event-wait seconds, submit->publish seconds, errors, max queue
    std::vector<double> stats() {
        return {(double) jobs_.load(), (double) ids_.load(), (double) uniq_.load(), read_ns_.load() * 1e-9,
                evwait_ns_.load() * 1e-9, lag_ns_.load() * 1e-9, (double) errors_.load(), (double) maxq_.load()};
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (stop_) return;
            stop_ = true;
        }
        stop_now_.store(true);
        cv_.notify_all();
        cv_space_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

  private:
    static constexpr int kEvents = 16;
    struct Job {
        int slot;
        int64_t num_ids;
        uint64_t seq;
        double t_submit;
    };

    void run() {
        cudaSetDevice(device_);
        if (cpu_ >= 0) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpu_, &set);
            pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        }
        pthread_setname_np(pthread_self(), "edge0-ple");
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this] { return !q_.empty() || stop_; });
                if (q_.empty()) return;
                j = q_.front();
                maxq_.store(std::max<int64_t>(maxq_.load(), (int64_t) q_.size()));
            }
            const double t0 = now_s();
            if (ids_flag_ != nullptr) {
                // doorbell: the GPU's post_ids kernel wrote the ids into mapped memory, then this flag; spin on
                // plain memory (no CUDA driver call from this thread while the main thread launches work)
                uint32_t n = 0;
                while ((int64_t) __atomic_load_n((const uint64_t*) ids_flag_, __ATOMIC_ACQUIRE) < (int64_t) j.seq) {
                    __builtin_ia32_pause();
                    if ((++n & 1023) == 0 && stop_now_.load(std::memory_order_relaxed)) return;
                }
            } else {
                cudaEventSynchronize(events_[j.slot]);
            }
            const double t1 = now_s();
            read_step(j.num_ids);
            const double t2 = now_s();
            std::atomic_thread_fence(std::memory_order_release);
            __atomic_store_n((uint64_t*) flag_, j.seq, __ATOMIC_RELEASE);
            {
                std::lock_guard<std::mutex> lk(mu_);
                q_.pop_front();
            }
            cv_space_.notify_all();
            jobs_ += 1;
            evwait_ns_ += (int64_t) ((t1 - t0) * 1e9);
            read_ns_ += (int64_t) ((t2 - t1) * 1e9);
            lag_ns_ += (int64_t) ((t2 - j.t_submit) * 1e9);
        }
    }

    void read_step(int64_t n) {
        if (n <= 0) return;
        order_.clear();
        for (int64_t i = 0; i < n; ++i) order_.emplace_back(host_ids_[i], (int32_t) i);
        std::sort(order_.begin(), order_.end());
        // unique rows -> tmp_[u]; inv_[i] = u
        inv_.resize(n);
        uniq_rows_.clear();
        for (int64_t k = 0; k < n; ++k) {
            if (k == 0 || order_[k].first != order_[k - 1].first) uniq_rows_.push_back(order_[k].first);
            inv_[order_[k].second] = (int32_t) uniq_rows_.size() - 1;
        }
        const int64_t u = (int64_t) uniq_rows_.size();
        ids_ += n;                      // both counters at once: a log snapshot never splits a (prefill) step
        uniq_ += u;
        const long page = sysconf(_SC_PAGESIZE);
        // queue every page read first (the drive serves them in parallel), then read each row
        for (int64_t k = 0; k < u; ++k) {
            const int64_t r = uniq_rows_[k];
            if (r < 0 || r >= n_rows_) continue;
            int fd;
            int64_t off;
            locate(r, &fd, &off);
            const int64_t lo = off / page * page;
            const int64_t hi = (off + row_bytes_ + page - 1) / page * page;
            posix_fadvise(fd, lo, hi - lo, POSIX_FADV_WILLNEED);
        }
        for (int64_t k = 0; k < u; ++k) {
            const int64_t r = uniq_rows_[k];
            uint8_t* dst = tmp_.data() + (size_t) k * row_bytes_;
            if (r < 0 || r >= n_rows_) {
                memset(dst, 0, row_bytes_);
                errors_ += 1;
                continue;
            }
            int fd;
            int64_t off;
            locate(r, &fd, &off);
            ssize_t got = pread(fd, dst, row_bytes_, off);
            if (got != row_bytes_) {
                memset(dst, 0, row_bytes_);
                errors_ += 1;
            }
        }
        for (int64_t i = 0; i < n; ++i)
            memcpy(rows_ + (size_t) i * row_bytes_, tmp_.data() + (size_t) inv_[i] * row_bytes_, row_bytes_);
    }

    void locate(int64_t r, int* fd, int64_t* off) const {
        const int64_t shard = r / rows_per_shard_;
        *fd = (int) fds_[shard];
        *off = bases_[shard] + (r - shard * rows_per_shard_) * row_bytes_;
    }

    std::vector<int64_t> fds_, bases_;
    int64_t rows_per_shard_, row_bytes_, n_rows_;
    const int64_t* host_ids_;
    uint8_t* rows_;
    volatile uint64_t* flag_;
    int64_t max_ids_;
    int device_, cpu_;
    volatile uint64_t* ids_flag_;
    std::atomic<bool> stop_now_{false};
    std::vector<uint8_t> tmp_;
    std::vector<std::pair<int64_t, int32_t>> order_;
    std::vector<int32_t> inv_;
    std::vector<int64_t> uniq_rows_;
    cudaEvent_t events_[kEvents];
    std::thread worker_;
    std::mutex mu_;
    std::condition_variable cv_, cv_space_;
    std::deque<Job> q_;
    bool stop_ = false;
    std::atomic<int64_t> jobs_{0}, ids_{0}, uniq_{0}, read_ns_{0}, evwait_ns_{0}, lag_ns_{0}, errors_{0}, maxq_{0};
};

}  // namespace e0ple

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using e0ple::Reader;
    pybind11::class_<Reader>(m, "Reader")
        .def(pybind11::init<std::vector<int64_t>, std::vector<int64_t>, int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t, int64_t, int64_t, int64_t>())
        .def("submit", &Reader::submit, pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("stats", &Reader::stats)
        .def("stop", &Reader::stop, pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("wait_copy", &e0ple::cuda_wait_copy);
    m.def("post_ids", &e0ple::cuda_post_ids);
}
