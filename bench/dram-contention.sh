#!/usr/bin/env bash
# DRAM bandwidth shared by the GPU's PCIe reads of cold experts and CPU expert compute (docs/PROFILE.md section 5).
# Runs ON the 5090 host with the GPU lock held and no server running (needs ~14 GiB free RAM). ~4 minutes.
#
#   bench/dram-contention.sh            -> logs/dram-contention.log
#
# 1. CPU DRAM read bandwidth alone (AVX-512 streaming loads, 1..32 threads)
# 2. GPU reads of pinned host memory alone: SM zero-copy (what the cold Marlin call does) and copy engine
# 3. both at once (GPU SM zero-copy + CPU read with 4/8/16/32 threads): what each side keeps
# 4. Strata's NVFP4 AVX-512 expert kernel (tools/cpu_experts/cpu_expert_bench.cpp) on Edge0-sized experts, alone and
#    while the GPU reads over PCIe
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG=$REPO/logs/dram-contention.log
S=${STRATA_SRC:-$HOME/work/edge0/edge0-strata}
BIN=/tmp/cpu_expert_bench
IMAGE=${IMAGE:-edge0-vllm:dev}
G=$REPO/logs/.dram-gpu.txt
C=$REPO/logs/.dram-cpu.txt
say() { echo "# $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
g++ -O3 -march=native -fopenmp -std=c++17 -I"$S/include" -I"$S/third_party/ggml" \
  "$REPO/tools/cpu_experts/cpu_expert_bench.cpp" "$S/src/kernels/cpu/nvfp4_avx512.cpp" -o "$BIN" || exit 1

window() {  # window <gpu file> <cpu file>: mean CPU GB/s and mean GPU GB/s over the CPU samples' time window
  python3 -I - "$1" "$2" <<'PY'
import sys
g = [l.split() for l in open(sys.argv[1]) if l[:1].isdigit() and " gpu-" in l]
c = [l.split() for l in open(sys.argv[2]) if l[:1].isdigit()]
if not c:
    print("no cpu samples"); sys.exit()
t0, t1 = float(c[0][0]) - 0.5, float(c[-1][0])
gin = [float(x[2]) for x in g if t0 <= float(x[0]) <= t1]
cpu = [float(x[-2]) for x in c]
print(f"CPU {sum(cpu) / len(cpu):.1f} GB/s ({c[0][1]}), GPU {sum(gin) / max(1, len(gin)):.1f} GB/s over the same "
      f"{t1 - t0:.1f} s ({len(gin)} GPU samples)")
PY
}

gpu() {   # gpu <mode> <seconds> : GPU read loop in the image
  docker run --rm --name edge0-dram-gpu --gpus all --ipc=host --ulimit memlock=-1 \
    -v "$REPO:/workspace/edge0" -e PYTHONPATH=/workspace/edge0/plugin --entrypoint python "$IMAGE" \
    -I /workspace/edge0/tools/cpu_experts/h2d_loop.py "$1" 4 "$2" 2>&1 | grep -E "gpu-|Error|error"
}

say "free: $(free -g | awk '/Mem:/{print $7}') GiB available; $(lscpu | grep 'Model name' | sed 's/  */ /g')"
say "1. CPU read alone"
for t in 1 4 8 16 32; do
  echo "threads $t: $($BIN read 8 "$t" 3 | tail -1)" | tee -a "$LOG"
done
say "2. GPU alone"
gpu sm 6 | tail -1 | tee -a "$LOG"
gpu ce 6 | tail -1 | tee -a "$LOG"
say "3. GPU SM zero-copy + CPU read concurrently"
for t in 4 8 16 32; do
  gpu sm 14 >"$G" &
  g=$!
  sleep 7       # container start + warm-up
  $BIN read 8 "$t" 4 >"$C"
  wait "$g"
  echo "threads $t read: $(window "$G" "$C")" | tee -a "$LOG"
done
say "4. CPU NVFP4 expert kernel (Strata nvfp4_avx512), 8 GiB pool, 24 experts per call"
for t in 16 32; do
  for nt in 1 2 4; do
    echo "threads $t tokens/expert $nt: $($BIN expert 8 "$t" 24 "$nt" 3 | tail -1)" | tee -a "$LOG"
  done
done
say "4b. the same while the GPU reads over PCIe (SM zero-copy)"
for t in 16 32; do
  gpu sm 14 >"$G" &
  g=$!
  sleep 7
  $BIN expert 8 "$t" 24 1 4 >"$C"
  wait "$g"
  echo "threads $t expert x1 token: $(window "$G" "$C") | $(tail -1 "$C")" | tee -a "$LOG"
done
rm -f "$G" "$C"
say "DRAM_DONE"
