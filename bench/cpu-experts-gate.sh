#!/usr/bin/env bash
# EDGE0_CPU_EXPERTS gate (docs/CPU-EXPERTS.md section 4). Runs ON the 5090 host with the GPU lock held and no server
# running: GPU unit tests (CPU kernel vs Marlin vs fp32 reference, pool, captured-graph op) and the split
# microbenchmark (tools/cpu_experts/bench_cpu_split.py). Log: logs/cpu-experts-gate.log, JSON results/cpu-experts/.
#
#   bench/cpu-experts-gate.sh [tests|bench|all] [extra bench_cpu_split.py args...]
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WHAT=${1:-all}; shift || true
LOG=$REPO/logs/cpu-experts-gate.log
mkdir -p "$REPO/logs" "$REPO/results/cpu-experts"
say() { echo "# $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
RUN=(docker run --rm --name edge0-cpuexp-gpu --gpus all --ipc=host --network host --ulimit memlock=-1
     -v /home/andychen/models:/models -v "$REPO:/workspace/edge0" -v /home/andychen/.cache/edge0-vllm:/root/.cache
     -e PYTHONPATH=/workspace/edge0/plugin -e EDGE0_CKPT=/models/Edge0-145B-vllm --entrypoint bash edge0-vllm:dev -c)
if [ "$WHAT" = tests ] || [ "$WHAT" = all ]; then
  say "unit tests"
  "${RUN[@]}" "cd /workspace/edge0/plugin && python -m pytest -q -s -p no:cacheprovider tests/test_cpu_experts_host.py tests/test_cpu_experts.py" 2>&1 \
    | grep -vE "^W1|cpp_extension" | tee -a "$LOG" | tail -60
fi
if [ "$WHAT" = bench ] || [ "$WHAT" = all ]; then
  tag=$(date +%H%M)
  say "split benchmark $*"
  "${RUN[@]}" "python /workspace/edge0/tools/cpu_experts/bench_cpu_split.py --out /workspace/edge0/results/cpu-experts/split-$tag.json $*" 2>&1 \
    | grep -vE "^W1|cpp_extension" | tee -a "$LOG" | tail -200
fi
say "GATE_DONE"
