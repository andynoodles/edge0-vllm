#!/usr/bin/env bash
# EDGE0_TIER GPU unit tests (plugin/tests/test_tier.py + the pool / CPU-experts regressions). Runs ON the 5090 host
# with the GPU lock held and no server running. Log: logs/fc-gate.log.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG=$REPO/logs/fc-gate.log
mkdir -p "$REPO/logs"
echo "# $(date +%H:%M:%S) tier GPU tests" | tee -a "$LOG"
docker run --rm --name edge0-fc-gpu --gpus all --ipc=host --network host --ulimit memlock=-1 \
  -v /home/andychen/models:/models -v "$REPO:/workspace/edge0" -v /home/andychen/.cache/edge0-vllm:/root/.cache \
  -e PYTHONPATH=/workspace/edge0/plugin -e EDGE0_CKPT=/models/Edge0-145B-vllm --entrypoint bash edge0-vllm:dev -c \
  "cd /workspace/edge0/plugin && python -m pytest -q -s -p no:cacheprovider tests/test_tier.py ${*:-tests/test_gpu_experts.py tests/test_cpu_experts.py}" 2>&1 \
  | grep -vE "^W1|cpp_extension" | tee -a "$LOG" | tail -40
echo "# $(date +%H:%M:%S) FC_GATE_DONE" | tee -a "$LOG"
